# -*- coding: utf-8 -*-
# 文件：waypoint_pub/waypoint_pub/env/uav_ros.py

import numpy as np
import rclpy
from pathlib import Path as Filepath
from datetime import datetime
import pandas as pd

from .agent_init import AgentInit


class UAV_ROS(AgentInit):
    """
    UAV ROS Control Class for state acquisition, dynamics calculation, control command publishing, and data logging
    (ROS2 版：把 rospy.* 全部替换为 rclpy/node 接口，保持原 API)
    """

    def __init__(self, m: float = 0.72,
                 g: float = 9.8,
                 kt: float = 1e-3,
                 dt: float = 0.02,
                 use_gazebo: bool = True,
                 control_name: str = "",
                 ):
        super().__init__(dt=dt, use_gazebo=use_gazebo)

        self.m = m
        self.g = g
        self.kt = kt

        self.x = 0.0; self.y = 0.0; self.z = 0.0
        self.vx = 0.0; self.vy = 0.0; self.vz = 0.0
        self.phi = 0.0; self.theta = 0.0; self.psi = 0.0
        self.p = 0.0; self.q = 0.0; self.r = 0.0

        self.dt = dt
        self.n = 0
        self.time = 0.0

        self.is_record_ref = False
        self.is_record_obs = False

        self.ref_data_log = []
        self.obs_data_log = []
        self.data_log = []

        self.control_name = control_name

        # 控制参数
        self.throttle = self.m * self.g
        self.phi_d = 0.0
        self.theta_d = 0.0
        self.psi_d = 0.0

    # -------------------- State acquisition methods --------------------
    def uav_state_callback(self) -> np.ndarray:
        return self.uav_states

    def uav_pos(self) -> np.ndarray:
        return self.uav_states[0:3]

    def uav_vel(self) -> np.ndarray:
        return self.uav_states[3:6]

    def uav_att(self) -> np.ndarray:
        return self.uav_states[6:9]

    def uav_pqr(self) -> np.ndarray:
        return self.uav_states[9:12]

    # -------------------- Dynamics & kinematics methods --------------------
    def T_pqr_2_dot_att(self) -> np.ndarray:
        [self.phi, self.theta, self.psi] = self.uav_att()
        assert all(isinstance(angle, (int, float, np.floating)) for angle in [self.phi, self.theta]), \
            "Attitude angles must be scalars"
        return np.array([
            [1, np.sin(self.phi) * np.tan(self.theta), np.cos(self.phi) * np.tan(self.theta)],
            [0, np.cos(self.phi), -np.sin(self.phi)],
            [0, np.sin(self.phi) / np.cos(self.theta), np.cos(self.phi) / np.cos(self.theta)]
        ])

    def uav_dot_att(self) -> np.ndarray:
        return np.dot(self.T_pqr_2_dot_att(), self.uav_pqr())

    def A(self) -> np.ndarray:
        [self.phi, self.theta, self.psi] = self.uav_att()
        thrust_orientation = np.array([
            np.cos(self.phi) * np.cos(self.psi) * np.sin(self.theta) + np.sin(self.phi) * np.sin(self.psi),
            np.cos(self.phi) * np.sin(self.psi) * np.sin(self.theta) - np.sin(self.phi) * np.cos(self.psi),
            np.cos(self.phi) * np.cos(self.theta)
        ])
        return (self.throttle / self.m) * thrust_orientation - np.array([0.0, 0.0, self.g])

    def u_to_angle_dir(self, uo: np.ndarray, is_idea=True):
        [self.phi, self.theta, self.psi] = self.uav_att()

        u1 = uo + np.array([0., 0., self.g])  # gravity comp
        uf = self.m * np.linalg.norm(u1)

        if uo[2] + self.g <= 0:
            self.logger.warn("Vertical acceleration <= -g, keep current attitude")
            return (self.phi, self.theta, 0.0)

        if uf == 0:
            self.logger.warn("Total thrust == 0, keep current attitude")
            return (self.phi, self.theta, 0.0)

        sin_phi_dir = np.clip(
            (self.m * (uo[0] * np.sin(self.psi) - uo[1] * np.cos(self.psi))) / uf,
            -1.0, 1.0
        )
        phi_d = np.arcsin(sin_phi_dir)

        tan_theta_dir = (uo[0] * np.cos(self.psi) + uo[1] * np.sin(self.psi)) / u1[2]
        theta_d = np.arctan(tan_theta_dir)

        return (phi_d, theta_d, uf)

    # -------------------- Logging and shutdown handling --------------------
    def data_record(self, sim_t: float, thrust: float = 0.,
                    ref_state: np.ndarray = np.zeros(9),
                    obs_state: np.ndarray = np.zeros(3)) -> None:
        self.data_log.append({
            "timestamp": sim_t,
            "x": self.uav_states[0], "y": self.uav_states[1], "z": self.uav_states[2],
            "vx": self.uav_states[3], "vy": self.uav_states[4], "vz": self.uav_states[5],
            "phi": self.uav_states[6], "theta": self.uav_states[7], "psi": self.uav_states[8],
            "p": self.uav_states[9], "q": self.uav_states[10], "r": self.uav_states[11],
            "thrust": thrust
        })

        if self.is_record_ref:
            if ref_state.ndim != 1 or ref_state.size != 9:
                raise ValueError(f"Reference state dimension error, expected (9,), got {ref_state.shape}")
            self.ref_data_log.append({
                "timestamp": sim_t,
                "ref_x": ref_state[0], "ref_y": ref_state[1], "ref_z": ref_state[2],
                "ref_vx": ref_state[3], "ref_vy": ref_state[4], "ref_vz": ref_state[5],
                "ref_ax": ref_state[6], "ref_ay": ref_state[7], "ref_az": ref_state[8]
            })

        if self.is_record_obs:
            if obs_state.ndim != 1 or obs_state.size != 3:
                raise ValueError(f"Observer state dimension error, expected (3,), got {obs_state.shape}")
            self.obs_data_log.append({
                "timestamp": sim_t,
                "obs_dx": obs_state[0], "obs_dy": obs_state[1], "obs_dz": obs_state[2]
            })

    def shutdown_handler(self):
        try:
            if len(self.data_log) == 0:
                self.logger.info("No data to save")
                return

            if not isinstance(self.control_name, str) or self.control_name.strip() == "":
                self.logger.error("Controller name is invalid or empty")
                return

            save_choice = input("\nData recording completed. Enter '1' to save data, '0' to discard: ")
            while save_choice not in ['0', '1']:
                save_choice = input("Invalid input. Please enter '1' to save or '0' to discard: ")

            if save_choice == '1':
                script_dir = Filepath(__file__).parent
                data_dir = script_dir.parent / "data"
                data_dir.mkdir(exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                base_name = f"{timestamp}_{self.control_name}"

                file_paths = []

                uav_path = data_dir / f"{base_name}_uav_data.csv"
                pd.DataFrame(self.data_log).to_csv(uav_path, index=False)
                file_paths.append(uav_path)

                if len(self.ref_data_log) != 0:
                    ref_path = data_dir / f"{base_name}_ref_data.csv"
                    pd.DataFrame(self.ref_data_log).to_csv(ref_path, index=False)
                    file_paths.append(ref_path)

                if len(self.obs_data_log) != 0:
                    obs_path = data_dir / f"{base_name}_obs_data.csv"
                    pd.DataFrame(self.obs_data_log).to_csv(obs_path, index=False)
                    file_paths.append(obs_path)

                self.logger.info(f"Data saved to: {[str(p) for p in file_paths]}")

                zip_choice = input("Do you want to compress these files into a zip? (0/1): ").strip().upper()
                while zip_choice not in ['0', '1']:
                    zip_choice = input("Invalid input. Please enter '1' to compress or '0' to skip: ").strip().upper()

                if zip_choice == '1':
                    zip_path = data_dir / f"{base_name}_data.zip"
                    try:
                        import zipfile
                        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                            for file in file_paths:
                                if file.exists():
                                    zf.write(file, arcname=file.name)
                        self.logger.info(f"Successfully compressed to: {zip_path}")
                        for file in file_paths:
                            file.unlink(missing_ok=True)
                    except Exception as e:
                        self.logger.error(f"Failed to create zip file: {str(e)}")
            else:
                self.logger.info("Data discarded")

        except Exception as e:
            self.logger.error(f"Error during shutdown handling: {str(e)}")

    def reach_target_point_by_pid(self):
        self.logger.info(f"Moving to target position: {self.target_xyzYaw} by pid")

    # -------------------- 主循环（保持与 ROS1 run() 兼容） --------------------
    def run(self, attitude_controller, simulation_time: float):
        """
        :param attitude_controller: 返回 (phi_d, theta_d, psi_d, thrust)
        :param simulation_time: 仿真时长(s)
        """
        self.initialize_system()
        self.logger.info(f"Control loop started at {1/self.dt:.1f}Hz")
        self.logger.info("Starting position guidance...")

        self.reach_target_position()
        self.logger.info("Waiting for 1 second...")
        for _ in range(100):
            self.pub_position_keep()
            self.rate.sleep()

        self.logger.info("Switching to attitude control...")

        self.is_record_ref = True
        self.is_show_rviz = True

        t0 = self.node.get_clock().now()
        sim_t = 0.0

        while rclpy.ok() and sim_t < simulation_time:
            # 让订阅/服务回调得以执行
            rclpy.spin_once(self.node, timeout_sec=0.0)

            phi_d, theta_d, psi_d, thrust = attitude_controller()
            self.publish_attitude_command(phi_d, theta_d, psi_d, thrust)

            now = self.node.get_clock().now()
            sim_t = (now - t0).nanoseconds * 1e-9

            self.data_record(sim_t=sim_t, ref_state=np.zeros(9))
            self.rate.sleep()

        self.is_show_rviz = False
        self.target_xyzYaw = (0.0, 0.0, 0.5, 0.0)
        self.reach_target_position()
