# -*- coding: utf-8 -*-
# 文件：waypoint_pub/waypoint_pub/env/agent_init.py

from typing import Tuple, Optional
import time
import atexit
import numpy as np
import pandas as pd
from pathlib import Path as Filepath
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from mavros_msgs.msg import State, PositionTarget, AttitudeTarget
from mavros_msgs.srv import CommandBool, SetMode

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import BatteryState

# -----------------------------------------------------------------------------
# 欧拉角/四元数转换
# -----------------------------------------------------------------------------
def quaternion_from_euler(roll: float, pitch: float, yaw: float):
    cr = np.cos(roll * 0.5);  sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5); sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5);   sy = np.sin(yaw * 0.5)
    # XYZ (sxyz) 与 ROS 常用的 ENU 对齐
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    w = cr * cp * cy + sr * sp * sy
    return [x, y, z, w]

def euler_from_quaternion(q):
    x, y, z, w = q
    # roll
    t0 = +2.0*(w*x + y*z)
    t1 = +1.0 - 2.0*(x*x + y*y)
    roll = np.arctan2(t0, t1)
    # pitch
    t2 = +2.0*(w*y - z*x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = np.arcsin(t2)
    # yaw
    t3 = +2.0*(w*z + x*y)
    t4 = +1.0 - 2.0*(y*y + z*z)
    yaw = np.arctan2(t3, t4)
    return (roll, pitch, yaw)

# -----------------------------------------------------------------------------
# 简易 Rate，兼容 rospy.Rate.sleep()
# -----------------------------------------------------------------------------
class _SimpleRate:
    def __init__(self, hz: float):
        self._period = 1.0 / float(hz)
        self._next = time.monotonic() + self._period
    def sleep(self):
        now = time.monotonic()
        if self._next > now:
            time.sleep(self._next - now)
        self._next += self._period

# -----------------------------------------------------------------------------
# AgentInit (ROS2 版)
# -----------------------------------------------------------------------------
class AgentInit:
    def __init__(self, dt: float = 0.01, use_gazebo: bool = True):
        # 基础参数
        self.dt = float(dt)
        self.use_gazebo = use_gazebo
        # thrust 映射保持你原始比例（仿真/实机不同）
        self.thrust_scale = (0.704 / 1.5 / 9.8) if use_gazebo else (0.41 / 0.72 / 9.8)

        # 状态标志
        self.is_show_rviz = False

        # 控制参数
        yaw = 0.0
        self.target_xyzYaw = (0.0, 0.0, 1.0, yaw)
        self.pos_tolerance = 0.2
        self.vel_tolerance = 0.2

        # 实时状态存储
        self.current_state = State()
        self.current_pose = PoseStamped()
        self.target_pose = PoseStamped()
        self.uav_odom = Odometry()
        self.uav_states = np.zeros(12)    # [x y z vx vy vz roll pitch yaw p q r]
        self.battery_voltage = 0.0
        self.targetPosition = PositionTarget()

        # 轨迹可视化
        self.trajectory = Path()
        self.trajectory.header.frame_id = "map"
        self.trajectory_queue = []

        # 数据记录
        self.data_log = []
        self.prev_velocity = np.zeros(3)
        self.prev_time: Optional[float] = None

        # 创建 Node
        self.node: Node = rclpy.create_node("px4_attitude_controller")
        self.logger = self.node.get_logger()
        self.rate = _SimpleRate(1.0 / self.dt)
        self.t0 = self.node.get_clock().now().nanoseconds * 1e-9
        self.current_time = self.node.get_clock().now()

        # 命名空间组：与原版保持逻辑
        self.group = "" if use_gazebo else "uav0/"

        # QoS（MAVROS 常用 BestEffort）
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # 订阅器
        self.state_sub = self.node.create_subscription(
            State, self.group + "mavros/state", self.state_cb, 10
        )
        self.odom_sub = self.node.create_subscription(
            Odometry, self.group + "mavros/local_position/odom", self.uav_odom_cb, sensor_qos
        )
        self.battery_sub = self.node.create_subscription(
            BatteryState, self.group + "mavros/battery", self.uav_battery_cb, sensor_qos
        )
        # self.uav_rate_sub 可根据需要恢复 IMU 订阅

        # 发布器
        self.att_cmd_pub = self.node.create_publisher(
            AttitudeTarget, self.group + "mavros/setpoint_raw/attitude", 10
        )
        self.pos_cmd_pub = self.node.create_publisher(
            PoseStamped, self.group + "mavros/setpoint_position/local", 10
        )
        self.trajectory_pub = self.node.create_publisher(
            Path, self.group + "reference_path", 10
        )
        self.positionTarget_pub = self.node.create_publisher(
            PositionTarget, self.group + "mavros/setpoint_raw/local", 10
        )

        # 服务客户端（异步）
        self.cli_arm = self.node.create_client(CommandBool, self.group + "mavros/cmd/arming")
        self.cli_mode = self.node.create_client(SetMode, self.group + "mavros/set_mode")

        # 控制指令初始化
        self.att_target = AttitudeTarget()
        self.att_target.type_mask = (
            AttitudeTarget.IGNORE_ROLL_RATE |
            AttitudeTarget.IGNORE_PITCH_RATE |
            AttitudeTarget.IGNORE_YAW_RATE
        )

        # 限速参数
        self.max_linear_vel = 1.0   # m/s
        self.max_angular_vel = 0.5  # rad/s

        # 关闭时保存数据
        atexit.register(self.shutdown_handler)

    # -------------------- Callbacks --------------------
    def state_cb(self, msg: State):
        self.current_state = msg

    def uav_battery_cb(self, msg: BatteryState):
        self.battery_voltage = msg.voltage

    def uav_odom_cb(self, msg: Odometry):
        self.uav_odom = msg
        self.current_pose.header = msg.header
        self.current_pose.pose = msg.pose.pose

        p = msg.pose.pose.position
        v = msg.twist.twist.linear
        w = msg.twist.twist.angular
        q = msg.pose.pose.orientation
        roll, pitch, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])

        self.uav_states = np.array([
            p.x, p.y, p.z,
            v.x, v.y, v.z,
            roll, pitch, yaw,
            w.x, w.y, w.z
        ])

        if self.is_show_rviz:
            self.rviz_show_path_limit()

    # -------------------- RViz Path --------------------
    def rviz_show_path(self):
        _odom = Odometry()
        _odom.header.stamp = self.node.get_clock().now().to_msg()
        _odom.header.frame_id = 'map'
        _odom.child_frame_id = 'base_link'
        _odom.pose.pose.position = self.current_pose.pose.position
        # 这里不再发布 _odom（如需可添加 odom_pub）

    def rviz_show_path_limit(self):
        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = self.node.get_clock().now().to_msg()
        pose_stamped.header.frame_id = "map"
        pose_stamped.pose = self.current_pose.pose

        self.trajectory_queue.append(pose_stamped)
        if len(self.trajectory_queue) > 1000:
            self.trajectory_queue.pop(0)

        self.trajectory.poses = self.trajectory_queue
        self.trajectory.header.stamp = self.node.get_clock().now().to_msg()
        self.trajectory_pub.publish(self.trajectory)

    # -------------------- Helpers --------------------
    def set_target_position(self, xyzYaw):
        pose = PoseStamped()
        pose.header.stamp = self.node.get_clock().now().to_msg()
        pose.header.frame_id = "map"
        pose.pose.position.x = float(xyzYaw[0])
        pose.pose.position.y = float(xyzYaw[1])
        pose.pose.position.z = float(xyzYaw[2])
        yaw = float(xyzYaw[3])
        q = quaternion_from_euler(0.0, 0.0, yaw)
        pose.pose.orientation.x = q[0]
        pose.pose.orientation.y = q[1]
        pose.pose.orientation.z = q[2]
        pose.pose.orientation.w = q[3]
        return pose

    # -------------------- Arming / Mode --------------------
    def _spin_until_future(self, future, timeout: float = 2.0) -> bool:
        """等待异步服务返回"""
        start = time.monotonic()
        while rclpy.ok() and (time.monotonic() - start) < timeout:
            rclpy.spin_once(self.node, timeout_sec=0.05)
            if future.done():
                return True
        return future.done()

    def arm_safe(self, max_attempts=50, wait_sec=2.0) -> bool:
        self.logger.info("Attempting to arm...")
        req = CommandBool.Request()
        req.value = True

        for attempt in range(max_attempts):
            if not rclpy.ok():
                self.logger.warn("ROS is shutting down - abort arming")
                return False

            if self.current_state.armed:
                self.logger.info("Drone already armed")
                return True

            try:
                if not self.cli_arm.service_is_ready():
                    self.logger.warn("Arming service not ready, wait...")
                    self.cli_arm.wait_for_service(timeout_sec=1.0)

                future = self.cli_arm.call_async(req)
                ok = self._spin_until_future(future, timeout=2.0)
                if not ok or not future.result() or not future.result().success:
                    self.logger.warn(f"Arm attempt {attempt+1} failed (service)")
                    self.pub_position_keep()
                    continue

                # 等待状态刷新
                t_end = time.monotonic() + wait_sec
                while rclpy.ok() and time.monotonic() < t_end:
                    rclpy.spin_once(self.node, timeout_sec=0.05)
                    self.rate.sleep()

                if self.current_state.armed:
                    self.logger.info("Drone armed successfully")
                    return True

                # 维持 setpoint 流再重试
                for _ in range(50):
                    self.pub_position_keep()
                    self.rate.sleep()
                self.logger.warn(f"Arm attempt {attempt+1} failed (status not updated)")

            except Exception as e:
                self.logger.warn(f"Arm attempt {attempt+1} exception: {e}")

        self.logger.error("Max arm attempts reached - please check connection")
        return False

    def set_offboard_mode(self, max_attempts=10) -> bool:
        self.logger.info("Attempting to switch to OFFBOARD mode...")
        req = SetMode.Request()
        req.custom_mode = "OFFBOARD"

        for attempt in range(max_attempts):
            if not rclpy.ok():
                return False
            if self.current_state.mode == "OFFBOARD":
                self.logger.info("Already in OFFBOARD mode")
                return True

            try:
                if not self.cli_mode.service_is_ready():
                    self.cli_mode.wait_for_service(timeout_sec=1.0)

                future = self.cli_mode.call_async(req)
                self._spin_until_future(future, timeout=2.0)
                if future.result() and future.result().mode_sent:
                    self.logger.info(f"OFFBOARD mode request sent (attempt {attempt+1})")
                else:
                    self.logger.warn(f"Mode switch failed (attempt {attempt+1})")

                self.pub_position_keep()
                self.rate.sleep()

            except Exception as e:
                self.logger.error(f"Set mode service failed: {e}")

        self.logger.error("Max OFFBOARD attempts reached")
        return False

    # -------------------- Thrust / Commands --------------------
    def thrust_to_throttle(self, thrust):
        """推力(N) → 油门值（原始代码是 0~3.0 clip）"""
        return float(np.clip(self.thrust_scale * float(thrust), 0.01, 3.0))

    def pub_position_keep(self):
        """以当前位置保持（满足 OFFBOARD setpoint 要求）"""
        self.current_pose.header.stamp = self.node.get_clock().now().to_msg()
        self.pos_cmd_pub.publish(self.current_pose)

    def publish_attitude_command(self, phi_d, theta_d, psi_d, thrust):
        q = quaternion_from_euler(float(phi_d), float(theta_d), float(psi_d))
        self.att_target.header.stamp = self.node.get_clock().now().to_msg()
        self.att_target.orientation.x = q[0]
        self.att_target.orientation.y = q[1]
        self.att_target.orientation.z = q[2]
        self.att_target.orientation.w = q[3]
        self.att_target.thrust = self.thrust_to_throttle(thrust)
        self.att_cmd_pub.publish(self.att_target)

    def pub_target_position(self):
        self.target_pose = self.set_target_position(self.target_xyzYaw)
        self.target_pose.header.stamp = self.node.get_clock().now().to_msg()
        self.pos_cmd_pub.publish(self.target_pose)

    def pub_target_position_small(self):
        """限速/限角速度的小步逼近发布"""
        # 目标
        tx, ty, tz, tyaw = self.target_xyzYaw
        # 当前
        cx, cy, cz = self.uav_states[0:3]
        cyaw = self.uav_states[8]

        # 位置限速
        dt = 2.0
        err = np.array([tx - cx, ty - cy, tz - cz], dtype=float)
        req_v = np.linalg.norm(err) / dt
        if req_v > self.max_linear_vel and req_v > 1e-6:
            err *= (self.max_linear_vel * dt / np.linalg.norm(err))
        nx, ny, nz = (cx + err[0], cy + err[1], cz + err[2])

        # 偏航限角速度
        max_yaw_step = self.max_angular_vel * dt
        yaw_diff = (tyaw - cyaw + np.pi) % (2*np.pi) - np.pi
        if abs(yaw_diff) > max_yaw_step:
            yaw_diff = np.sign(yaw_diff) * max_yaw_step
        nyaw = cyaw + yaw_diff

        # 发送
        msg = PoseStamped()
        msg.header.frame_id = "map"
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.pose.position.x = float(nx)
        msg.pose.position.y = float(ny)
        msg.pose.position.z = float(nz)
        q = quaternion_from_euler(0.0, 0.0, float(nyaw))
        msg.pose.orientation.x = q[0]
        msg.pose.orientation.y = q[1]
        msg.pose.orientation.z = q[2]
        msg.pose.orientation.w = q[3]
        self.pos_cmd_pub.publish(msg)
        print(f"x:{nx:.2f}, y:{ny:.2f}, z:{nz:.2f}")

    def reach_target_position(self):
        self.logger.info(f"Moving to target position: {self.target_xyzYaw}")
        self.target_pose = self.set_target_position(self.target_xyzYaw)

        while rclpy.ok():
            self.target_pose.header.stamp = self.node.get_clock().now().to_msg()
            self.pos_cmd_pub.publish(self.target_pose)

            # 处理订阅
            rclpy.spin_once(self.node, timeout_sec=0.02)

            current_pos = self.uav_odom.pose.pose.position
            dist = np.sqrt(
                (current_pos.x - self.target_xyzYaw[0])**2 +
                (current_pos.y - self.target_xyzYaw[1])**2 +
                (current_pos.z - self.target_xyzYaw[2])**2
            )
            if dist < self.pos_tolerance:
                self.logger.info("Reached target position!")
                return True

            self.rate.sleep()

    # -------------------- 数据保存（退出时） --------------------
    def shutdown_handler(self):
        try:
            if len(self.data_log) == 0:
                self.logger.info("No data to save")
                return

            user_input = input("\nData recording completed. Enter '1' to save data, '0' to discard: ")
            while user_input not in ['0', '1']:
                user_input = input("Invalid input. Please enter '1' or '0': ")

            if user_input == '1':
                # 保存到包内 scripts/data
                base_dir = Filepath(__file__).parent.parent.parent / "scripts" / "data"
                base_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = base_dir / f"uav_data_{timestamp}.csv"
                pd.DataFrame(self.data_log).to_csv(filename, index=False)
                self.logger.info(f"Data saved to {filename}")
            else:
                self.logger.info("Data discarded")
        except Exception as e:
            self.logger.error(f"Error during shutdown handling: {e}")

    # -------------------- 系统初始化流程 --------------------
    def initialize_system(self):
        try:
            self.logger.info("initialization...")

            # 等待 FCU 连接
            self.logger.info("1, Waiting for FCU connection...")
            while rclpy.ok() and not self.current_state.connected:
                rclpy.spin_once(self.node, timeout_sec=0.05)
                self.rate.sleep()
            if not rclpy.ok():
                return

            # 发送初始 setpoint（OFFBOARD 要求）
            self.logger.info("2, Sending initial position command...")
            for _ in range(100):
                if not rclpy.ok(): break
                self.pub_position_keep()
                rclpy.spin_once(self.node, timeout_sec=0.0)
                self.rate.sleep()

            # 切换 OFFBOARD
            self.logger.info("3, Switching to OFFBOARD mode...")
            if not self.set_offboard_mode():
                self.logger.error("Failed to switch to OFFBOARD mode")
                return

            # 交互式解锁
            self.logger.info("4, Please enter '1' to arm and start flight (Ctrl+C to exit)")
            while True:
                try:
                    user_input = input("Please enter '1' to continue: ")
                    if user_input == '1':
                        break
                    else:
                        self.logger.warn("Invalid input, please try again")
                except KeyboardInterrupt:
                    self.logger.info("User interrupted, exiting...")
                    return

            self.arm_safe()
            self.logger.info("System initialization complete")

        except Exception as e:
            self.logger.error(f"Unexpected error during initialization: {e}")
