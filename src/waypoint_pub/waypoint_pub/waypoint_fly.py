import os
import sys
import yaml
import numpy as np

import rclpy
from rclpy.node import Node

# 让 Python 能找到 env/uav_ros.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from env.uav_ros import UAV_ROS  


# 打印 2 位小数
np.set_printoptions(precision=2, floatmode='fixed')


def split_waypoints(waypoints, max_distance=0.9):
    """在 [x,y,z,yaw] 航点之间插值，保证相邻距离不超过 max_distance"""
    if not waypoints:
        return []

    interpolated = [waypoints[0]]

    for i in range(1, len(waypoints)):
        p0 = waypoints[i - 1][:3]
        p1 = waypoints[i][:3]
        yaw0 = waypoints[i - 1][3]
        yaw1 = waypoints[i][3]

        dist = np.linalg.norm(p1 - p0)

        if dist <= max_distance:
            interpolated.append(waypoints[i])
        else:
            num_points = int(np.ceil(dist / max_distance))
            for j in range(1, num_points + 1):
                alpha = j / (num_points + 1)
                pos = p0 + alpha * (p1 - p0)

                # 插值最短角度
                yaw_diff = (yaw1 - yaw0 + np.pi) % (2 * np.pi) - np.pi
                yaw = yaw0 + alpha * yaw_diff

                interpolated.append(np.array([pos[0], pos[1], pos[2], yaw]))

            interpolated.append(waypoints[i])

    return interpolated


def load_waypoints(config_file="waypoint_pub/waypoint_yaml/hover.yaml"):
    """从 YAML 加载航点，格式：
    waypoints:
      - [x, y, z, yaw_deg]   # yaw 是角度
      - [x, y, z, yaw_deg]
    """
    package_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(package_path, config_file)
    print(f"Looking for waypoints at: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    waypoints = []
    for wp in config["waypoints"]:
        if len(wp) == 4:
            x, y, z, yaw_deg = wp
            yaw_rad = np.radians(yaw_deg % 360)
            waypoints.append(np.array([x, y, z, yaw_rad]))
        else:
            waypoints.append(np.array(wp))

    print("Successfully loaded waypoints (x,y,z,yaw_rad):")
    for i, wp in enumerate(waypoints):
        print(f"Waypoint before split {i}: {wp}")

    splited = split_waypoints(waypoints)

    for i, wp in enumerate(splited):
        print(f"Waypoint after  split {i}: {wp}")

    return splited


def now_sec(node: Node) -> float:
    """ROS2 的当前时间（秒）"""
    return node.get_clock().now().nanoseconds * 1e-9


class WaypointFly(Node):
    def __init__(self):
        super().__init__("waypoint_pub")

        # ====== 可调参数 ======
        self.dt = 0.01                       # 控制周期
        self.threshold_distance = 0.315      # 航点到达判据（米）
        self.wait_after_reach = 2.0          # 到达后等待秒数
        self.control_name = "PdT_fix"        # 复用你原控制名
        self.show_rviz_if_gazebo = True
        # =====================

        self.get_logger().info("we are in main")

        # 加载航点
        self.waypoints = load_waypoints()
        self.get_logger().info("after load_waypoints")

        # 初始化 UAV
        self.uav = UAV_ROS(m=0.72, dt=self.dt, use_gazebo=False, control_name=self.control_name)

        # 记录状态
        self.reached_final_waypoint = False
        self.current_waypoint_index = 0
        self.start_curr_point_time = now_sec(self)

        # 系统初始化
        self.get_logger().info("system init")
        self.uav.initialize_system()

        self.get_logger().info(f"Control loop started at {1.0 / self.dt:.1f} Hz")
        self.get_logger().info("Starting position guidance...")

        # 先保持 1 秒（10ms × 100 次）
        self.get_logger().info("Wait 1 sec")
        self.keep_count = 100

        # gazebo / rviz 开关
        self.uav.is_show_rviz = self.uav.use_gazebo and self.show_rviz_if_gazebo

        print("Successfully loaded waypoints:\n x y z yaw")
        for i, wp in enumerate(self.waypoints):
            print(f"Waypoint {i}: {wp}")

        # 定时器：主控制循环
        self.timer = self.create_timer(self.dt, self.control_step)

    # 主循环
    def control_step(self):
        # 先做 1 秒 “position_keep”
        if self.keep_count > 0:
            self.uav.pub_position_keep()
            self.keep_count -= 1
            return

        # 已结束则收尾
        if self.reached_final_waypoint:
            self.get_logger().info("Simulation is finished")
            self.uav.is_show_rviz = False
            self.uav.target_xyzYaw = (0.0, 0.0, 0.25, 0.0)
            self.uav.reach_target_position()
            self.destroy_node()
            rclpy.shutdown()
            return

        # 当前状态与目标
        current_position = self.uav.uav_states[0:3]  # [x, y, z]
        current_target = self.waypoints[self.current_waypoint_index]
        target_position = current_target[:3]
        distance = np.linalg.norm(current_position - target_position)

        tnow = now_sec(self)

        # 还没到达 - 继续飞
        if distance > self.threshold_distance and self.current_waypoint_index < len(self.waypoints) - 1:
            print(f"approaching waypoint {self.current_waypoint_index}, curr dist is {distance:.2f}")

        #到了但等待时间未够 原地悬停
        elif (
            distance < self.threshold_distance
            and self.current_waypoint_index < len(self.waypoints) - 1
            and (tnow - self.start_curr_point_time) < self.wait_after_reach
        ):
            print(f"reached waypoint {self.current_waypoint_index}, need to wait for {self.wait_after_reach:.0f} sec")

        #到达且等待时间到了 -切换到下一个点
        elif (
            distance < self.threshold_distance
            and self.current_waypoint_index < len(self.waypoints) - 1
            and (tnow - self.start_curr_point_time) >= self.wait_after_reach
        ):
            self.current_waypoint_index += 1
            self.start_curr_point_time = tnow
            print(f"Switching to waypoint {self.current_waypoint_index}, the point is {self.waypoints[self.current_waypoint_index]}")

         #....
        elif distance < self.threshold_distance and self.current_waypoint_index == len(self.waypoints) - 1:
            self.reached_final_waypoint = True
            print("Reached final waypoint! Prepare to land")

        # 发布目标
        self.uav.target_xyzYaw = (
            float(current_target[0]),
            float(current_target[1]),
            float(current_target[2]),
            float(current_target[3]),
        )
        self.uav.pub_target_position()
        # 限速控制
        # self.uav.pub_velocity_limited_target()

def main(args=None):
    rclpy.init(args=args)
    node = WaypointFly()
    try:
        rclpy.spin(node)  # 用定时器驱动
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()
