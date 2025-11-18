# -*- coding: utf-8 -*-


import math
import numpy as np

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from env.uav_ros import UAV_ROS  

from std_msgs.msg import Bool

def now_sec(node: Node) -> float:
    return node.get_clock().now().nanoseconds * 1e-9


class GraspMission(Node):
    def __init__(self):
        super().__init__("grasp_mission")

        self.dt = 0.01                    
        self.threshold_distance = 0.25    
        self.wait_after_reach = 2.0       
        self.control_name = "GraspMission" 

        
        self.offset_x = -0.2
        self.offset_y = 0.0
        self.offset_z = 0.2
     
        self.get_logger().info("GraspMission node started")

     
        self.has_object_pose = False
        self.object_pose = PoseStamped()

        
        self.object_sub = self.create_subscription(
            PoseStamped,
            "/vrpn_mocap/object/pose",
            self.object_pose_cb,
            10
        )

        
        self.pre_grasp_point = None  # np.array([x, y, z, yaw])

       
        self.reached_pre_grasp = False
        self.start_reach_time = now_sec(self)

        
        self.grasp_start_sent = False
        
        self.grasp_done = False

        
        self.home_point = np.array([0.0, 0.0, 0.2, 0.0])

        self.grasp_start_pub = self.create_publisher(
            Bool, "grasp/start", 10
        )
        self.grasp_done_sub = self.create_subscription(
            Bool, "grasp/done", self.grasp_done_cb, 10
        )


        
        self.uav = UAV_ROS(m=0.72, dt=self.dt, use_gazebo=False, control_name=self.control_name)
        self.reached_pre_grasp = False
        self.start_reach_time = None

       
        self.get_logger().info("Calling UAV system initialization ...")
        self.uav.initialize_system()
        self.get_logger().info("UAV system initialization done")

        
        self.keep_count = int(1.0 / self.dt)
        self.get_logger().info(f"Will keep position for {self.keep_count} steps (~1s)")

        self.uav.is_show_rviz = False
        self.timer = self.create_timer(self.dt, self.control_step)

    def object_pose_cb(self, msg: PoseStamped):
        self.object_pose = msg
        self.has_object_pose = True


    def grasp_done_cb(self, msg: Bool):
        if msg.data:
            self.get_logger().info("[GraspMission] Received grasp done from arm node.")
            self.grasp_done = True


    def update_pre_grasp_from_object(self):
        if not self.has_object_pose:
            return False

        p = self.object_pose.pose.position
        x_pre = p.x + self.offset_x
        y_pre = p.y + self.offset_y
        z_pre = p.z + self.offset_z
        yaw_pre = 0.0

        self.pre_grasp_point = np.array([x_pre, y_pre, z_pre, yaw_pre])

        self.get_logger().info(
            f"Pre-grasp point computed from object: "
            f"obj=({p.x:.2f}, {p.y:.2f}, {p.z:.2f}) -> "
            f"pre=({x_pre:.2f}, {y_pre:.2f}, {z_pre:.2f}, yaw={yaw_pre:.2f})"
        )
        return True


    def control_step(self):

        if self.keep_count > 0:
            self.uav.pub_position_keep()
            self.keep_count -= 1
            return

        
        if self.pre_grasp_point is None:
            if not self.update_pre_grasp_from_object():
                self.get_logger().warn(
                    "Waiting for /vrpn_mocap/object/pose ... "
                    "UAV is holding current position."
                )
                self.uav.pub_position_keep()
                return
            else:
                self.get_logger().info(
                    "Pre-grasp point ready, start flying to pre-grasp point."
                )

        
        current_position = self.uav.uav_states[0:3]   # [x, y, z]
        target_position = self.pre_grasp_point[:3]
        distance = np.linalg.norm(current_position - target_position)
        tnow = now_sec(self)

        
        if not self.reached_pre_grasp:
            if distance > self.threshold_distance:
                
                print(f"[GraspMission] approaching pre-grasp point, "
                      f"dist={distance:.2f} m, "
                      f"curr=({current_position[0]:.2f}, "
                      f"{current_position[1]:.2f}, "
                      f"{current_position[2]:.2f})")
               
                self.uav.target_xyzYaw = (
                    float(self.pre_grasp_point[0]),
                    float(self.pre_grasp_point[1]),
                    float(self.pre_grasp_point[2]),
                    float(self.pre_grasp_point[3]),
                )
                self.uav.pub_target_position()
                return
            else:
                
                self.reached_pre_grasp = True
                self.start_reach_time = tnow
                self.get_logger().info("[GraspMission] reached pre-grasp point, start hovering...")
                
        
        if self.reached_pre_grasp:
            elapsed = tnow - self.start_reach_time

            
            if elapsed < self.wait_after_reach:
                print(f"[GraspMission] hovering at pre-grasp point, "
                      f"wait {self.wait_after_reach - elapsed:.1f}s")
                self.uav.target_xyzYaw = (
                    float(self.pre_grasp_point[0]),
                    float(self.pre_grasp_point[1]),
                    float(self.pre_grasp_point[2]),
                    float(self.pre_grasp_point[3]),
                )
                self.uav.pub_target_position()
                return

    
            if not self.grasp_start_sent:
                self.get_logger().info("[GraspMission] pre-grasp stable, sending grasp/start=True")
                msg = Bool()
                msg.data = True
                self.grasp_start_pub.publish(msg)
                self.grasp_start_sent = True

            
            if not self.grasp_done:
                print("[GraspMission] waiting for arm to finish grasp... (hovering)")
                self.uav.target_xyzYaw = (
                    float(self.pre_grasp_point[0]),
                    float(self.pre_grasp_point[1]),
                    float(self.pre_grasp_point[2]),
                    float(self.pre_grasp_point[3]),
                )
                self.uav.pub_target_position()
                return

            
            home_pos = self.home_point[:3]
            distance_home = np.linalg.norm(current_position - home_pos)

            if distance_home > self.threshold_distance:
                print(f"[GraspMission] returning home, dist={distance_home:.2f} m")
                self.uav.target_xyzYaw = (
                    float(self.home_point[0]),
                    float(self.home_point[1]),
                    float(self.home_point[2]),
                    float(self.home_point[3]),
                )
                self.uav.pub_target_position()
            else:
                
                self.get_logger().info("[GraspMission] reached home point after grasp. Holding position.")
                self.uav.target_xyzYaw = (
                    float(self.home_point[0]),
                    float(self.home_point[1]),
                    float(self.home_point[2]),
                    float(self.home_point[3]),
                )
                self.uav.pub_target_position()
                


def main(args=None):
    rclpy.init(args=args)
    node = GraspMission()
    try:
        rclpy.spin(node)  
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
