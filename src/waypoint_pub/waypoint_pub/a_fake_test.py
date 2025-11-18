#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool
import time

class FakeGraspTest(Node):
    def __init__(self):
        super().__init__("fake_grasp_test")
        self.obj_pub = self.create_publisher(
            PoseStamped,
            "/vrpn_mocap/object/pose",
            10
        )


        self.grasp_start_sub = self.create_subscription(
            Bool, "grasp/start", self.grasp_start_cb, 10
        )

      
        self.grasp_done_pub = self.create_publisher(
            Bool, "grasp/done", 10
        )

       
        self.fake_pose = PoseStamped()
        self.fake_pose.header.frame_id = "world"
        self.fake_pose.pose.position.x = 2.0
        self.fake_pose.pose.position.y = 2.0
        self.fake_pose.pose.position.z = 0.8
        self.fake_pose.pose.orientation.w = 1.0

        self.get_logger().info("Fake Vicon & Fake Arm node started.")
        self.timer = self.create_timer(0.02, self.pub_fake_pose)  
        self.waiting_for_done = False
        self.wait_start_time = 0.0


    def pub_fake_pose(self):
        self.fake_pose.header.stamp = self.get_clock().now().to_msg()
        self.obj_pub.publish(self.fake_pose)


    def grasp_start_cb(self, msg: Bool):
        if msg.data and not self.waiting_for_done:
            self.get_logger().info("Received grasp/start → simulate grasping for 20s...")
            self.waiting_for_done = True
            self.wait_start_time = time.time()
            self.create_timer(0.5, self.check_grasp_done)


    def check_grasp_done(self):
        if not self.waiting_for_done:
            return

        if time.time() - self.wait_start_time >= 20.0:
            self.get_logger().info("20s passed → publish grasp/done=True")
            self.grasp_done_pub.publish(Bool(data=True))
            self.waiting_for_done = False


def main(args=None):
    rclpy.init(args=args)
    node = FakeGraspTest()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
