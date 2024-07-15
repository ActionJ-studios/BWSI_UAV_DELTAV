#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from turtlesim.msg import Pose

class PoseSubscriberNode(Node):

    def __init__(self):
        super().__init__("pose_subscriber")
        self.pose_subsciber_ = self.create_subscription(Pose, "/turtle1/pose", self.pose_calback, 10)

    def pose_calback(self, msg: Pose):
        self.get_logger().info("("+str(msg.x)+", "+str(msg.y)+")")

def main(args=None):
    rclpy.init(args=args)
    node = PoseSubscriberNode()
    rclpy.spin(node)
    rclpy.shutdown()