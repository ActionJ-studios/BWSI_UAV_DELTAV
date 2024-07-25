import rclpy
from rclpy.node import Node
from px4_msgs.msg import VehicleLocalPosition
from px4_msgs.msg import OffboardControlMode
from px4_msgs.msg import TrajectorySetpoint
from px4_msgs.msg import VehicleCommand
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy

class Hover(Node):
    def __init__(self):
        super().__init__('height_control_node')
        self.target_height = 5.0  # Desired height in meters
        self.height_tolerance = 0.5  # Tolerance in meters

        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=10
        )

        self.create_subscription(VehicleLocalPosition, '/fmu/out/vehicle_local_position', self.local_position_callback, qos_profile)
        self.control_publisher = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', 10)
        self.offboard_control_mode_publisher = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', 10)
        self.vehicle_command_publisher = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', 10)

        self.timer = self.create_timer(0.1, self.control_loop)

        self.current_height = None
        self.offboard_mode_set = False
        self.armed = False

    def local_position_callback(self, msg):
        self.current_height = msg.z

    def control_loop(self):
        if self.current_height is None:
            self.get_logger().info(f'current height none.')
            return

        if not self.armed:
            self.arm_drone()
            return

        height_error = self.target_height - self.current_height

        if abs(height_error) > self.height_tolerance:
            trajectory_setpoint_msg = TrajectorySetpoint()
            trajectory_setpoint_msg.timestamp = self.get_clock().now().nanoseconds // 1000  # PX4 timestamp in microseconds
            trajectory_setpoint_msg.position = [float('nan'), float('nan'), -self.target_height]  # PX4 NED coordinate frame, z is negative up
            trajectory_setpoint_msg.yaw = float('nan')  # NaN to ignore yaw
            self.control_publisher.publish(trajectory_setpoint_msg)

            if not self.offboard_mode_set:
                offboard_control_mode_msg = OffboardControlMode()
                offboard_control_mode_msg.timestamp = self.get_clock().now().nanoseconds // 1000  # PX4 timestamp in microseconds
                offboard_control_mode_msg.position = True
                offboard_control_mode_msg.velocity = False
                offboard_control_mode_msg.acceleration = False
                offboard_control_mode_msg.attitude = False
                offboard_control_mode_msg.body_rate = False
                self.offboard_control_mode_publisher.publish(offboard_control_mode_msg)
                self.offboard_mode_set = True

            self.get_logger().info(f'Adjusting height to {self.target_height} meters.')
        else:
            self.get_logger().info('Height within tolerance, no adjustment needed.')

    def arm_drone(self):
        self.get_logger().info('Arming drone...')
        arm_cmd = VehicleCommand()
        arm_cmd.timestamp = self.get_clock().now().nanoseconds // 1000
        arm_cmd.param1 = 1.0  # 1 to arm, 0 to disarm
        arm_cmd.command = VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM
        arm_cmd.target_system = 1
        arm_cmd.target_component = 1
        arm_cmd.source_system = 1
        arm_cmd.source_component = 1
        arm_cmd.from_external = True
        self.vehicle_command_publisher.publish(arm_cmd)
        self.armed = True

def main(args=None):
    rclpy.init(args=args)
    hover_node = Hover()
    rclpy.spin(hover_node)
    hover_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
