import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/devil/Desktop/Coding/Robotics/bwsi-uav/BWSI_UAV_DELTAV/install/my_robot_controller'
