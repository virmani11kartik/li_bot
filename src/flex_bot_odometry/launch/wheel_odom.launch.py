from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_share = get_package_share_directory("flex_bot_odometry")
    default_params = os.path.join(pkg_share, "config", "wheel_odom.yaml")

    params_file = LaunchConfiguration("params_file")

    return LaunchDescription([
        DeclareLaunchArgument(
            "params_file",
            default_value=default_params,
            description="Path to wheel odometry parameter YAML file"
        ),

        Node(
            package="flex_bot_odometry",
            executable="wheel_odom_node",
            name="wheel_odom_node",
            output="screen",
            parameters=[params_file],
        ),
    ])
