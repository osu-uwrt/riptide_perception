from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration as LC

import os

from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    ld = LaunchDescription()

    tensorrt_wrapper_dir = get_package_share_directory("tensor_detector")

    params_path = os.path.join(tensorrt_wrapper_dir, 'config', 'tensorrt.yaml')

    weights_path = os.path.join(
        tensorrt_wrapper_dir, 'weights', 'best200generated.engine')

    ld.add_action(DeclareLaunchArgument(
        name="robot",
        default_value="tempest",
        description="name of the robot"
    ))

    ld.add_action(Node(
        package="tensor_detector",
        namespace=LC("robot"),
        executable="tensorrtWrapper",
        name="tensor_detector",
        parameters=[
            {"engine_path": weights_path},
            params_path,
        ]
    ))

    return ld
