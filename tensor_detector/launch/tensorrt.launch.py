import argparse
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node

import os
import sys

from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    ld = LaunchDescription()

    tensorrt_wrapper_dir = get_package_share_directory("tensor_detector")

    params_path = os.path.join(tensorrt_wrapper_dir, 'config', 'tensorrt.yaml')

    node_name = "tensor_detector"

    ld.add_action(Node(
        package="tensor_detector", executable="tensorrtWrapper", name=node_name, parameters=[params_path]
    ))

    ld.add_action(DeclareLaunchArgument(
            name = "robot",
            default_value = "tempest",
            description = "name of the robot"
        ),)

    return ld