from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration as LC

import os

from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    ld = LaunchDescription()

    yolov8_detector_dir = get_package_share_directory("yolov8_detector")

    params_path = os.path.join(yolov8_detector_dir, 'config', 'yolov8.yaml')

    weights_path = os.path.join(
        yolov8_detector_dir, 'weights', 'yolov8n-seg.trt')

    ld.add_action(DeclareLaunchArgument(
        name="robot",
        default_value="tempest",
        description="name of the robot"
    ))

    ld.add_action(Node(
        package="yolov8_detector",
        namespace=LC("robot"),
        executable="yolov8_detector",
        name="yolov8_detector",
        parameters=[
            {"model_path": weights_path},
            params_path,
        ]
    ))

    return ld