from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node, PushRosNamespace
from launch.substitutions import LaunchConfiguration as LC
import os
from ament_index_python.packages import get_package_share_directory

cfg_36h11 = {
    "image_transport": "raw",
    "family": "36h11",
    "size": 0.508,
    "max_hamming": 0,
    "z_up": True
}

def generate_launch_description():
    ld = LaunchDescription()

    tensorrt_wrapper_dir = get_package_share_directory("tensor_detector")

    params_path = os.path.join(tensorrt_wrapper_dir, 'config', 'yolo_orientation.yaml')

    ld.add_action(PushRosNamespace(LC("robot")))

    ld.add_action(DeclareLaunchArgument(
        name="robot",
        default_value="tempest",
        description="name of the robot"
    ))

    ld.add_action(Node(
        package='tensor_detector',
        executable='yolo_orientation.py',
        name='yolo_orientation',
        output='screen',
        parameters=[params_path]
    ))

    return ld
