from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python import get_package_share_directory
import os

def generate_launch_description():
    cfg = os.path.join(get_package_share_directory("video_compression"), "config", "compressor_config.yaml")
    
    return LaunchDescription([
        Node(
            package="video_compression",
            executable="compressor",
            name="video_compressor",
            parameters=[
                cfg
            ]
        )
    ])
