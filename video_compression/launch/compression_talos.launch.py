from launch import LaunchDescription
from launch_ros.actions import Node, PushRosNamespace
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.substitutions import LaunchConfiguration as LC
from ament_index_python import get_package_share_directory
import os

def generate_launch_description():
    cfg = os.path.join(get_package_share_directory("video_compression"), "config", "compressor_config_talos.yaml")
    
    return LaunchDescription([
        DeclareLaunchArgument(
            name="robot",
            default_value="tempest",
            description="name of the robot"
        ),
        
        GroupAction([            
            PushRosNamespace(
                LC("robot")
            ),
            
            Node(
                package="video_compression",
                executable="compressor",
                name="video_compressor",
                parameters=[
                    cfg
                ]
            )
        ], scoped=True)
    ])
