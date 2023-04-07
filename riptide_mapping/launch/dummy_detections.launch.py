import os
import launch
from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument, GroupAction
from launch_ros.actions import Node, PushRosNamespace
from launch.substitutions import LaunchConfiguration as LC

def generate_launch_description():
    # declare the launch args to read for this file
    config = os.path.join(
        get_package_share_directory('riptide_mapping2'),
        'config',
        'dummy_detections.yaml'
    )

    return launch.LaunchDescription([        
        DeclareLaunchArgument(
            "log_level", 
            default_value="INFO",
            description="log level to use",
        ),
        
        DeclareLaunchArgument(
            "robot",
            default_value="tempest",
            description="name of the robot"
        ),
        
        GroupAction(actions=[
            PushRosNamespace(
                LC("robot")
            ),
            
            # create the nodes    
            Node(
                package='riptide_mapping2',
                executable='dummydetections.py',
                name='dummydetections',
                respawn=True,
                output='screen',
                
                parameters = [
                    config
                ]
            )
        ], scoped=True)
    ])