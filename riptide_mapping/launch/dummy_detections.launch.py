import os
import launch
from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument, GroupAction
from launch_ros.actions import Node, PushRosNamespace
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration as LC

config_dir = os.path.join(
    get_package_share_directory('riptide_mapping2'),
    'config'
)

def generate_launch_description():
    # declare the launch args to read for this file

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

        DeclareLaunchArgument(
            "config",
            default_value="dummy_detections",
            description="name of maps to use"
        ),

        DeclareLaunchArgument(
            "config_yaml",
            default_value = [LC('config'), '.yaml']
        ),
        
        DeclareLaunchArgument(
            "base_link_name",
            default_value=[LC("robot"), "/base_link"]
        ),
        
        DeclareLaunchArgument(
            "dummy_dfc_link_name",
            default_value=[LC("robot"), "/dummy_dfc_link"]
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
                output='screen',
                
                parameters = [
                    PathJoinSubstitution([
                        config_dir,
                        LC('config_yaml')
                    ])
                ],
                
                arguments=[
                    "--ros-args", "--log-level", LC("log_level")
                ]
            ),
            
            
            
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="dummy_dfc_link_publisher",
                arguments=[
                    "--frame-id", LC("base_link_name"),
                    "--child-frame-id", LC('dummy_dfc_link_name'),
                    "--pitch", "1.5707"
                ]
            )
        ], scoped=True)
    ])