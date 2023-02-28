import os
import launch
from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node, PushRosNamespace
from launch.substitutions import LaunchConfiguration as LC

def generate_launch_description():
    # declare the launch args to read for this file
    DeclareLaunchArgument(
        "robot", 
        default_value="tempest",
        description="Namespace of the vehicle",
    ),

    config = os.path.join(
        get_package_share_directory('riptide_mapping2'),
        'config',
        'config.yaml'
    )
    
    dependentFrames = os.path.join(
        get_package_share_directory('riptide_mapping2'),
        'config',
        'dependent_frames.yaml'
    )
    
    orientedFrames = os.path.join(
        get_package_share_directory('riptide_mapping2'),
        'config',
        'oriented_frames.yaml'
    )

    return launch.LaunchDescription([
        PushRosNamespace(
            LC("robot")
        ),

        DeclareLaunchArgument(
            "log_level", 
            default_value="INFO",
            description="log level to use",
        ),

        # create the nodes    
        Node(
            package='riptide_mapping2',
            executable='mapping.py',
            name='riptide_mapping2',
            respawn=True,
            output='screen',
            
            # use the parameters on the node
            parameters = [
                config
            ]
        ),
        
        Node(
            package='riptide_mapping2',
            executable='dependentFramePublisher',
            name='dependent_frame_publisher',
            respawn=True,
            output='screen',
            
            parameters = [
                dependentFrames
            ]
        ),
        
        Node(
            package='riptide_mapping2',
            executable='orientedFramePublisher',
            name='oriented_frame_publisher',
            respawn=True,
            output='screen',
            
            parameters = [
                orientedFrames
            ]
        ),

        Node(
        package="chameleon_tf",
        executable="chameleon_tf",
        name="world_to_map",
        output="screen",
        respawn=True,
        parameters=[
            {"source_frame": "world"},
            {"target_frame": "map"},
            {"initial_translation": [
                0.0,
                0.0,
                0.0
            ]},
            {"initial_rotation": [
                0.0,
                0.0,
                0.0
            ]},
        ]
    )
    ])