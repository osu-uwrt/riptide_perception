import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch_ros.actions import Node, PushRosNamespace
from launch.substitutions import LaunchConfiguration as LC
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    tensorrt_wrapper_dir = get_package_share_directory("tensor_detector")
    params_path = os.path.join(tensorrt_wrapper_dir, 'config', 'yolo_detection.yaml')

    return LaunchDescription([
        DeclareLaunchArgument(
            name="robot",
            default_value="tempest",
            description="name of the robot"
        ),
        
        GroupAction([
            PushRosNamespace(LC("robot")),
            
            Node(
                package='tensor_detector',
                executable='yolo_detection.py',
                name='yolo_detection',
                output='screen',
                parameters=[
                    params_path,
                    {"robot_namespace": LC("robot")}]
            )
        ], scoped=True)
    ])
