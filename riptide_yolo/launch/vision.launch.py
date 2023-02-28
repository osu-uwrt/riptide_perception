import launch
import os
from launch_ros.actions import Node, PushRosNamespace
from ament_index_python.packages import get_package_share_directory
from launch.substitutions import LaunchConfiguration as LC
from launch.actions import DeclareLaunchArgument

def generate_launch_description():
    riptide_vision_share_dir = get_package_share_directory('riptide_yolo')

    riptide_vision = Node(
        package="riptide_yolo", executable="vision",
        parameters=[
                       {"weights":os.path.join(riptide_vision_share_dir,"weights/last.pt")},
                       {"data":os.path.join(riptide_vision_share_dir,"data/pool.yaml")}
                   ],
        remappings=[('yolo/detected_objects', 'detected_objects')]
    )

    return launch.LaunchDescription([
        DeclareLaunchArgument(
            "namespace", 
            default_value="talos",
            description="Namespace of the vehicle",
        ),

        PushRosNamespace(
            LC("namespace"),
        ),

        launch.actions.GroupAction([
            riptide_vision,
        ], scoped=True)
    ])