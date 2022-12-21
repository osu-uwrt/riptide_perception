import launch
import os
import launch_ros.actions
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    riptide_vision_share_dir = get_package_share_directory('riptide_yolo')

    riptide_vision = launch_ros.actions.Node(
        package="riptide_yolo", executable="vision",
        parameters=[
                       {"weights":os.path.join(riptide_vision_share_dir,"weights/last.pt")},
                       {"data":os.path.join(riptide_vision_share_dir,"data/pool.yaml")}
                   ],
        remappings=[('yolo/detected_objects', 'detected_objects')]
    )

    return launch.LaunchDescription([
        launch_ros.actions.PushRosNamespace("tempest"),
        riptide_vision,
    ])