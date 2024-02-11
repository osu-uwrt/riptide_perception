from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch_ros.actions import Node, PushRosNamespace
from launch_ros.actions.composable_node_container import ComposableNode, ComposableNodeContainer
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

    params_path = os.path.join(tensorrt_wrapper_dir, 'config', 'tensorrt.yaml')

    weights_path = os.path.join(
        tensorrt_wrapper_dir, 'weights', 'best_200_uwrtarvp.engine')
    
    ld.add_action(PushRosNamespace(LC("robot")))

    ld.add_action(DeclareLaunchArgument(
        name="robot",
        default_value="tempest",
        description="name of the robot"
    ))

    ld.add_action(Node(
            package='tensor_detector',  # Assuming test_orientation.py is in the tensor_detector package
            executable='yolo_orientation.py',  # The name you've configured in setup.py under entry_points/console_scripts
            name='yolo_orientation_node',  # Optional: a name for the node
            output='screen',  # Optional: if you want the node's output to appear in the console
    ))


    
    #
    # ZED LAUNCH
    #
    ld.add_action(IncludeLaunchDescription(
        AnyLaunchDescriptionSource(os.path.join(
            get_package_share_directory('riptide_hardware2'),
            "launch", "zed.launch.py"
        ))
    ))

    #
    # APRILTAG STUFF
    #

    ld.add_action(ComposableNodeContainer(
        name='tag_container',
        namespace="apriltag",
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[
            ComposableNode(
                name='apriltag_36h11',
                package='apriltag_ros', plugin='AprilTagNode',
                remappings=[
                    # This maps the 'raw' images for simplicity of demonstration.
                    # In practice, this will have to be the rectified 'rect' images.
                    ("image_rect",
                    "zed2i/zed_node/left/image_rect_color"),
                    ("camera_info",
                    "zed2i/zed_node/left/camera_info"),
                ],
                parameters=[cfg_36h11],
                extra_arguments=[{'use_intra_process_comms': True}],
            )
        ],
        output='screen'
    )),

    ld.add_action(Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='surface_frame_node',
        arguments=["0", "0.4572", "0", "0", "-1.5707", "-1.5707",
                "tag36h11:0", "estimated_origin_frame"]
    ))

    return ld
