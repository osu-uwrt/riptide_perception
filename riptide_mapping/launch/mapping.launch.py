import os
import launch
from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument, GroupAction
from launch_ros.actions import Node, PushRosNamespace, ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration as LC

# cfg_36h11 = {
#     "image_transport": "raw",
#     "family": "36h11",
#     "size": 0.508,
#     "max_hamming": 0,
#     "z_up": True
# }

config_dir = os.path.join(
    get_package_share_directory('riptide_mapping2'),
    'config'
)

def generate_launch_description():
    # declare the launch args to read for this file
    DeclareLaunchArgument(
        "robot",
        default_value="tempest",
        description="Namespace of the vehicle",
    ),

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
            default_value="config",
            description="name of maps to use"
        ),

        DeclareLaunchArgument(
            "config_yaml",
            default_value=[LC('config'), ".yaml"]
        ),

        GroupAction([
            PushRosNamespace(
                LC("robot")
            ),

            # create the nodes
            Node(
                package='riptide_mapping2',
                executable='mapping.py',
                name='riptide_mapping2',
                respawn=True,
                output='screen',

                # use the parameters on the node
                parameters=[
                    PathJoinSubstitution([
                        config_dir,
                        LC("config_yaml")
                    ])
                ]
            ),
            
            Node(
                package="chameleon_tf",
                executable="chameleon_tf",
                name="world_to_map",
                output="screen",
                respawn=True,
                parameters=[
                    {"stddev_threshold": 0.5},
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
                    {"transform_locks": [
                        False,      # unlock x
                        False,      # unlock y
                        True,       # lock z
                        True,       # lock roll
                        True,       # lock pitch
                        False       # unlock yaw
                    ]}
                ]
            ),

            # ComposableNodeContainer(
            #     name='tag_container',
            #     namespace="apriltag",
            #     package='rclcpp_components',
            #     executable='component_container',
            #     composable_node_descriptions=[
            #         ComposableNode(
            #             name='apriltag_36h11',
            #             package='apriltag_ros', plugin='AprilTagNode',
            #             remappings=[
            #                 # This maps the 'raw' images for simplicity of demonstration.
            #                 # In practice, this will have to be the rectified 'rect' images.
            #                 ("image_rect",
            #                 "ffc/zed_node/left/image_rect_color"),
            #                 ("camera_info",
            #                 "ffc/zed_node/left/camera_info"),
            #             ],
            #             parameters=[cfg_36h11],
            #             extra_arguments=[{'use_intra_process_comms': True}],
            #         )
            #     ],
            #     output='screen'
            # ),

            # Node(
            #     package='tf2_ros',
            #     executable='static_transform_publisher',
            #     name='surface_frame_node',
            #     arguments=["0", "0.4572", "0", "0", "-1.5707", "-1.5707",
            #             "tag36h11:0", "estimated_origin_frame"]
            # )
        ], scoped=True)
    ])
