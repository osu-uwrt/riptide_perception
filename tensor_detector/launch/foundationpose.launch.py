import launch
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import (
    EnvironmentVariable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode


def zed_topic(robot, camera, relative_topic):
    """Absolute ZED topic name: /<robot>/<camera>/zed_node/<relative_topic>."""
    return ["/", robot, "/", camera, "/zed_node/", relative_topic]


def generate_launch_description():

    # Launch configurations. These are lazy: each one is resolved when an
    # action that uses it runs, so they can appear in argument defaults below.
    launch_foundationpose = LaunchConfiguration("launch_foundationpose")
    isaac_enabled = IfCondition(launch_foundationpose)

    robot = LaunchConfiguration("robot")
    camera = LaunchConfiguration("camera")
    zed_rgb_topic = LaunchConfiguration("zed_rgb_topic")
    zed_camera_info_topic = LaunchConfiguration("zed_camera_info_topic")
    zed_depth_topic = LaunchConfiguration("zed_depth_topic")
    image_topic = LaunchConfiguration("image_topic")

    target_class = LaunchConfiguration("target_class")
    yolo_model_path = LaunchConfiguration("yolo_model_path")

    image_width = LaunchConfiguration("image_width")
    image_height = LaunchConfiguration("image_height")
    zed_qos = LaunchConfiguration("zed_qos")

    models_dir = LaunchConfiguration("models_dir")
    mesh_file_path = LaunchConfiguration("mesh_file_path")
    object_config_path = LaunchConfiguration("object_config_path")
    debug = LaunchConfiguration("debug")

    # Derived values (not arguments).
    rgb_topic = zed_topic(robot, camera, zed_rgb_topic)
    camera_info_topic = zed_topic(robot, camera, zed_camera_info_topic)
    depth_topic = zed_topic(robot, camera, zed_depth_topic)

    refine_model_file_path = PathJoinSubstitution([models_dir, "refine_model.onnx"])
    refine_engine_file_path = PathJoinSubstitution(
        [models_dir, "refine_trt_engine.plan"]
    )
    score_model_file_path = PathJoinSubstitution([models_dir, "score_model.onnx"])
    score_engine_file_path = PathJoinSubstitution([models_dir, "score_trt_engine.plan"])

    launch_args = [
        # --- Switch (declared first: the Isaac-only arguments below use it) ---
        DeclareLaunchArgument(
            "launch_foundationpose",
            default_value="false",
            description="Start the Isaac container (format converter + FoundationPose). "
            "false runs yolo_2d alone, e.g. on the VM.",
        ),
        # --- Camera topics ---
        DeclareLaunchArgument(
            "robot",
            default_value="tempest",
            description="Robot namespace the ZED runs under",
        ),
        DeclareLaunchArgument(
            "camera",
            default_value="ffc",
            description="ZED camera name (ffc or dfc)",
        ),
        DeclareLaunchArgument(
            "zed_rgb_topic",
            default_value="rgb/image_rect_color",
            description="Rectified RGB topic relative to zed_node. "
            "Wrapper >= 5.1: rgb/color/rect/image",
        ),
        DeclareLaunchArgument(
            "zed_camera_info_topic",
            default_value="rgb/camera_info",
            description="RGB camera_info topic relative to zed_node. "
            "Wrapper >= 5.1: rgb/color/rect/camera_info",
        ),
        DeclareLaunchArgument(
            "zed_depth_topic",
            default_value="depth/depth_registered",
            description="Depth topic (32FC1, meters) relative to zed_node",
        ),
        DeclareLaunchArgument(
            "image_topic",
            default_value=rgb_topic + ["/compressed"],
            description="Compressed image yolo_2d subscribes to. Defaults to the "
            "ZED's compressed RGB; on the VM set it to /image_raw/compressed",
        ),
        # --- yolo_2d ---
        DeclareLaunchArgument(
            "target_class",
            description="Class yolo_2d builds the FoundationPose mask from",
        ),
        DeclareLaunchArgument(
            "yolo_model_path",
            description="Absolute path to the YOLO segmentation model (.pt or .engine)",
        ),
        # --- Image size and QoS (Isaac only) ---
        DeclareLaunchArgument(
            "image_width",
            default_value="1920",
            description="Width of the ZED RGB image (HD1200: 1920)",
        ),
        DeclareLaunchArgument(
            "image_height",
            default_value="1200",
            description="Height of the ZED RGB image (HD1200: 1200)",
        ),
        DeclareLaunchArgument(
            "zed_qos",
            default_value="DEFAULT",
            description="Isaac QoS profile for subscriptions to ZED topics. "
            "SENSOR_DATA if the ZED publishes best effort",
        ),
        # --- FoundationPose (Isaac only) ---
        DeclareLaunchArgument(
            "models_dir",
            default_value=PathJoinSubstitution(
                [
                    EnvironmentVariable(
                        "ISAAC_ROS_WS", default_value="/workspaces/isaac_ros-dev"
                    ),
                    "isaac_ros_assets",
                    "models",
                    "foundationpose",
                ]
            ),
            description="Directory holding the refine/score .onnx models and "
            "TensorRT .plan engines",
        ),
        # Required only when Isaac runs, so the VM (launch_foundationpose:=false)
        # doesn't need a mesh path.
        DeclareLaunchArgument(
            "mesh_file_path",
            description="Absolute path to the object's .obj mesh (meters)",
            condition=isaac_enabled,
        ),
        DeclareLaunchArgument(
            "object_config_path",
            description="Absolute path to the object's YAML configuration",
            condition=isaac_enabled,
        ),
        DeclareLaunchArgument(
            "debug",
            default_value="false",
            description="FoundationPose writes intermediate results to debug_dir",
        ),
    ]

    foundationpose_node = ComposableNode(
        name="foundationpose",
        package="isaac_ros_foundationpose",
        plugin="nvidia::isaac_ros::foundationpose::FoundationPoseNode",
        parameters=[
            object_config_path,
            {
                "debug": debug,
                "mesh_file_path": mesh_file_path,
                "refine_model_file_path": refine_model_file_path,
                "refine_engine_file_path": refine_engine_file_path,
                "refine_input_tensor_names": ["input_tensor1", "input_tensor2"],
                "refine_input_binding_names": ["input1", "input2"],
                "refine_output_tensor_names": ["output_tensor1", "output_tensor2"],
                "refine_output_binding_names": ["output1", "output2"],
                "score_model_file_path": score_model_file_path,
                "score_engine_file_path": score_engine_file_path,
                "score_input_tensor_names": ["input_tensor1", "input_tensor2"],
                "score_input_binding_names": ["input1", "input2"],
                "score_output_tensor_names": ["output_tensor"],
                "score_output_binding_names": ["output1"],
                "max_depth": 4.0,
                "min_depth": 0.1,
                "depth_qos": zed_qos,
                "color_info_qos": zed_qos,
            },
        ],
        remappings=[
            ("pose_estimation/depth_image", depth_topic),
            ("pose_estimation/image", "foundationpose/rgb8"),
            ("pose_estimation/camera_info", camera_info_topic),
            ("pose_estimation/segmentation", "foundationpose/segmentation"),
            ("pose_estimation/output", "foundationpose/output"),
        ],
    )

    format_converter_node = ComposableNode(
        name="format_converter_node",
        package="isaac_ros_image_proc",
        plugin="nvidia::isaac_ros::image_proc::ImageFormatConverterNode",
        remappings=[("image_raw", rgb_topic), ("image", "foundationpose/rgb8")],
        parameters=[
            {
                "encoding_desired": "rgb8",
                "image_width": image_width,
                "image_height": image_height,
                "input_qos": zed_qos,
            }
        ],
    )

    foundationpose_container = ComposableNodeContainer(
        name="foundationpose_container",
        namespace="",
        package="rclcpp_components",
        executable="component_container_mt",
        condition=isaac_enabled,
        composable_node_descriptions=[foundationpose_node, format_converter_node],
        output="screen",
    )

    yolo_2d_node = Node(
        name="yolo_2d",
        package="tensor_detector",
        executable="yolo_2d.py",
        output="screen",
        parameters=[
            {
                "target_class": target_class,
                "model_path": yolo_model_path,
            }
        ],
        remappings=[("image/compressed", image_topic)],
    )

    return launch.LaunchDescription(
        launch_args + [yolo_2d_node, foundationpose_container]
    )
