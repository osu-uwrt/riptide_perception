#!/usr/bin/env python3
"""yolo_orientation node.

ROS node 
parameters, publishers/subscribers, services, the camera lifecycle, and the image/depth/camera_info callbacks

Per frame it builds a Frame bundle and hands it to a DetectionProcessor
The processor returns detections + markers, which this node publishes.
"""
import os
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, CameraInfo, PointCloud2, Image
from visualization_msgs.msg import MarkerArray, Marker
from vision_msgs.msg import Detection3DArray
from cv_bridge import CvBridge
import numpy as np
import yaml
from ament_index_python.packages import get_package_share_directory
from std_srvs.srv import SetBool
from riptide_msgs2.srv import SetString
from tf2_ros.transform_listener import TransformListener
from tf2_ros import Buffer

from detection import DetectionProcessor, ProcessorConfig, Frame
from pointcloud import CloudColorMode
from yolo_model import YoloModel


class YOLONode(Node):
    def __init__(self):
        super().__init__('yolo_orientation')
        self.declare_parameters(
            namespace='',
            parameters=[
                ### YAML Params
                ('active_camera', 'ffc'),
                ('ffc_model', ''),
                ('dfc_model', ''),
                ('ffc_class_id_map', ''),
                ('dfc_class_id_map', ''),
                ('ffc_threshold', 0.9),
                ('dfc_threshold', 0.9),
                ('ffc_iou', 0.9),
                ('dfc_iou', 0.9),
                ('robot_namespace', 'talos'),

                ### Tuning (can be in YAML, but don't need to be)
                ('class_detect_shrink', 0.0),       # Shrinks the mask around the class
                ('min_points', 5),                  # Minimum points required for SVD
                ('publish_interval', 0.1),          # For visualization markers (also drives lifetime)
                ('marker_lifetime', 5.0),           # Marker lifetime in seconds (0 = persist until replaced/deleted)
                ('slalom_history_size', 10),        # The closest slalom from history is published
                ('use_incoming_timestamp', True),   # Timestamp for detection comes from image callback msg
                ('log_processing_time', False),
                ('export', True),                   # Export model
                ('print_camera_info', False),
                ('torpedo_task_camera', 'ffc'),     # Determines which camera will do weird stuff with blood/fire for now (should only be ffc)
                ('grid_step', 8),                   # Pixel spacing for the surface grid sample (smaller = denser)
                ('max_sample_points', 50),          # Cap points per patch fed to SVD/cloud (0 = uncapped)
                ('cloud_color_mode', 'class'),      # Point cloud coloring: 'class' (flat COLOR_MAP color) or 'pixel' (sampled from image)
                ('publish_box_markers', False),
            ]
        )

        self.robot_ns = self.get_parameter('robot_namespace').get_parameter_value().string_value
        self.active_camera = self.get_parameter('active_camera').get_parameter_value().string_value

        # Node-level tunables
        self.log_processing_time = self.get_parameter('log_processing_time').get_parameter_value().bool_value
        self.use_incoming_timestamp = self.get_parameter('use_incoming_timestamp').get_parameter_value().bool_value
        self.export = self.get_parameter('export').get_parameter_value().bool_value
        self.print_camera_info = self.get_parameter('print_camera_info').get_parameter_value().bool_value
        self.publish_interval = self.get_parameter('publish_interval').get_parameter_value().double_value
        self.torpedo_task_camera = self.get_parameter('torpedo_task_camera').get_parameter_value().string_value

        self.create_publishers()

        self.bridge = CvBridge()
        self.depth_image = None
        self.camera_info_gathered = False

        # tf
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # The detection logic class
        self.detector = DetectionProcessor(
            config=self._build_processor_config(),
            logger=self.get_logger(),
            now_stamp=lambda: self.get_clock().now().to_msg(),
            tf_buffer=self.tf_buffer,
        )

        self.create_switch_service()
        self.create_slalom_switch_service()

        self.setup_camera()

    def _build_processor_config(self):
        mode_str = self.get_parameter('cloud_color_mode').get_parameter_value().string_value
        try:
            color_mode = CloudColorMode(mode_str)
        except ValueError:
            self.get_logger().warning(
                f"Unknown cloud_color_mode '{mode_str}', falling back to 'class'")
            color_mode = CloudColorMode.CLASS
        return ProcessorConfig(
            class_detect_shrink=self.get_parameter('class_detect_shrink').get_parameter_value().double_value,
            min_points=self.get_parameter('min_points').get_parameter_value().integer_value,
            slalom_history_size=self.get_parameter('slalom_history_size').get_parameter_value().integer_value,
            use_incoming_timestamp=self.use_incoming_timestamp,
            publish_interval=self.publish_interval,
            marker_lifetime=self.get_parameter('marker_lifetime').get_parameter_value().double_value,
            grid_step=self.get_parameter('grid_step').get_parameter_value().integer_value,
            max_sample_points=self.get_parameter('max_sample_points').get_parameter_value().integer_value,
            cloud_color_mode=color_mode,
            publish_box_markers=self.get_parameter('publish_box_markers').get_parameter_value().bool_value,
        )

    def create_publishers(self):
        self.marker_array_publisher = self.create_publisher(MarkerArray, '~/visualization_marker_array', 10)
        self.annotated_image_publisher = self.create_publisher(CompressedImage, '~/annotated/compressed', 10)
        self.point_cloud_publisher = self.create_publisher(PointCloud2, '~/point_cloud', 10)
        self.detection_publisher = self.create_publisher(Detection3DArray, 'detected_objects', 10)

    ### Services
    def create_switch_service(self):
        self.srv = self.create_service(SetBool, 'set_camera_is_dfc', self.switch_camera_callback)
        self.get_logger().info("Camera switch service created. Call to toggle between ffc and dfc cameras")

    def create_slalom_switch_service(self):
        self.slalom_srv = self.create_service(SetString, 'set_slalom_type', self.switch_slalom_callback)
        self.get_logger().info("Slalom switch service created. Call to change name of pubbed slalom det")

    def switch_camera_callback(self, request, response):
        if getattr(self, 'camera_switch_in_progress', False):
            response.success = False
            response.message = "Camera switch already in progress."
            return response

        self.camera_switch_in_progress = True

        new_camera = 'dfc' if request.data else 'ffc'
        old_camera = self.active_camera

        if new_camera == old_camera:
            response.success = True
            response.message = f"Camera already set to {new_camera}, no change needed"
            self.camera_switch_in_progress = False
            return response

        self.get_logger().info(f"Switching from {self.active_camera} to {new_camera}")
        old_camera = self.active_camera
        self.active_camera = new_camera

        # Reconfigure after a short delay.
        self.delayed_timer = self.create_timer(0.1, self.delayed_setup)

        response.success = True
        response.message = f"Successfully switched from {old_camera} to {new_camera}"
        return response

    def switch_slalom_callback(self, request, response):
        self.detector.set_slalom_name(request.data)
        response.success = True
        response.message = f"Successfully set slalom type to {request.data}"
        return response

    def delayed_setup(self):
        try:
            self.setup_camera()
        finally:
            self.camera_switch_in_progress = False
            self.delayed_timer.cancel()

    ### Cammera lifecycle
    def setup_camera(self):
        self.get_logger().info(f"Active camera: {self.active_camera}")
        self.camera_prefix = self.active_camera
        self.frame_id = f'{self.robot_ns}/{self.camera_prefix}_left_camera_frame_optical'

        yolo_model = self.get_parameter(f'{self.active_camera}_model').get_parameter_value().string_value
        class_id_map_str = self.get_parameter(f'{self.active_camera}_class_id_map').get_parameter_value().string_value
        self.conf = self.get_parameter(f'{self.active_camera}_threshold').get_parameter_value().double_value
        self.iou = self.get_parameter(f'{self.active_camera}_iou').get_parameter_value().double_value

        self.get_logger().info(f"Yolo Model: {yolo_model}")
        self.get_logger().info(f"Class id map str: {class_id_map_str}")
        self.get_logger().info(f"Confidence Threshold: {self.conf}")
        self.get_logger().info(f"IOU: {self.iou}")

        self.load_class_id_map(class_id_map_str)

        # Task profile: torpedo (fire/blood) logic runs only on the chosen camera (ffc really, but generic here cause why not)
        # On the other camera those classes fall through to the standard path (so bins works like normal)
        self.detector.set_torpedo_enabled(self.active_camera == self.torpedo_task_camera)

        weights_dir = os.path.join(get_package_share_directory("tensor_detector"), 'weights')
        model_path = os.path.join(weights_dir, yolo_model)
        self.get_logger().info(f"Loading model path: {model_path}")
        self.model = YoloModel(model_path=model_path, export=self.export)

        self.reset_collection_variables()
        self.destroy_subscriptions()
        self.create_subscriptions()

    def load_class_id_map(self, class_id_map_str):
        # YAML is the single source of truth for the class map
        self.class_id_map = yaml.safe_load(class_id_map_str) if class_id_map_str else {}
        if not self.class_id_map:
            # Not crashing out here because there's if the other camera config is empty after switch the node would crash
            self.get_logger().warning("No class_id_map provided in params; no detections will be produced.")

        self.get_logger().info(f"Class id map: {self.class_id_map}")

    def reset_collection_variables(self):
        self.depth_image = None
        self.camera_info_gathered = False

    def destroy_subscriptions(self):
        for attr in ('zed_info_subscription', 'image_subscription', 'depth_subscription'):
            if hasattr(self, attr):
                sub = getattr(self, attr)
                try:
                    topic = sub.topic_name
                except Exception:
                    topic = "unknown"
                self.destroy_subscription(sub)
                self.get_logger().info(f"Destroying subscription: {topic}")

    def create_subscriptions(self):
        #TODO: Move these to config file, but it's type dependent
        base = f'/{self.robot_ns}/{self.camera_prefix}/zed_node'

        info_topic = f'{base}/left/camera_info'
        self.zed_info_subscription = self.create_subscription(
            CameraInfo, info_topic, self.camera_info_callback, 1)
        self.get_logger().info(f"Creating camera info subcription: {info_topic}")

        image_topic = f'{base}/left/image_rect_color/compressed'
        self.image_subscription = self.create_subscription(
            CompressedImage, image_topic, self.image_callback, 1)
        self.get_logger().info(f"Creating image subcription: {image_topic}")

        depth_topic = f'{base}/depth/depth_registered'
        self.depth_subscription = self.create_subscription(
            Image, depth_topic, self.depth_callback, 1)
        self.get_logger().info(f"Creating depth subcription: {depth_topic}")

    ### Callbacks
    def has_subscribers(self, publisher):
        return publisher.get_subscription_count() > 0

    def camera_info_callback(self, msg):
        if not self.camera_info_gathered:
            if self.print_camera_info:
                self.get_logger().info(f"Camera info: {msg}")
            self.intrinsic_matrix = np.array(msg.k).reshape((3, 3))
            self.fx = msg.k[0]
            self.cx = msg.k[2]
            self.fy = msg.k[4]
            self.cy = msg.k[5]
            self.camera_info_gathered = True

    def depth_callback(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def _stamp(self, msg):
        if self.use_incoming_timestamp:
            return msg.header.stamp
        return self.get_clock().now().to_msg()

    ### THE MEAT
    def image_callback(self, msg: CompressedImage):
        start_time = time.perf_counter() if self.log_processing_time else None

        if self.depth_image is None or not self.camera_info_gathered:
            self.get_logger().warning(
                "Skipping image because either no depth image or camera info is available.",
                throttle_duration_sec=1)
            return

        # Get the image from the msg
        cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        if cv_image is None:
            return

        # Run inference
        results = self.model.infer(cv_image, conf=self.conf, iou=self.iou)

        # Store everything we need in the frame dataclass 🤌
        frame = Frame(
            image=cv_image,
            depth=self.depth_image,
            fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy,
            K=self.intrinsic_matrix,
            frame_id=self.frame_id,
            timestamp=msg.header.stamp,
            class_id_map=self.class_id_map,
            conf=self.conf,
            want_markers=self.has_subscribers(self.marker_array_publisher),
            want_cloud=self.has_subscribers(self.point_cloud_publisher),
            want_overlay_points=self.has_subscribers(self.annotated_image_publisher)
        )

        # The generic planar and task specific detection logic
        detections, markers = self.detector.process(results, frame)

        # Publish markers: clear the previous set, then add the current one
        if self.has_subscribers(self.marker_array_publisher):
            marker_array = MarkerArray()
            clear = Marker()
            clear.action = Marker.DELETEALL
            marker_array.markers.append(clear)
            if markers:
                marker_array.markers.extend(markers)
            self.marker_array_publisher.publish(marker_array)

        # Publish point cloud for visualization
        if self.has_subscribers(self.point_cloud_publisher):
            cloud = self.detector.take_point_cloud(self.frame_id, self._stamp(msg))
            if cloud is not None:
                self.point_cloud_publisher.publish(cloud)

        # Publish annotated frame for visualization
        if self.has_subscribers(self.annotated_image_publisher):
            annotated_frame = results[0].plot()
            self.annotated_image_publisher.publish(self.bridge.cv2_to_compressed_imgmsg(annotated_frame))

        # Publish detections for mapping
        if self.has_subscribers(self.detection_publisher):
            self.detection_publisher.publish(detections)

        # Debug log
        if self.log_processing_time:
            elapsed = time.perf_counter() - start_time
            self.get_logger().info(f"total={elapsed * 1000:.0f}ms ({1 / elapsed:.1f} fps)")



def main(args=None):
    rclpy.init(args=args)
    yolo_node = YOLONode()
    rclpy.spin(yolo_node)
    yolo_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()