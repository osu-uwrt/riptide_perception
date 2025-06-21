#!/usr/bin/env python3

# Standard library imports
import os
import time
import yaml

# Third-party imports
import numpy as np
import rclpy
from ultralytics import YOLO

# ROS2 imports
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_srvs.srv import SetBool
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesis

class CoreYOLONode(Node):
    def __init__(self):
        super().__init__('core_yolo_detection')
        
        # Declare parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                # Robot configuration
                ('robot_namespace', 'talos'),
                
                # Camera selection and models
                ('active_camera', 'ffc'),
                ('ffc_model', ''),
                ('dfc_model', ''),
                ('ffc_class_id_map', ''),
                ('dfc_class_id_map', ''),
                ('ffc_threshold', 0.9),
                ('dfc_threshold', 0.9),
                ('ffc_iou', 0.9),
                ('dfc_iou', 0.9),
                
                # Processing options
                ('log_processing_time', False),
                ('use_incoming_timestamp', True),
                ('export_to_tensorrt', False),
                ('enable_annotated_output', True),
                
                # Topic names
                ('annotated_image_topic', 'yolo_annotated'),
                ('detections_topic', 'yolo_detections'),
                ('camera_switch_service', 'set_camera_is_dfc'),
                
                # Camera topic patterns (use {robot} and {camera} as placeholders)
                ('camera_info_topic_pattern', '/{robot}/{camera}/zed_node/left/camera_info'),
                ('image_topic_pattern', '/{robot}/{camera}/zed_node/left/image_rect_color'),
                ('frame_id_pattern', '{robot}/{camera}_left_camera_optical_frame'),
                
                # Queue sizes
                ('camera_info_queue_size', 1),
                ('image_queue_size', 10),
                ('publisher_queue_size', 10),
                
                # Camera switch timing
                ('camera_switch_delay', 0.1),
            ]
        )

        # Load all parameters
        self.load_parameters()
        
        # Initialize components
        self.bridge = CvBridge()
        self.camera_info_gathered = False
        self.detection_id_counter = 0
        self.camera_switch_in_progress = False
        
        # Create publishers
        self.create_publishers()
        
        # Create camera switch service
        self.create_switch_service()
        
        # Setup initial camera
        self.setup_camera()

    def load_parameters(self):
        """Load all parameters from ROS parameter server"""
        # Robot configuration
        self.robot_namespace = self.get_parameter('robot_namespace').get_parameter_value().string_value
        
        # Camera configuration
        self.active_camera = self.get_parameter('active_camera').get_parameter_value().string_value
        
        # Processing options
        self.log_processing_time = self.get_parameter('log_processing_time').get_parameter_value().bool_value
        self.use_incoming_timestamp = self.get_parameter('use_incoming_timestamp').get_parameter_value().bool_value
        self.export = self.get_parameter('export_to_tensorrt').get_parameter_value().bool_value
        self.enable_annotated_output = self.get_parameter('enable_annotated_output').get_parameter_value().bool_value
        
        # Topic names
        self.annotated_topic = self.get_parameter('annotated_image_topic').get_parameter_value().string_value
        self.detections_topic = self.get_parameter('detections_topic').get_parameter_value().string_value
        self.camera_switch_service_name = self.get_parameter('camera_switch_service').get_parameter_value().string_value
        
        # Topic patterns
        self.camera_info_topic_pattern = self.get_parameter('camera_info_topic_pattern').get_parameter_value().string_value
        self.image_topic_pattern = self.get_parameter('image_topic_pattern').get_parameter_value().string_value
        self.frame_id_pattern = self.get_parameter('frame_id_pattern').get_parameter_value().string_value
        
        # Queue sizes
        self.camera_info_queue_size = self.get_parameter('camera_info_queue_size').get_parameter_value().integer_value
        self.image_queue_size = self.get_parameter('image_queue_size').get_parameter_value().integer_value
        self.publisher_queue_size = self.get_parameter('publisher_queue_size').get_parameter_value().integer_value
        
        # Timing
        self.camera_switch_delay = self.get_parameter('camera_switch_delay').get_parameter_value().double_value

    def create_publishers(self):
        """Create ROS2 publishers"""
        self.annotated_image_publisher = self.create_publisher(
            Image, self.annotated_topic, self.publisher_queue_size
        )
        self.detection_publisher = self.create_publisher(
            Detection2DArray, self.detections_topic, self.publisher_queue_size
        )

    def create_switch_service(self):
        """Create service for camera switching"""
        self.srv = self.create_service(
            SetBool, self.camera_switch_service_name, self.switch_camera_callback
        )
        self.get_logger().info(f"Camera switch service created: {self.camera_switch_service_name}")

    def switch_camera_callback(self, request, response):
        """Handle camera switching requests"""
        if self.camera_switch_in_progress:
            response.success = False
            response.message = "Camera switch already in progress"
            return response

        self.camera_switch_in_progress = True
        new_camera = 'dfc' if request.data else 'ffc'
        old_camera = self.active_camera

        if new_camera == old_camera:
            response.success = True
            response.message = f"Camera already set to {new_camera}"
            self.camera_switch_in_progress = False
            return response

        self.get_logger().info(f"Switching from {old_camera} to {new_camera}")
        self.active_camera = new_camera
        
        # Schedule reconfiguration
        self.delayed_timer = self.create_timer(self.camera_switch_delay, self.delayed_setup)
        
        response.success = True
        response.message = f"Successfully switched from {old_camera} to {new_camera}"
        return response

    def delayed_setup(self):
        """Delayed camera setup after switching"""
        try:
            self.setup_camera()
        finally:
            self.camera_switch_in_progress = False
            self.delayed_timer.cancel()

    def setup_camera(self):
        """Setup camera-specific configuration and subscriptions"""
        self.get_logger().info(f"Setting up camera: {self.active_camera}")
        
        # Set frame ID using pattern with robot namespace
        self.frame_id = self.frame_id_pattern.format(
            robot=self.robot_namespace, 
            camera=self.active_camera
        )
        
        # Load camera-specific parameters
        self.load_camera_parameters()
        
        # Load YOLO model
        self.load_model()
        
        # Reset state
        self.reset_camera_state()
        
        # Recreate subscriptions
        self.destroy_subscriptions()
        self.create_subscriptions()

    def load_camera_parameters(self):
        """Load camera-specific parameters"""
        yolo_model = self.get_parameter(f'{self.active_camera}_model').get_parameter_value().string_value
        class_id_map_str = self.get_parameter(f'{self.active_camera}_class_id_map').get_parameter_value().string_value
        self.conf_threshold = self.get_parameter(f'{self.active_camera}_threshold').get_parameter_value().double_value
        self.iou_threshold = self.get_parameter(f'{self.active_camera}_iou').get_parameter_value().double_value
        
        self.get_logger().info(f"Model: {yolo_model}")
        self.get_logger().info(f"Confidence: {self.conf_threshold}, IOU: {self.iou_threshold}")
        
        # Load class ID mapping
        self.load_class_id_map(class_id_map_str)
        
        # Store model path for loading
        self.yolo_model_path = yolo_model

    def load_class_id_map(self, class_id_map_str):
        """Load and setup class ID mapping"""
        self.class_id_map = yaml.safe_load(class_id_map_str) if class_id_map_str else {}
        
        # Require explicit class ID mapping
        if not self.class_id_map:
            raise ValueError(
                f"No class ID mapping provided for camera '{self.active_camera}'. "
                f"Please set the '{self.active_camera}_class_id_map' parameter with a valid YAML mapping."
            )
        
        self.get_logger().info(f"Class mapping: {self.class_id_map}")

    def load_model(self):
        """Load YOLO model"""
        tensorrt_wrapper_dir = get_package_share_directory("tensor_detector")
        yolo_model_path = os.path.join(tensorrt_wrapper_dir, 'weights', self.yolo_model_path)
        
        # Check for .engine version
        engine_model_path = yolo_model_path.replace('.pt', '.engine')
        if yolo_model_path.endswith(".pt") and os.path.exists(engine_model_path):
            yolo_model_path = engine_model_path
        
        self.get_logger().info(f"Loading model: {yolo_model_path}")
        self.model = YOLO(yolo_model_path, task="segment")
        
        # Export to engine if requested
        if self.export and yolo_model_path.endswith(".pt"):
            self.model.export(format="engine")

    def reset_camera_state(self):
        """Reset camera-related state variables"""
        self.camera_info_gathered = False
        self.detection_id_counter = 0

    def destroy_subscriptions(self):
        """Destroy existing subscriptions"""
        for attr_name in ['camera_info_subscription', 'image_subscription']:
            if hasattr(self, attr_name):
                subscription = getattr(self, attr_name)
                try:
                    topic = subscription.topic_name
                except:
                    topic = "unknown"
                self.destroy_subscription(subscription)
                self.get_logger().info(f"Destroyed subscription: {topic}")

    def create_subscriptions(self):
        """Create camera subscriptions"""
        # Camera info subscription
        camera_info_topic = self.camera_info_topic_pattern.format(
            robot=self.robot_namespace, 
            camera=self.active_camera
        )
        self.camera_info_subscription = self.create_subscription(
            CameraInfo, 
            camera_info_topic, 
            self.camera_info_callback, 
            self.camera_info_queue_size
        )
        self.get_logger().info(f"Subscribed to: {camera_info_topic}")
        
        # Image subscription
        image_topic = self.image_topic_pattern.format(
            robot=self.robot_namespace, 
            camera=self.active_camera
        )
        self.image_subscription = self.create_subscription(
            Image, 
            image_topic, 
            self.image_callback, 
            self.image_queue_size
        )
        self.get_logger().info(f"Subscribed to: {image_topic}")

    def camera_info_callback(self, msg):
        """Process camera info message"""
        if not self.camera_info_gathered:
            self.intrinsic_matrix = np.array(msg.k).reshape((3, 3))
            self.fx, self.fy = msg.k[0], msg.k[4]
            self.cx, self.cy = msg.k[2], msg.k[5]
            self.distortion_matrix = np.array(msg.d)
            self.camera_info_gathered = True
            self.get_logger().info("Camera info gathered")

    def image_callback(self, msg: Image):
        """Main image processing callback"""
        # Skip inference if no one is listening
        if not self.has_any_subscribers():
            return
            
        if self.log_processing_time:
            start_time = time.time()

        if not self.camera_info_gathered:
            self.get_logger().warning("Skipping - no camera info", throttle_duration_sec=1)
            return

        # Convert ROS image to OpenCV
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        if cv_image is None:
            return

        # Run YOLO inference
        results = self.model(cv_image, verbose=False, iou=self.iou_threshold, conf=self.conf_threshold)
        
        # Process detections
        detections = self.process_detections(results, msg.header.stamp)
        
        # Publish results
        self.publish_results(results, detections, msg.header.stamp)
        
        if self.log_processing_time:
            processing_time = time.time() - start_time
            self.get_logger().info(f"Processing time: {processing_time*1000:.1f}ms, FPS: {1/processing_time:.1f}")

    def process_detections(self, results, timestamp):
        """Process YOLO results into Detection2D messages"""
        detections = Detection2DArray()
        detections.header.frame_id = self.frame_id
        detections.header.stamp = timestamp if self.use_incoming_timestamp else self.get_clock().now().to_msg()
        
        for result in results:
            if result.boxes is None:
                continue
                
            for box in result.boxes.cpu().numpy():
                if box.conf[0] <= self.conf_threshold:
                    continue
                
                class_id = int(box.cls[0])
                if class_id not in self.class_id_map:
                    continue
                
                detection = self.create_detection2d(box, class_id, timestamp)
                if detection:
                    detections.detections.append(detection)
        
        return detections

    def create_detection2d(self, box, class_id, timestamp):
        """Create a Detection2D message from YOLO box"""
        x_min, y_min, x_max, y_max = map(float, box.xyxy[0])
        confidence = float(box.conf[0])
        class_name = self.class_id_map[class_id]
        
        # Create Detection2D message
        detection = Detection2D()
        detection.header.frame_id = self.frame_id
        detection.header.stamp = timestamp if self.use_incoming_timestamp else self.get_clock().now().to_msg()
        
        # Set bounding box
        detection.bbox.center.position.x = (x_min + x_max) / 2.0
        detection.bbox.center.position.y = (y_min + y_max) / 2.0
        detection.bbox.center.theta = 0.0
        detection.bbox.size_x = x_max - x_min
        detection.bbox.size_y = y_max - y_min
        
        # Set hypothesis
        hypothesis = ObjectHypothesis()
        hypothesis.class_id = class_name
        hypothesis.score = confidence
        detection.results.append(hypothesis)
        
        # Store detection ID
        detection.id = str(self.generate_detection_id())
        
        return detection

    def generate_detection_id(self):
        """Generate unique detection ID"""
        self.detection_id_counter += 1
        return self.detection_id_counter

    def publish_results(self, results, detections, timestamp):
        """Publish detection results and annotated image"""
        # Publish detections
        if self.has_subscribers(self.detection_publisher):
            self.detection_publisher.publish(detections)
        
        # Publish annotated image (if enabled)
        if self.enable_annotated_output and self.has_subscribers(self.annotated_image_publisher):
            annotated_frame = results[0].plot()
            annotated_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding="bgr8")
            annotated_msg.header.stamp = timestamp if self.use_incoming_timestamp else self.get_clock().now().to_msg()
            annotated_msg.header.frame_id = self.frame_id
            self.annotated_image_publisher.publish(annotated_msg)

    def has_subscribers(self, publisher):
        """Check if publisher has subscribers"""
        return publisher.get_subscription_count() > 0

    def has_any_subscribers(self):
        """Check if any output publisher has subscribers"""
        has_detection_subs = self.has_subscribers(self.detection_publisher)
        has_annotated_subs = self.enable_annotated_output and self.has_subscribers(self.annotated_image_publisher)
        
        return has_detection_subs or has_annotated_subs


def main(args=None):
    rclpy.init(args=args)
    node = CoreYOLONode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()