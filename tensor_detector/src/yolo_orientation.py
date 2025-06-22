#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import numpy as np
from visualization_msgs.msg import Marker, MarkerArray
from vision_msgs.msg import Detection3DArray, Detection3D, ObjectHypothesisWithPose, ObjectHypothesis
from geometry_msgs.msg import Point32
from scipy.spatial.transform import Rotation as R
from ament_index_python.packages import get_package_share_directory
import time
import os
import yaml
import math
from std_srvs.srv import Trigger

class YOLONode(Node):
	def __init__(self):
		super().__init__('yolo_orientation')
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
				('depth_info_topic_pattern', '/{robot}/{camera}/zed_node/depth/camera_info'),
                ('image_topic_pattern', '/{robot}/{camera}/zed_node/left/image_rect_color'),
				('depth_topic_pattern', '/{robot}/{camera}/zed_node/depth/depth_registered'),
                ('frame_id_pattern', '{robot}/{camera}_left_camera_optical_frame'),
                
                # Queue sizes
                ('camera_info_queue_size', 1),
				('depth_info_queue_size', 1),
                ('image_queue_size', 10),
				('depth_queue_size', 10),
                ('publisher_queue_size', 10),
                
                # Camera switch timing
                ('camera_switch_delay', 0.1),
                
                # Detection and processing parameters
                ('print_camera_info', False),
                ('class_detect_shrink', 0.15),
                ('min_points', 5),
                ('publish_interval', 0.1),
                ('history_size', 10),
                ('default_normal_x', 0.0),
                ('default_normal_y', 0.0),
                ('default_normal_z', 1.0),
                ('map_min_area', 50),
            ]
		)

		# Load all parameters
		self.load_parameters()

		# Color map for classes published to markers
		self.color_map = {
			'bin_target': (1.0, 0.0, 0.0),
			'mapping_map': (0.0, 1.0, 0.0),
			'mapping_hole': (0.0, 0.0, 1.0),
			'mapping_largest_hole': (0.0, 0.0, 1.0),
			'mapping_smallest_hole': (1.0, 0.0, 0.0),
			'gate_hot': (1.0, 1.0, 1.0)
		}

		self.create_publishers()

		# CV and bridge init
		self.bridge = CvBridge()

		# Init global vars
		self.depth_image = None
		self.camera_info_gathered = False
		self.depth_info_gathered = False
		self.gray_image = None
		self.mask = None
		self.accumulated_points = []
		self.detection_id_counter = 0
		self.centroid_history = {}
		self.orientation_history = {}
		self.temp_markers = []
		self.last_publish_time = time.time()
		self.open_torpedo_centroid = None
		self.open_torpedo_quat = None
		self.closed_torpedo_centroid = None
		self.closed_torpedo_quat = None
		self.holes = []
		self.latest_bbox_class_7 = None
		self.latest_bbox_class_8 = None
		self.detection_timestamp = None
		self.detection_time = None
		self.mapping_holes = []
		self.mapping_map_centroid = None
		self.mapping_map_quat = None
		self.largest_hole = None
		self.smallest_hole = None
		self.latest_buoy = None
		self.plane_normal = None

		self.create_switch_service()

		# Set up the camera based on the active_camera parameter
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
		self.depth_info_topic_pattern = self.get_parameter('depth_info_topic_pattern').get_parameter_value().string_value
		self.image_topic_pattern = self.get_parameter('image_topic_pattern').get_parameter_value().string_value
		self.depth_topic_pattern = self.get_parameter('depth_topic_pattern').get_parameter_value().string_value
		self.frame_id_pattern = self.get_parameter('frame_id_pattern').get_parameter_value().string_value
		
		# Queue sizes
		self.camera_info_queue_size = self.get_parameter('camera_info_queue_size').get_parameter_value().integer_value
		self.depth_info_queue_size = self.get_parameter('depth_info_queue_size').get_parameter_value().integer_value
		self.image_queue_size = self.get_parameter('image_queue_size').get_parameter_value().integer_value
		self.depth_queue_size = self.get_parameter('depth_queue_size').get_parameter_value().integer_value
		self.publisher_queue_size = self.get_parameter('publisher_queue_size').get_parameter_value().integer_value
		
		# Timing
		self.camera_switch_delay = self.get_parameter('camera_switch_delay').get_parameter_value().double_value
		
		# Detection and processing parameters
		self.print_camera_info = self.get_parameter('print_camera_info').get_parameter_value().bool_value
		self.class_detect_shrink = self.get_parameter('class_detect_shrink').get_parameter_value().double_value
		self.min_points = self.get_parameter('min_points').get_parameter_value().integer_value
		self.publish_interval = self.get_parameter('publish_interval').get_parameter_value().double_value
		self.history_size = self.get_parameter('history_size').get_parameter_value().integer_value
		
		# Default normal vector
		default_normal_x = self.get_parameter('default_normal_x').get_parameter_value().double_value
		default_normal_y = self.get_parameter('default_normal_y').get_parameter_value().double_value
		default_normal_z = self.get_parameter('default_normal_z').get_parameter_value().double_value
		self.default_normal = np.array([default_normal_x, default_normal_y, default_normal_z])
		
		self.map_min_area = self.get_parameter('map_min_area').get_parameter_value().integer_value
		
	def create_publishers(self):
		# Creating publishers
		self.marker_array_publisher = self.create_publisher(MarkerArray, 'visualization_marker_array', 10)
		self.publisher = self.create_publisher(Image, 'yolo', 10)
		self.point_cloud_publisher = self.create_publisher(PointCloud, 'point_cloud', 10)
		self.detection_publisher = self.create_publisher(Detection3DArray, 'detected_objects', 10)

	def delayed_setup(self):
		try:
			self.setup_camera()
		finally:
			self.camera_switch_in_progress = False
			self.delayed_timer.cancel()

	def create_switch_service(self):
		# Create the service for camera switching
		self.srv = self.create_service(Trigger, 'switch_camera', self.switch_camera_callback)
		self.get_logger().info("Camera switch service created. Call to toggle between ffc and dfc cameras")

	def switch_camera_callback(self, request, response):
		if getattr(self, 'camera_switch_in_progress', False):
			response.success = False
			response.message = "Camera switch already in progress."
			return response

		self.camera_switch_in_progress = True

		new_camera = 'dfc' if self.active_camera == 'ffc' else 'ffc'
		self.get_logger().info(f"Switching from {self.active_camera} to {new_camera}")
		old_camera = self.active_camera
		self.active_camera = new_camera

		# Schedule reconfiguration after a short delay (e.g., 0.1 seconds)
		self.delayed_timer = self.create_timer(self.camera_switch_delay, self.delayed_setup)

		response.success = True
		response.message = f"Successfully switched from {old_camera} to {new_camera}"
		return response

	def setup_camera(self):
		self.get_logger().info(f"Active camera: {self.active_camera}")

		# Set the camera prefix
		self.camera_prefix = self.active_camera

		# Set frame ID
		self.frame_id = self.frame_id_pattern.format(robot=self.robot_namespace, camera=self.active_camera)

		# Get camera-specific parameters
		yolo_model = self.get_parameter(f'{self.active_camera}_model').get_parameter_value().string_value
		class_id_map_str = self.get_parameter(f'{self.active_camera}_class_id_map').get_parameter_value().string_value
		self.conf = self.get_parameter(f'{self.active_camera}_threshold').get_parameter_value().double_value
		self.iou = self.get_parameter(f'{self.active_camera}_iou').get_parameter_value().double_value

		self.get_logger().info(f"Yolo Model: {yolo_model}")
		self.get_logger().info(f"Class id map str: {class_id_map_str}")
		self.get_logger().info(f"Confidence Threshold: {self.conf}")
		self.get_logger().info(f"IOU: {self.iou}")

		self.load_class_id_map(class_id_map_str)
		self.load_model(yolo_model)
		
		self.reset_collection_variables()
		self.reset_subscriptions()
		
	def load_class_id_map(self, class_id_map_str):
		# Load class ID map
		self.class_id_map = yaml.safe_load(class_id_map_str) if class_id_map_str else {}

		# Add default class ID map if none provided
		if not self.class_id_map:
			self.get_logger().info(f"No class id map found, defaulting to:")
			if self.active_camera == 'ffc':
				self.class_id_map = {
					0: 'bin_target',
					1: 'mapping_map', 
					2: 'mapping_hole', 
					3: 'gate_hot',
					4: 'gate_cold',
					5: 'bin_temperature',
					6: 'bin'
				}
			else:  # dfc
				self.class_id_map = {
					0: 'bin_target'
				}
		else:
			self.get_logger().info(f"Class id map found:")

		if self.active_camera == 'ffc':
			# Update internal class_id_map
			self.class_id_map.update({
				21: "mapping_largest_hole",
				22: "mapping_smallest_hole"
			})

		self.get_logger().info(f"{self.class_id_map}")

	def reset_collection_variables(self):
		# Reset camera-related variables
		self.depth_image = None
		self.camera_info_gathered = False
		self.depth_info_gathered = False

	def reset_subscriptions(self):
		self.destroy_subscriptions()
		self.create_subscriptions()

	def destroy_subscriptions(self):
		"""Destroy all subscriptions"""
		self.destroy_sub('zed_info_subscription', 'camera info')
		self.destroy_sub('depth_info_subscription', 'depth info') 
		self.destroy_sub('image_subscription', 'image')
		self.destroy_sub('depth_subscription', 'depth')

	def destroy_sub(self, subscription_attr, description):
		"""Destroy subscription with logging"""
		if hasattr(self, subscription_attr):
			subscription = getattr(self, subscription_attr)
			try:
				topic = subscription.topic_name
			except Exception:
				topic = "unknown"
			self.destroy_subscription(subscription)
			self.get_logger().info(f"Destroying {description} subscription: {topic}")

	def create_subscriptions(self):
		"""Create new subscriptions"""
		self.camera_info_subscription = self.create_sub(
			self.camera_info_topic_pattern, CameraInfo, 
			self.camera_info_callback, self.camera_info_queue_size)
		
		self.depth_info_subscription = self.create_sub(
			self.depth_info_topic_pattern, CameraInfo,
			self.depth_info_callback, self.depth_info_queue_size)
		
		self.image_subscription = self.create_sub(
			self.image_topic_pattern, Image,
			self.image_callback, self.image_queue_size)
		
		self.depth_subscription = self.create_sub(
			self.depth_topic_pattern, Image,
			self.depth_callback, self.depth_queue_size)

	def create_sub(self, topic_pattern, msg_type, callback, queue_size):
		"""Create subscription with topic formatting and logging"""
		topic = topic_pattern.format(robot=self.robot_namespace, camera=self.active_camera)
		subscription = self.create_subscription(msg_type, topic, callback, queue_size)
		self.get_logger().info(f"Subscribed to: {topic}")
		return subscription

	def load_model(self, yolo_model):
		# Load model
		tensorrt_wrapper_dir = get_package_share_directory("tensor_detector")
		yolo_model_path = os.path.join(tensorrt_wrapper_dir, 'weights', yolo_model)
		self.get_logger().info(f"Loading model path: {yolo_model_path}")
		self.initialize_yolo(yolo_model_path)

	def initialize_yolo(self, yolo_model_path):
		# Check if the .engine version of the model exists
		engine_model_path = yolo_model_path.replace('.pt', '.engine')
		if yolo_model_path.endswith(".pt") and os.path.exists(engine_model_path):
			# If the .engine file exists, use it instead
			yolo_model_path = engine_model_path
 
		self.model = YOLO(yolo_model_path, task="segment")
 
		# Check if the model needs to be exported
		if self.export and yolo_model_path.endswith(".pt"):
			self.model.export(format="engine")
			# Update the model path to use the .engine file
			# Note: This will create a new .engine file if it didn't exist before
			self.initialize_yolo(engine_model_path)  # Recursive call with the new .engine path
		
	def is_inside_bbox(self, inner_bbox, outer_bbox):
		inner_x_min, inner_y_min, inner_x_max, inner_y_max = inner_bbox
		outer_x_min, outer_y_min, outer_x_max, outer_y_max = outer_bbox
 
		return (inner_x_min >= outer_x_min and inner_x_max <= outer_x_max and
				inner_y_min >= outer_y_min and inner_y_max <= outer_y_max)

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
 
			self.distortion_matrix = np.array(msg.d)
 
			self.camera_info_gathered = True
 
	def depth_info_callback(self, msg):
		if not self.depth_info_gathered:
			self.depth_intrinsic_matrix = np.array(msg.k).reshape((3, 3))
			self.depth_distortion_matrix = np.array(msg.d)
			self.depth_info_gathered = True
 
	def depth_callback(self, msg):
		self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

	def image_callback(self, msg: Image):
		"""Simplified image callback that delegates to class-specific handlers"""
		
		if self.log_processing_time:
			self.detection_time = time.time()

		if self.depth_image is None or not self.camera_info_gathered:
			self.get_logger().warning("Skipping image because either no depth image or camera info is available.", throttle_duration_sec=1)
			return

		cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
		self.gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
		
		if cv_image is None:
			return
		
		results = self.model(cv_image, verbose=False, iou=self.iou, conf=self.conf)

		# Initialize detection array
		detections = Detection3DArray()
		detections.header.frame_id = self.frame_id
		self.detection_timestamp = msg.header.stamp
		
		if self.use_incoming_timestamp:
			detections.header.stamp = msg.header.stamp
		else:
			detections.header.stamp = self.get_clock().now().to_msg()

		# Initialize mask
		if self.mask is None or self.mask.shape[:2] != cv_image.shape[:2]:
			self.mask = np.zeros(cv_image.shape[:2], dtype=np.uint8)

		# Reset mapping variables for each frame
		self.reset_frame_variables()

		# Process each detection
		for result in results:
			for box in result.boxes.cpu().numpy():
				if box.conf[0] <= self.conf:
					continue
					
				class_id = box.cls[0]
				if class_id not in self.class_id_map:
					continue
					
				detection = self.process_detection_by_class(box, cv_image, result)
				if detection:
					detections.detections.append(detection)

		# Handle special case for mapping holes
		self.process_mapping_holes(detections, cv_image)

		# Publish results
		self.publish_frame_results(results, detections)

	def reset_frame_variables(self):
		"""Reset variables that need to be cleared each frame"""
		self.mapping_holes = []
		self.largest_hole = None
		self.smallest_hole = None

	def process_detection_by_class(self, box, cv_image, result):
		"""Route detection processing to appropriate class handler"""
		class_id = int(box.cls[0])
		class_name = self.class_id_map.get(class_id, "Unknown")
		conf = box.conf[0]
		
		# Route to specific handlers
		if class_name == "mapping_hole":
			return self.handle_mapping_hole(box)
		elif class_name == "mapping_map":
			return self.handle_mapping_map(box, cv_image, result, conf)
		else:
			return self.handle_generic_detection(box, cv_image, result, conf, class_name)

	def handle_mapping_hole(self, box):
		"""Handle mapping hole detections (store for later processing)"""
		x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
		
		# Store hole for later processing
		timestamp = self.detection_timestamp if self.use_incoming_timestamp else self.get_clock().now().to_msg()
		self.holes.append(((x_min, y_min, x_max, y_max), timestamp))
		self.mapping_holes.append(box)
		
		return None  # Will be processed later in process_mapping_holes

	def handle_mapping_map(self, box, cv_image, result, conf):
		"""Handle mapping map detections"""
		x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
		
		# Check minimum area requirement
		map_width = x_max - x_min
		map_height = y_max - y_min
		map_area = max(map_width, map_height)
		
		if map_area < self.map_min_area:
			self.get_logger().info(f"Not Publishing: map area {map_area} < {self.map_min_area}")
			return None
		
		self.get_logger().info(f"Publishing: map area {map_area} >= {self.map_min_area}")
		self.latest_bbox_class_1 = (x_min, y_min, x_max, y_max)
		
		# Process as plane detection with hole exclusion
		detection = self.create_plane_detection(box, cv_image, result, conf, "torpedo", exclude_holes=True)
		
		if detection:
			# Store mapping data for hole processing
			centroid = [detection.results[0].pose.pose.position.x,
					   detection.results[0].pose.pose.position.y, 
					   detection.results[0].pose.pose.position.z]
			quat = [detection.results[0].pose.pose.orientation.x,
					detection.results[0].pose.pose.orientation.y,
					detection.results[0].pose.pose.orientation.z,
					detection.results[0].pose.pose.orientation.w]
			
			self.mapping_map_centroid = centroid
			self.mapping_map_quat = quat
		
		return detection

	def handle_generic_detection(self, box, cv_image, result, conf, class_name):
		"""Handle any other detection types"""
		return self.create_plane_detection(box, cv_image, result, conf, class_name)

	def create_plane_detection(self, box, cv_image, result, conf, class_name, exclude_holes=False):
		"""Create a detection using plane fitting from feature points"""
		x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
		bbox_center_x = (x_min + x_max) / 2
		bbox_center_y = (y_min + y_max) / 2
		bbox_width = x_max - x_min
		bbox_height = y_max - y_min
		
		# Apply shrinking to exclude edges
		shrink_x = bbox_width * self.class_detect_shrink
		shrink_y = bbox_height * self.class_detect_shrink
		
		x_min_shrunk = int(x_min + shrink_x)
		x_max_shrunk = int(x_max - shrink_x)
		y_min_shrunk = int(y_min + shrink_y)
		y_max_shrunk = int(y_max - shrink_y)
		
		# Create mask for region of interest
		mask_roi = self.create_detection_mask(x_min_shrunk, y_min_shrunk, x_max_shrunk, y_max_shrunk, exclude_holes, result)
		
		if mask_roi is None:
			return None
		
		# Extract and mask the gray image region
		cropped_gray_image = self.gray_image[y_min_shrunk:y_max_shrunk, x_min_shrunk:x_max_shrunk]
		masked_gray_image = cv2.bitwise_and(cropped_gray_image, cropped_gray_image, mask=mask_roi)
		
		# Detect features
		good_features = cv2.goodFeaturesToTrack(masked_gray_image, maxCorners=0, qualityLevel=0.02, minDistance=1)
		
		if good_features is None:
			return None
		
		# Adjust feature coordinates to full image space
		good_features[:, 0, 0] += x_min_shrunk
		good_features[:, 0, 1] += y_min_shrunk
		
		feature_points = [pt[0] for pt in good_features]
		
		# Get 3D points and fit plane
		points_3d = self.get_3d_points(feature_points, cv_image)
		
		if points_3d is None or len(points_3d) < self.min_points:
			return None
		
		plane_result = self.fit_plane_to_points(points_3d)
		if plane_result is None:
			return None
			
		normal, _, centroid = plane_result
		
		# Calculate centroid using bbox center and plane centroid z
		final_centroid = self.calculate_centroid(bbox_center_x, bbox_center_y, centroid[2])
		if final_centroid is None:
			return None
		
		if normal[2] > 0:
			normal = -normal
		
		self.plane_normal = normal
		quat, _ = self.calculate_quaternion_and_euler_angles(normal)
		
		# Publish marker
		self.publish_marker(quat, final_centroid, class_name, bbox_width, bbox_height)
		
		# Create detection message
		return self.create_detection_message(class_name, final_centroid, quat, conf)

	def create_detection_mask(self, x_min, y_min, x_max, y_max, exclude_holes=False, result=None):
		"""Create a mask for the detection region, optionally excluding holes"""
		if result and hasattr(result, 'masks') and result.masks is not None:
			self.mask.fill(0)
			for contour in result.masks.xy:
				contour = np.array(contour, dtype=np.int32)
				cv2.fillPoly(self.mask, [contour], 255)
		
		mask_roi = self.mask[y_min:y_max, x_min:x_max].copy()
		
		if exclude_holes and self.holes:
			# Calculate padding based on bounding box size
			padding_x = int((x_max - x_min) * 0.1)
			padding_y = int((y_max - y_min) * 0.1)
			
			for hole_bbox, _ in self.holes:
				hole_x_min, hole_y_min, hole_x_max, hole_y_max = hole_bbox
				
				# Adjust hole coordinates relative to the ROI
				adj_hole_x_min = max(hole_x_min - x_min - padding_x, 0)
				adj_hole_y_min = max(hole_y_min - y_min - padding_y, 0)
				adj_hole_x_max = min(hole_x_max - x_min + padding_x, mask_roi.shape[1])
				adj_hole_y_max = min(hole_y_max - y_min + padding_y, mask_roi.shape[0])
				
				# Exclude hole region from mask
				mask_roi[adj_hole_y_min:adj_hole_y_max, adj_hole_x_min:adj_hole_x_max] = 0
			
			# Apply morphological operations to refine exclusion zones
			kernel = np.ones((5, 5), np.uint8)
			mask_roi = cv2.dilate(mask_roi, kernel, iterations=1)
			mask_roi = cv2.erode(mask_roi, kernel, iterations=1)
		
		return mask_roi

	def process_mapping_holes(self, detections, cv_image):
		"""Process mapping holes after all detections are complete"""
		if len(self.mapping_holes) != 4:
			return
		
		self.find_smallest_and_largest_holes()
		
		# Create detections for smallest and largest holes
		if self.smallest_hole is not None:
			detection = self.create_hole_detection(self.smallest_hole, cv_image, "smallest")
			if detection:
				detections.detections.append(detection)
		
		if self.largest_hole is not None:
			detection = self.create_hole_detection(self.largest_hole, cv_image, "largest")
			if detection:
				detections.detections.append(detection)

	def create_hole_detection(self, hole_box, cv_image, hole_type):
		"""Create a detection for a specific hole (smallest/largest)"""
		if (self.plane_normal is None or self.mapping_map_centroid is None or 
			self.mapping_map_quat is None or not hasattr(self, 'latest_bbox_class_1')):
			return None
		
		x_min, y_min, x_max, y_max = map(int, hole_box.xyxy[0])
		bbox = (x_min, y_min, x_max, y_max)
		
		# Check if hole is inside the mapping area
		if not self.is_inside_bbox(bbox, self.latest_bbox_class_1):
			return None
		
		bbox_center_x = (x_min + x_max) / 2
		bbox_center_y = (y_min + y_max) / 2
		bbox_width = x_max - x_min
		bbox_height = y_max - y_min
		
		# Project hole center onto the mapping plane
		d = np.linalg.inv(self.intrinsic_matrix) @ np.array([bbox_center_x, bbox_center_y, 1.0])
		d = d / np.linalg.norm(d)
		
		n = self.plane_normal
		p0 = self.mapping_map_centroid
		
		numerator = np.dot(n, p0)
		denominator = np.dot(n, d)
		
		if denominator == 0:
			return None
		
		t = numerator / denominator
		hole_position = t * d
		
		# Determine class name based on hole type
		class_name = f"torpedo_{'small' if hole_type == 'smallest' else 'large'}_hole"
		
		# Publish marker
		self.publish_marker(self.mapping_map_quat, hole_position, class_name, bbox_width, bbox_height)
		
		# Create detection message
		conf = hole_box.conf[0]
		return self.create_detection_message(class_name, hole_position, self.mapping_map_quat, conf)

	def create_detection_message(self, class_name, centroid, quat, conf):
		"""Create a Detection3D message"""
		detection = Detection3D()
		detection.header.frame_id = self.frame_id
		
		if self.use_incoming_timestamp:
			detection.header.stamp = self.detection_timestamp
		else:
			detection.header.stamp = self.get_clock().now().to_msg()
		
		detection.results.append(self.create_object_hypothesis_with_pose(class_name, centroid, quat, conf))
		return detection

	def publish_frame_results(self, results, detections):
		"""Publish all results for the current frame"""
		# Publish markers
		if self.temp_markers:
			self.publish_markers(self.temp_markers)
			self.temp_markers = []
		
		# Publish annotated image
		annotated_frame = results[0].plot()
		self.publish_accumulated_point_cloud()
		
		if self.has_subscribers(self.publisher):
			annotated_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding="bgr8")
			self.publisher.publish(annotated_msg)
		
		# Cleanup and timing
		self.cleanup_old_holes(age_threshold=2.0)
		
		if self.log_processing_time:
			self.detection_time = time.time() - self.detection_time
			self.get_logger().info(f"Total time (ms): {self.detection_time * 1000}")
			self.get_logger().info(f"FPS: {1/self.detection_time}")
		
		# Publish detections
		if self.has_subscribers(self.detection_publisher):
			self.detection_publisher.publish(detections)
		
		self.detection_id_counter = 0
 
	def cleanup_old_holes(self, age_threshold=2.0):
		# Get the current time as a builtin_interfaces.msg.Time object
		current_time = self.get_clock().now().to_msg()
 
		# Filter out holes older than the specified age threshold
		self.holes = [(bbox, timestamp) for bbox, timestamp in self.holes
					if ((current_time.sec - timestamp.sec) + (current_time.nanosec - timestamp.nanosec) * 1e-9) < age_threshold]
 
	def generate_unique_detection_id(self):
			# Increment and return the counter to get a unique ID for each detection
			self.detection_id_counter += 1
			return self.detection_id_counter

	def get_hole_size(self, hole):
		x_min, y_min, x_max, y_max = map(int, hole.xyxy[0])
		hole_width = x_max - x_min
		hole_height = y_max - y_min
		hole_size = hole_height*hole_width
		return hole_size

	def find_smallest_and_largest_holes(self):
		if self.plane_normal is None or self.mapping_map_centroid is None:
			self.get_logger().warning("Plane normal or centroid not defined. Cannot compute hole sizes.")
			return

		hole_sizes = []
		for hole in self.mapping_holes:
			x_min, y_min, x_max, y_max = map(int, hole.xyxy[0])
			corners_2d = [
				(x_min, y_min),
				(x_max, y_min),
				(x_max, y_max),
				(x_min, y_max)
			]
			corners_3d = []
			for (u, v) in corners_2d:
				d = np.linalg.inv(self.intrinsic_matrix) @ np.array([u, v, 1.0])
				n = self.plane_normal
				p0 = self.mapping_map_centroid

				numerator = np.dot(n, p0)
				denominator = np.dot(n, d)
				if denominator == 0:
					self.get_logger().warning(f"Denominator zero for point ({u}, {v}). Skipping this corner.")
					continue
				t = numerator / denominator
				if t <= 0:
					self.get_logger().warning(f"Intersection behind the camera for point ({u}, {v}). Skipping this corner.")
					continue
				point_3d = t * d
				corners_3d.append(point_3d)

			if len(corners_3d) == 4:
				width_vector = corners_3d[1] - corners_3d[0]
				height_vector = corners_3d[3] - corners_3d[0]
				width = np.linalg.norm(width_vector)
				height = np.linalg.norm(height_vector)
				hole_size = width * height
				hole_sizes.append((hole, hole_size))
			else:
				self.get_logger().warning(f"Not enough valid corners for hole. Expected 4, got {len(corners_3d)}")
				continue

		if not hole_sizes:
			self.get_logger().warning("No valid hole sizes computed.")
			return
 
		# Sort holes based on hole_size
		hole_sizes.sort(key=lambda x: x[1])
 
		self.smallest_hole = hole_sizes[0][0]
		self.largest_hole = hole_sizes[-1][0]
 
	def calculate_centroid(self, center_x, center_y, z):
		# Validate inputs
		if z <= 0 or np.isnan(z) or np.isinf(z):
			return None
			
		if np.isnan(center_x) or np.isnan(center_y) or np.isinf(center_x) or np.isinf(center_y):
			return None
			
		center_3d_x = (center_x - self.cx) * z / self.fx
		center_3d_y = (center_y - self.cy) * z / self.fy
		
		# Validate results
		if np.isnan(center_3d_x) or np.isnan(center_3d_y) or np.isinf(center_3d_x) or np.isinf(center_3d_y):
			return None
			
		return [center_3d_x, center_3d_y, z]
 
	def create_object_hypothesis_with_pose(self, class_name, centroid, quat, conf):
		hypothesis_with_pose = ObjectHypothesisWithPose()
		hypothesis = ObjectHypothesis()
 
		hypothesis.class_id = class_name
		hypothesis.score = conf.item() # Convert from numpy float to float
 
		hypothesis_with_pose.hypothesis = hypothesis
		hypothesis_with_pose.pose.pose.position.x = centroid[0]
		hypothesis_with_pose.pose.pose.position.y = centroid[1]
		hypothesis_with_pose.pose.pose.position.z = centroid[2]
		hypothesis_with_pose.pose.pose.orientation.x = quat[0]
		hypothesis_with_pose.pose.pose.orientation.y = quat[1]
		hypothesis_with_pose.pose.pose.orientation.z = quat[2]
		hypothesis_with_pose.pose.pose.orientation.w = quat[3]
 
		return hypothesis_with_pose
 
	def publish_accumulated_point_cloud(self):
			if not self.has_subscribers(self.point_cloud_publisher):
				return # Skip if no subscribers
			
			if not self.accumulated_points:
				return  # Skip if there are no points
 
			# Prepare the PointCloud message
			cloud = PointCloud()
			cloud.header.frame_id = self.frame_id
			if self.use_incoming_timestamp:
				cloud.header.stamp = self.detection_timestamp
			else:
				cloud.header.stamp = self.get_clock().now().to_msg()
 
 
			# Convert accumulated 3D points to Point32 messages and add to the PointCloud
			for point in self.accumulated_points:
				cloud.points.append(Point32(x=float(point[0]), y=float(point[1]), z=float(point[2])))
 
			# Publish the accumulated point cloud
			self.point_cloud_publisher.publish(cloud)
 
			# Clear the accumulated points after publishing
			self.accumulated_points.clear()
 
	def overlay_points_on_image(self, image, points):
		# Draw circles on the image for each point
		for point in points:
			try:
				# Check for valid z value to avoid division by zero
				if point[2] <= 0 or np.isnan(point[2]) or np.isinf(point[2]):
					continue
				
				# Check for valid x and y values
				if np.isnan(point[0]) or np.isnan(point[1]) or np.isinf(point[0]) or np.isinf(point[1]):
					continue
				
				# Transform the 3D point back to 2D
				x2d = int(point[0] * self.fx / point[2] + self.cx)
				y2d = int(point[1] * self.fy / point[2] + self.cy)
				
				# Check if the projected point is within image bounds
				if 0 <= x2d < image.shape[1] and 0 <= y2d < image.shape[0]:
					cv2.circle(image, (x2d, y2d), radius=3, color=(0, 255, 0), thickness=-1)
			except Exception as e:
				# Log the specific error for debugging
				self.get_logger().debug(f"Error overlaying point {point}: {e}")
				continue
 
	def get_3d_points(self, feature_points, cv_image):
		points_3d = []

		for x, y in feature_points:
			xi = int(x)
			yi = int(y)

			# Make sure the point is on the image
			if yi >= self.depth_image.shape[0] or xi >= self.depth_image.shape[1] or yi < 0 or xi < 0:
				continue

			# Make sure the point is on the mask
			if self.mask[yi, xi] != 255:
				continue

			z = self.depth_image[yi, xi]
			
			# Check for valid depth values
			if np.isnan(z) or z <= 0 or np.isinf(z):
				continue

			point_3d = self.calculate_centroid(xi, yi, z)
			
			# Validate the calculated 3D point
			if any(np.isnan(coord) or np.isinf(coord) for coord in point_3d):
				continue
				
			points_3d.append(point_3d)

		# Only proceed if we have valid points
		if not points_3d:
			return None

		# Now, overlay points on the image
		self.overlay_points_on_image(cv_image, points_3d)

		# Prepare PointCloud message
		cloud = PointCloud()
		cloud.header.frame_id = self.frame_id  # Adjust the frame ID as necessary

		if self.use_incoming_timestamp:
			cloud.header.stamp = self.detection_timestamp
		else:
			cloud.header.stamp = self.get_clock().now().to_msg()

		# Convert points to a numpy array
		points_3d = np.array(points_3d)

		# Validate points_3d array before filtering
		if len(points_3d) == 0:
			return None

		# Filter outlier points
		points_3d = self.radius_outlier_removal(points_3d, min_neighbors=min(10, int(len(points_3d) * 0.8)))
		if points_3d is None or len(points_3d) == 0:
			return None
			
		points_3d = self.statistical_outlier_removal(points_3d, k=min(10, int(len(points_3d) * 0.8)))
		if points_3d is None or len(points_3d) == 0:
			return None
		
		self.accumulated_points.extend(points_3d)  # Add the new points to the accumulated list

		# Optionally, limit the size of the accumulated points to prevent unbounded growth
		max_points = 10000  # Example limit, adjust based on your needs
		if len(self.accumulated_points) > max_points:
			self.accumulated_points = self.accumulated_points[-max_points:]

		if len(points_3d) < self.min_points:
			return None

		return points_3d
 
	def statistical_outlier_removal(self, points_3d, k=10, std_ratio=1.0):
		"""
		Remove statistical outliers from the point cloud.

		:param points_3d: Numpy array of 3D points
		:param k: Number of nearest neighbors to use for mean distance calculation
		:param std_ratio: Standard deviation ratio threshold
		:return: Filtered array of 3D points
		"""
		if len(points_3d) == 0:
			return points_3d
			
		# Validate input points
		valid_points = []
		for point in points_3d:
			if not any(np.isnan(coord) or np.isinf(coord) for coord in point):
				valid_points.append(point)
		
		if len(valid_points) == 0:
			return np.array([])
			
		points_3d = np.array(valid_points)
		
		if len(points_3d) <= k:
			return points_3d  # Not enough points for filtering
			
		mean_distances = np.zeros(len(points_3d))
		for i, point in enumerate(points_3d):
			try:
				distances = np.linalg.norm(points_3d - point, axis=1)
				# Check for any invalid distances
				valid_distances = distances[~(np.isnan(distances) | np.isinf(distances))]
				if len(valid_distances) > k:
					sorted_distances = np.sort(valid_distances)
					mean_distances[i] = np.mean(sorted_distances[1:k+1])
				else:
					mean_distances[i] = 0  # Default value for insufficient valid neighbors
			except Exception as e:
				self.get_logger().debug(f"Error calculating distances for point {i}: {e}")
				mean_distances[i] = 0

		# Filter out invalid mean distances
		valid_mean_distances = mean_distances[~(np.isnan(mean_distances) | np.isinf(mean_distances))]
		if len(valid_mean_distances) == 0:
			return points_3d  # Return original if no valid distances

		mean_dist_global = np.mean(valid_mean_distances)
		std_dev = np.std(valid_mean_distances)

		if std_dev == 0:
			return points_3d  # No variance, return all points

		threshold = mean_dist_global + std_ratio * std_dev
		filtered_indices = np.where((mean_distances < threshold) & 
								   (~np.isnan(mean_distances)) & 
								   (~np.isinf(mean_distances)))[0]
		
		return points_3d[filtered_indices]
 
	def radius_outlier_removal(self, points_3d, radius=1.0, min_neighbors=10):
		"""
		Remove radius outliers from the point cloud.

		:param points_3d: Numpy array of 3D points
		:param radius: The radius within which to count neighbors
		:param min_neighbors: Minimum number of neighbors within the radius for the point to be kept
		:return: Filtered array of 3D points
		"""
		if len(points_3d) == 0:
			return points_3d
			
		# Validate input points
		valid_points = []
		for point in points_3d:
			if not any(np.isnan(coord) or np.isinf(coord) for coord in point):
				valid_points.append(point)
		
		if len(valid_points) == 0:
			return np.array([])
			
		points_3d = np.array(valid_points)
		
		filtered_indices = []
		for i, point in enumerate(points_3d):
			try:
				distances = np.linalg.norm(points_3d - point, axis=1)
				# Check for valid distances
				valid_distances = distances[~(np.isnan(distances) | np.isinf(distances))]
				neighbor_count = len(np.where(valid_distances <= radius)[0])
				if neighbor_count > min_neighbors:
					filtered_indices.append(i)
			except Exception as e:
				self.get_logger().debug(f"Error calculating radius neighbors for point {i}: {e}")
				continue

		return points_3d[filtered_indices] if filtered_indices else np.array([])
 
	def fit_plane_to_points(self, points_3d):
		try:
			# Validate input points
			if len(points_3d) < 3:
				return None
				
			# Filter out any invalid points
			valid_points = []
			for point in points_3d:
				if not any(np.isnan(coord) or np.isinf(coord) for coord in point):
					valid_points.append(point)
			
			if len(valid_points) < 3:
				return None
				
			points_array = np.array(valid_points)
			centroid = np.mean(points_array, axis=0)
			
			# Check if centroid is valid
			if any(np.isnan(coord) or np.isinf(coord) for coord in centroid):
				return None
				
			u, s, vh = np.linalg.svd(points_array - centroid)
			normal = vh[-1]
			
			# Validate normal vector
			if any(np.isnan(coord) or np.isinf(coord) for coord in normal):
				return None
				
			normal_magnitude = np.linalg.norm(normal)
			if normal_magnitude == 0:
				return None
				
			normal = normal / normal_magnitude
			d = -np.dot(normal, centroid)
			
			return normal, d, centroid
		except Exception as e:
			self.get_logger().debug(f"Error fitting plane to points: {e}")
			return None
 
	def calculate_quaternion_and_euler_angles(self, normal):
 
		if np.allclose(normal, self.default_normal):
			quat = [0.0, 0.0, 0.0, 1.0]  # No rotation needed
			euler_angles = None
		else:
			rotation = self.calculate_rotation(normal)
			quat = rotation.as_quat()
			euler_angles = rotation.as_euler('xyz',degrees=True)
 
		return quat, euler_angles
 
	def calculate_rotation(self, normal):
 
		# Compute rotation axis (cross product) and angle (dot product)
		axis = np.cross(self.default_normal, normal)
		axis_length = np.linalg.norm(axis)
		if axis_length == 0:
			# Normal is in the opposite direction
			axis = np.array([1, 0, 0])
			angle = np.pi
		else:
			axis /= axis_length  # Normalize the rotation axis
			angle = np.arccos(np.dot(self.default_normal, normal))
 
		# Convert axis-angle to quaternion
		rotation = R.from_rotvec(axis * angle)
 
		return rotation
 
	def publish_marker(self, quat, centroid, class_name, bbox_width, bbox_height):
		
		if not self.has_subscribers(self.marker_array_publisher):
			return
		
		# Create a plane marker
		plane_marker = Marker()
		plane_marker.header.frame_id = self.frame_id
		if self.use_incoming_timestamp:
			plane_marker.header.stamp = self.detection_timestamp
		else:
			plane_marker.header.stamp = self.get_clock().now().to_msg()
		plane_marker.ns = "detection_markers"  # Namespace for all detection markers
		plane_marker.id = self.generate_unique_detection_id()  # Unique ID for each marker
		plane_marker.type = Marker.CUBE
		plane_marker.action = Marker.ADD
 
		# Set the position of the plane marker to be the centroid of the plane
		plane_marker.pose.position.x = centroid[0]
		plane_marker.pose.position.y = centroid[1]
		plane_marker.pose.position.z = centroid[2]
 
		# Set the plane marker's orientation
		plane_marker.pose.orientation.x = quat[0]
		plane_marker.pose.orientation.y = quat[1]
		plane_marker.pose.orientation.z = quat[2]
		plane_marker.pose.orientation.w = quat[3]
 
		# Set the scale of the plane marker based on the bounding box size
		plane_marker.scale.x = float(bbox_width) / 150.0
		plane_marker.scale.y = float(bbox_height) / 150.0
		if class_name == "buoy":
			plane_marker.scale.z = 0.01
		else:
			plane_marker.scale.z = 0.05
 
		# Set the color and transparency (alpha) of the plane marker
		color = self.get_color_for_class(class_name)
		plane_marker.color.r = color[0]
		plane_marker.color.g = color[1]
		plane_marker.color.b = color[2]
		plane_marker.color.a = 0.8  # Semi-transparent
 
		# Append the plane marker to publish all at once
		self.temp_markers.append(plane_marker)
 
		# Create an arrow marker
		arrow_marker = Marker()
		arrow_marker.header.frame_id = self.frame_id
		if self.use_incoming_timestamp:
			arrow_marker.header.stamp = self.detection_timestamp
		else:
			arrow_marker.header.stamp = self.get_clock().now().to_msg()
		arrow_marker.ns = "orientation_markers"  # Namespace for all detection markers
		arrow_marker.id = self.generate_unique_detection_id()  # Unique ID for each marker
		arrow_marker.type = Marker.ARROW
		arrow_marker.action = Marker.ADD
 
		# Set the position of the arrow marker to be the centroid of the plane
		arrow_marker.pose.position.x = centroid[0]
		arrow_marker.pose.position.y = centroid[1]
		arrow_marker.pose.position.z = centroid[2]
 
		# Create a rotation for -90 degrees around the y-axis (or another axis as needed)
		additional_rotation = R.from_euler('y', -90, degrees=True).as_quat()
 
		# Apply the additional rotation to the plane's quaternion
		arrow_quat = R.from_quat(quat) * R.from_quat(additional_rotation)
		arrow_quat = arrow_quat.as_quat()
 
		# Set the arrow marker's orientation
		arrow_marker.pose.orientation.x = arrow_quat[0]
		arrow_marker.pose.orientation.y = arrow_quat[1]
		arrow_marker.pose.orientation.z = arrow_quat[2]
		arrow_marker.pose.orientation.w = arrow_quat[3]
 
		# Set the scale for the arrow marker
		arrow_marker.scale.x = 1.0  # Length of the arrow
		arrow_marker.scale.y = 0.05  # Width of the arrow
		arrow_marker.scale.z = 0.05  # Height of the arrow
 
		# Set the color and transparency (alpha) of the arrow marker
		arrow_marker.color.r = color[0]
		arrow_marker.color.g = color[1]
		arrow_marker.color.b = color[2]
		arrow_marker.color.a = 0.8  # Semi-transparent
 
		# Append the arrow marker to publish all at once
		self.temp_markers.append(arrow_marker)
 
	def publish_markers(self, markers):

		if not self.has_subscribers(self.marker_array_publisher):
			return
		
		current_time = time.time()
		if current_time - self.last_publish_time > self.publish_interval:
			marker_array = MarkerArray()
			marker_array.markers = markers
			self.last_publish_time = current_time
			self.marker_array_publisher.publish(marker_array)
			# Clear the markers after publishing
			markers.clear()
 
	def get_color_for_class(self, class_name):
		return self.color_map.get(class_name, (1.0, 1.0, 1.0))  # Default to white
 
def main(args=None):
	rclpy.init(args=args)
	yolo_node = YOLONode()
	rclpy.spin(yolo_node)
	yolo_node.destroy_node()
	rclpy.shutdown()
 
if __name__ == '__main__':
	main()