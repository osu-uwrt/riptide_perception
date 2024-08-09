#! /usr/bin/env python3
 
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
 
class YOLONode(Node):
	def __init__(self):
		super().__init__('yolo_orientation')
		self.declare_parameters(
            namespace='',
            parameters=[
                ('yolo_model', ''),
                ('class_id_map', ''),
                ('threshold', 0.9),
                ('iou', 0.9),
            ]
        )
 
		tensorrt_wrapper_dir = get_package_share_directory("tensor_detector")
 
	    # Load parameters
		yolo_model = self.get_parameter('yolo_model').get_parameter_value().string_value
		yolo_model_path = os.path.join(tensorrt_wrapper_dir, 'weights', yolo_model)
 
 
		self.get_logger().info(f"Model path: {yolo_model_path}") 
 
 
		class_id_map_str = self.get_parameter('class_id_map').get_parameter_value().string_value
		self.class_id_map = yaml.safe_load(class_id_map_str) if class_id_map_str else {}
 
		self.get_logger().info(f"Class id map info: {self.class_id_map}") 
 
		self.conf = self.get_parameter('threshold').get_parameter_value().double_value
		self.iou = self.get_parameter('iou').get_parameter_value().double_value
 
		##########################
		# USER DEFINED PARAMS    #
		##########################
		self.log_processing_time = False
		self.use_incoming_timestamp = True
		self.export = True # Whether or not to export .pt file to engine
		self.print_camera_info = False # Print the camera info recieved
		self.frame_id = 'talos/zed_left_camera_optical_frame' 
		self.class_detect_shrink = 0.15 # Shrink the detection area around the class (% Between 0 and 1, 1 being full shrink)
		self.min_points = 5 # Minimum number of points for SVD
		self.publish_interval = 0.1  # 100 milliseconds
		self.history_size = 10 # Window size for rolling average smoothing
		self.default_normal = np.array([0.0, 0.0, 1.0]) # Default normal for quaternion calculation
		self.class_id_map = {
					0: 'buoy', 
					1: 'mapping_map', 
					2: 'mapping_hole', 
					3: 'gate_hot'
					}
		# Update internal class_id_map
		self.class_id_map.update({
            21: "mapping_largest_hole",
            22: "mapping_smallest_hole"
        })  # Internal mappings
		self.color_map = { # Color map for classes published to markers
			'buoy': (1.0, 0.0, 0.0),  
			'mapping_map': (0.0, 1.0, 0.0),  
			'mapping_hole': (0.0, 0.0, 1.0),
			'gate_hot': (1.0, 1.0, 1.0)
		}
 
 
		# Creating subscriptions
		self.zed_info_subscription = self.create_subscription(CameraInfo, '/talos/zed/zed_node/left/camera_info', self.camera_info_callback, 1)
		self.depth_info_subscription = self.create_subscription(CameraInfo, '/talos/zed/zed_node/depth/camera_info', self.depth_info_callback, 1)
		self.image_subscription = self.create_subscription(Image, '/talos/zed/zed_node/left/image_rect_color', self.image_callback, 10)
		self.depth_subscription = self.create_subscription(Image, '/talos/zed/zed_node/depth/depth_registered', self.depth_callback, 10)
 
		# Creating publishers
		self.marker_publisher = self.create_publisher(Marker, 'visualization_marker', 10)
		self.marker_array_publisher = self.create_publisher(MarkerArray, 'visualization_marker_array', 10)
		self.publisher = self.create_publisher(Image, 'yolo', 10)
		self.point_cloud_publisher = self.create_publisher(PointCloud, 'point_cloud', 10)
		self.mask_publisher = self.create_publisher(Image, 'yolo_mask', 10)
		self.detection_publisher = self.create_publisher(Detection3DArray, 'detected_objects', 10)
 
		# CV and Yolo init
		self.bridge = CvBridge()
		self.initialize_yolo(yolo_model_path)
 
 
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
 
		detections = Detection3DArray()
		detections.header.frame_id = self.frame_id
		self.detection_timestamp = msg.header.stamp
		if self.use_incoming_timestamp:
			detections.header.stamp = msg.header.stamp
		else:
			detections.header.stamp = self.get_clock().now().to_msg()
 
		if self.mask is None or self.mask.shape[:2] != cv_image.shape[:2]:
			self.mask = np.zeros(cv_image.shape[:2], dtype=np.uint8)
 
		# Reset mapping holes each image
		self.mapping_holes = []
		self.largest_hole = None
		self.smallest_hole = None
 
		for result in results:
			for box in result.boxes.cpu().numpy():
				if box.conf[0] <= self.conf:
					continue
				class_id = box.cls[0]
				if class_id in self.class_id_map:
					conf = box.conf[0]
					
					# If its a hole, store it, otherwise make the detection message
					if class_id == 2:
						#self.get_logger().info(f"class id: {class_id}")
						x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
						if self.use_incoming_timestamp:
							self.holes.append(((x_min, y_min, x_max, y_max), self.detection_timestamp))
						else:
							self.holes.append(((x_min, y_min, x_max, y_max), self.get_clock().now().to_msg()))
						self.mapping_holes.append(box)
						#self.get_logger().info(f"holes: {len(self.mapping_holes)}")
					else:
						detection = self.create_detection3d_message(box, cv_image, conf)
 
						if detection:
							detections.detections.append(detection)
 
					self.mask.fill(0)
					for contour in result.masks.xy:
						contour = np.array(contour, dtype=np.int32)
						cv2.fillPoly(self.mask, [contour], 255)
					mask_msg = self.bridge.cv2_to_imgmsg(self.mask,encoding="mono8")
					self.mask_publisher.publish(mask_msg)
 
 
		# Create detection3d for the holes if there are 4
		if len(self.mapping_holes) == 4:
			#self.get_logger().info(f"holes: {len(self.mapping_holes)}")
			self.find_smallest_and_largest_holes()
 
			# Publish smallest
			class_id = self.smallest_hole.cls[0]
			conf = self.smallest_hole.conf[0]
			detection = self.create_detection3d_message(self.smallest_hole, cv_image, conf, "smallest")
   
			if detection:
				detections.detections.append(detection)
 
			# Publish largest
			class_id = self.largest_hole.cls[0]
			conf = self.largest_hole.conf[0]
			detection = self.create_detection3d_message(self.largest_hole, cv_image, conf, "largest")
			if detection:
				detections.detections.append(detection)
 
 
		if self.temp_markers:
			self.publish_markers(self.temp_markers)
			self.temp_markers = []  # Clear the list for the next frame
 
		annotated_frame = results[0].plot()
		annotated_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding="bgr8")
		self.publish_accumulated_point_cloud()
		self.publisher.publish(annotated_msg)
		# if not self.torpedo_seen and len(self.holes) > 1 and self.torpedo_centroid is not None and self.torpedo_quat is not None:
		# 	detections.detections.append(self.spoof_torpedo())
		self.torpedo_seen = False
		self.cleanup_old_holes(age_threshold=2.0)
		if self.log_processing_time:
			self.detection_time = time.time() - self.detection_time
			self.get_logger().info(f"Total time (ms): {self.detection_time * 1000}")
			self.get_logger().info(f"FPS: {1/self.detection_time}")
		self.detection_publisher.publish(detections)
		self.detection_id_counter = 0
 
	def get_hole_size(self, hole):
		x_min, y_min, x_max, y_max = map(int, hole.xyxy[0])
		hole_width = x_max - x_min
		hole_height = y_max - y_min
		hole_size = max(hole_width, hole_height)
		return hole_size
 
	def find_smallest_and_largest_holes(self):
		for hole in self.mapping_holes:
			hole_size = self.get_hole_size(hole)
 
			if self.largest_hole is None:
				self.largest_hole = hole
			else:
				largest_hole_size = self.get_hole_size(self.largest_hole)
				if hole_size > largest_hole_size:
					self.largest_hole = hole
 
			if self.smallest_hole is None:
				self.smallest_hole = hole
			else:
				smallest_hole_size = self.get_hole_size(self.smallest_hole)
				if hole_size < smallest_hole_size:
					self.smallest_hole = hole
 
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
 
	def create_detection3d_message(self, box, cv_image, conf, hole_scale=None):
		x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
		bbox = (x_min, y_min, x_max, y_max)
		bbox_width = x_max - x_min
		bbox_height = y_max - y_min
 
		class_id = int(box.cls[0])
 
		bbox_center_x = (x_min + x_max) / 2
		bbox_center_y = (y_min + y_max) / 2
 
		class_name = self.class_id_map.get(class_id, "Unknown")
 
		if class_name == "mapping_map":
			self.latest_bbox_class_1 = (x_min, y_min, x_max, y_max)
		elif class_name == "mapping_hole":
			if self.mapping_map_centroid is not None and self.mapping_map_quat is not None and self.latest_bbox_class_1 and self.is_inside_bbox(bbox, self.latest_bbox_class_1):
				hole_quat = self.mapping_map_quat
				hole_centroid = self.calculate_centroid(bbox_center_x, bbox_center_y, self.mapping_map_centroid[2])

				

				if hole_scale == "smallest":
					class_name = "torpedo_small_hole"
				elif hole_scale == "largest":
					class_name = "torpedo_large_hole"
				else:
					return None
				
				
 
				self.publish_marker(hole_quat, hole_centroid, class_name, bbox_width, bbox_height)
 
				# Create Detection3D message
				detection = Detection3D()
				detection.header.frame_id = self.frame_id
				if self.use_incoming_timestamp:
					detection.header.stamp = self.detection_timestamp
				else:
					detection.header.stamp = self.get_clock().now().to_msg()
				detection.results.append(self.create_object_hypothesis_with_pose(class_name, hole_centroid, hole_quat, conf))
				return detection

		elif class_name == "torpedo_open":
			self.latest_bbox_class_7 = (x_min, y_min, x_max, y_max)
		elif class_name == "torpedo_closed":
			self.latest_bbox_class_8 = (x_min, y_min, x_max, y_max)
		elif class_name == "torpedo_hole":
			if self.open_torpedo_centroid is not None and self.open_torpedo_quat is not None and self.latest_bbox_class_7 and self.is_inside_bbox(bbox, self.latest_bbox_class_7):
				class_name = "torpedo_open_hole"
				hole_quat = self.open_torpedo_quat
				hole_centroid = self.calculate_centroid(bbox_center_x, bbox_center_y, self.open_torpedo_centroid[2])
			elif self.closed_torpedo_centroid is not None and self.closed_torpedo_quat is not None and self.latest_bbox_class_8 and self.is_inside_bbox(bbox, self.latest_bbox_class_8):
				class_name = "torpedo_closed_hole"
				hole_quat = self.closed_torpedo_quat
				hole_centroid = self.calculate_centroid(bbox_center_x, bbox_center_y, self.closed_torpedo_centroid[2])
			else:
				return None
 
			if self.use_incoming_timestamp:
				self.holes.append(((x_min, y_min, x_max, y_max), self.detection_timestamp))
			else:
				self.holes.append(((x_min, y_min, x_max, y_max), self.get_clock().now().to_msg()))
 
 
			self.publish_marker(hole_quat, hole_centroid, class_name, bbox_width, bbox_height)
 
			# Create Detection3D message
			detection = Detection3D()
			detection.header.frame_id = self.frame_id
			if self.use_incoming_timestamp:
				detection.header.stamp = self.detection_timestamp
			else:
				detection.header.stamp = self.get_clock().now().to_msg()
			detection.results.append(self.create_object_hypothesis_with_pose(class_name, hole_centroid, hole_quat, conf))
			return detection
 
 
		# Calculate the shrink size based on the class_detect_shrink percentage
		shrink_x = (x_max - x_min) * self.class_detect_shrink  
		shrink_y = (y_max - y_min) * self.class_detect_shrink  
 
		# Adjust the bounding box coordinates to exclude the edges
		x_min = int(x_min + shrink_x)
		x_max = int(x_max - shrink_x)
		y_min = int(y_min + shrink_y)
		y_max = int(y_max - shrink_y)
 
		# Extract the region of interest based on the bounding box
		mask_roi = self.mask[y_min:y_max, x_min:x_max]
		cropped_gray_image = self.gray_image[y_min:y_max, x_min:x_max]
		masked_gray_image = cv2.bitwise_and(cropped_gray_image, cropped_gray_image, mask=mask_roi)
 
		if class_name == "mapping_map":
			# Prepare the ROI mask, excluding the holes
			mask_roi = self.mask[y_min:y_max, x_min:x_max].copy()  # Work on a copy to avoid modifying the original

			# Dynamic padding calculation based on bounding box size
			padding_x = int((x_max - x_min) * 0.1)  # 10% of the bounding box width
			padding_y = int((y_max - y_min) * 0.1)  # 10% of the bounding box height
			#self.get_logger().info(f"holes for exclusion count: {len(self.holes)}")

			for hole_bbox, _ in self.holes:
				hole_x_min, hole_y_min, hole_x_max, hole_y_max = hole_bbox
				adjusted_hole_x_min = max(hole_x_min - x_min - padding_x, 0)
				adjusted_hole_y_min = max(hole_y_min - y_min - padding_y, 0)
				adjusted_hole_x_max = min(hole_x_max - x_min + padding_x, mask_roi.shape[1])
				adjusted_hole_y_max = min(hole_y_max - y_min + padding_y, mask_roi.shape[0])

				# Set the hole region in mask_roi to 0 to exclude it from feature detection
				mask_roi[adjusted_hole_y_min:adjusted_hole_y_max, adjusted_hole_x_min:adjusted_hole_x_max] = 0

			# Apply morphological operations to refine the exclusion zones
			kernel = np.ones((5, 5), np.uint8)
			mask_roi = cv2.dilate(mask_roi, kernel, iterations=1)
			mask_roi = cv2.erode(mask_roi, kernel, iterations=1)

			# Continue with feature detection using the adjusted mask_roi
			masked_gray_image = cv2.bitwise_and(cropped_gray_image, cropped_gray_image, mask=mask_roi)
		elif class_name == "buoy":
			# Sample the depth value at the center of the bounding box
			depth_value = self.depth_image[int(bbox_center_y), int(bbox_center_x)]
			self.get_logger().info(f"bbox_center_x: {bbox_center_x}") 
			self.get_logger().info(f"bbox_center_y: {bbox_center_y}")
			self.get_logger().info(f"depth: {depth_value}")
			if np.isnan(depth_value) or math.isinf(bbox_center_x) or math.isinf(bbox_center_y) or math.isinf(self.depth_image):
				return None
			centroid = self.calculate_centroid(bbox_center_x, bbox_center_y, float(depth_value))
			quat, _ = self.calculate_quaternion_and_euler_angles(-self.default_normal)
			self.publish_marker(quat, centroid, class_name, bbox_width, bbox_height)

			# Create Detection3D message
			detection = Detection3D()
			detection.header.frame_id = self.frame_id
			if self.use_incoming_timestamp:
				detection.header.stamp = self.detection_timestamp
			else:
				detection.header.stamp = self.get_clock().now().to_msg()

			# Set the pose
			detection.results.append(self.create_object_hypothesis_with_pose(class_name, centroid, quat, conf))

			return detection
		elif class_name in ["torpedo_open", "torpedo_closed"]:
			# Prepare the ROI mask, excluding the holes
			mask_roi = self.mask[y_min:y_max, x_min:x_max].copy()  # Work on a copy to avoid modifying the original
 
			# Padding for exclusion zone
			padding = 10
 
			for hole_bbox, _ in self.holes:
				# For simplicity, let's assume hole_bbox is a tuple of (hole_x_min, hole_y_min, hole_x_max, hole_y_max)
				# You might need to adjust the coordinates based on the ROI's position
				hole_x_min, hole_y_min, hole_x_max, hole_y_max = hole_bbox
				adjusted_hole_x_min = max(hole_x_min - x_min - padding, 0)
				adjusted_hole_y_min = max(hole_y_min - y_min - padding, 0)
				adjusted_hole_x_max = min(hole_x_max - x_min + padding, mask_roi.shape[1])
				adjusted_hole_y_max = min(hole_y_max - y_min + padding, mask_roi.shape[0])
 
				# Set the hole region in mask_roi to 0 to exclude it from feature detection
				mask_roi[adjusted_hole_y_min:adjusted_hole_y_max, adjusted_hole_x_min:adjusted_hole_x_max] = 0
 
			# Continue with feature detection using the adjusted mask_roi
			masked_gray_image = cv2.bitwise_and(cropped_gray_image, cropped_gray_image, mask=mask_roi)
 
		# Detect features within the object's bounding box
		good_features = cv2.goodFeaturesToTrack(masked_gray_image, maxCorners=0, qualityLevel=0.02, minDistance=1)
 
		if good_features is not None:
			good_features[:, 0, 0] += x_min  # Adjust X coordinates
			good_features[:, 0, 1] += y_min  # Adjust Y coordinates
 
			# Convert features to a list of (x, y) points
			feature_points = [pt[0] for pt in good_features]
 
			# Get 3D points from feature points
			points_3d = self.get_3d_points(feature_points, cv_image)
 
			if points_3d is not None and len(points_3d) >= self.min_points:
				normal, _, centroid = self.fit_plane_to_points(points_3d)
 
				centroid = self.calculate_centroid(bbox_center_x, bbox_center_y, centroid[2])
 
				if normal[2] > 0:
					normal = -normal
				

 

				quat, _ = self.calculate_quaternion_and_euler_angles(normal)
 
 
 
				if class_name == "torpedo_open":  
					self.open_torpedo_centroid = centroid
					self.open_torpedo_quat = quat
				elif class_name == "torpedo_closed":
					self.closed_torpedo_centroid = centroid
					self.closed_torpedo_quat = quat
				elif class_name == "mapping_map":
					self.mapping_map_centroid = centroid
					self.mapping_map_quat = quat
					class_name = "torpedo"
 
 
				# When calling publish_marker, pass these dimensions along with other required information
				self.publish_marker(quat, centroid, class_name, bbox_width, bbox_height)
 
				# Create Detection3D message
				detection = Detection3D()
				detection.header.frame_id = self.frame_id
				if self.use_incoming_timestamp:
					detection.header.stamp = self.detection_timestamp
				else:
					detection.header.stamp = self.get_clock().now().to_msg()
 
				# Set the pose
				detection.results.append(self.create_object_hypothesis_with_pose(class_name, centroid, quat, conf))
 
				return detection
		
		if self.latest_buoy is not None and class_name == "buoy":
			centroid = self.calculate_centroid(bbox_center_x, bbox_center_y, self.latest_buoy[2])
			quat, _ = self.calculate_quaternion_and_euler_angles(-self.default_normal)
			self.publish_marker(quat, centroid, class_name, bbox_width, bbox_height)
			detection = Detection3D()
			detection.header.frame_id = self.frame_id
			if self.use_incoming_timestamp:
				detection.header.stamp = self.detection_timestamp
			else:
				detection.header.stamp = self.get_clock().now().to_msg()

			# Set the pose
			detection.results.append(self.create_object_hypothesis_with_pose(class_name, centroid, quat, conf))

			return detection

		return None
 
	def calculate_centroid(self, center_x, center_y, z):
		center_3d_x = (center_x - self.cx) * z / self.fx
		center_3d_y = (center_y - self.cy) * z / self.fy
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
				# Transform the 3D point back to 2D
				x2d = int(point[0] * self.fx / point[2] + self.cx)
				y2d = int(point[1] * self.fy / point[2] + self.cy)
 
				# Draw the circle on the image
				cv2.circle(image, (x2d, y2d), radius=3, color=(0, 255, 0), thickness=-1)
			except:
				pass
 
	def get_3d_points(self, feature_points, cv_image):
		points_3d = []
 
 
		for x, y in feature_points:
			xi = int(x)
			yi = int(y)
 
			# Make sure the point is on the image
			if yi >= self.depth_image.shape[0] or xi >= self.depth_image.shape[1]:
				continue
 
			# Make sure the point is on the mask
			if self.mask[yi, xi] != 255:
				continue
 
			z = self.depth_image[yi, xi]
			if np.isnan(z) or z == 0:
				continue
 
			point_3d = self.calculate_centroid(xi, yi, z)
			points_3d.append(point_3d)
 
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
 
		# Filter outlier points
		points_3d = self.radius_outlier_removal(points_3d, min_neighbors=min(10,int(len(points_3d)*0.8)))
		points_3d = self.statistical_outlier_removal(points_3d, k=min(10,int(len(points_3d) * 0.8)))
 
		if points_3d is not None:
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
		mean_distances = np.zeros(len(points_3d))
		for i, point in enumerate(points_3d):
			distances = np.linalg.norm(points_3d - point, axis=1)
			sorted_distances = np.sort(distances)
			mean_distances[i] = np.mean(sorted_distances[1:k+1])
 
		mean_dist_global = np.mean(mean_distances)
		std_dev = np.std(mean_distances)
 
		threshold = mean_dist_global + std_ratio * std_dev
		filtered_indices = np.where(mean_distances < threshold)[0]
		return points_3d[filtered_indices]
 
	def radius_outlier_removal(self, points_3d, radius=1.0, min_neighbors=10):
		"""
		Remove radius outliers from the point cloud.
 
		:param points_3d: Numpy array of 3D points
		:param radius: The radius within which to count neighbors
		:param min_neighbors: Minimum number of neighbors within the radius for the point to be kept
		:return: Filtered array of 3D points
		"""
		filtered_indices = []
		for i, point in enumerate(points_3d):
			distances = np.linalg.norm(points_3d - point, axis=1)
			if len(np.where(distances <= radius)[0]) > min_neighbors:
				filtered_indices.append(i)
 
		return points_3d[filtered_indices]
 
	def fit_plane_to_points(self, points_3d):
		try:
			centroid = np.mean(points_3d, axis=0)
			u, s, vh = np.linalg.svd(points_3d - centroid)
			normal = vh[-1]
			normal = normal / np.linalg.norm(normal)
			d = -np.dot(normal, centroid)
			return normal, d, centroid
		except:
			pass
 
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
		current_time = time.time()
		if current_time - self.last_publish_time > self.publish_interval:
			marker_array = MarkerArray()
			marker_array.markers = markers
			self.marker_array_publisher.publish(marker_array)
			self.last_publish_time = current_time
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