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
from geometry_msgs.msg import Point32, Vector3
from scipy.spatial.transform import Rotation as R
from collections import deque
import time
import os

class YOLONode(Node):
	def __init__(self):
		super().__init__('yolo_node')
		self.declare_parameters(
			namespace='',
			parameters=[
				('yolo_model_path', '200.engine'),
				('specific_class_id', [0,1,2,3,4,5,6,7,8,9,10]),
		])

		# USER DEFINED PARAMS
		self.export = True # Whether or not to export .pt file to engine
		self.conf = 0.6 # Confidence threshold for yolo detections
		self.iou = 0.9 # Intersection over union for yolo detections
		self.frame_id = 'talos/zed_left_camera_optical_frame' 
		self.class_detect_shrink = 0.15 # Shrink the detection area around the class (% Between 0 and 1, 1 being full shrink)
		self.min_points = 5 # Minimum number of points for SVD
		self.publish_interval = 0.1  # 100 milliseconds
		self.history_size = 10 # Window size for rolling average smoothing
		self.class_id_map = {
			0: "buoy",
			1: "buoy_glyph_1",
			2: "buoy_glyph_2",
			3: "buoy_glyph_3",
			4: "buoy_glyph_4",
			5: "gate",
			6: "earth_glyph",
			7: "torpedo_open",
			8: "torpedo_closed",
			9: "torpedo_hole",
			91: "torpedo_open_hole",
			92: "torpedo_closed_hole",
			10: "bin"
			# Add more class IDs and their corresponding names as needed
		}
		self.default_normal = np.array([0, 0, 1]) # Default normal for quaternion calculation
		self.print_camera_info = False # Print the camera info recieved

		# Setting up initial params
		yolo_model_path = self.get_parameter('yolo_model_path').get_parameter_value().string_value
		self.specific_class_id = self.get_parameter('specific_class_id').get_parameter_value()._integer_array_value

		# Creating subscriptions
		self.zed_info_subscription = self.create_subscription(CameraInfo, '/talos/zed/zed_node/left/camera_info', self.camera_info_callback, 1)
		self.depth_info_subscription = self.create_subscription(CameraInfo, '/talos/zed/zed_node/depth/camera_info', self.depth_info_callback, 1)
		self.image_subscription = self.create_subscription(Image, '/talos/zed/zed_node/left_raw/image_raw_color', self.image_callback, 10)
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
				print(f"camera info: {msg}")
			self.fx = msg.k[0]
			self.cx = msg.k[2]
			self.fy = msg.k[4]
			self.cy = msg.k[5]
			self.camera_info_gathered = True

	def depth_info_callback(self, msg):
		if not self.depth_info_gathered:
			self.depth_info_gathered = True

	def depth_callback(self, msg):
		self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

	def image_callback(self, msg: Image):
		if self.depth_image is None or not self.camera_info_gathered:
			return

		cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
		self.gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
		if cv_image is None:
			return
		results = self.model(cv_image, verbose=False, iou=self.iou, conf=self.conf)

		detections = Detection3DArray()
		# detections.header.frame_id = self.frame_id
		# detections.header.stamp = msg.header.stamp
		detections.header = msg.header

		if self.mask is None or self.mask.shape[:2] != cv_image.shape[:2]:
			self.mask = np.zeros(cv_image.shape[:2], dtype=np.uint8)

		for result in results:
			for box in result.boxes.cpu().numpy():
				if box.conf[0] <= self.conf:
					continue
				class_id = box.cls[0]
				if class_id in self.specific_class_id:
					conf = box.conf[0]
					detection = self.create_detection3d_message(msg.header, box, cv_image, conf)
					if detection:
						detections.detections.append(detection)

					self.mask.fill(0)
					for contour in result.masks.xy:
						contour = np.array(contour, dtype=np.int32)
						cv2.fillPoly(self.mask, [contour], 255)
					mask_msg = self.bridge.cv2_to_imgmsg(self.mask,encoding="mono8")
					self.mask_publisher.publish(mask_msg)
					

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
	

	# def spoof_torpedo(self):

	# 	torpedo_spoof = [
	# 		0,
	# 		0,
	# 		self.torpedo_centroid[2]
	# 	]
	# 	for hole in self.holes:
	# 		torpedo_spoof[0] += hole[0]
	# 		torpedo_spoof[1] += hole[1]

	# 	torpedo_spoof[0] /= len(self.holes)
	# 	torpedo_spoof[1] /= len(self.holes)

	# 	self.publish_plane_marker(self.torpedo_quat, torpedo_spoof, 7, 100, 100)
					
	# 	# Create Detection3D message
	# 	detection = Detection3D()
	# 	detection.header.frame_id = self.frame_id
	# 	detection.header.stamp = self.get_clock().now().to_msg()
	# 	detection.results.append(self.create_object_hypothesis_with_pose(7, torpedo_spoof, self.torpedo_quat, np.float32(1)))
	# 	return detection
	
	def create_detection3d_message(self, header, box, cv_image, conf):
		x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
		bbox = (x_min, y_min, x_max, y_max)
		bbox_width = x_max - x_min
		bbox_height = y_max - y_min
		
		class_id = int(box.cls[0])
		#print(class_id, flush=True)

		if class_id == 7:
			self.latest_bbox_class_7 = (x_min, y_min, x_max, y_max)
		elif class_id == 8:
			self.latest_bbox_class_8 = (x_min, y_min, x_max, y_max)
		elif class_id == 9:
			if self.open_torpedo_centroid is not None and self.open_torpedo_quat is not None and self.latest_bbox_class_7 and self.is_inside_bbox(bbox, self.latest_bbox_class_7):
				class_id = 91
				hole_quat = self.open_torpedo_quat
				hole_centroid = [
					(x_min + bbox_width/2 - self.cx) * self.open_torpedo_centroid[2] / self.fx,
					(y_min + bbox_width/2 - self.cy) * self.open_torpedo_centroid[2] / self.fy,
					self.open_torpedo_centroid[2]
				]
			elif self.closed_torpedo_centroid is not None and self.closed_torpedo_quat is not None and self.latest_bbox_class_8 and self.is_inside_bbox(bbox, self.latest_bbox_class_8):
				class_id = 92
				hole_quat = self.closed_torpedo_quat
				hole_centroid = [
					(x_min + bbox_width/2 - self.cx) * self.closed_torpedo_centroid[2] / self.fx,
					(y_min + bbox_width/2 - self.cy) * self.closed_torpedo_centroid[2] / self.fy,
					self.closed_torpedo_centroid[2]
				]
			else:
				return None
			
			self.holes.append(((x_min, y_min, x_max, y_max), self.get_clock().now().to_msg()))

			#print(hole_centroid ,flush=True)
			#print(self.torpedo_centroid, flush=True)

			self.publish_plane_marker(hole_quat, hole_centroid, class_id, bbox_width, bbox_height)
			
			# Create Detection3D message
			detection = Detection3D()
			# detection.header.frame_id = self.frame_id
			# detection.header.stamp = self.get_clock().now().to_msg()
			detection.header = header
			detection.results.append(self.create_object_hypothesis_with_pose(class_id, hole_centroid, hole_quat, conf))
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

		if class_id in [7, 8]:
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
			good_features = cv2.goodFeaturesToTrack(masked_gray_image, maxCorners=0, qualityLevel=0.02, minDistance=1)

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
				if normal[2] > 0:
					normal = -normal

				quat, _ = self.calculate_quaternion_and_euler_angles(normal)

				# Temporal smoothing of quaternion and centroid using rolling average/history
				#smoothed_quat = self.smooth_orientation(class_id, quat)
				smoothed_quat = quat
				#smoothed_centroid = self.smooth_centroid(class_id, centroid)	
				smoothed_centroid = centroid	
			
				if class_id == 7:  # Assuming class ID 7 is for upper torpedo
					self.open_torpedo_centroid = smoothed_centroid
					self.open_torpedo_quat = smoothed_quat
				elif class_id == 8:  # Assuming class ID 8 is for lower torpedo
					self.closed_torpedo_centroid = smoothed_centroid
					self.closed_torpedo_quat = smoothed_quat


				# When calling publish_plane_marker, pass these dimensions along with other required information
				self.publish_plane_marker(smoothed_quat, smoothed_centroid, class_id, bbox_width, bbox_height)

				# Create Detection3D message
				detection = Detection3D()
				detection.header.frame_id = self.frame_id
				detection.header.stamp = self.get_clock().now().to_msg()

				# Set the pose
				detection.results.append(self.create_object_hypothesis_with_pose(class_id, smoothed_centroid, smoothed_quat, conf))

				return detection
		return None

	def create_object_hypothesis_with_pose(self, class_id, centroid, quat, conf):
		hypothesis_with_pose = ObjectHypothesisWithPose()
		hypothesis = ObjectHypothesis()

		class_name = self.class_id_map.get(class_id, "Unknown")
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

			x3d = (xi - self.cx) * z / self.fx
			y3d = (yi - self.cy) * z / self.fy
			points_3d.append([x3d, y3d, z])

		# Now, overlay points on the image
		self.overlay_points_on_image(cv_image, points_3d)

		# Prepare PointCloud message
		cloud = PointCloud()
		cloud.header.frame_id = self.frame_id  # Adjust the frame ID as necessary
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
			quat = [0, 0, 0, 1]  # No rotation needed
			euler_angles = normal.as_euler('xyz',degrees=True)
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
	
	def publish_plane_marker(self, quat, centroid, class_id, bbox_width, bbox_height):

		marker = Marker()
		marker.header.frame_id = self.frame_id
		marker.header.stamp = self.get_clock().now().to_msg()
		marker.ns = "detection_markers"  # Namespace for all detection markers
		marker.id = self.generate_unique_detection_id()  # Unique ID for each marker
		marker.type = Marker.CUBE
		marker.action = Marker.ADD

		# Set the position of the marker to be the centroid of the plane
		marker.pose.position.x = centroid[0]
		marker.pose.position.y = centroid[1]
		marker.pose.position.z = centroid[2]

		# Set the marker's orientation
		marker.pose.orientation.x = quat[0]
		marker.pose.orientation.y = quat[1]
		marker.pose.orientation.z = quat[2]
		marker.pose.orientation.w = quat[3]

		# Set a scale for the marker (you might want to adjust this based on the object size)
		# Set the scale of the marker based on the bounding box size
		marker.scale.x = float(bbox_width)/150.0
		marker.scale.y = float(bbox_height)/150.0
		if class_id == 0 :
			marker.scale.z = 0.01 
		else:
			marker.scale.z = 0.05
		
		# Set the color and transparency (alpha) of the marker
		# You might want to use different colors for different classes
		color = self.get_color_for_class(class_id)
		marker.color.r = color[0]
		marker.color.g = color[1]
		marker.color.b = color[2]
		marker.color.a = 0.8  # Semi-transparent

		# Append the marker to publish all at once
		self.temp_markers.append(marker)

	def smooth_orientation(self, class_id, current_orientation):
		if class_id not in self.orientation_history:
			self.orientation_history[class_id] = deque(maxlen=self.history_size)

		# Add the current orientation to the history
		self.orientation_history[class_id].append(current_orientation)

		# Calculate the average orientation
		average_orientation = np.mean(self.orientation_history[class_id], axis=0)

		# Normalize the averaged orientation to ensure it's a unit vector
		norm = np.linalg.norm(average_orientation)
		if norm == 0:  # Avoid division by zero
			return current_orientation
		averaged_normalized_orientation = average_orientation / norm

		return averaged_normalized_orientation


	def smooth_centroid(self, class_id, current_centroid):
		if class_id not in self.centroid_history:
			self.centroid_history[class_id] = deque(maxlen=self.history_size)

		# Add the current centroid to the history
		self.centroid_history[class_id].append(current_centroid)

		# Calculate the average centroid
		average_centroid = np.mean(self.centroid_history[class_id], axis=0)

		return average_centroid


	def publish_markers(self, markers):
		current_time = time.time()
		if current_time - self.last_publish_time > self.publish_interval:
			marker_array = MarkerArray()
			marker_array.markers = markers
			self.marker_array_publisher.publish(marker_array)
			self.last_publish_time = current_time
			# Clear the markers after publishing
			markers.clear()

	def get_color_for_class(self, class_id):
		# Define a simple color map for classes, or use a more sophisticated method as needed
		color_map = {
			0: (1.0, 0.0, 0.0),  # Red for class 0
			1: (0.0, 1.0, 0.0),  # Green for class 1
			2: (0.0, 0.0, 1.0),  # Blue for class 2
			3: (1.0, 1.0, 1.0),  # Green for class 1
			4: (1.0, 1.0, 0.0),  # Red for class 0
			5: (0.0, 1.0, 0.0),  # Green for class 1
			6: (0.0, 1.0, 0.0),  # Green for class 1
			7: (0.5, 0.5, 0.0),
			8: (0.0, 0.5, 0.5),
			9: (0.5, 0.0, 0.5),
			# Add more class-color mappings as needed
		}
		return color_map.get(class_id, (1.0, 1.0, 1.0))  # Default to white

def main(args=None):
	rclpy.init(args=args)
	yolo_node = YOLONode()
	rclpy.spin(yolo_node)
	yolo_node.destroy_node()
	rclpy.shutdown()

if __name__ == '__main__':
	main()
