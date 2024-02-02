import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from sensor_msgs.msg import Image, CameraInfo, PointCloud
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import numpy as np
from visualization_msgs.msg import Marker
from vision_msgs.msg import Detection3DArray, Detection3D, ObjectHypothesisWithPose, ObjectHypothesis
from geometry_msgs.msg import Point32, Vector3
from scipy.spatial.transform import Rotation as R
from collections import deque

class YOLONode(Node):
	def __init__(self):
		super().__init__('yolo_node')
		self.declare_parameters(
			namespace='',
			parameters=[
				('yolo_model_path', 'yolov8n-seg-200Epoch.engine'),
				('specific_class_id', [0,1]),
		])

		yolo_model_path = self.get_parameter('yolo_model_path').get_parameter_value().string_value
		self.specific_class_id = self.get_parameter('specific_class_id').get_parameter_value()._integer_array_value

		self.zed_info_subscription = self.create_subscription(CameraInfo, '/zed/zed_node/left/camera_info', self.camera_info_callback, 1)
		self.depth_info_subscription = self.create_subscription(CameraInfo, '/zed/zed_node/depth/camera_info', self.depth_info_callback, 1)
		self.image_subscription = self.create_subscription(Image, '/zed/zed_node/left_raw/image_raw_color', self.image_callback, 10)
		self.depth_subscription = self.create_subscription(Image, '/zed/zed_node/depth/depth_registered', self.depth_callback, 10)
		self.marker_publisher = self.create_publisher(Marker, '/visualization_marker', 10)
		self.euler_publisher = self.create_publisher(Vector3, '/euler_angles', 10)
		self.publisher = self.create_publisher(Image, '/yolo', 10)
		self.point_cloud_publisher = self.create_publisher(PointCloud, '/point_cloud', 10)
		self.mask_publisher = self.create_publisher(Image, '/yolo_mask', 10)
		self.bridge = CvBridge()
		self.model = YOLO(yolo_model_path)
		self.depth_image = None
		self.camera_info_gathered = False
		self.depth_info_gathered = False
		self.previous_normal = None
		self.cv_image = None
		self.gray_image = None
		self.class_detect_shrink = 10
		self.mask = None
		self.frame_id = 'zed_left_camera_optical_frame'
		self.centroid_publisher = self.create_publisher(Detection3DArray, '/talos/detected_objects', 10)
		self.previous_normal = None
		self.previous_d = None
		self.previous_centroid = None
		self.cumulative_normal = None
		self.sample_count = 0
		self.window_size = 5
		self.plane_parameters_window = deque(maxlen=self.window_size)
		self.default_normal = np.array([0, 0, 1])
		self.class_id_map = {
			0: "buoy",
			1: "buoy_glyph_1",
			# Add more class IDs and their corresponding names as needed
		}
		self.accumulated_points = []
		self.detection_id_counter = 0

	def camera_info_callback(self, msg):
		if not self.camera_info_gathered:
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

	def image_callback(self, msg):
		if self.depth_image is None or not self.camera_info_gathered:
			return

		cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
		self.cv_image = cv_image
		self.gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
		results = self.model(cv_image)

		detections = Detection3DArray()
		detections.header.frame_id = self.frame_id
		detections.header.stamp = self.get_clock().now().to_msg()

		if self.mask is None or self.mask.shape[:2] != cv_image.shape[:2]:
			self.mask = np.zeros(cv_image.shape[:2], dtype=np.uint8)

		for result in results:
			for box in result.boxes.cpu().numpy():
				class_id = int(box.cls[0])
				if class_id in self.specific_class_id:
					detection = self.create_detection3d_message(box, cv_image)
					if detection:
						#print(detection)
						detections.detections.append(detection)

					self.mask.fill(0)
					for contour in result.masks.xy:
						contour = np.array(contour, dtype=np.int32)
						cv2.fillPoly(self.mask, [contour], 255)
					mask_msg = self.bridge.cv2_to_imgmsg(self.mask,encoding="mono8")
					self.mask_publisher.publish(mask_msg)
					

		annotated_frame = results[0].plot()
		annotated_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding="bgr8")
		self.publish_accumulated_point_cloud()
		self.publisher.publish(annotated_msg)
		self.centroid_publisher.publish(detections)
		self.detection_id_counter = 0

	def generate_unique_detection_id(self):
			# Increment and return the counter to get a unique ID for each detection
			self.detection_id_counter += 1
			return self.detection_id_counter
	
	def create_detection3d_message(self, box, cv_image):
		x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
		class_id = int(box.cls[0])

		# Extract the region of interest based on the bounding box
		mask_roi = self.mask[y_min:y_max, x_min:x_max]
		cropped_gray_image = self.gray_image[y_min:y_max, x_min:x_max]
		masked_gray_image = cv2.bitwise_and(cropped_gray_image, cropped_gray_image, mask=mask_roi)

		# Detect features within the object's bounding box
		good_features = cv2.goodFeaturesToTrack(masked_gray_image, maxCorners=2500, qualityLevel=0.02, minDistance=1)
		
		if good_features is not None:
			good_features[:, 0, 0] += x_min  # Adjust X coordinates
			good_features[:, 0, 1] += y_min  # Adjust Y coordinates

			# Convert features to a list of (x, y) points
			feature_points = [pt[0] for pt in good_features]

			# Get 3D points from feature points
			points_3d = self.get_3d_points(feature_points, cv_image)
			
			if points_3d is not None and len(points_3d) >= 5:
				normal, _, centroid = self.fit_plane_to_points(points_3d)
				if normal[2] > 0:
					normal = -normal

				quat, _ = self.calculate_quaternion_and_euler_angles(normal)

				# Get a unique ID for this detection
				detection_id = self.generate_unique_detection_id()

				bbox_width = x_max - x_min
				bbox_height = y_max - y_min

				# When calling publish_plane_marker, pass these dimensions along with other required information
				self.publish_plane_marker(normal, centroid, detection_id, class_id, bbox_width, bbox_height)

				# Create Detection3D message
				detection = Detection3D()
				detection.header.frame_id = self.frame_id
				detection.header.stamp = self.get_clock().now().to_msg()

				# Set the pose
				detection.results.append(self.create_object_hypothesis_with_pose(class_id, centroid, quat))
				#print(self.class_id_map.get(class_id, "Unknown"))
				return detection
		return None

	def create_object_hypothesis_with_pose(self, class_id, centroid, quat):
		hypothesis_with_pose = ObjectHypothesisWithPose()
		hypothesis = ObjectHypothesis()

		class_name = self.class_id_map.get(class_id, "Unknown")
		#print(class_name)
		hypothesis.class_id = class_name
		hypothesis.score = 1.0  # You might want to use the detection score here

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
				#print("point not in shape")
				continue

			# Make sure the point is on the mask
			if self.mask[yi, xi] != 255:
				#print("point not in mask")
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
		points_3d = self.radius_outlier_removal(points_3d)
		points_3d = self.statistical_outlier_removal(points_3d)

		if points_3d is not None:
			self.accumulated_points.extend(points_3d)  # Add the new points to the accumulated list

			# Optionally, limit the size of the accumulated points to prevent unbounded growth
			max_points = 10000  # Example limit, adjust based on your needs
			if len(self.accumulated_points) > max_points:
				self.accumulated_points = self.accumulated_points[-max_points:]

			# Convert 3D points to Point32 messages and add to the PointCloud
			#for point in points_3d:
			#	cloud.points.append(Point32(x=float(point[0]), y=float(point[1]), z=float(point[2])))

			# Publish the PointCloud message
			#self.point_cloud_publisher.publish(cloud)

			#print(f"pointCount: {len(points_3d)}")
			if len(points_3d) < 5:
				print("Not enough points for SVD.")
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

	def radius_outlier_removal(self, points_3d, radius=0.5, min_neighbors=10):
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

	def smooth_plane_exponential_smoothing(self, current_normal, current_centroid,  alpha=0.9):
		"""
		Smooth the plane parameters (normal, d, centroid) using exponential smoothing.

		:param current_normal: The current plane normal vector estimate.
		:param current_d: The current d parameter of the plane equation.
		:param current_centroid: The current centroid of the plane.
		:param previous_normal: The previous plane normal vector estimate.
		:param previous_d: The previous d parameter of the plane equation.
		:param previous_centroid: The previous centroid of the plane.
		:param alpha: The smoothing factor, between 0 and 1.
		:return: Smoothed plane normal vector, d parameter, and centroid.
		"""
		if self.previous_normal is None:
			return current_normal, current_centroid
		else:
			smoothed_normal = alpha * np.array(current_normal) + (1 - alpha) * np.array(self.previous_normal)
			smoothed_normal = smoothed_normal / np.linalg.norm(smoothed_normal)
			smoothed_centroid = alpha * np.array(current_centroid) + (1 - alpha) * np.array(self.previous_centroid)

			return smoothed_normal.tolist(), smoothed_centroid.tolist()

	def add_to_moving_average(self, normal, centroid):
		"""
		Add the most recent plane parameters to the moving average and compute the smoothed plane and centroid.

		:param normal: The current normal vector of the plane.
		:param d: The current d parameter of the plane equation.
		:param centroid: The current centroid of the plane points.
		:return: Tuple of (smoothed_normal, smoothed_d, smoothed_centroid).
		"""
		self.plane_parameters_window.append((normal, centroid))

		# Sum all normals, d values, and centroids
		sum_normals = np.sum([params[0] for params in self.plane_parameters_window], axis=0)
		sum_centroids = np.sum([params[1] for params in self.plane_parameters_window], axis=0)

		# Compute the average
		avg_normal = sum_normals / len(self.plane_parameters_window)
		avg_centroid = sum_centroids / len(self.plane_parameters_window)

		# Normalize the average normal vector
		avg_normal /= np.linalg.norm(avg_normal)

		return avg_normal, avg_centroid

	def fit_plane_to_points(self, points_3d):
		try:
			centroid = np.mean(points_3d, axis=0)
			u, s, vh = np.linalg.svd(points_3d - centroid)
			normal = vh[-1]
			normal = normal / np.linalg.norm(normal)
			d = -np.dot(normal, centroid)
			return normal, d, centroid
		except:
			print("SVD did not converge")

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
	
	def publish_plane_marker(self, normal, centroid, detection_id, class_id, bbox_width, bbox_height):
		marker = Marker()
		marker.header.frame_id = self.frame_id
		marker.header.stamp = self.get_clock().now().to_msg()
		marker.ns = "detection_markers"  # Namespace for all detection markers
		marker.id = detection_id  # Unique ID for each marker
		marker.type = Marker.CUBE
		marker.action = Marker.ADD

		# Set the position of the marker to be the centroid of the plane
		marker.pose.position.x = centroid[0]
		marker.pose.position.y = centroid[1]
		marker.pose.position.z = centroid[2]

		quat, euler_angles = self.calculate_quaternion_and_euler_angles(normal)

		# Set the marker's orientation
		marker.pose.orientation.x = quat[0]
		marker.pose.orientation.y = quat[1]
		marker.pose.orientation.z = quat[2]
		marker.pose.orientation.w = quat[3]

		# Set a scale for the marker (you might want to adjust this based on the object size)
		# Set the scale of the marker based on the bounding box size
		marker.scale.x = float(bbox_width)/100.0
		marker.scale.y = float(bbox_height)/100.0
		marker.scale.z = 0.1  


		# Set the color and transparency (alpha) of the marker
		# You might want to use different colors for different classes
		color = self.get_color_for_class(class_id)
		marker.color.r = color[0]
		marker.color.g = color[1]
		marker.color.b = color[2]
		marker.color.a = 0.8  # Semi-transparent
		marker.lifetime = Duration(seconds=1.5).to_msg()  # Marker persists for 0.5 seconds

		# Publish the marker
		self.marker_publisher.publish(marker)

	def get_color_for_class(self, class_id):
		# Define a simple color map for classes, or use a more sophisticated method as needed
		color_map = {
			0: (1.0, 0.0, 0.0),  # Red for class 0
			1: (0.0, 1.0, 0.0),  # Green for class 1
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

