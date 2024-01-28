import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import numpy as np
from visualization_msgs.msg import Marker
from vision_msgs.msg import Detection3DArray, Detection3D
from geometry_msgs.msg import Quaternion, Point32, Pose
from scipy.spatial.transform import Rotation as R
from collections import deque
import math

class YOLONode(Node):
	def __init__(self):
		super().__init__('yolo_node')
		self.declare_parameters(
			namespace='',
			parameters=[
				('yolo_model_path', 'yolov8n-seg-200Epoch.engine'),
				('specific_class_id', 0),
		])

		yolo_model_path = self.get_parameter('yolo_model_path').get_parameter_value().string_value
		self.specific_class_id = self.get_parameter('specific_class_id').get_parameter_value().integer_value

		self.zed_info_subscription = self.create_subscription(CameraInfo, '/zed/zed_node/left/camera_info', self.camera_info_callback, 1)
		self.depth_info_subscription = self.create_subscription(CameraInfo, '/zed/zed_node/depth/camera_info', self.depth_info_callback, 1)
		self.image_subscription = self.create_subscription(Image, '/zed/zed_node/left_raw/image_raw_color', self.image_callback, 10)
		self.depth_subscription = self.create_subscription(Image, '/zed/zed_node/depth/depth_registered', self.depth_callback, 10)
		self.marker_publisher = self.create_publisher(Marker, '/visualization_marker', 10)
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
		#self.frame_id = 'talos/zed2i_base_link'
		self.frame_id = 'zed_left_camera_optical_frame'
		self.centroid_publisher = self.create_publisher(Detection3DArray, '/talos/detected_objects', 10)
		self.previous_normal = None
		self.previous_d = None
		self.previous_centroid = None
		self.cumulative_normal = None
		self.sample_count = 0
		self.window_size = 5
		self.plane_parameters_window = deque(maxlen=self.window_size)

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

		if self.mask is None or self.mask.shape[:2] != cv_image.shape[:2]:
			self.mask = np.zeros(cv_image.shape[:2], dtype=np.uint8)

		for result in results:
			for box in result.boxes.cpu().numpy():
				if int(box.cls[0]) == self.specific_class_id:
					self.mask.fill(0)
					for contour in result.masks.xy:
						contour = np.array(contour, dtype=np.int32)
						cv2.fillPoly(self.mask, [contour], 255)
					mask_msg = self.bridge.cv2_to_imgmsg(self.mask,encoding="mono8")
					self.mask_publisher.publish(mask_msg)
					self.orientation(box, cv_image)

		annotated_frame = results[0].plot()
		annotated_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding="bgr8")
		self.publisher.publish(annotated_msg)

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

	def orientation(self, box, cv_image):
		x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
		mask_roi = self.mask[y_min:y_max, x_min:x_max]

		cropped_gray_image = self.gray_image[y_min:y_max, x_min:x_max]
		masked_gray_image = cv2.bitwise_and(cropped_gray_image, cropped_gray_image, mask=mask_roi)

		# Detect good features to track within the object's bounding box
		good_features = cv2.goodFeaturesToTrack(masked_gray_image, maxCorners=1500, qualityLevel=0.02, minDistance=1)

		if good_features is not None:
			good_features[:, 0, 0] += x_min  # Adjust X coordinates
			good_features[:, 0, 1] += y_min  # Adjust Y coordinates

			# Convert features to a list of (x, y) points
			feature_points = [pt[0] for pt in good_features]

			# Get 3D points from feature points
			points_3d = self.get_3d_points(feature_points, cv_image)

			if points_3d is not None:
				try:
					normal, d, centroid = self.fit_plane_to_points(points_3d)
					if normal[2] > 0:
						normal = -normal # If the normal is the other direction of the plane
					# normal, d, centroid = self.smooth_plane_exponential_smoothing(normal, d, centroid)
					# normal, d, centroid = self.add_to_moving_average(normal, d, centroid)
					self.previous_normal = normal
					self.previous_d = d
					self.previous_centroid = centroid
					if normal is not None:
						self.previous_normal = normal  # Store the current normal for the next iteration
						self.publish_plane_marker(normal, d, centroid)
						self.publish_centroid_marker(centroid)
				except Exception as e:
					print(f"SVD Error: {e}")

	def get_3d_points(self, feature_points, cv_image):
		points_3d = []

		for x, y in feature_points:
			xi = int(x)
			yi = int(y)

			if self.mask[yi, xi] != 255:
				continue

			if yi >= self.depth_image.shape[0] or xi >= self.depth_image.shape[1]:
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
			# Convert 3D points to Point32 messages and add to the PointCloud
			for point in points_3d:
				cloud.points.append(Point32(x=float(point[0]), y=float(point[1]), z=float(point[2])))

			# Publish the PointCloud message
			self.point_cloud_publisher.publish(cloud)

			print(len(points_3d),"pointCount:")
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
			mean_distances[i] = np.mean(sorted_distances[1:k+1])  # Exclude the point itself

		mean_dist_global = np.mean(mean_distances)
		std_dev = np.std(mean_distances)

		threshold = mean_dist_global + std_ratio * std_dev
		filtered_indices = np.where(mean_distances < threshold)[0]
		return points_3d[filtered_indices]

	def radius_outlier_removal(self, points_3d, radius=0.5, min_neighbors=5):
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

	def smooth_plane_exponential_smoothing(self, current_normal, current_d, current_centroid,  alpha=0.1):
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
		previous_normal = self.previous_normal
		if previous_normal is None:
			return current_normal, current_d, current_centroid
		else:
			previous_normal = np.array(previous_normal)
			previous_centroid = np.array(self.previous_centroid)
			previous_d = self.previous_d
			print(previous_normal,"prevNorm:")
			print(previous_d,"prevD:")
			print(previous_centroid,"prevCentroid:")
			smoothed_normal = alpha * np.array(current_normal) + (1 - alpha) * previous_normal
			smoothed_normal = smoothed_normal / np.linalg.norm(smoothed_normal)  # Normalize the normal vector

			smoothed_d = alpha * current_d + (1 - alpha) * previous_d  # Smooth the d parameter

			smoothed_centroid = alpha * np.array(current_centroid) + (1 - alpha) * previous_centroid  # Smooth the centroid

			return smoothed_normal.tolist(), smoothed_d, smoothed_centroid.tolist()

	def add_to_moving_average(self, normal, d, centroid):
		"""
		Add the most recent plane parameters to the moving average and compute the smoothed plane and centroid.

		:param normal: The current normal vector of the plane.
		:param d: The current d parameter of the plane equation.
		:param centroid: The current centroid of the plane points.
		:return: Tuple of (smoothed_normal, smoothed_d, smoothed_centroid).
		"""
		self.plane_parameters_window.append((normal, d, centroid))

		# Sum all normals, d values, and centroids
		sum_normals = np.sum([params[0] for params in self.plane_parameters_window], axis=0)
		sum_d = sum(params[1] for params in self.plane_parameters_window)
		sum_centroids = np.sum([params[2] for params in self.plane_parameters_window], axis=0)

		# Compute the average
		avg_normal = sum_normals / len(self.plane_parameters_window)
		avg_d = sum_d / len(self.plane_parameters_window)
		avg_centroid = sum_centroids / len(self.plane_parameters_window)

		# Normalize the average normal vector
		avg_normal /= np.linalg.norm(avg_normal)

		return avg_normal, avg_d, avg_centroid

	def publish_centroid_marker(self, centroid):
		marker = Detection3D()
		marker.header.frame_id = self.frame_id
		marker.header.stamp = self.get_clock().now().to_msg()
		#try:
			#marker.ns = "centroid_marker2"
#		except Exception as e:
#			print(f"Failed: {e}")
		#marker.id = 1
		#marker.type = Detection3D

#		marker.action = Detection3D.ADD
#		marker.pose = Pose()
		try:
			marker.bbox.center.position.x = centroid[0]
			marker.bbox.center.position.y = centroid[1]
			marker.bbox.center.position.z = centroid[2]
		except Exception as e:
			print(f"welp: {e}")
		#quaternion = self.vector_to_quaternion(normal)
		#marker.pose.orientation = quaternion
		print(centroid)
		print(marker,"Marker")
		print(centroid,"Centroid")
		markers = Detection3DArray()
		markers.detections.append(marker)
		marker.id = 'buoy'
#		marker.scale.x = 0.1  # Plane width
#		marker.scale.y = 0.1  # Plane height
#		marker.scale.z = 0.1  # Very thin to represent a plane
#		marker.color.a = 0.5  # Semi-transparent
#		marker.color.r = 0.0
#		marker.color.g = 1.0
#		marker.color.b = 0.0
		self.centroid_publisher.publish(markers)

	def fit_plane_least_squares(self, points_3d):
		try:
			# Points matrix with an additional ones column for the offset term
			A = np.c_[points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], np.ones(points_3d.shape[0])]

			# Coefficients for the plane equation Ax + By + Cz = 1
			# This assumes that the points are normalized such that D=1
			C = np.ones((points_3d.shape[0], 1))

			# Solve the least squares problem to find the plane coefficients
			plane_coeffs, _, _, _ = np.linalg.lstsq(A, C, rcond=None)

			# The normal vector to the plane is given by the coefficients A, B, C
			normal = plane_coeffs[:3].flatten()

			# Normalize the normal vector
			normal = normal / np.linalg.norm(normal)

			# The offset D is the last coefficient in the solution
			d = -plane_coeffs[3][0]

			# The centroid is not directly given by this method, but you can compute it if needed
			centroid = np.mean(points_3d, axis=0)

			return normal, d, centroid
		except np.linalg.LinAlgError:
			print("Least Squares did not converge")
			return None, None, None

	def fit_plane_to_points(self, points_3d):
		try:
			centroid = np.mean(points_3d, axis=0)
			u, s, vh = np.linalg.svd(points_3d - centroid)
			normal = vh[2, :]
			normal = normal / np.linalg.norm(normal)
			d = -np.dot(normal, centroid)
			return normal, d, centroid
		except:
			print("SVD did not converge")

	def publish_plane_marker(self, normal, d, centroid):
		marker = Marker()
		marker.header.frame_id = self.frame_id
		marker.header.stamp = self.get_clock().now().to_msg()
		marker.ns = "plane_marker"
		marker.id = 1
		marker.type = Marker.CUBE
		marker.action = Marker.ADD
		marker.pose.position.x = centroid[0]
		marker.pose.position.y = centroid[1]
		marker.pose.position.z = centroid[2]
		print("normal",normal)
		if (normal[0] < 0):
			rotation = R.from_euler('xz', [math.acos(normal[2]), -math.acos(normal[0])])	
		else:
			rotation = R.from_euler('xz', [math.acos(normal[2]), math.acos(normal[0])])
		quaternion = self.vector_to_quaternion(normal)
		print("quaternion",quaternion)
		# marker.pose.orientation = rotation.as_quat()
		marker.pose.orientation.x = rotation.as_quat()[0]
		marker.pose.orientation.y = rotation.as_quat()[1]
		marker.pose.orientation.z = rotation.as_quat()[2]
		marker.pose.orientation.w = rotation.as_quat()[3]
		marker.scale.x = 2.1  # Plane width
		marker.scale.y = 2.1  # Plane height
		marker.scale.z = 0.01  # Very thin to represent a plane
		marker.color.a = 0.5  # Semi-transparent
		marker.color.r = 0.0
		marker.color.g = 1.0
		marker.color.b = 0.0
		self.marker_publisher.publish(marker)

	def vector_to_quaternion(self, normal):
		# Assume that the Z-axis is the reference vector
		z_axis = np.array([0, 0, 1])

		# Compute the cross product between Z-axis and the normal vector
		cross_prod = np.cross(z_axis, normal)

		# Compute the dot product for the angle
		dot_prod = np.dot(z_axis, normal)

		# Compute the angle between the normal and the Z-axis
		angle = np.arccos(dot_prod)

		if not np.allclose(angle,0):
			axis = cross_prod / np.linalg.norm(cross_prod)

#		if dot_prod < 0:
#			axis = -axis
#			angle = 2 * np.pi - angle

		# If the normal is not aligned with Z-axis, compute the rotation
		if not np.allclose(angle, 0):
			# Create the rotation using the axis-angle representation
			rotation = R.from_rotvec(axis * angle)
		else:
			# If the normal is already the Z-axis, no rotation is needed
			rotation = R.from_quat([0, 0, 0, 1])

		quaternion = rotation.as_quat()
		return Quaternion(x=quaternion[0], y=quaternion[1], z=quaternion[2], w=quaternion[3])
def main(args=None):
	rclpy.init(args=args)
	yolo_node = YOLONode()
	rclpy.spin(yolo_node)
	yolo_node.destroy_node()
	rclpy.shutdown()

if __name__ == '__main__':
	main()
