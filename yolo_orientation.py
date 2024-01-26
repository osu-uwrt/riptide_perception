import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import numpy as np
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Quaternion, Point32
from scipy.spatial.transform import Rotation as R

class YOLONode(Node):
	def __init__(self):
		super().__init__('yolo_node')
		self.declare_parameters(
			namespace='',
			parameters=[
				('yolo_model_path', 'yolov8n-seg.engine'),
				('specific_class_id', 63),
		])

		yolo_model_path = self.get_parameter('yolo_model_path').get_parameter_value().string_value
		self.specific_class_id = self.get_parameter('specific_class_id').get_parameter_value().integer_value

		self.zed_info_subscription = self.create_subscription(CameraInfo, '/zed/zed_node/left/camera_info', self.camera_info_callback, 1)
		self.depth_info_subscription = self.create_subscription(CameraInfo, '/zed/zed_node/depth/camera_info', self.depth_info_callback, 1)
		self.image_subscription = self.create_subscription(Image, '/zed/zed_node/right_raw/image_raw_color', self.image_callback, 10)
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
				cv2.circle(image, (x2d, y2d), radius=5, color=(0, 255, 0), thickness=-1)
			except:
				pass

	def orientation(self, box, cv_image):
		x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
		mask_roi = self.mask[y_min:y_max, x_min:x_max]

		cropped_gray_image = self.gray_image[y_min:y_max, x_min:x_max]
		masked_gray_image = cv2.bitwise_and(cropped_gray_image, cropped_gray_image, mask=mask_roi)

		# Detect good features to track within the object's bounding box
		good_features = cv2.goodFeaturesToTrack(masked_gray_image, maxCorners=500, qualityLevel=0.02, minDistance=15)

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
					if normal is not None:
						self.previous_normal = normal  # Store the current normal for the next iteration
						self.publish_plane_marker(normal, d, centroid)
				except:
					print("A")

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
		cloud.header.frame_id = "zed_camera_link"  # Adjust the frame ID as necessary
		cloud.header.stamp = self.get_clock().now().to_msg()

		# Convert 3D points to Point32 messages and add to the PointCloud
		for point in points_3d:
			cloud.points.append(Point32(x=float(point[0]), y=float(point[1]), z=float(point[2])))

		# Publish the PointCloud message
		self.point_cloud_publisher.publish(cloud)

		if len(points_3d) < 5:
			print("Not enough points for SVD.")
			return None

		return np.array(points_3d)

	def fit_plane_to_points(self, points_3d):
		try:
			centroid = np.mean(points_3d, axis=0)
			u, s, vh = np.linalg.svd(points_3d - centroid)
			normal = vh[2, :]
			d = -np.dot(normal, centroid)
			return normal, d, centroid
		except:
			print("SVD did not converge")

	def publish_plane_marker(self, normal, d, centroid):
		marker = Marker()
		marker.header.frame_id = "zed_camera_link"
		marker.header.stamp = self.get_clock().now().to_msg()
		marker.ns = "plane_marker"
		marker.id = 1
		marker.type = Marker.CUBE
		marker.action = Marker.ADD
		marker.pose.position.x = centroid[0]
		marker.pose.position.y = centroid[1]
		marker.pose.position.z = centroid[2]
		quaternion = self.vector_to_quaternion(normal)
		marker.pose.orientation = quaternion
		marker.scale.x = 2.0  # Plane width
		marker.scale.y = 2.0  # Plane height
		marker.scale.z = 0.01  # Very thin to represent a plane
		marker.color.a = 0.5  # Semi-transparent
		marker.color.r = 0.0
		marker.color.g = 1.0
		marker.color.b = 0.0
		self.marker_publisher.publish(marker)

	def vector_to_quaternion(self, normal):
		# Align the normal vector with the Z-axis
		rotation = R.align_vectors([[0, 0, 1]], [normal])[0]
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
