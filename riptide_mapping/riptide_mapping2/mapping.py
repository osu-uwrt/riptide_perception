#!/usr/bin/env python3
# THE LINE ABOVE IS NEEEDED FOR NODE TO WORK

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_system_default
from rclpy.parameter import Parameter
from rclpy.time import Time

from geometry_msgs.msg import PoseWithCovariance, PoseWithCovarianceStamped, Pose, Vector3
from vision_msgs.msg import Detection3DArray
from tf2_geometry_msgs import do_transform_pose_stamped

import tf2_ros
from tf2_ros import TransformException, TransformStamped

from location import Location

import math

# Instead of updating the location for individual objects we apply a global offset to account for robot drift as we
# are confident in deadly reckoning the relative location of objects. The only objects that we keep track of in the translational
# system are the objects in active_objects. Rotational estimates are kept for all objects as well as they aren't relavant to robot drift.
class MappingNode(Node):

    def __init__(self):
        # Init the ROS Node
        super().__init__('riptide_mapping2')

        self.objects = {
            "gate": dict(),
            "earth_glyph": dict(),
            "buoy": dict(),
            "buoy_glyph_1": dict(),
            "buoy_glyph_2": dict(),
            "buoy_glyph_3": dict(),
            "buoy_glyph_4": dict(),
            "torpedo": dict(),
            "torpedo_upper_hole": dict(),
            "torpedo_lower_hole": dict(),
            "table": dict(),
            "prequal_gate": dict(),
            "prequal_pole": dict()
        }

        # Manually declare all the parameters from yaml config bc ros2 is sick
        for object in self.objects.keys():
            self.declare_parameters(
                namespace="",
                parameters=[
                    ('init_data.{}.parent'.format(object), "map"),
                    ('init_data.{}.pose.x'.format(object), 0.0),
                    ('init_data.{}.pose.y'.format(object), 0.0),
                    ('init_data.{}.pose.z'.format(object), 0.0),
                    ('init_data.{}.pose.yaw'.format(object), 0.0),
                    ('init_data.{}.covar.x'.format(object), 1.0),
                    ('init_data.{}.covar.y'.format(object), 1.0),
                    ('init_data.{}.covar.z'.format(object), 1.0),
                    ('init_data.{}.covar.yaw'.format(object), 1.0)
                ]
            )
        
        self.declare_parameters(
            namespace="",
            parameters=[
                ("buffer_size", 100),
                ("quantile", [0.01, 0.99]),
                ("confidence_cutoff", 0.7),
                ("minimum_distance", 1.0)
            ]
        )

        self.offset_object = None

        for object in self.objects.keys():
            self.create_location(object)
            self.add_publisher(object)

        # Create the buffer to send 
        self.tf_buffer = tf2_ros.buffer.Buffer()
        self.tf_listener = tf2_ros.transform_listener.TransformListener(self.tf_buffer, self)
        self.tf_brod = tf2_ros.transform_broadcaster.TransformBroadcaster(self)

        self.publish_timer = self.create_timer(1.0, self.publish_pose)

        self.add_on_set_parameters_callback(self.param_callback)
        self.create_subscription(Detection3DArray, "detected_objects".format(self.get_namespace()), self.vision_callback, qos_profile_system_default)
        
    def create_location(self, object: str):

        pose = Pose()

        pose.position.x = float(self.get_parameter('init_data.{}.pose.x'.format(object)).value)
        pose.position.y = float(self.get_parameter('init_data.{}.pose.y'.format(object)).value)
        pose.position.z = float(self.get_parameter('init_data.{}.pose.z'.format(object)).value)

        pose.orientation.x = 0.0
        pose.orientation.y = 0.0
        pose.orientation.z = float(self.get_parameter('init_data.{}.pose.yaw'.format(object)).value)

        self.objects[object]["location"] = Location(pose, int(self.get_parameter("buffer_size").value), tuple(self.get_parameter("quantile").value))

    # Creates a publisher to publish PoseWithCovariance
    def add_publisher(self, object: str):
        self.objects[object]["publisher"] = self.create_publisher(PoseWithCovarianceStamped, "mapping/{}".format(object), qos_profile_system_default)

    # Check which params need updated and update them via the create_location method
    def param_callback(self, params: 'list[Parameter]'):
        updates = set()

        for param in params:
            updates.add(str(param.name).split(".")[1])

        for object in updates:
            self.create_location(object)

    def vision_callback(self, detections: Detection3DArray):
        distance: 'dict[str, float]' = dict()
        closest_obj_id = ""
        for detection in detections.detections:
            for result in detection.results:
                if not result.hypothesis.class_id in self.objects.keys():
                    self.get_logger().info(f"Received detection on unknown class {result.hypothesis.class_id}!")
                    continue
                
                # Skip this detection if confidence is to low
                if result.hypothesis.score < float(self.get_parameter("confidence_cutoff").value):
                    self.get_logger().info(f"Rejecting detection of {result.hypothesis.class_id} because confidence {result.hypothesis.score} is too low")
                    continue

                if closest_obj_id == "":
                    closest_obj_id = result.hypothesis.class_id
                    
                parent: str = str(self.get_parameter("init_data.{}.parent".format(result.hypothesis.class_id)).value)

                # Calculate distance using that formula from pythagoras
                # We can do this because the pose is a transform between 2 objects therefore it is relative not absolute location
                distance[result.hypothesis.class_id] = math.sqrt(
                    math.pow(result.pose.pose.position.x, 2) +
                    math.pow(result.pose.pose.position.y, 2) +
                    math.pow(result.pose.pose.position.z, 2)
                )

                if parent == "map" and distance[closest_obj_id] > distance[result.hypothesis.class_id] and distance[result.hypothesis.class_id] > float(self.get_parameter("minimum_distance").value):
                    closest_obj_id = result.hypothesis.class_id

        for detection in detections.detections:
            for result in detection.results:
                if not result.hypothesis.class_id in self.objects.keys():
                    continue #already did print, just continue here
                
                # Skip this detection if confidence is to low
                if result.hypothesis.score < float(self.get_parameter("confidence_cutoff").value):
                    continue # already did print, just continue here

                # We have a transform from camera to child we need to transform so
                # that we have a transform from parrent to child
                camera: str = detection.header.frame_id
                parent: str = str(self.get_parameter("init_data.{}.parent".format(result.hypothesis.class_id)).value)
                child: str = result.hypothesis.class_id

                # Get the pose that is a transform from camera to child
                pose: PoseWithCovariance = result.pose

                # If the current object isnt the closest object and its parent is map we
                # aren't going to track its location in favor of offsetting the entire map
                update_object = parent != "map" or child == closest_obj_id

                if update_object:
                    try:
                        transform = self.tf_buffer.lookup_transform(
                            parent,
                            camera,
                            Time()
                        )
                    except TransformException as ex:
                        self.get_logger().error(f"Can't transform from {camera} to {parent}: {ex}")
                        continue

                    trans_pose = do_transform_pose_stamped(pose, transform)
                    self.objects[child]["location"].add_pose(trans_pose.pose, True, True)

    def publish_pose(self):
        for object in self.objects.keys():
            pose = PoseWithCovarianceStamped()

            pose.header.stamp = self.get_clock().now().to_msg()
            pose.header.frame_id = str(self.get_parameter("init_data.{}.parent".format(object)).value)
            pose.pose = self.objects[object]["location"].get_pose()

            self.objects[object]["publisher"].publish(pose)

            transform = TransformStamped()

            transform.header = pose.header
            transform.child_frame_id = object + "_frame"
            transform.header.stamp = pose.header.stamp
            transform.transform.translation = Vector3(x=pose.pose.pose.position.x, y=pose.pose.pose.position.y, z=pose.pose.pose.position.z)
            transform.transform.rotation = pose.pose.pose.orientation

            self.tf_brod.sendTransform(transform)

def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(MappingNode())
    rclpy.shutdown()

if __name__ == '__main__':
    main()
