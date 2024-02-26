#!/usr/bin/env python3
# THE LINE ABOVE IS NEEEDED FOR NODE TO WORK

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_system_default
from rclpy.parameter import Parameter
from rclpy.time import Time

from geometry_msgs.msg import PoseWithCovariance, PoseWithCovarianceStamped, Pose, Vector3, Quaternion
from vision_msgs.msg import Detection3DArray
from tf2_geometry_msgs import do_transform_pose_stamped

import tf2_ros
from tf2_ros import TransformException, TransformStamped

from transforms3d.euler import quat2euler, euler2quat

from location import Location

import math
from typing import cast

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
                ("target_object", "gate")
            ]
        )

        for object in self.objects.keys():
            self.create_location(object)
            self.add_publisher(object)

        # Create the buffer to send 
        self.tf_buffer = tf2_ros.buffer.Buffer()
        self.tf_brod = tf2_ros.transform_broadcaster.TransformBroadcaster(self)

        self.target_object = str(self.get_parameter("target_object").value)

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
    def param_callback(self, params: list[Parameter]):
        updates = set()
        self.get_logger().info(str(params))
        for param in params:
            if(str(param.name).split(".")) == "init_data":
                updates.add(str(param.name).split(".")[1])

        for object in updates:
            self.create_location(object)

    def vision_callback(self, detections: Detection3DArray):

        if self.target_object == None:
            return

        for detection in detections:
            for result in detection.results:
                # Skip this detection if confidence is to low
                if result.hypothesis.score < float(self.get_parameter("confidence_cutoff").value):
                    continue

                # We have a transform from camera to child we need to transform so
                # that we have a transform from parrent to child
                camera: str = detection.header.frame_id
                parent: str = str(self.get_parameter("init_data.{}.parent".format(result.hypothesis.class_id)).value)
                child: str = result.hypothesis.class_id

                # Get the pose that is a transform from camera to child
                pose: PoseWithCovariance = result.pose

                try:
                    transform = self.tf_buffer.lookup_transform(
                        parent,
                        camera,
                        Time()
                    )
                except TransformException as ex:
                    self.get_logger().error("Can't transform from " + camera + " to " + parent)
                    continue

                # If the current object isnt the closest object and its parent is map we
                # aren't going to track its location in favor of offsetting the entire map
                update_position = parent != "map" or child == self.target_object
                update_orientation = True

                trans_pose = do_transform_pose_stamped(pose, transform)
                        
                self.objects[child]["location"].add_pose(trans_pose.pose, update_position, update_orientation)

    # TODO clean up this ugly method
    def publish_pose(self):
        for object in self.objects.keys():
            pose = PoseWithCovarianceStamped()

            pose.pose = cast(Location, self.objects[object]["location"]).get_pose()
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.header.frame_id = object + "_frame"

            #self.objects[object]["publisher"].publish(pose)

            transform = TransformStamped()

            transform.header.stamp = pose.header.stamp
            transform.transform.translation = Vector3(x=pose.pose.pose.position.x, y=pose.pose.pose.position.y, z=pose.pose.pose.position.z)
            transform.transform.rotation = pose.pose.pose.orientation
            
            # If an object has the parent of anything other than map just apply the transform regularly
            if str(self.get_parameter("init_data.{}.parent".format(object)).value) != "map":
                transform.header.frame_id = str(self.get_parameter("init_data.{}.parent".format(object)).value)
                transform.child_frame_id = object + "_frame"
            # If the object is our target then we set its transform to offset the offset frame
            elif object == self.target_object:
                transform.header.frame_id = "map"
                transform.child_frame_id = "offset"
                transform.transform.rotation = Quaternion(w=1.0, x=0.0, y=0.0, z=0.0)
                self.tf_brod.sendTransform(transform)

                transform.header.frame_id = "offset"
                transform.child_frame_id = object + "_frame"
                transform.transform.translation = Vector3(x=0.0, y=0.0, z=0.0)
                transform.transform.rotation = pose.pose.pose.orientation
            # If the object is a child of map and is not our target we set it as the child of the offset frame
            else:
                transform.header.frame_id = "offset"
                transform.child_frame_id = object + "_frame"

            self.objects[object]["publisher"].publish(pose)
            self.tf_brod.sendTransform(transform)

def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(MappingNode())
    rclpy.shutdown()

if __name__ == '__main__':
    main()
