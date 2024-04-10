#!/usr/bin/env python3
# THE LINE ABOVE IS NEEEDED FOR NODE TO WORK

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_system_default
from rclpy.parameter import Parameter
from rclpy.time import Time

from geometry_msgs.msg import PoseWithCovariance, PoseWithCovarianceStamped, Pose, Vector3, Point
from vision_msgs.msg import Detection3DArray
from tf2_geometry_msgs import do_transform_pose_stamped
from riptide_msgs2.srv import MappingTarget
from riptide_msgs2.msg import MappingTargetInfo

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
            "buoy_glyph_3": dict(),
            "buoy_glyph_4": dict(),
            "torpedo_open": dict(),
            "torpedo_open_hole": dict(),
            "torpedo_closed": dict(),
            "torpedo_closed_hole": dict(),
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
                ("confidence_cutoff", 0.7),
                ("buffer_size", 100),
                ("quantile", [0.01, 0.99])
            ]
        )

        for object in self.objects.keys():
            self.create_location(object)
            self.add_publisher(object)

        # Create the buffer to send 
        self.tf_buffer = tf2_ros.buffer.Buffer()
        self.tf_listener = tf2_ros.transform_listener.TransformListener(self.tf_buffer, self)
        self.tf_brod = tf2_ros.transform_broadcaster.TransformBroadcaster(self)

        self.target_object = ""
        self.lock_map = False
        self.offset = Location(Point(), Vector3(), int(self.get_parameter("buffer_size").value), tuple(self.get_parameter("quantile").value))


        self.add_on_set_parameters_callback(self.param_callback)
        self.create_subscription(Detection3DArray, "detected_objects".format(self.get_namespace()), self.vision_callback, qos_profile_system_default)
        self.status_pub = self.create_publisher(MappingTargetInfo, "state/mapping", qos_profile_system_default)
        self.create_service(MappingTarget, "mapping_target", self.target_callback)
        
        self.publish_pose()
        self.publish_timer = self.create_timer(1.0, self.publish_pose)
        
    def create_location(self, object: str):
        #create the Location object using two vector3s describing coordinates and euler rotation
        xyz = Point()
        rpy = Vector3()

        xyz.x = float(self.get_parameter('init_data.{}.pose.x'.format(object)).value)
        xyz.y = float(self.get_parameter('init_data.{}.pose.y'.format(object)).value)
        xyz.z = float(self.get_parameter('init_data.{}.pose.z'.format(object)).value)

        rpy.x = 0.0
        rpy.y = 0.0
        rpy.z = float(self.get_parameter('init_data.{}.pose.yaw'.format(object)).value)

        self.objects[object]["location"] = Location(xyz, rpy, int(self.get_parameter("buffer_size").value), tuple(self.get_parameter("quantile").value))
        
        #create pose to store as initial. Used to publish objects with the map offset
        pose = Pose()
        pose.position = xyz
        (pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z) = euler2quat(rpy.x, rpy.y, rpy.z)
        self.objects[object]["init_pose"] = pose

    # Creates a publisher to publish PoseWithCovariance
    def add_publisher(self, object: str):
        self.objects[object]["publisher"] = self.create_publisher(PoseWithCovarianceStamped, "mapping/{}".format(object), qos_profile_system_default)

    # Check which params need updated and update them via the create_location method
    def param_callback(self, params):
        updates = set()
        self.get_logger().info(str(params))
        for param in params:
            if(str(param.name).split(".")) == "init_data":
                updates.add(str(param.name).split(".")[1])

        for object in updates:
            self.create_location(object)

    def target_callback(self, request: MappingTarget.Request, response: MappingTarget.Response):
        self.target_object = str(request.target_info.target_object)
        self.lock_map = bool(request.target_info.lock_map)

        return response

    def vision_callback(self, detections: Detection3DArray):
        if self.lock_map:
            return
        
        # Send the Poses for each location to their Location class
        for detection in detections.detections:
            for result in detection.results:

                if not result.hypothesis.class_id in self.objects.keys():
                    self.get_logger().warning(f"Unknown class id {result.hypothesis.class_id}")
                    continue #already did print, just continue here

                # Skip this detection if confidence is to low
                if result.hypothesis.score < float(self.get_parameter("confidence_cutoff").value):
                    self.get_logger().info(f"Rejecting detection of {result.hypothesis.class_id} because confidence {result.hypothesis.score} is too low")
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
                        detections.header.stamp
                    )
                except TransformException as ex:
                    self.get_logger().error(f"Can't transform from {camera} to {parent}: {ex}")
                    continue

                # If the current object isnt the closest object and its parent is map we
                # aren't going to track its location in favor of offsetting the entire map
                update_position = True
                update_orientation = True

                trans_pose = do_transform_pose_stamped(pose, transform)
                
                object_location: Location = self.objects[child]["location"]
                object_location.add_pose(trans_pose.pose, update_position, update_orientation)

                closest_object = self.closest_object(detections)
                if child == self.target_object or (self.target_object == "" and child == closest_object):
                    offset_pose = Pose()

                    offset_pose.position.x = object_location.get_pose().pose.position.x - float(self.get_parameter("init_data.{}.pose.x".format(child)).value)
                    offset_pose.position.y = object_location.get_pose().pose.position.y - float(self.get_parameter("init_data.{}.pose.y".format(child)).value)
                    offset_pose.position.z = object_location.get_pose().pose.position.z - float(self.get_parameter("init_data.{}.pose.z".format(child)).value)

                    # Rotational will never be changed because we don't want to offset that
                    # FOG go brrrrrrrrrrrrrrrr
                    self.offset.add_pose(offset_pose, True, False)
                    
    def closest_object(self, detections: Detection3DArray) -> str:
        object = ""
        closest_dist: float = 1000
        
        for detection in detections.detections:
            for result in detection.results:
                if not result.hypothesis.class_id in self.objects.keys():
                    continue
                
                if detection.header.frame_id != "zed_left_camera_optical_frame" or self.get_parameter("init_data.{}.parent".format(result.hypothesis.class_id)).value != "map":
                    continue

                pose: Pose = result.pose.pose
                dist = math.sqrt(pose.position.x**2 + pose.position.y**2 + pose.position.z**2)

                if dist > 1 and dist < closest_dist:
                    object = result.hypothesis.class_id
                    closest_dist = dist

        return object

    # Publishes stuff
    def publish_pose(self):
        # Send the transform between offset and map which is tracked in
        # self.offset which is a Location class
        offset_transform = TransformStamped()
        offset_pose = self.offset.get_pose()

        offset_transform.header.stamp = self.get_clock().now().to_msg()
        offset_transform.transform.translation = Vector3(x=offset_pose.pose.position.x, y=offset_pose.pose.position.y, z=offset_pose.pose.position.z)
        offset_transform.header.frame_id = "map"
        offset_transform.child_frame_id = "offset"

        self.tf_brod.sendTransform(offset_transform)

        # For every object send the covariance and transform
        for object in self.objects.keys():
            parent = str(self.get_parameter("init_data.{}.parent".format(object)).value)
            pose = PoseWithCovarianceStamped()

            pose.pose = cast(Location, self.objects[object]["location"]).get_pose()
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.header.frame_id = parent

            # If the object is the target object the translational covariance will be in the offset object.
            if object == self.target_object:
                offset_covar = offset_pose.covariance

                pose.pose.covariance[0] = offset_covar[0]
                pose.pose.covariance[7] = offset_covar[7]
                pose.pose.covariance[14] = offset_covar[14]
            
            self.objects[object]["publisher"].publish(pose)

            transform = TransformStamped()
            transform.header.stamp = pose.header.stamp
            transform.transform.translation = Vector3(x=pose.pose.pose.position.x, y=pose.pose.pose.position.y, z=pose.pose.pose.position.z)
            transform.transform.rotation = pose.pose.pose.orientation
            transform.child_frame_id = object + "_frame"

            # If an object has the parent of anything other than map just apply the transform regularly
            # This will eventually be changed when chamelon_tf is absorbed by mapping and the offset tf frame is removed
            if parent == "map":
                transform.header.frame_id = "offset"
                
                # assign initial position because that offset is taken care of by map offset as long as this object is the 
                # target object. DONT assign orientation because map offset doesn't cover that
                init_pose: Pose = self.objects[object]["init_pose"]
                transform.transform.translation.x = init_pose.position.x # need to assign individual components because a vector3 is not a point
                transform.transform.translation.y = init_pose.position.y
                transform.transform.translation.z = init_pose.position.z
            else:
                transform.header.frame_id = str(self.get_parameter("init_data.{}.parent".format(object)).value)

            self.tf_brod.sendTransform(transform)
        
        # publish status
        stat = MappingTargetInfo()
        stat.target_object = self.target_object
        stat.lock_map = self.lock_map
        self.status_pub.publish(stat)

def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(MappingNode())
    rclpy.shutdown()

if __name__ == '__main__':
    main()
