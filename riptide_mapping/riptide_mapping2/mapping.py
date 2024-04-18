#!/usr/bin/env python3
# THE LINE ABOVE IS NEEEDED FOR NODE TO WORK

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_system_default, qos_profile_sensor_data
from rclpy.parameter import Parameter
from rclpy.time import Time
from rclpy.executors import MultiThreadedExecutor

from std_msgs.msg import Header
from geometry_msgs.msg import PoseWithCovariance, PoseWithCovarianceStamped, Pose, Vector3, Point
from vision_msgs.msg import Detection3DArray, ObjectHypothesisWithPose
from tf2_geometry_msgs import do_transform_pose_stamped
from riptide_msgs2.srv import MappingTarget
from riptide_msgs2.msg import MappingTargetInfo

import tf2_ros
from tf2_ros import TransformException, TransformStamped

from transforms3d.euler import quat2euler, euler2quat

from location import Location

from tf2_msgs.msg import TFMessage

from rclpy.qos import DurabilityPolicy
from rclpy.qos import Duration
from rclpy.qos import HistoryPolicy
from rclpy.qos import QoSProfile

import math
from typing import cast

STALE_TIME = 2 #seconds

class TransformListenerWithHook(tf2_ros.TransformListener):
    def __init__(self, buffer: tf2_ros.buffer.Buffer, node: Node, hook):
        super().__init__(buffer=buffer, node=node, qos=qos_profile_sensor_data)
        self.hook = hook
    
    def callback(self, data: TFMessage) -> None:
        super().callback(data)
        # for transform in data.transforms:
        #     if transform.child_frame_id == "talos/base_link":
        #         print("Im a little piss boy", flush=True)
                
        self.hook()


class OutstandingDetectionInfo:
    def __init__(self, det_result: ObjectHypothesisWithPose, det_header: Header, closest_object: str):
        self.det_result = det_result
        self.det_header = det_header
        self.closest_object = closest_object
        

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
        
        self.outstanding_detections: list[OutstandingDetectionInfo] = []

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
        self.tf_buffer = tf2_ros.buffer.Buffer(node=self)
        self.tf_listener = TransformListenerWithHook(self.tf_buffer, self, self.update_outstanding_detections)
        self.tf_brod = tf2_ros.transform_broadcaster.TransformBroadcaster(self)

        self.target_object = ""
        self.lock_map = False
        self.offset = Location(Point(), Vector3(), int(self.get_parameter("buffer_size").value), tuple(self.get_parameter("quantile").value))

        self.add_on_set_parameters_callback(self.param_callback)
        self.create_subscription(Detection3DArray, "detected_objects".format(self.get_namespace()), self.vision_callback, qos_profile_system_default)
        self.status_pub = self.create_publisher(MappingTargetInfo, "state/mapping", qos_profile_system_default)
        self.create_service(MappingTarget, "mapping_target", self.target_callback)
        
        self.last_pub_time = Time()
        self.publish_pose()
        self.publish_timer = self.create_timer(0.25, self.publish_pose_if_stale)
        
        # qos = QoSProfile(
        #         depth=100,
        #         durability=DurabilityPolicy.VOLATILE,
        #         history=HistoryPolicy.KEEP_ALL,
        #         # deadline=Duration(nanoseconds=50000000)
        #         )
        
        # static_qos = QoSProfile(
        #     depth=100,
        #     durability=DurabilityPolicy.TRANSIENT_LOCAL,
        #     history=HistoryPolicy.KEEP_LAST,
        #     )
        
        # self.tfSub = self.create_subscription(TFMessage, "/tf", self.tf_cb, qos_profile=qos)
        # self.tfStaticSub = self.create_subscription(TFMessage, "/tf_static", self.tf_static_cb, qos_profile=static_qos)
    
    
    # def tf_cb(self, msg: TFMessage):
    #     for tf in msg.transforms:
    #         self.tf_buffer.set_transform(tf, "default_authority")
    #         if tf.child_frame_id == "buoy_frame":
    #             self.get_logger().info(f"GOT BUOY TRANS AT TIME {self.get_clock().now().to_msg()}, trans stamp is {tf.header.stamp}")
    #     # pass
    
    # def tf_static_cb(self, msg: TFMessage):
    #     for tf in msg.transforms:
    #         self.tf_buffer.set_transform_static(tf, "default_authority")
    #         # self.get_logger().info(f"GOT CHILD ID {tf.child_frame_id}")
    #         # if "zed_left" in tf.child_frame_id:
    
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
        # self.get_logger().info(str(params))
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
        
        closest_object = self.closest_object(detections)
        
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
                
                update_success, _ = self.try_update_pose(result, detections.header, closest_object)
                if not update_success:
                    self.outstanding_detections.append(OutstandingDetectionInfo(result, detections.header, closest_object))
            
        self.publish_pose()
    
    
    def update_outstanding_detections(self):
        current_time = self.get_clock().now()
        oustanding_detections_remaining: list[OutstandingDetectionInfo] = []
        for outstanding in self.outstanding_detections:
            elapsed_nanoseconds = current_time.nanoseconds - (outstanding.det_header.stamp.sec * 1e9) - outstanding.det_header.stamp.nanosec
            elapsed_seconds = elapsed_nanoseconds / float(1e9)
            
            update_success, error_msg = self.try_update_pose(outstanding.det_result, outstanding.det_header, outstanding.closest_object)
            if not update_success and not elapsed_seconds > STALE_TIME:     
                oustanding_detections_remaining.append(outstanding)
            
            if elapsed_seconds > STALE_TIME:
                self.get_logger().error(f"Timing out result for detection for class {outstanding.det_result.hypothesis.class_id} because the tf2 lookup could " + \
                    f"not be completed. Result originated at time {outstanding.det_header.stamp}. Final TF lookup error: {error_msg}")
                
                # self.get_logger().info(f"frames: {self.tf_buffer.all_frames_as_yaml()}", throttle_duration_sec=1)
                pass

                
        self.outstanding_detections = oustanding_detections_remaining
    
    
    def try_update_pose(self, result: ObjectHypothesisWithPose, detection_header: Header, closest_object: str):        
        # We have a transform from camera to child we need to transform so
        # that we have a transform from parrent to child
        parent: str = str(self.get_parameter("init_data.{}.parent".format(result.hypothesis.class_id)).value)
        child: str = result.hypothesis.class_id

        # Get the pose that is a transform from camera to child
        pose: PoseWithCovariance = result.pose

        try:
            transform = self.tf_buffer.lookup_transform(
                parent,
                detection_header.frame_id,
                detection_header.stamp
            )
        except TransformException as ex:
            # self.get_logger().error(f"When processing {result.hypothesis.class_id}: Can't look up transform from {detection_header.frame_id} to {parent}: {ex}")
            return False, str(ex)

        # If the current object isnt the closest object and its parent is map we
        # aren't going to track its location in favor of offsetting the entire map
        update_position = True
        update_orientation = True

        trans_pose = do_transform_pose_stamped(pose, transform)
        
        object_location: Location = self.objects[child]["location"]
        object_location.add_pose(trans_pose.pose, update_position, update_orientation)

        if child == self.target_object or (self.target_object == "" and child == closest_object):
            offset_pose = Pose()

            # offset_pose.position.x = object_location.get_pose().pose.position.x - float(self.get_parameter("init_data.{}.pose.x".format(child)).value)
            # offset_pose.position.y = object_location.get_pose().pose.position.y - float(self.get_parameter("init_data.{}.pose.y".format(child)).value)
            # offset_pose.position.z = object_location.get_pose().pose.position.z - float(self.get_parameter("init_data.{}.pose.z".format(child)).value)
            
            offset_pose.position.x = trans_pose.pose.position.x - float(self.get_parameter("init_data.{}.pose.x".format(child)).value)
            offset_pose.position.y = trans_pose.pose.position.y - float(self.get_parameter("init_data.{}.pose.y".format(child)).value)
            offset_pose.position.z = trans_pose.pose.position.z - float(self.get_parameter("init_data.{}.pose.z".format(child)).value)
            
            
            # Rotational will never be changed because we don't want to offset that
            # FOG go brrrrrrrrrrrrrrrr
            self.offset.add_pose(offset_pose, True, False)
        
        return True, ""
    
                    
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
    
    def publish_pose_if_stale(self):
        elapsed = (self.get_clock().now() - self.last_pub_time).to_msg()
        if elapsed.sec + float(elapsed.nanosec / 1e9) >= STALE_TIME:
            self.publish_pose()

    # Publishes stuff
    def publish_pose(self):
        # Send the transform between offset and map which is tracked in
        # self.offset which is a Location class
        offset_transform = TransformStamped()
        offset_pose = self.offset.get_pose()

        now = self.get_clock().now().to_msg()
        offset_transform.transform.translation = Vector3(x=offset_pose.pose.position.x, y=offset_pose.pose.position.y, z=offset_pose.pose.position.z)
        offset_transform.header.stamp = now
        offset_transform.header.frame_id = "map"
        offset_transform.child_frame_id = "offset"
        
        transforms = []

        transforms.append(offset_transform)

        # For every object send the covariance and transform
        for object in self.objects.keys():
            parent = str(self.get_parameter("init_data.{}.parent".format(object)).value)
            pose = PoseWithCovarianceStamped()

            pose.pose = cast(Location, self.objects[object]["location"]).get_pose()
            pose.header.stamp = now
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
            # This will eventually be changed when chameleon_tf is absorbed by mapping and the offset tf frame is removed
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

            transforms.append(transform)
        
        self.tf_brod.sendTransform(transforms)
        
        # feed the buffer
        for transform in transforms:
            self.tf_buffer.set_transform(transform, "default_authority")
        
        # publish status
        stat = MappingTargetInfo()
        stat.target_object = self.target_object
        stat.lock_map = self.lock_map
        self.status_pub.publish(stat)
        
        self.last_pub_time = self.get_clock().now()


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(MappingNode())
    rclpy.shutdown()

if __name__ == '__main__':
    main()
