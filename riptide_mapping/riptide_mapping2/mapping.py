#!/usr/bin/env python3
# THE LINE ABOVE IS NEEEDED FOR NODE TO WORK

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_system_default, qos_profile_sensor_data
from rclpy.time import Time

from std_msgs.msg import Header, Int8
from std_srvs.srv import Trigger
from geometry_msgs.msg import PoseWithCovariance, PoseWithCovarianceStamped, Pose, Vector3, Point, PoseStamped
from vision_msgs.msg import Detection3DArray, ObjectHypothesisWithPose
from tf2_geometry_msgs import do_transform_pose_stamped
from riptide_msgs2.srv import MappingTarget, StartBinaryClassifier, SeedObjectPose
from riptide_msgs2.msg import MappingTargetInfo, LedCommand

import tf2_ros
from tf2_ros import TransformException, TransformStamped

from transforms3d.euler import euler2quat

from location import Location
from binary_classifier import BinaryClassifier, DetectionSample
from bin_geometry import BinGeometryFitter

from tf2_msgs.msg import TFMessage
import math
from typing import cast

STALE_TIME = 2 #seconds
BINARY_CLASSIFIER_CLASSES = {"fire", "blood", "magnet"}

class TransformListenerWithHook(tf2_ros.TransformListener):
    def __init__(self, buffer: tf2_ros.buffer.Buffer, node: Node, hook):
        super().__init__(buffer=buffer, node=node, qos=qos_profile_sensor_data)
        self.hook = hook
    
    def callback(self, data: TFMessage) -> None:
        super().callback(data)
        self.hook()


class OutstandingDetectionInfo:
    def __init__(self, det_result: ObjectHypothesisWithPose, det_header: Header, closest_object: str, binary_classifier_detection=False):
        self.det_result = det_result
        self.det_header = det_header
        self.closest_object = closest_object
        self.binary_classifier_detection = binary_classifier_detection
        

# Instead of updating the location for individual objects we apply a global offset to account for robot drift as we
# are confident in deadly reckoning the relative location of objects. The only objects that we keep track of in the translational
# system are the objects in active_objects. Rotational estimates are kept for all objects as well as they aren't relavant to robot drift.
class MappingNode(Node):

    def __init__(self):
        # Init the ROS Node
        super().__init__('riptide_mapping2')

        self.objects = {
            # Gate
            "pre_return_home": dict(),
            "gate": dict(),
            "gate_repair": dict(),
            "gate_rescue": dict(),

            # Slalom (left the same for autonomy compatability)
            "slalom_parent": dict(),
            "slalom_front": dict(),
            "slalom_middle": dict(),
            "slalom_back": dict(),

            # Torpedo
            "pre_torpedo_pose": dict(),
            "torpedo": dict(),
            "fire_hole_large": dict(),
            "fire_hole_small": dict(),
            "blood_hole_large": dict(),
            "blood_hole_small": dict(),

            # Bin
            "pre_bin_pose": dict(),
            "post_bin_pose": dict(),
            "bin": dict(),
            "bin_target1": dict(),
            "bin_target2": dict(),

            # Octagon
            "octagon": dict(),
            "compass": dict(),
            "hammer_and_wrench": dict(),
            "buoy": dict(),
            "sos": dict(),

            # Table
            "table": dict(),
            "pill": dict(),
            "plug": dict(),
            "nut_and_bolt": dict(),
            "bandage": dict(),
            "helmet": dict(),
            "warning": dict(),
            "pre_table_pose": dict(),
            "post_table_pose": dict(),

            # Prequal
            "prequal_gate": dict(),
            "prequal_pole": dict(),
        }

        # Bin fit cursed ah objects left out of mapping so we can get rid of this bunk code after comp
        self.objects.update({name: dict() for name in BinGeometryFitter.EXTRA_OBJECTS})

        self.downwards_objects = {
            # Bin
            "bin": dict(),
            "bin_target1": dict(),
            "bin_target2": dict(),

            # Table
            "table": dict(),
            # "pill": dict(),
            # "plug": dict(),
            # "nut_and_bolt": dict(),
            # "bandage": dict(),
            # "helmet": dict(),
            # "warning": dict(),
        }
                
        self.outstanding_detections: list[OutstandingDetectionInfo] = []

        # Manually declare all the parameters from yaml config bc ros2 is sick
        for object in self.objects.keys():
            self.declare_parameters(
                namespace="",
                parameters=[
                    ('init_data.{}.parent'.format(object), "world"),
                    ('init_data.{}.pose.x'.format(object), 0.0),
                    ('init_data.{}.pose.y'.format(object), 0.0),
                    ('init_data.{}.pose.z'.format(object), 0.0),
                    ('init_data.{}.pose.yaw'.format(object), 0.0),
                    ('init_data.{}.covar.x'.format(object), 1.0),
                    ('init_data.{}.covar.y'.format(object), 1.0),
                    ('init_data.{}.covar.z'.format(object), 1.0),
                    ('init_data.{}.covar.yaw'.format(object), 1.0),
                    ('init_data.{}.lock_orientation_to_config'.format(object), False),
                    ('init_data.{}.point_yaw_at_parent'.format(object), False),
                    
                    # Only the bin vinyls use it, everything else stays ""
                    ('init_data.{}.class'.format(object), ""),
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
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_brod = tf2_ros.transform_broadcaster.TransformBroadcaster(self)

        self.target_object = ""
        self.lock_map = False
        self.offset = Location(Point(), Vector3(), int(self.get_parameter("buffer_size").value), tuple(self.get_parameter("quantile").value))
        self.binary_classifier = BinaryClassifier(self)

        # Bin geometry shitter cursed ah subsystem
        self.bin_fitter = BinGeometryFitter(self)

        # Store seeds (which move tf and reset cov once we enter tracking)
        self.instance1_seeded = False
        self.instance2_seeded = False

        self.add_on_set_parameters_callback(self.param_callback)
        self.create_subscription(Detection3DArray, "detected_objects".format(self.get_namespace()), self.vision_callback, qos_profile_system_default)
        self.status_pub = self.create_publisher(MappingTargetInfo, "state/mapping", qos_profile_system_default)
        self.create_service(MappingTarget, "mapping_target", self.target_callback) # Should prob be mapping ns but not changing for compatability for now
        self.create_service(Trigger, "mapping/reset_mapping", self.reset_mapping_callback)
        self.create_service(SeedObjectPose, "mapping/seed_object_pose", self.seed_object_pose_callback)

        # binary classifier services (resolve under this nodes ns)
        self.create_service(StartBinaryClassifier, "mapping/start_binary_classifier", self.start_binary_classifier_callback)
        self.create_service(Trigger, "mapping/freeze_binary_classifier_buffer", self.freeze_binary_classifier_buffer_callback)
        self.create_service(Trigger, "mapping/start_binary_classifier_second", self.start_binary_classifier_second_callback)
        self.create_service(Trigger, "mapping/stop_binary_classifier", self.stop_binary_classifier_callback)

        self.led_pulse_pub = self.create_publisher(LedCommand, "command/led", qos_profile_system_default)
        
        self.last_pub_time = Time()
        self.publish_pose()
        self.publish_timer = self.create_timer(0.125, self.update_oustanding_items)
        
    def reset_runtime_state(self):
        # Reset all configured map objects back to init_data params.
        # This also clears each Location's internal sample buffer because create_location() constructs a new Location object
        for object_name in self.objects.keys():
            self.create_location(object_name)
            # Drop any stale bin fit cov override
            self.objects[object_name].pop("fit_covar", None)

        # Reset global map drift/offset buffer
        self.offset = Location(
            Point(),
            Vector3(),
            int(self.get_parameter("buffer_size").value),
            tuple(self.get_parameter("quantile").value),
        )

        # Drop queued detections waiting on TF
        self.outstanding_detections.clear()

        # Stop/clear binary classifier state
        try:
            self.binary_classifier.stop()
        except Exception as ex:
            self.get_logger().warning(f"Binary classifier stop during reset failed: {ex}")

        # Reset mapping mode
        self.target_object = ""
        self.lock_map = False
        self.instance1_seeded = False
        self.instance2_seeded = False

        # Immediately publish reset topics/TF/status
        self.publish_pose()

    def reset_mapping_callback(self, request: Trigger.Request, response: Trigger.Response):
        self.get_logger().info("Resetting mapping runtime state to init_data")
        self.reset_runtime_state()

        response.success = True
        response.message = "Mapping reset to init_data"
        return response
    
    def pulse_detection_led(self, red=0, green=255, blue=0):
        # Pulse LEDs to indicate a detection was received
        ledPulse = LedCommand()
        ledPulse.target = LedCommand.TARGET_ALU
        ledPulse.mode = LedCommand.SINGLETON_FLASH

        ledPulse.red = red
        ledPulse.green = green
        ledPulse.blue = blue

        self.led_pulse_pub.publish(ledPulse)

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
        
        if self.target_object in self.objects.keys():
            self.get_logger().info(f"reset {self.target_object}")
            self.objects[self.target_object]["location"].reset()
            self.offset.cool_buffer()

        return response

    def start_binary_classifier_callback(self, request: StartBinaryClassifier.Request, response: StartBinaryClassifier.Response):
        class_name = str(request.class_name).strip()
        target1 = str(request.object1_name).strip()
        target2 = str(request.object2_name).strip()

        if self.binary_classifier.running:
            response.success = False
            response.message = "BinaryClassifier is already running"
            return response

        if class_name == "":
            response.success = False
            response.message = "class_name cannot be empty"
            return response

        if class_name not in BINARY_CLASSIFIER_CLASSES:
            response.success = False
            response.message = f"Unknown class_name {class_name}"
            return response

        # both instance targets are caller-supplied; they must be real mapping objects we can publish/seed
        if target1 not in self.objects.keys() or target2 not in self.objects.keys():
            response.success = False
            response.message = f"instance targets must exist in mapping objects (got '{target1}', '{target2}')"
            return response

        success, message = self.binary_classifier.start(
            class_name,
            target1,
            target2,
        )

        if not success:
            response.success = False
            response.message = message
            return response

        self.objects[target1]["location"].reset()
        self.objects[target2]["location"].reset()

        self.instance1_seeded = False
        self.instance2_seeded = False

        self.target_object = target1
        self.lock_map = False
        self.offset.cool_buffer()
        self.outstanding_detections.clear()

        response.success = True
        response.message = message
        self.get_logger().info(message)
        return response

    def freeze_binary_classifier_buffer_callback(self, request: Trigger.Request, response: Trigger.Response):
        # Freeze the classifier buffer while autonomy goes to do the first task, so the
        # second-instance detections already gathered survive the short TTL. Unfrozen by start_second.
        success, message = self.binary_classifier.freeze_buffer()

        if success:
            self.get_logger().info(message)
        else:
            self.get_logger().warning(message)

        response.success = success
        response.message = message
        return response

    def start_binary_classifier_second_callback(self, request: Trigger.Request, response: Trigger.Response):
        success, message = self.binary_classifier.start_second()

        if success:
            # If start_second locked instance 2 from the (preserved) buffer, publish it now as a TF for autonomy to navigate to, 
            # but high covariance so it isn't treated as confirmed.
            if self.binary_classifier.second_locked:
                self.seed_object_estimate(self.binary_classifier.instance2_name, self.binary_classifier.instance2_centroid)
                self.instance2_seeded = True
                self.publish_pose()
            else:
                self.get_logger().info("start_second: no buffered cluster yet, will lock on live detection")

            self.get_logger().info(message)
        else:
            self.get_logger().warning(message)

        response.success = success
        response.message = message
        return response

    def stop_binary_classifier_callback(self, request: Trigger.Request, response: Trigger.Response):
        success, message = self.binary_classifier.stop()

        if success:
            self.target_object = ""
            self.outstanding_detections.clear()
            self.get_logger().info(message)
        else:
            self.get_logger().warning(message)

        response.success = success
        response.message = message
        return response

    def seed_object_pose_callback(self, request: SeedObjectPose.Request, response: SeedObjectPose.Response):
        # Snap an object's estimate to an externally-measured position (e.g. autonomy's computed
        # table center) using the same seeding logic as the binary classifier.
        child = str(request.object_name).strip()

        if child not in self.objects.keys():
            response.success = False
            response.message = f"Unknown mapping object {child}"
            return response

        parent = str(self.get_parameter("init_data.{}.parent".format(child)).value)
        if parent != "map":
            response.success = False
            response.message = f"seed_object_pose only supports map-parented objects ({child} has parent {parent})"
            return response

        # keep the orientation from the init config; seeding only moves position
        config_rpy_deg = Vector3()
        config_rpy_deg.z = float(self.get_parameter("init_data.{}.pose.yaw".format(child)).value)

        self.get_logger().info(f"Seeding {child} to externally measured position "
                               f"({request.position.x}, {request.position.y}, {request.position.z})")
        self.seed_object_estimate(child, (request.position.x, request.position.y, request.position.z), config_rpy_deg)
        self.publish_pose()

        response.success = True
        response.message = f"Seeded {child}"
        return response

    def maybe_seed_binary_instances(self):
        # Fire once on each aquire -> track transition
        # Rebuild the target's Location centered on the classifier centroid so the published pose snaps to the cluster instead of crawling
        # from the reckoned init pose (needs to only fire once, or cov would never drop)
        bc = self.binary_classifier

        if bc.first_locked and not self.instance1_seeded:
            self.seed_object_estimate(bc.instance1_name, bc.instance1_centroid)
            self.instance1_seeded = True

        if bc.second_locked and not self.instance2_seeded:
            self.seed_object_estimate(bc.instance2_name, bc.instance2_centroid)
            self.instance2_seeded = True
            
    def seed_object_estimate(self, child: str, centroid_map, rpy_deg: Vector3 = None):
        # Seed a map-parented object's pose so both its Location and TF frame land on the same centroid.
        # Rebuilt Location stays soft/unwarmed (cov=1.0) until its buffer fills.
        # rpy_deg (degrees) seeds the orientation buffer; zero rotation when omitted (classifier has no orientation info).
        if child not in self.objects.keys():
            self.get_logger().error(f"seed_object_estimate: unknown object {child}")
            return

        offset_pos = self.offset.get_pose().pose.position

        cx = float(centroid_map[0])
        cy = float(centroid_map[1])
        cz = float(centroid_map[2])

        # rebuild Location centered on the centroid -> topic position = centroid, cov = soft 1.0
        xyz = Point(x=cx, y=cy, z=cz)
        rpy = rpy_deg if rpy_deg is not None else Vector3()
        self.objects[child]["location"] = Location(
            xyz, rpy,
            int(self.get_parameter("buffer_size").value),
            tuple(self.get_parameter("quantile").value),
        )

        # TF init_pose cancels the shared offset so map -> offset -> child lands on the centroid
        init_pose = Pose()
        init_pose.position.x = cx - offset_pos.x
        init_pose.position.y = cy - offset_pos.y
        init_pose.position.z = cz - offset_pos.z
        (init_pose.orientation.w,
         init_pose.orientation.x,
         init_pose.orientation.y,
         init_pose.orientation.z) = euler2quat(math.radians(rpy.x), math.radians(rpy.y), math.radians(rpy.z))
        self.objects[child]["init_pose"] = init_pose

    def vision_callback(self, detections: Detection3DArray):
        if self.lock_map:
            return
        
        # Bypass normal mapping behavior for binary classifier
        if self.binary_classifier.running:
            self.handle_binary_classifier_detections(detections)
            self.publish_pose()
            return
        
        closest_object = self.closest_object(detections)
        
        # if no target object set, use closest
        closest_or_target = closest_object if self.target_object == "" else self.target_object
                
        # Send the Poses for each location to their Location class
        for detection in detections.detections:
            result_ids = [r.hypothesis.class_id for r in detection.results]
            
            try:
                result_idx = result_ids.index(closest_or_target)
            except ValueError:
                continue
                        
            # if we reached here, then result is in the result list at result_idx    
            result = detection.results[result_idx]
                        
            if not result.hypothesis.class_id in self.objects.keys() \
                    and not result.hypothesis.class_id in self.downwards_objects.keys():
                self.get_logger().warning(f"Unknown class id {result.hypothesis.class_id}")
                continue #already did print, just continue here

            # Skip this detection if confidence is too low
            if result.hypothesis.score < float(self.get_parameter("confidence_cutoff").value):
                self.get_logger().info(f"Rejecting detection of {result.hypothesis.class_id} because confidence {result.hypothesis.score} is too low")
                continue
            
            self.pulse_detection_led()
            
            # update pose of object in map
            update_success, _ = self.try_update_pose(result, detections.header, closest_object)
            if not update_success:
                self.outstanding_detections.append(OutstandingDetectionInfo(result, detections.header, closest_object))
            
        self.publish_pose()

    def handle_binary_classifier_detections(self, detections: Detection3DArray):
        for detection in detections.detections:
            result_ids = [r.hypothesis.class_id for r in detection.results]

            try:
                result_idx = result_ids.index(self.binary_classifier.class_name)
            except ValueError:
                continue

            result = detection.results[result_idx]

            if result.hypothesis.score < float(self.get_parameter("confidence_cutoff").value):
                self.get_logger().info(f"Rejecting detection of {result.hypothesis.class_id} because confidence {result.hypothesis.score} is too low")
                continue

            self.pulse_detection_led()

            update_success, _ = self.try_update_binary_classifier_pose(result, detections.header)

            if not update_success:
                self.outstanding_detections.append(OutstandingDetectionInfo(result, detections.header, "", True))

        self.maybe_seed_binary_instances()

    def update_outstanding_detections(self):
        current_time = self.get_clock().now()
        oustanding_detections_remaining: list[OutstandingDetectionInfo] = []
        for outstanding in self.outstanding_detections:
            elapsed_nanoseconds = current_time.nanoseconds - (outstanding.det_header.stamp.sec * 1e9) - outstanding.det_header.stamp.nanosec
            elapsed_seconds = elapsed_nanoseconds / float(1e9)
            
            if outstanding.binary_classifier_detection:
                if self.binary_classifier.running:
                    update_success, error_msg = self.try_update_binary_classifier_pose(outstanding.det_result, outstanding.det_header)
                    now_sec = float(self.get_clock().now().nanoseconds) / 1e9
                    self.binary_classifier.age_buffer(now_sec)
                    self.maybe_seed_binary_instances()
                else:
                    update_success = True
                    error_msg = "BinaryClassifier stopped"
            else:
                update_success, error_msg = self.try_update_pose(outstanding.det_result, outstanding.det_header, outstanding.closest_object)

            if not update_success and not elapsed_seconds > STALE_TIME:     
                oustanding_detections_remaining.append(outstanding)
            
            if elapsed_seconds > STALE_TIME:
                self.get_logger().error(f"Timing out result for detection for class {outstanding.det_result.hypothesis.class_id} because the tf2 lookup could " + \
                    f"not be completed. Result originated at time {outstanding.det_header.stamp}. Final TF lookup error: {error_msg}")
                
        self.outstanding_detections = oustanding_detections_remaining
    
    def transform_detection_to_object_parent(self, result: ObjectHypothesisWithPose, detection_header: Header, child: str):
        # We have a transform from camera to child we need to transform so
        # that we have a transform from parrent to child
        parent: str = str(self.get_parameter("init_data.{}.parent".format(child)).value)

        # Get the pose that is a transform from camera to child
        pose: PoseWithCovariance = result.pose

        try:
            transform = self.tf_buffer.lookup_transform(
                parent,
                detection_header.frame_id,
                detection_header.stamp
            )
        except TransformException as ex:
            return False, None, None, str(ex)

        trans_pose = do_transform_pose_stamped(pose, transform)
        if child in self.downwards_objects.keys():
            trans_pose.pose.orientation.x = 0.0
            trans_pose.pose.orientation.y = 0.0
            trans_pose.pose.orientation.z = 0.0
            trans_pose.pose.orientation.w = 1.0

        return True, trans_pose, parent, ""

    def update_object_with_pose(self, trans_pose, parent: str, child: str, closest_object: str):
        if not child in self.objects.keys():
            return False, f"Unknown mapping object {child}"

        update_position = True
        update_orientation = True

        # These objects keep their orientation from config (detection yaw is unreliable)
        if bool(self.get_parameter("init_data.{}.lock_orientation_to_config".format(child)).value):
            update_orientation = False
        
        object_location: Location = self.objects[child]["location"]
        object_location.add_pose(trans_pose.pose, update_position, update_orientation)

        # update the offset pose if necessary (target or otherwise closest object, parent is map)
        if (child == self.target_object or (self.target_object == "" and child == closest_object)) \
           and parent == "map":
               
            offset_pose = Pose()

            offset_pose.position.x = trans_pose.pose.position.x - float(self.get_parameter("init_data.{}.pose.x".format(child)).value)
            offset_pose.position.y = trans_pose.pose.position.y - float(self.get_parameter("init_data.{}.pose.y".format(child)).value)
            offset_pose.position.z = trans_pose.pose.position.z - float(self.get_parameter("init_data.{}.pose.z".format(child)).value)
            
            # Rotational will never be changed because we don't want to offset that
            # FOG go brrrrrrrrrrrrrrrr
            self.offset.add_pose(offset_pose, True, False)
        
        return True, ""

    def try_update_pose(self, result: ObjectHypothesisWithPose, detection_header: Header, closest_object: str):        
        child: str = result.hypothesis.class_id

        update_success, trans_pose, parent, error_msg = self.transform_detection_to_object_parent(
            result,
            detection_header,
            child,
        )

        if not update_success:
            return False, error_msg

        return self.update_object_with_pose(
            trans_pose,
            parent,
            child,
            closest_object,
        )

    def try_update_binary_classifier_pose(self, result: ObjectHypothesisWithPose, detection_header: Header):
        common_target = self.binary_classifier.instance1_name

        update_success, trans_pose, parent, error_msg = self.transform_detection_to_object_parent(
            result,
            detection_header,
            common_target,
        )

        if not update_success:
            return False, error_msg

        stamp_sec = float(detection_header.stamp.sec) + float(detection_header.stamp.nanosec) / float(1e9)
        now_sec = float(self.get_clock().now().nanoseconds) / float(1e9)

        sample = DetectionSample(
            trans_pose.pose.position.x,
            trans_pose.pose.position.y,
            trans_pose.pose.position.z,
            result.hypothesis.score,
            stamp_sec,
        )

        assignment = self.binary_classifier.observe(sample, now_sec)

        if not assignment.accepted:
            self.get_logger().debug(f"Binary classifier rejected sample: {assignment.reason}")
            return True, assignment.reason

        child: str = assignment.target_name

        if not child in self.objects.keys():
            return False, f"BinaryClassifier assigned unknown target {child}"

        child_parent: str = str(self.get_parameter("init_data.{}.parent".format(child)).value)

        # If both binary targets share the same parent, reuse the already-transformed pose
        if child_parent != parent:
            update_success, trans_pose, parent, error_msg = self.transform_detection_to_object_parent(
                result,
                detection_header,
                child,
            )

            if not update_success:
                return False, error_msg

        return self.update_object_with_pose(
            trans_pose,
            parent,
            child,
            "",
        )
               
    def closest_object(self, detections: Detection3DArray) -> str:
        object = ""
        closest_dist: float = 1000
        
        for detection in detections.detections:
            for result in detection.results:
                if not result.hypothesis.class_id in self.objects.keys():
                    continue
                
                # if detection.header.frame_id != "zed_left_camera_optical_frame" or self.get_parameter("init_data.{}.parent".format(result.hypothesis.class_id)).value != "map":
                #     continue

                pose: Pose = result.pose.pose
                dist = math.sqrt(pose.position.x**2 + pose.position.y**2 + pose.position.z**2)

                if dist > 1 and dist < closest_dist:
                    object = result.hypothesis.class_id
                    closest_dist = dist

        return object
    
    def update_oustanding_items(self):
        self.update_outstanding_detections()
        
        #publish poses if they are stale
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
            lock_orientation = bool(self.get_parameter("init_data.{}.lock_orientation_to_config".format(object)).value)
            is_downward = object in self.downwards_objects
            pose = PoseWithCovarianceStamped()

            pose.pose = cast(Location, self.objects[object]["location"]).get_pose()
            pose.header.stamp = now
            pose.header.frame_id = parent

            # point yaw at parent on the PUBLISHED pose too, so the cov arrow matches the tf.
            # only applies to non-map parents; position here is in parent frame.
            if parent != "map" and not lock_orientation and not is_downward \
                and bool(self.get_parameter("init_data.{}.point_yaw_at_parent".format(object)).value):
                
                yaw = math.atan2(-pose.pose.pose.position.y, -pose.pose.pose.position.x)
                (pose.pose.pose.orientation.w,
                pose.pose.pose.orientation.x,
                pose.pose.pose.orientation.y,
                pose.pose.pose.orientation.z) = euler2quat(0.0, 0.0, yaw)
            
            # If the object is the target object the translational covariance will be in the offset object.
            if object == self.target_object and parent == "map":
                offset_covar = self.offset.get_pose().covariance

                pose.pose.covariance[0] = offset_covar[0]
                pose.pose.covariance[7] = offset_covar[7]
                pose.pose.covariance[14] = offset_covar[14]

            # Cov stuff for cursed ah bin fit, won't touch other objects since they dont have "fit_covar"
            fit_covar = self.objects[object].get("fit_covar")
            if fit_covar is not None:
                pose.pose.covariance[0] = fit_covar
                pose.pose.covariance[7] = fit_covar
                pose.pose.covariance[14] = fit_covar

            self.objects[object]["publisher"].publish(pose)

            transform = TransformStamped()
            transform.header.stamp = pose.header.stamp
            transform.transform.translation = Vector3(x=pose.pose.pose.position.x, y=pose.pose.pose.position.y, z=pose.pose.pose.position.z)
            transform.transform.rotation = pose.pose.pose.orientation
            transform.child_frame_id = object + "_frame"

            # If an object has the parent of anything other than map just apply the transform regularly
            # This will eventually be changed when chameleon_tf is absorbed by mapping and the offset tf frame is removed

            frame_valid = True; #if the mapping frame is valid / nans have occured

            if parent == "map":
                transform.header.frame_id = "offset"
                
                # assign initial position because that offset is taken care of by map offset as long as this object is the 
                # target object. DONT assign orientation because map offset doesn't cover that
                # The actual offset transform is added to the array earlier in this function. We just need to pub init pose
                init_pose: Pose = self.objects[object]["init_pose"]

                # check init pose for nans and infs
                if not (math.isfinite(init_pose.position.x) and math.isfinite(init_pose.position.y) and math.isfinite(init_pose.position.z)):
                    frame_valid = False

                transform.transform.translation.x = init_pose.position.x # need to assign individual components because a vector3 is not a point
                transform.transform.translation.y = init_pose.position.y
                transform.transform.translation.z = init_pose.position.z
            else:
                transform.header.frame_id = str(self.get_parameter("init_data.{}.parent".format(object)).value)

            if(frame_valid):
                transforms.append(transform)
            else:
                self.get_logger().error("Recieving NAN in position vector from objects:init_pose")
        
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
