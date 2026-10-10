#!/usr/bin/env python3
# THE LINE ABOVE IS NEEEDED FOR NODE TO WORK

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_system_default, qos_profile_sensor_data
from rclpy.time import Time

from std_msgs.msg import Header, Int8
from std_srvs.srv import Trigger
from geometry_msgs.msg import PoseWithCovariance, PoseWithCovarianceStamped, Pose, Vector3, Point, PoseStamped
from nav_msgs.msg import Odometry
from vision_msgs.msg import Detection3DArray, ObjectHypothesisWithPose
from tf2_geometry_msgs import do_transform_pose_stamped
from rcl_interfaces.msg import SetParametersResult
from riptide_msgs2.srv import MappingTarget, StartBinaryClassifier
from riptide_msgs2.msg import MappingTargetInfo, LedCommand

import tf2_ros
from tf2_ros import TransformException, TransformStamped

from transforms3d.euler import euler2quat
from transforms3d.quaternions import quat2mat

from map_estimator import (MapEstimator, ObjectSpec, EstimatorSettings, NoiseSettings, detection_covariance,
                           slew_vector, slew_angle)
from binary_classifier import BinaryClassifier, DetectionSample
from bin_geometry import BinGeometryFitter

from tf2_msgs.msg import TFMessage
from collections import deque
from dataclasses import fields
import math
import numpy

STALE_TIME = 2 #seconds
BINARY_CLASSIFIER_CLASSES = {"fire", "blood", "magnet"}
TABLE_BASKETS = ("helmet", "warning")
MAX_TABLE_YAW_CORRECTION = math.radians(30)
MAX_ODOM_STEP = 1.0 # m, bigger jumps between odometry messages are resets, not travel
MOTION_HISTORY = 200 # odometry samples kept to look up the vehicle's rates at a detection's stamp
MAX_SLEW_DT = 0.2 # s, longest step a published frame takes toward the estimate in one publish

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


class ProjectedDetection:
    # A detection moved into an object's parent frame, plus what the noise model needs
    def __init__(self, pose: PoseStamped, parent: str, rotation, position_camera, stamp_sec: float):
        self.pose = pose
        self.parent = parent
        self.rotation = rotation # camera -> parent rotation matrix
        self.position_camera = position_camera
        self.stamp_sec = stamp_sec


# The config is a dead-reckoned layout of the course, and the odometry frame drifts away from it. Map-parented objects sit at
# prior + offset + deviation: one shared offset (map -> offset TF, random-walks with distance traveled) and a small per-object
# layout deviation, all in one Kalman filter (map_estimator.py). Any targeted map object refines the offset and with it the rest of
# the map. Objects parented to other mapping frames are filtered in that frame. Only the target objects are fused, because
# lookalike props get each other's class (slalom poles on the gate).
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
                    # covar is left over from the 2022 estimator and unused, the priors are the estimator.* params
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
                ("odom_topic", "odometry/filtered"),
                # Published frames chase the estimate at most this fast so a correction never steps a frame the
                # controller is tracking (<= 0: jump). The lag still left is added to the published covariance
                ("output.max_speed", 0.3),
                ("output.max_yaw_rate_deg", 30.0),
            ]
        )

        # Estimator and detection noise tuning, see map_estimator.py for what each one means
        for prefix, settings in (("estimator", EstimatorSettings), ("noise", NoiseSettings)):
            self.declare_parameters(
                namespace="",
                parameters=[("{}.{}".format(prefix, f.name), f.default) for f in fields(settings)]
            )

        for object in self.objects.keys():
            self.add_publisher(object)

        self.estimator = self.create_estimator()

        # Published (slewed) poses: object -> (position in map for map objects, else in the parent frame; yaw), and map -> offset
        self.output = {}
        self.output_offset = None
        self.last_output_time = None
        self.output_settling = False

        # Odometry: distance traveled drives the offset's drift, rates scale the detection noise
        self.last_odom_position = None
        self.pending_distance = 0.0
        self.last_predict_time = self.get_clock().now()
        self.motion_history = deque(maxlen=MOTION_HISTORY) # (stamp sec, angular speed, linear speed)

        # Create the buffer to send
        self.tf_buffer = tf2_ros.buffer.Buffer(node=self)
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_brod = tf2_ros.transform_broadcaster.TransformBroadcaster(self)

        # Objects fused from detections. The service takes "a,b,c"; the first is the primary target
        self.target_objects: 'list[str]' = []
        self.target_object = ""
        self.lock_map = False
        # Set once the table is fit to the baskets, after that basket positions are frozen (yaw still points at the table)
        self.table_centered = False
        self.binary_classifier = BinaryClassifier(self)

        # Bin geometry shitter cursed ah subsystem
        self.bin_fitter = BinGeometryFitter(self)

        # Store seeds (which move tf and reset cov once we enter tracking)
        self.instance1_seeded = False
        self.instance2_seeded = False

        self.add_on_set_parameters_callback(self.param_callback)
        self.create_subscription(Detection3DArray, "detected_objects", self.vision_callback, qos_profile_system_default)
        self.create_subscription(Odometry, str(self.get_parameter("odom_topic").value), self.odom_callback, qos_profile_sensor_data)
        self.status_pub = self.create_publisher(MappingTargetInfo, "state/mapping", qos_profile_system_default)
        self.create_service(MappingTarget, "mapping_target", self.target_callback) # Should prob be mapping ns but not changing for compatability for now
        self.create_service(Trigger, "mapping/reset_mapping", self.reset_mapping_callback)
        self.create_service(Trigger, "mapping/center_table_on_baskets", self.center_table_on_baskets_callback)

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
        # Reset all configured map objects and the offset back to init_data params and the estimator priors
        self.estimator = self.create_estimator()
        for object_name in self.objects.keys():
            # Drop any stale basket fit cov override
            self.objects[object_name].pop("fit_covar", None)

        # Frames jump to the reset map instead of sliding there
        self.output.clear()
        self.output_offset = None

        # Drop queued detections waiting on TF
        self.outstanding_detections.clear()

        # Stop/clear binary classifier state
        try:
            self.binary_classifier.stop()
        except Exception as ex:
            self.get_logger().warning(f"Binary classifier stop during reset failed: {ex}")

        # Reset mapping mode
        self.set_targets("")
        self.lock_map = False
        self.table_centered = False
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

    def center_table_on_baskets_callback(self, request: Trigger.Request, response: Trigger.Response):
        # Move the table so the baskets sit at their config offsets from it. Other table children ride along with table_frame, the baskets get
        # shifted back so they stay put in the world (point_yaw_at_parent then re-aims them at the new table center)
        if str(self.get_parameter("init_data.table.parent").value) != "map":
            response.success = False
            response.message = "center_table_on_baskets only supports a map-parented table"
            return response

        # Midpoint of the baskets in table_frame
        positions = [self.estimator.position(basket)[0] for basket in TABLE_BASKETS]
        mx = sum(p[0] for p in positions) / len(positions)
        my = sum(p[1] for p in positions) / len(positions)
        mz = sum(p[2] for p in positions) / len(positions)

        if not all(math.isfinite(v) for v in (mx, my, mz)):
            response.success = False
            response.message = f"Basket midpoint is not finite ({mx}, {my}, {mz})"
            return response

        # Yaw the table so the measured basket axis lines up with the config basket axis, that way the other children
        # (which are fixed offsets in table_frame) land where they are relative to the baskets in config
        cfg = [[float(self.get_parameter(f"init_data.{basket}.pose.{axis}").value) for axis in "xyz"] for basket in TABLE_BASKETS]
        cfg_axis = math.atan2(cfg[0][1] - cfg[1][1], cfg[0][0] - cfg[1][0])
        meas_axis = math.atan2(positions[0][1] - positions[1][1], positions[0][0] - positions[1][0])
        dyaw = math.atan2(math.sin(meas_axis - cfg_axis), math.cos(meas_axis - cfg_axis))

        yaw_msg = f"yawed table {math.degrees(dyaw):.1f} deg"
        if abs(dyaw) > MAX_TABLE_YAW_CORRECTION:
            # Probably a bad/unmapped basket, don't swing all the table objects around on it
            yaw_msg = f"skipped yaw correction of {math.degrees(dyaw):.1f} deg (limit {math.degrees(MAX_TABLE_YAW_CORRECTION):.0f})"
            dyaw = 0.0

        # New table origin in the old table_frame: the measured basket midpoint minus the (yawed) config basket midpoint,
        # so the baskets end up at their config offsets from the table
        cmx = sum(c[0] for c in cfg) / len(cfg)
        cmy = sum(c[1] for c in cfg) / len(cfg)
        cmz = sum(c[2] for c in cfg) / len(cfg)
        ox = mx - (cmx * math.cos(dyaw) - cmy * math.sin(dyaw))
        oy = my - (cmx * math.sin(dyaw) + cmy * math.cos(dyaw))
        oz = mz - cmz

        # Rotate the origin shift into map using the table's yaw (offset frame is translation only)
        table_yaw, _ = self.estimator.yaw("table")
        dx = ox * math.cos(table_yaw) - oy * math.sin(table_yaw)
        dy = ox * math.sin(table_yaw) + oy * math.cos(table_yaw)
        dz = oz

        self.estimator.shift("table", (dx, dy, dz))
        self.estimator.add_yaw("table", dyaw)

        # Re-express the baskets in the new table_frame so they stay put in the world
        for basket in TABLE_BASKETS:
            self.estimator.shift(basket, (-ox, -oy, -oz))
            self.estimator.rotate_position(basket, -dyaw)

        # The frames jump to the fit, autonomy reads them right after this call
        for name in ("table",) + TABLE_BASKETS:
            self.output.pop(name, None)

        # The table is now fit to these basket positions, stop detections from dragging them around
        self.table_centered = True

        self.publish_pose()

        response.success = True
        response.message = f"Moved table by ({dx:.3f}, {dy:.3f}, {dz:.3f}) in map to fit the baskets, {yaw_msg}"
        self.get_logger().info(response.message)
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

    def object_spec(self, object: str, overrides=None) -> ObjectSpec:
        # Prior for one object from its init_data params (overrides: {param name: value} not applied yet)
        def param(name):
            name = 'init_data.{}.{}'.format(object, name)
            if overrides is not None and name in overrides:
                return overrides[name]
            return self.get_parameter(name).value

        position = numpy.array([float(param('pose.x')), float(param('pose.y')), float(param('pose.z'))])
        return ObjectSpec(
            parent=str(param('parent')),
            position=position,
            yaw=math.radians(float(param('pose.yaw'))),
            lock_yaw=bool(param('lock_orientation_to_config')),
        )

    def create_estimator(self) -> MapEstimator:
        specs = {object: self.object_spec(object) for object in self.objects.keys()}
        return MapEstimator(specs, self.read_settings(EstimatorSettings, "estimator"), self.read_settings(NoiseSettings, "noise"))

    def read_settings(self, settings_type, prefix: str, overrides=None):
        values = {}
        for f in fields(settings_type):
            name = "{}.{}".format(prefix, f.name)
            value = overrides[name] if overrides is not None and name in overrides else self.get_parameter(name).value
            values[f.name] = type(f.default)(value)
        return settings_type(**values)

    # Creates a publisher to publish PoseWithCovariance
    def add_publisher(self, object: str):
        self.objects[object]["publisher"] = self.create_publisher(PoseWithCovarianceStamped, "mapping/{}".format(object), qos_profile_system_default)

    # Apply init_data, estimator and noise changes. Runs before the new values are stored, so read them from params
    def param_callback(self, params):
        overrides = {str(param.name): param.value for param in params}
        priors = set()
        for name in overrides:
            parts = name.split(".")
            if parts[0] == "init_data" and len(parts) > 2 and parts[1] in self.objects and parts[2] in ("pose", "parent", "lock_orientation_to_config"):
                priors.add(parts[1])

        try:
            if any(name.startswith("estimator.") for name in overrides):
                self.estimator.settings = self.read_settings(EstimatorSettings, "estimator", overrides)
            if any(name.startswith("noise.") for name in overrides):
                self.estimator.noise = self.read_settings(NoiseSettings, "noise", overrides)
        except (TypeError, ValueError) as ex:
            return SetParametersResult(successful=False, reason=str(ex))

        for object in priors:
            spec = self.object_spec(object, overrides)
            if spec.parent != self.estimator.specs[object].parent:
                # Moving an object between map and a parent frame changes the filter layout, rebuild on the next reset
                self.get_logger().warning(f"{object} parent changed to {spec.parent}, applies after mapping/reset_mapping")
                continue
            self.estimator.specs[object].lock_yaw = spec.lock_yaw
            self.estimator.set_prior(object, spec.position, spec.yaw)
            self.output.pop(object, None)
            # The timer publishes the moved frame on its next tick
            self.output_settling = True

        return SetParametersResult(successful=True)

    def set_targets(self, targets: str):
        # "torpedo" or "torpedo, bin": every listed object is fused, the first is the primary target
        self.target_objects = [t.strip() for t in str(targets).split(",") if t.strip() != ""]
        self.target_object = self.target_objects[0] if self.target_objects else ""

    def target_callback(self, request: MappingTarget.Request, response: MappingTarget.Response):
        self.set_targets(request.target_info.target_object)
        self.lock_map = bool(request.target_info.lock_map)

        for target in self.target_objects:
            # Resetting a pinned basket would let detections drag it off the table fit
            if self.table_centered and target in TABLE_BASKETS:
                self.get_logger().info(f"{target} is pinned to the table, not resetting it")
            elif target in self.objects.keys():
                # Back to the prior covariance (keeps the position) so a cov check on it needs fresh detections
                self.get_logger().info(f"reset {target}")
                self.estimator.reset(target)
            else:
                self.get_logger().warning(f"Unknown target object {target}")

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

        self.estimator.reset(target1)
        self.estimator.reset(target2)

        self.instance1_seeded = False
        self.instance2_seeded = False

        self.set_targets(target1)
        self.lock_map = False
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
            self.set_targets("")
            self.outstanding_detections.clear()
            self.get_logger().info(message)
        else:
            self.get_logger().warning(message)

        response.success = success
        response.message = message
        return response

    def maybe_seed_binary_instances(self):
        # Fire once on each aquire -> track transition
        # Seed the target on the classifier centroid so the published pose snaps to the cluster instead of crawling
        # from the reckoned init pose (needs to only fire once, or cov would never drop)
        bc = self.binary_classifier

        if bc.first_locked and not self.instance1_seeded:
            self.seed_object_estimate(bc.instance1_name, bc.instance1_centroid)
            self.instance1_seeded = True

        if bc.second_locked and not self.instance2_seeded:
            self.seed_object_estimate(bc.instance2_name, bc.instance2_centroid)
            self.instance2_seeded = True
            
    def seed_object_estimate(self, child: str, centroid):
        # Put the object (and its TF frame) on the classifier centroid, given in the object's parent frame, keeping its yaw.
        # It starts from its prior covariance, so it reads as soft until detections tighten it
        if child not in self.objects.keys():
            self.get_logger().error(f"seed_object_estimate: unknown object {child}")
            return

        self.estimator.seed(child, (float(centroid[0]), float(centroid[1]), float(centroid[2])))
        self.output.pop(child, None)

    def vision_callback(self, detections: Detection3DArray):
        if self.lock_map:
            return
        
        # Bypass normal mapping behavior for binary classifier
        if self.binary_classifier.running:
            self.handle_binary_classifier_detections(detections)
            self.publish_pose()
            return
        
        closest_object = self.closest_object(detections)
        
        # Fuse the target objects, or the closest object if none is set. Other classes are left alone because lookalike props
        # get each other's class
        fused = set(self.target_objects) if self.target_objects else {closest_object}
                
        # Send each targeted detection to the estimator
        for detection in detections.detections:
            result = next((r for r in detection.results if r.hypothesis.class_id in fused), None)
            if result is None:
                continue
                        
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
            return False, None, str(ex)

        trans_pose = do_transform_pose_stamped(pose, transform)
        if child in self.downwards_objects.keys():
            trans_pose.pose.orientation.x = 0.0
            trans_pose.pose.orientation.y = 0.0
            trans_pose.pose.orientation.z = 0.0
            trans_pose.pose.orientation.w = 1.0

        q = transform.transform.rotation
        rotation = quat2mat([q.w, q.x, q.y, q.z])
        position_camera = (pose.pose.position.x, pose.pose.position.y, pose.pose.position.z)
        stamp_sec = float(detection_header.stamp.sec) + float(detection_header.stamp.nanosec) / 1e9

        return True, ProjectedDetection(trans_pose, parent, rotation, position_camera, stamp_sec), ""

    def update_object_with_pose(self, projected: ProjectedDetection, child: str):
        # Fuse one detection. A detection the estimator rejects still counts as handled (it is not retried)
        if not child in self.objects.keys():
            return False, f"Unknown mapping object {child}"

        update_position = True
        update_orientation = True

        # These objects keep their orientation from config (detection yaw is unreliable)
        if bool(self.get_parameter("init_data.{}.lock_orientation_to_config".format(child)).value):
            update_orientation = False

        # Baskets are pinned once the table has been fit to them
        if self.table_centered and child in TABLE_BASKETS:
            update_position = False

        pose = projected.pose.pose
        position = None
        position_cov = None
        if update_position:
            angular_speed, linear_speed = self.motion_at(projected.stamp_sec)
            position = (pose.position.x, pose.position.y, pose.position.z)
            position_cov = detection_covariance(projected.position_camera, projected.rotation, angular_speed, linear_speed,
                                                self.estimator.noise)

        yaw = None
        yaw_var = None
        if update_orientation:
            # Vision detects the normal vector as the Z axis, its heading is the object's yaw
            # (maybe someday roll and pitch will be nonzero)
            det_rotm = quat2mat([pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z])
            normal = det_rotm @ numpy.array([0.0, 0.0, 1.0])
            yaw = math.atan2(normal[1], normal[0])
            yaw_var = math.radians(self.estimator.noise.yaw_std_deg) ** 2

        self.predict_offset()
        accepted, reason = self.estimator.observe(child, projected.stamp_sec, position, position_cov, yaw, yaw_var)
        if not accepted:
            self.get_logger().debug(f"Rejected detection of {child}: {reason}")
            if "reacquiring" in reason:
                self.get_logger().warning(f"{child}: {reason}")

        return True, reason

    def try_update_pose(self, result: ObjectHypothesisWithPose, detection_header: Header, closest_object: str):        
        child: str = result.hypothesis.class_id

        update_success, projected, error_msg = self.transform_detection_to_object_parent(
            result,
            detection_header,
            child,
        )

        if not update_success:
            return False, error_msg

        return self.update_object_with_pose(projected, child)

    def try_update_binary_classifier_pose(self, result: ObjectHypothesisWithPose, detection_header: Header):
        common_target = self.binary_classifier.instance1_name

        update_success, projected, error_msg = self.transform_detection_to_object_parent(
            result,
            detection_header,
            common_target,
        )

        if not update_success:
            return False, error_msg

        now_sec = float(self.get_clock().now().nanoseconds) / float(1e9)

        sample = DetectionSample(
            projected.pose.pose.position.x,
            projected.pose.pose.position.y,
            projected.pose.pose.position.z,
            result.hypothesis.score,
            projected.stamp_sec,
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
        if child_parent != projected.parent:
            update_success, projected, error_msg = self.transform_detection_to_object_parent(
                result,
                detection_header,
                child,
            )

            if not update_success:
                return False, error_msg

        return self.update_object_with_pose(projected, child)
               
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
    
    def odom_callback(self, msg: Odometry):
        p = msg.pose.pose.position
        position = numpy.array([p.x, p.y, p.z])
        if self.last_odom_position is not None:
            step = float(numpy.linalg.norm(position - self.last_odom_position))
            if step < MAX_ODOM_STEP:
                self.pending_distance += step
        self.last_odom_position = position

        w = msg.twist.twist.angular
        v = msg.twist.twist.linear
        stamp = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) / 1e9
        self.motion_history.append((stamp, math.sqrt(w.x**2 + w.y**2 + w.z**2), math.sqrt(v.x**2 + v.y**2 + v.z**2)))

    def motion_at(self, stamp_sec: float):
        # Vehicle angular and linear speed at the odometry sample nearest stamp_sec (still if there is no odometry)
        if not self.motion_history:
            return 0.0, 0.0
        nearest = min(self.motion_history, key=lambda m: abs(m[0] - stamp_sec))
        return nearest[1], nearest[2]

    def predict_offset(self):
        # Grow the offset's uncertainty by the dead-reckoning drift since the last call
        now = self.get_clock().now()
        dt = (now - self.last_predict_time).nanoseconds / 1e9
        self.estimator.predict(self.pending_distance, dt)
        self.pending_distance = 0.0
        self.last_predict_time = now

    def update_oustanding_items(self):
        self.update_outstanding_detections()
        
        # publish while frames are still sliding to the estimate, otherwise when the poses go stale
        elapsed = (self.get_clock().now() - self.last_pub_time).nanoseconds / 1e9
        if self.output_settling or elapsed >= STALE_TIME:
            self.publish_pose()

    # Publishes stuff
    def publish_pose(self):
        self.predict_offset()

        now_time = self.get_clock().now()
        now = now_time.to_msg()
        # Capped so a publish after a quiet spell (only stale publishes) can't jump a frame that just started moving
        dt = 0.0 if self.last_output_time is None else min(max((now_time - self.last_output_time).nanoseconds / 1e9, 0.0), MAX_SLEW_DT)
        self.last_output_time = now_time
        max_speed = float(self.get_parameter("output.max_speed").value)
        max_yaw_rate = math.radians(float(self.get_parameter("output.max_yaw_rate_deg").value))
        settling = False

        # map -> offset is where the course sits in odometry. Map objects hang off it at prior + deviation
        offset, _ = self.estimator.offset()
        if self.output_offset is None:
            self.output_offset = offset.copy()
        else:
            self.output_offset = slew_vector(self.output_offset, offset, max_speed, dt)
        settling = settling or float(numpy.linalg.norm(offset - self.output_offset)) > 1e-4

        offset_transform = TransformStamped()
        offset_transform.transform.translation = Vector3(x=float(self.output_offset[0]), y=float(self.output_offset[1]), z=float(self.output_offset[2]))
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
            is_map = self.estimator.is_map_object(object)

            # Slide the published pose toward the estimate. Map objects slide in map (offset included) so a frame never moves
            # faster than max_speed in the world, the rest in their parent frame
            target_position, cov = self.estimator.position(object)
            target_yaw, yaw_var = self.estimator.yaw(object)
            out = self.output.get(object)
            if out is None:
                out = (target_position.copy(), target_yaw)
            else:
                out = (slew_vector(out[0], target_position, max_speed, dt), slew_angle(out[1], target_yaw, max_yaw_rate, dt))
            self.output[object] = out
            position, yaw = out
            lag = target_position - position
            yaw_lag = math.atan2(math.sin(target_yaw - yaw), math.cos(target_yaw - yaw))
            settling = settling or float(numpy.linalg.norm(lag)) > 1e-4 or abs(yaw_lag) > 1e-4

            # Map objects' frames hang off the offset frame, the rest off their parent
            translation = position - self.output_offset if is_map else position

            # A frame that has not caught up with the estimate is that much less certain
            cov = cov + numpy.diag(lag ** 2)
            yaw_var = yaw_var + yaw_lag ** 2

            pose = PoseWithCovarianceStamped()
            pose.header.stamp = now
            pose.header.frame_id = parent
            pose.pose.pose.position = Point(x=float(position[0]), y=float(position[1]), z=float(position[2]))
            (pose.pose.pose.orientation.w,
            pose.pose.pose.orientation.x,
            pose.pose.pose.orientation.y,
            pose.pose.pose.orientation.z) = euler2quat(0.0, 0.0, yaw)

            # point yaw at parent on the PUBLISHED pose too, so the cov arrow matches the tf.
            # only applies to non-map parents; position here is in parent frame.
            if parent != "map" and not lock_orientation and not is_downward \
                and bool(self.get_parameter("init_data.{}.point_yaw_at_parent".format(object)).value):
                
                yaw = math.atan2(-pose.pose.pose.position.y, -pose.pose.pose.position.x)
                (pose.pose.pose.orientation.w,
                pose.pose.pose.orientation.x,
                pose.pose.pose.orientation.y,
                pose.pose.pose.orientation.z) = euler2quat(0.0, 0.0, yaw)

            covariance = [0.0] * 36
            for i in range(3):
                for j in range(3):
                    covariance[6 * i + j] = float(cov[i, j])
            covariance[35] = float(yaw_var)

            # Cov stuff for cursed ah bin fit, won't touch other objects since they dont have "fit_covar"
            fit_covar = self.objects[object].get("fit_covar")
            if fit_covar is not None:
                covariance[0] = fit_covar
                covariance[7] = fit_covar
                covariance[14] = fit_covar

            pose.pose.covariance = covariance
            self.objects[object]["publisher"].publish(pose)

            transform = TransformStamped()
            transform.header.stamp = pose.header.stamp
            transform.header.frame_id = "offset" if is_map else parent
            transform.child_frame_id = object + "_frame"
            transform.transform.translation = Vector3(x=float(translation[0]), y=float(translation[1]), z=float(translation[2]))
            transform.transform.rotation = pose.pose.pose.orientation

            if all(math.isfinite(v) for v in translation):
                transforms.append(transform)
            else:
                self.get_logger().error(f"Not publishing {object}_frame, its translation is not finite")
        
        self.tf_brod.sendTransform(transforms)
        
        # feed the buffer
        for transform in transforms:
            self.tf_buffer.set_transform(transform, "default_authority")
        
        # publish status
        stat = MappingTargetInfo()
        stat.target_object = ",".join(self.target_objects)
        stat.lock_map = self.lock_map
        self.status_pub.publish(stat)
        
        self.last_pub_time = self.get_clock().now()
        self.output_settling = settling


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(MappingNode())
    rclpy.shutdown()

if __name__ == '__main__':
    main()
