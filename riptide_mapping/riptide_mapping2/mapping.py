#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import qos_profile_system_default # can replace this with others

from rcl_interfaces.msg import SetParametersResult
from vision_msgs.msg import Detection3DArray
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, TransformStamped, Vector3

from estimate import KalmanEstimate, euclideanDist
from tf2_geometry_msgs import do_transform_pose_stamped
from transforms3d.euler import quat2euler, euler2quat
from tf2_ros import TransformException
import tf2_ros
import numpy as np
from math import pi

DEG_TO_RAD = (pi/180)

#Used to translate between DOPE ids and names of objects
object_ids = {
    0 : "gate",
    1 : "earth_glyph", 
    2 : "buoy",
    3 : "buoy_glyph_1",
    4 : "buoy_glyph_2",
    5 : "torpedo",
    6 : "torpedo_upper_hole",
    7 : "torpedo_lower_hole",
    8 : "table"
}

objects = {}
for key in object_ids.values():
    objects[key] = {
        "pose" : None,
        "publisher" : None
    }

class MappingNode(Node):

    def __init__(self):
        super().__init__('riptide_mapping2') 
                
        # class variables 
        self.tf_buffer = tf2_ros.buffer.Buffer()
        self.tl = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_buffer.transform
        self.mapFrame = "map"
        
        ## TODO CHANGE THIS TO WORK PROPERLY
        self.tf_brod = tf2_ros.transform_broadcaster.TransformBroadcaster(self)
        self.config = {}

        # declare the configuration data
        self.declare_parameters(
            namespace='',
            parameters=[
                # Covariance parameters for merging stats
                ('cov_limit', 1.0),
                ('k_value', 0.1),

                # filtering parameters in camera frame
                ('angle_cutoff', pi),
                ('distance_limit', 10.0),
                ('confidence_cutoff', .7),
                ('detection_cov_factor', 7.0)
            ])
        
        # declare the fields for all of the models in the dict
        for object in object_ids.values():
            self.declare_parameters(
                namespace='',
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
                ])

        # new parameter reconfigure call
        self.add_on_set_parameters_callback(self.paramUpdateCallback)

        # Creating publishers
        for field in objects:
            objects[field]["publisher"] = self.create_publisher(PoseWithCovarianceStamped, "mapping/{}".format(field), qos_profile_system_default)

        # Subscribers
        self.create_subscription(Detection3DArray, "detected_objects".format(self.get_namespace()), self.dopeCallback, qos_profile_system_default) # DOPE's information 

        # Timers
        self.publishTimer = self.create_timer(0.5, self.pubEstim) # publish the inital estimate
        self.paramUpdateTimer = self.create_timer(1.0, self.processParamUpdates) # propagate param updates
        self.paramUpdateTimer.cancel()

        # manually trigger the callback to load init params
        self.paramUpdateCallback(self.get_parameters(self._parameters.keys()))


    def pubEstim(self):
        for objectName in objects: 
            if not objects[objectName]["pose"] is None:
                # Publish that object's data out 
                output_pose = objects[objectName]["pose"].getPoseEstim()
                objects[objectName]["publisher"].publish(output_pose)

                # Publish /tf data for the given object 
                newTf = TransformStamped()
                newTf.transform.translation = Vector3(x=output_pose.pose.pose.position.x, 
                    y=output_pose.pose.pose.position.y, z=output_pose.pose.pose.position.z)
                newTf.transform.rotation = output_pose.pose.pose.orientation
                
                newTf.header.stamp = self.get_clock().now().to_msg()
                newTf.child_frame_id = objectName + "_frame"
                newTf.header.frame_id = output_pose.header.frame_id
                self.tf_brod.sendTransform(newTf)

    # This timer will fire 1 second after paramter updates
    # This lets the parameter system reload all changes before modifying the estimates that need to be updated
    def processParamUpdates(self):
        success = True
        for objectName in objects: 
            if('init_data.{}.needs_update'.format(objectName) in self.config 
                and self.config['init_data.{}.needs_update'.format(objectName)]):
                try:
                    self.get_logger().info("Resetting estimate for {}".format(objectName))
                    # reset the update flag
                    self.config['init_data.{}.needs_update'.format(objectName)] = False

                    # Get pose data from reconfig and update our map accordingly
                    object_pose = PoseWithCovarianceStamped()
                    object_pose.header.frame_id = self.config['init_data.{}.parent'.format(objectName)]
                    
                    object_pose.pose.pose.position.x = self.config['init_data.{}.pose.x'.format(objectName)]
                    object_pose.pose.pose.position.y = self.config['init_data.{}.pose.y'.format(objectName)]
                    object_pose.pose.pose.position.z = self.config['init_data.{}.pose.z'.format(objectName)]
                    
                    object_yaw = self.config['init_data.{}.pose.yaw'.format(objectName)] * DEG_TO_RAD # Need to convert this from degrees to radians.
                    
                    # convert rpy to quat
                    quat = euler2quat(0, 0, object_yaw) #order defaults to sxyz
                    
                    object_pose.pose.pose.orientation.w = quat[0]
                    object_pose.pose.pose.orientation.x = quat[1]
                    object_pose.pose.pose.orientation.y = quat[2]
                    object_pose.pose.pose.orientation.z = quat[3]
                    
                    object_pose.pose.covariance[0] = self.config['init_data.{}.covar.x'.format(objectName)]
                    object_pose.pose.covariance[7] = self.config['init_data.{}.covar.y'.format(objectName)]
                    object_pose.pose.covariance[14] = self.config['init_data.{}.covar.z'.format(objectName)]
                    object_pose.pose.covariance[35] = self.config['init_data.{}.covar.yaw'.format(objectName)]

                    # Create a new Estimate object on reconfig.
                    objects[objectName]["pose"] = KalmanEstimate(object_pose, self.config['k_value'], self.config['cov_limit'], self.config['detection_cov_factor'])

                except Exception as e:
                    eStr = "Exception: {}, Exception message: {}".format(type(e).__name__, e)
                    self.get_logger().error("Error while updating object estimate for {}. {}".format(objectName, eStr))
                    success = False

        # if we have sucessfully updated all params, cancel the timer
        if(success):
            self.paramUpdateTimer.cancel()
        


    # Handles reconfiguration for the mapping system.
    # NOTE: Reconfig reconfigures all values, not just the one specified in rqt.
    def paramUpdateCallback(self, params):
        # cancel the propagate timer
        self.paramUpdateTimer.cancel()

        # update config and mark for re-estimation
        for param in params:
            self.config[param.name] = param.value
            for objectName in objects: 
                if(objectName in param.name):
                    self.config['init_data.{}.needs_update'.format(objectName)] = True

        # allow the timer to run
        self.paramUpdateTimer.reset()

        return SetParametersResult(successful=True)

    # Handles merging DOPE's output into our representation
    # msg: Detection3DArray (http://docs.ros.org/en/lunar/api/vision_msgs/html/msg/Detection3DArray.html)
    def dopeCallback(self, msg: Detection3DArray):    
        # Context: This loop will run <number of different objects DOPE thinks it sees on screen> times
        # `detection` is of type Detection3D (http://docs.ros.org/en/lunar/api/vision_msgs/html/msg/Detection3D.html)
        for detection in msg.detections:
            # Note that we don't change the first loop to `detection in msg.detections.results` because we want the timestamp from the Detection3D object
            # Context: This loop will run <number of objects DOPE can identify> times 
            # `result` is of type ObjectHypothesisWithPose (http://docs.ros.org/en/lunar/api/vision_msgs/html/msg/ObjectHypothesisWithPose.html)
            for result in detection.results: 
                name = result.hypothesis.class_id 

                if not name in objects.keys():
                    self.get_logger().warn(f"Rejecting {name} because it is not registered")
                    continue
                
                if objects[name]["pose"] is None:
                    self.get_logger().warning(f"Rejected {name}: no estimate (not initialized)")
                    continue

                # DOPE's frame is the same as the camera frame, specifically the left lens of the camera.
                # We need to convert that to the map frame, which is what is used in our mapping system 
                # Tutorial on how this works @ http://wiki.ros.org/tf/TfUsingPython#TransformerROS_and_TransformListener
                # Transform the pose part 
                pose = PoseStamped()
                pose.header = detection.header
                pose.pose = result.pose.pose   

                # check the distance limits we have on the detection frame
                distance = euclideanDist(np.array([pose.pose.position.x, pose.pose.position.y, pose.pose.position.z]))
                dist_lim = self.config["distance_limit"]
                if(distance > dist_lim):
                    self.get_logger().warning(f"Rejected {name}: distance {distance}m outside limit of {dist_lim}", throttle_duration_sec = 1)
                    continue

                # check the angular difference between the robot and vision detection
                angleMax = self.config["angle_cutoff"]
                rpy = quat2euler([pose.pose.orientation.w, pose.pose.orientation.x,
                                             pose.pose.orientation.y, pose.pose.orientation.z])
                if(abs(rpy[2]) > angleMax):
                    self.get_logger().warning(f"Rejected {name}: relative angle {rpy[2]} outside {angleMax}", throttle_duration_sec = 1)
                    continue

                # theshold the confidence of the detection is above the min
                min = self.config["confidence_cutoff"]
                if(result.hypothesis.score < min):
                    self.get_logger().warning(f"Rejected {name}: confidence {result.hypothesis.score} below {min}", throttle_duration_sec = 1)
                    continue
                
                #transform pose into parent frame and feed to estimate
                parentFrame = objects[name]["pose"].getPoseEstim().header.frame_id
                try:
                    trans = self.tf_buffer.lookup_transform(
                        parentFrame,
                        detection.header.frame_id,
                        Time())
                except TransformException as ex:
                    self.get_logger().error(f'Could not transform {detection.header.frame_id} to {parentFrame}: {ex}', throttle_duration_sec = 1)
                    return

                # transform camera pose into map frame
                convertedPose = do_transform_pose_stamped(pose, trans)

                # Get the reading in the map frame message all together
                reading_parent_frame = PoseWithCovarianceStamped()
                reading_parent_frame.header.stamp = msg.header.stamp
                reading_parent_frame.header.frame_id = parentFrame
                reading_parent_frame.pose.pose = convertedPose.pose 
                
                # Merge the given position into our position for that object
                valid, errStr = objects[name]["pose"].addPosEstim(reading_parent_frame, result.pose.pose.orientation.w <= 1)
                if(not valid):
                    self.get_logger().warning(f"Rejected {name}: {errStr}", throttle_duration_sec = 1)
                else:
                    self.get_logger().info(f"FOUND {name}", throttle_duration_sec = 1)
            

def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(MappingNode())
    rclpy.shutdown()

if __name__ == '__main__':
    main()
