#! /usr/bin/env python3

#
# Robot gaslighting script :)
#

from math import sqrt, atan2, pi, acos

import numpy as np
import transforms3d as tf3d
import rclpy
import rclpy.time
from rcl_interfaces.msg import SetParametersResult
from geometry_msgs.msg import PoseWithCovariance, PoseWithCovarianceStamped, Pose, Quaternion
from rclpy.node import Node
from std_msgs.msg import Header
from tf2_ros import Buffer, TransformException, TransformListener
from tf2_geometry_msgs import do_transform_pose
from transforms3d.euler import euler2quat, quat2euler
from transforms3d.quaternions import qmult, qinverse, rotate_vector
from vision_msgs.msg import (Detection3D, Detection3DArray,
                             ObjectHypothesisWithPose)

TOPIC_NAME = "detected_objects"
CAMERA_ROTATION = tf3d.euler.euler2quat(-1.5707, 0, -1.5707) # makes orientations agree with camera

objects = [
    "gate",
    "gate_reefshark",
    "gate_sawfish",
    "slalom_front",
    "slalom_middle",
    "slalom_back",
    "torpedo",
    "torpedo_large_hole",
    "torpedo_small_hole",
    "bin_target",
    "table",
    "table_basket_pink",
    "table_basket_yellow",
    "table_spoon_pink",
    "table_bottle_yellow"
]

config = {}

class DummyDetectionNode(Node):
    def __init__(self):
        super().__init__("dummydetections")
        self.declareParams()
        
        name = self.get_namespace() + "/"
        self.robot = name[1 : name.find('/', 1)] #start substr at 1 to omit leading /
        
        self.timerPeriod   = self.get_parameter("timer_period").value
        self.pool        = self.get_parameter("simulate_pool").value
        
        if self.timerPeriod == 0:
            self.get_logger().error("Error reading params! Timer period is zero!")
        
        self.add_on_set_parameters_callback(self.updateParams)
            
        self.timer = self.create_timer(self.timerPeriod, self.timerCB) #timer rate will be updated        
        self.pub         = self.create_publisher(Detection3DArray, TOPIC_NAME, 10)
        self.tfBuffer    = Buffer()
        self.tfListener  = TransformListener(self.tfBuffer, self)
        self.pubs        = [ ]
        for object in objects:
            self.pubs.append(self.create_publisher(PoseWithCovarianceStamped, f"dummydetections/{object}", 10))
        
        self.updateParams(self.get_parameters(self._parameters.keys()))
        self.get_logger().info("Started dummy detection node.")
        
        
    def declareParams(self):
        self.declare_parameter("timer_period", 0.0)
        self.declare_parameter("simulate_pool", False)
        self.declare_parameter("forward_camera_hfov", 60)
        self.declare_parameter("forward_camera_vfov", 40)
        self.declare_parameter("forward_camera_frame", "stereo/left_link")
        self.declare_parameter("forward_camera_pub_frame", "stereo/left_link")
        self.declare_parameter("downward_camera_hfov", 60)
        self.declare_parameter("downward_camera_vfov", 40)
        self.declare_parameter("downward_camera_frame", "downward_link")
        self.declare_parameter("downward_camera_pub_frame", "downward_link")
        
        for object in objects:
            self.declare_parameter(f"detection_data.{object}.pose", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            self.declare_parameter(f"detection_data.{object}.noise", 0.0)
            self.declare_parameter(f"detection_data.{object}.score", 0.0)
            self.declare_parameter(f"detection_data.{object}.downward", False)
            self.declare_parameter(f"detection_data.{object}.publish_invalid_orientation", False)
            self.declare_parameter(f"detection_data.{object}.min_dist", 0.0)
            self.declare_parameter(f"detection_data.{object}.max_dist", 0.0)   
        
    
    #TODO: UPDATE ALL THE OTHER NON OBJECT PARAMS LIKE SIMULATE_POOL
    def updateParams(self, params):
        for param in params: #set timer to new rate if there is a new rate to set
            config[param.name] = param.value
            
            if param.name == "timer_rate":
                self.timer.timer_period_ns = param.value * 1e9
                
        return SetParametersResult(successful=True)
    
    
    def isVisibleByCamera(self, cameraName: str, objectPoseMap: Pose, objectMinDist: float, objectMaxDist: float, downward: bool):
        cameraFrame = config[f"{cameraName}_camera_frame"]
        cameraHFov = config[f"{cameraName}_camera_hfov"]
        cameraVfov = config[f"{cameraName}_camera_vfov"]
        
        cameraFrameName = cameraFrame.replace("<robot>", self.robot)
        
        try:
            mapToCameraTransform = self.tfBuffer.lookup_transform(cameraFrameName, "map", rclpy.time.Time())
            objectPoseCameraFrame = do_transform_pose(objectPoseMap, mapToCameraTransform)
        except TransformException as ex:
            self.get_logger().warn(f"Could not look up transform from {cameraFrameName} to map! {ex}", throttle_duration_sec = 0.5)
            return False
        
        #distance to object
        objectDist = sqrt(objectPoseCameraFrame.position.x ** 2 + \
                    objectPoseCameraFrame.position.y ** 2 + \
                    objectPoseCameraFrame.position.z ** 2)
                        
        #object position relative to camera. Measured in angles to make it easier to eval fov
        objectHAng = abs(atan2(objectPoseCameraFrame.position.y,
                            objectPoseCameraFrame.position.x) * 180 / pi)
        
        objectVAng = abs(atan2(objectPoseCameraFrame.position.z,
                            objectPoseCameraFrame.position.x) * 180 / pi)

        #object rotation relative to camera (if camera is next to object it cant see it)
        #invert direction of object rotation in camera frame so angles are < 180
        transformQuat = [objectPoseCameraFrame.orientation.w, objectPoseCameraFrame.orientation.x,
                         objectPoseCameraFrame.orientation.y, objectPoseCameraFrame.orientation.z]
        
        reverseQuat = [0.707, 0, 0.707, 0] if downward else [0, 0, 0, 1] #quats aimed to align normal vectors with the camera x axis. ordered wxyz
        cameraDiffQuat = qmult(transformQuat, reverseQuat)

        rotatedVector = rotate_vector([1, 0, 0], cameraDiffQuat)
        cosang = np.dot([1, 0, 0], rotatedVector) / np.linalg.norm(rotatedVector)
        camRotDiff = acos(cosang) * 180.0 / pi
        
        self.get_logger().debug(f"object dist: {objectDist} in [{objectMinDist}, {objectMaxDist}]")
        self.get_logger().debug(f"object hang: {objectHAng} < {cameraHFov / 2}")
        self.get_logger().debug(f"object vang: {objectVAng} < {cameraVfov / 2}")
        self.get_logger().debug(f"rotation diff: {camRotDiff} < 94")
        
        isVisible =  objectDist < objectMaxDist and objectDist > objectMinDist and \
                objectHAng < cameraHFov / 2 and objectVAng < cameraVfov / 2 and \
                camRotDiff < 94 # this number was arbitrarily chosen for liltank testing
        
        self.get_logger().debug(f"Object is visible: {isVisible}\n")
        return isVisible
    
    
    #takes objectPos as [x, y, z] and maxDist to determine whether or not the robot can see an object.
    def isVisibleByRobot(self, objectName: str):
        #look up the camera position in TF. start by resolving the robot name
        objectPose = config[f"detection_data.{objectName}.pose"]
        maxDist = config[f"detection_data.{objectName}.max_dist"]
        minDist = config[f"detection_data.{objectName}.min_dist"]
        downward = config[f"detection_data.{objectName}.downward"]
                
        objectQuatArr = euler2quat(
            objectPose[3],
            objectPose[4],
            objectPose[5]
        )
        objectQuat = Quaternion(
            w = objectQuatArr[0],
            x = objectQuatArr[1],
            y = objectQuatArr[2],
            z = objectQuatArr[3]
        )
        
        objectPoseInMap = Pose()
        objectPoseInMap.position.x = objectPose[0]
        objectPoseInMap.position.y = objectPose[1]
        objectPoseInMap.position.z = objectPose[2]
        objectPoseInMap.orientation = objectQuat
                
        return self.isVisibleByCamera("forward", objectPoseInMap, minDist, maxDist, downward), \
                self.isVisibleByCamera("downward", objectPoseInMap, minDist, maxDist, downward)
        
        
    def timerCB(self):
        #quick param update
        self.pool        = self.get_parameter("simulate_pool").value
        forwardsDetectArray = Detection3DArray()
        downwardsDetectArray = Detection3DArray()
        
        now = self.get_clock().now().to_msg()
        
        #formulate header
        fwdHeader = Header()
        fwdHeader.frame_id = self.get_parameter("forward_camera_pub_frame").value.replace("<robot>", self.robot)
        fwdHeader.stamp = now
        
        dwdHeader = Header()
        dwdHeader.frame_id = self.get_parameter("downward_camera_pub_frame").value.replace("<robot>", self.robot)
        dwdHeader.stamp = now
        
        forwardsDetectArray.header = fwdHeader
        downwardsDetectArray.header = dwdHeader
        
        for i in range(0, len(objects)):
            objectName = objects[i]
            self.get_logger().debug(f"Processing dummy detection for {objectName}")
            
            poseArr = self.get_parameter(f"detection_data.{objectName}.pose").value # this pose is in map frame
            noise = self.get_parameter(f"detection_data.{objectName}.noise").value
            score = self.get_parameter(f"detection_data.{objectName}.score").value
            publishInvalid = self.get_parameter(f"detection_data.{objectName}.publish_invalid_orientation").value
            
            self.get_logger().debug(f"Object name: {objectName}")
            
            if poseArr is not None:
                visibleForwards, visibleDownwards = self.isVisibleByRobot(objectName)
                if not self.pool or visibleForwards or visibleDownwards:
                    mapPose = Pose()
                
                    #generate noise
                    [r, p, y] = quat2euler([mapPose.orientation.w, mapPose.orientation.x, mapPose.orientation.y, mapPose.orientation.z]) #wxyz
                    noise = np.random.normal(0, noise, 7)
                    
                    #add noise to stuff
                    mapPose.position.x = poseArr[0] + noise[0]
                    mapPose.position.y = poseArr[1] + noise[1]
                    mapPose.position.z = poseArr[2] + noise[2]
                    r = poseArr[3] + noise[3]
                    p = poseArr[4] + noise[4]
                    y = poseArr[5] + noise[5]
                    
                    #rpy to quat that boi
                    newQuat = euler2quat(r, p, y) #returns in WXYZ order
                    if publishInvalid:
                        newQuat = [2.0, 2.0, 2.0, 2.0] #invalid quaternion indicating that mapping should not merge orientation
                    
                    mapPose.orientation.w = newQuat[0]
                    mapPose.orientation.x = newQuat[1]
                    mapPose.orientation.y = newQuat[2]
                    mapPose.orientation.z = newQuat[3]
                    
                    # now convert pose to the desired camera frame
                    fwd_camera_frame = self.get_parameter("forward_camera_frame").value.replace("<robot>", self.robot) #this is the frame that "detects" the object
                    dwd_camera_frame = self.get_parameter("downward_camera_frame").value.replace("<robot>", self.robot)
                    camera_frame = fwd_camera_frame if visibleForwards else dwd_camera_frame if visibleDownwards else None
                    
                    map2camera = self.tfBuffer.lookup_transform(camera_frame, "map", rclpy.time.Time())
                    framePose = do_transform_pose(mapPose, map2camera)
                    
                    pubPose = PoseWithCovarianceStamped()
                    pubPose.header = fwdHeader if visibleForwards else dwdHeader if visibleDownwards else None
                    pubPose.pose.pose = framePose
                    self.pubs[i].publish(pubPose)
                    
                    #rotate quaternion to the "z out" position
                    transformedQuat = [framePose.orientation.w, framePose.orientation.x, framePose.orientation.y, framePose.orientation.z]
                    rotatedQuat = tf3d.quaternions.qmult(transformedQuat, CAMERA_ROTATION) #wxyz
                    
                    framePose.orientation.w = rotatedQuat[0]
                    framePose.orientation.x = rotatedQuat[1]
                    framePose.orientation.y = rotatedQuat[2]
                    framePose.orientation.z = rotatedQuat[3]
                                        
                    #populate detection. looks like mapping only uses results so I'll just populate that and also header because its easy
                    detection = Detection3D()
                    detection.header = fwdHeader if visibleForwards else dwdHeader if visibleDownwards else None
                    
                    hypothesis = ObjectHypothesisWithPose()
                    hypothesis.hypothesis.class_id = objectName
                    hypothesis.hypothesis.score = score
                    hypothesis.pose.pose = framePose
                                        
                    detection.results.append(hypothesis)
                    
                    # now append the detection to the correct detection array
                    if visibleForwards:
                        forwardsDetectArray.detections.append(detection)
                    
                    elif visibleDownwards:
                        downwardsDetectArray.detections.append(detection)
                    
        self.pub.publish(forwardsDetectArray)
        
        # if len(downwardsDetectArray.detections) > 0:
        #     self.pub.publish(downwardsDetectArray)
        
        
def main(args = None):
    rclpy.init(args=args)
    rclpy.spin(DummyDetectionNode())
    rclpy.shutdown()
    

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
