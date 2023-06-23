#! /usr/bin/env python3

#
# Robot gaslighting script :)
#

from math import sqrt, atan2, pi

import numpy as np
import rclpy
import rclpy.time
from rcl_interfaces.msg import SetParametersResult
from geometry_msgs.msg import PoseWithCovariance, PoseWithCovarianceStamped
from rclpy.node import Node
from std_msgs.msg import Header
from tf2_ros import Buffer, TransformException, TransformListener
from transforms3d.euler import euler2quat, quat2euler
from vision_msgs.msg import (Detection3D, Detection3DArray,
                             ObjectHypothesisWithPose)

TOPIC_NAME = "detected_objects"

objects = [
    "gman",
    "bootlegger"
]

class DummyDetectionNode(Node):
    def __init__(self):
        super().__init__("dummydetections")
        self.declareParams()
        
        self.timerRate   = self.get_parameter("timer_rate").value
        self.pool        = self.get_parameter("simulate_pool").value
        self.cameraHFov  = self.get_parameter("camera_hfov").value
        self.cameraVFov  = self.get_parameter("camera_vfov").value
        self.cameraFrame = self.get_parameter("camera_frame").value
        
        if self.timerRate == 0:
            self.get_logger().error("Error reading params! Timer rate is zero!")
        
        self.add_on_set_parameters_callback(self.updateParams)
            
        self.timer = self.create_timer(self.timerRate, self.timerCB) #timer rate will be updated        
        self.pub         = self.create_publisher(Detection3DArray, TOPIC_NAME, 10)
        self.tfBuffer    = Buffer()
        self.tfListener  = TransformListener(self.tfBuffer, self)
        self.pubs        = [ ]
        for object in objects:
            self.pubs.append(self.create_publisher(PoseWithCovarianceStamped, f"dummydetections/{object}", 10))
                
        self.get_logger().info("Started dummy detection node.")
        
        
    def declareParams(self):
        self.declare_parameter("timer_rate", 0.0)
        self.declare_parameter("simulate_pool", False)
        self.declare_parameter("camera_hfov", 60)
        self.declare_parameter("camera_vfov", 40)
        self.declare_parameter("camera_frame", "stereo/left_link")
        
        for object in objects:
            self.declare_parameter(f"detection_data.{object}.pose", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            self.declare_parameter(f"detection_data.{object}.noise", 0.0)
            self.declare_parameter(f"detection_data.{object}.score", 0.0)
            self.declare_parameter(f"detection_data.{object}.publish_invalid_orientation", False)
            self.declare_parameter(f"detection_data.{object}.max_visible_dist", 0.0)   
        
        
    def updateParams(self, params):
        for param in params: #set timer to new rate if there is a new rate to set
            if param.name == "timer_rate":
                self.timer.timer_period_ns = param.value * 1e9
                
        return SetParametersResult(successful=True)
    
    
    #takes objectPos as [x, y, z] and maxDist to determine whether or not the robot can see an object.
    def isVisibleByRobot(self, objectPos: 'list[float]', maxDist: float):
        #look up the camera position in TF. start by resolving the robot name
        name = self.get_namespace() + "/"
        robot = name[1 : name.find('/', 1)] #start substr at 1 to omit leading /
                
        cameraFrameName = f"{robot}/{self.cameraFrame}"
        try:
            robotTransform = self.tfBuffer.lookup_transform("map", cameraFrameName, rclpy.time.Time())
            dX = objectPos[0] - robotTransform.transform.translation.x
            dY = objectPos[1] - robotTransform.transform.translation.y
            dZ = objectPos[2] - robotTransform.transform.translation.z
            
            #determine distance between camera and the object
            dist = sqrt((dX ** 2) + (dY ** 2) + (dZ ** 2))
            
            #determine the horizontal and vertical angles between the camera and the object
            _, p, y = quat2euler([robotTransform.transform.rotation.w,
                                  robotTransform.transform.rotation.x,
                                  robotTransform.transform.rotation.y,
                                  robotTransform.transform.rotation.z])
            
            horizHeadingToObj = atan2(dY, dX)
            horizHeadingToObj *= 180.0 / pi
            y *= 180.0 / pi
            hAngleDeg = y - horizHeadingToObj
            hAngleDeg = (hAngleDeg + 180) % 360 - 180
            
            vertHeadingToObj = atan2(dZ, dX)
            vertHeadingToObj *= 180.0 / pi
            p *= 180.0 / pi
            vAngleDeg = p - vertHeadingToObj
            vAngleDeg = (vAngleDeg + 180) % 360 - 180

            self.get_logger().info(f"dist: {dist}, hangledeg: {hAngleDeg}, vangledeg: {vAngleDeg}")
                        
            return dist < maxDist and abs(hAngleDeg) < self.cameraHFov and abs(vAngleDeg) < self.cameraVFov
        except TransformException as ex:
            self.get_logger().warn(f"Could not look up transform from {cameraFrameName} to world! {ex}", throttle_duration_sec = 1, skip_first = True)
            return False
          
        
    def timerCB(self):
        #quick param update
        self.pool        = self.get_parameter("simulate_pool").value
        self.cameraHFov  = self.get_parameter("camera_hfov").value
        self.cameraVFov  = self.get_parameter("camera_vfov").value
        self.cameraFrame = self.get_parameter("camera_frame").value
        
        detectArray = Detection3DArray()
        
        #formulate header
        header = Header()
        header.frame_id = "map"
        header.stamp = self.get_clock().now().to_msg()
        
        detectArray.header = header
        
        for i in range(0, len(objects)):
            object = objects[i]
            poseArr = self.get_parameter(f"detection_data.{object}.pose").value
            noise = self.get_parameter(f"detection_data.{object}.noise").value
            score = self.get_parameter(f"detection_data.{object}.score").value
            publishInvalid = self.get_parameter(f"detection_data.{object}.publish_invalid_orientation").value
            maxDist = self.get_parameter(f"detection_data.{object}.max_visible_dist").value
            
            if poseArr is not None:
                if not self.pool or self.isVisibleByRobot(poseArr, maxDist):
                    pose = PoseWithCovariance()
                
                    #generate noise
                    [r, p, y] = quat2euler([pose.pose.orientation.w, pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z]) #wxyz
                    noise = np.random.normal(0, noise, 7)
                    
                    #add noise to stuff
                    pose.pose.position.x = poseArr[0] + noise[0]
                    pose.pose.position.y = poseArr[1] + noise[1]
                    pose.pose.position.z = poseArr[2] + noise[2]
                    r = poseArr[3] + noise[3]
                    p = poseArr[4] + noise[4]
                    y = poseArr[5] + noise[5]
                    
                    #rpy to quat that boi
                    newQuat = euler2quat(r, p, y) #returns in WXYZ order
                    if publishInvalid:
                        newQuat = [2.0, 2.0, 2.0, 2.0] #invalid quaternion indicating that mapping should not merge orientation
                    
                    pose.pose.orientation.w = newQuat[0]
                    pose.pose.orientation.x = newQuat[1]
                    pose.pose.orientation.y = newQuat[2]
                    pose.pose.orientation.z = newQuat[3]
                    
                    pubPose = PoseWithCovarianceStamped()
                    pubPose.header = header
                    pubPose.pose = pose
                    self.pubs[i].publish(pubPose)
                    
                    #populate detection. looks like mapping only uses results so I'll just populate that and also header because its easy
                    detection = Detection3D()
                    detection.header = header
                    
                    hypothesis = ObjectHypothesisWithPose()
                    hypothesis.hypothesis.class_id = object
                    hypothesis.hypothesis.score = score
                    hypothesis.pose = pose
                    
                    detection.results.append(hypothesis)
                    detectArray.detections.append(detection)
        
        self.pub.publish(detectArray)
        
        
def main(args = None):
    rclpy.init(args=args)
    rclpy.spin(DummyDetectionNode())
    rclpy.shutdown()
    

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
