#! /usr/bin/env python3

import numpy as np
import rclpy
from geometry_msgs.msg import PoseWithCovariance, PoseWithCovarianceStamped
from rclpy.node import Node
from std_msgs.msg import Header
from transforms3d.euler import euler2quat, quat2euler
from vision_msgs.msg import (Detection3D, Detection3DArray,
                             ObjectHypothesisWithPose)

objects = [
    "gman",
    "bootlegger"
]

pubs = [ ]

class DummyDetectionNode(Node):
    def __init__(self):
        super().__init__("dummydetections")
        self.declareParams()
        self.timerRate = self.get_parameter("timer_rate").value
        if self.timerRate == 0:
            self.get_logger().error("Error reading params! Timer rate is zero!")
            
        self.topic     = self.get_parameter("topic").value
        self.timer     = self.create_timer(self.timerRate, self.timerCB)
        self.pub       = self.create_publisher(Detection3DArray, self.topic, 10)
        for object in objects:
            pubs.append(self.create_publisher(PoseWithCovarianceStamped, f"dummydetections/{object}", 10))
            
        self.get_logger().info("Started dummy detection node.")
        
        
    def declareParams(self):
        self.declare_parameter("timer_rate", 0.0)
        self.declare_parameter("topic", "detected_objects")
        
        for object in objects:
            self.declare_parameter(f"detection_data.{object}.pose", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            self.declare_parameter(f"detection_data.{object}.noise", 0.0)
            self.declare_parameter(f"detection_data.{object}.score", 0.0)
            self.declare_parameter(f"detection_data.{object}.publish_invalid_orientation", False)
            
          
    def timerCB(self):
        detectArray = Detection3DArray()
        
        #formulate header
        header = Header()
        header.frame_id = "world"
        header.stamp = self.get_clock().now().to_msg()
        
        detectArray.header = header
        
        for i in range(0, len(objects)):
            object = objects[i]
            poseArr = self.get_parameter(f"detection_data.{object}.pose").value
            noise = self.get_parameter(f"detection_data.{object}.noise").value
            score = self.get_parameter(f"detection_data.{object}.score").value
            publishInvalid = self.get_parameter(f"detection_data.{object}.publish_invalid_orientation").value
            
            if poseArr is not None:
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
                pubs[i].publish(pubPose)
                
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
    