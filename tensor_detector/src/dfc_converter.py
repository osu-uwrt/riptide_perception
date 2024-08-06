#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_system_default
from geometry_msgs.msg import PoseWithCovarianceStamped
from vision_msgs.msg import Detection3DArray, Detection3D, ObjectHypothesisWithPose, ObjectHypothesis

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from transforms3d.quaternions import quat2mat

import cv2
import numpy as np

class DFCConverter(Node):
    def __init__(self):
        super().__init__('dfc_converter')
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        self.objects = ["blah", "bin_temperature", "worm", "coral"]
        
        self.objectIds = [10, 2, 13, 14]
        
        self.objectDepths = [-10, -2, -10, -10]
        
        self.dfc_subscription = self.create_subscription(Detection3DArray, 'dfc_objects', self.dfc_callback, 1)
        
        self.detection_publisher = self.create_publisher(Detection3DArray, "detected_objects" ,qos_profile_system_default)
        
    def dfc_callback(self, msg: Detection3DArray):
        
        convertedDetections = Detection3DArray()
        convertedDetections.header.stamp = msg.header.stamp
        convertedDetections.header.frame_id = "world"
        
        for detection in msg.detections:
            
            convertedDetection = Detection3D()
            
            for result in detection.results:
                convertedResult = ObjectHypothesisWithPose()
                resultClassId = int(result.hypothesis.class_id)
                
                usedFrameId = "downwards_camera/left_optical"
                
                if resultClassId > 20:
                    usedFrameId = "downwards_camera/right_optical"
                    resultClassId -= 20
                try:
                    convertedResult.hypothesis.class_id = self.objects[self.objectIds.index(resultClassId)]
                    
                    try:
                        tf = self.tf_buffer.lookup_transform(usedFrameId, "world", msg.header.stamp)
                        transformedRay = (result.detection.pose.pose.x, result.detection.pose.pose.y, result.detection.pose.pose.z)
                        transformedRay = quat2mat(tf.transform.rotation) * transformedRay
                        
                        loc = self.line_plane_intersection(transformedRay, (tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z), self.objectDepths[self.objectIds.index(resultClassId)])
                        
                        convertedResult.pose.pose.position.x = loc[0]
                        convertedResult.pose.pose.position.y = loc[1]
                        convertedResult.pose.pose.position.z = loc[2]
                        convertedResult.pose.pose.orientation.w = 1
                        convertedResult.pose.pose.orientation.x = 0
                        convertedResult.pose.pose.orientation.y = 0
                        convertedResult.pose.pose.orientation.z = 0
                        
                    except TransformException as ex:
                        self.get_logger().info(
                            f'Could not transform {usedFrameId} to {"world"}: {ex}')
                except ValueError:
                    self.get_logger().warning(f"Unknown class id {result.hypothesis.class_id}") 
                    continue
                        
                convertedDetection.results.append(convertedResult)
            convertedDetections.detections.append(convertedDetection)
        self.detection_publisher.publish(convertedDetections)

    def line_plane_intersection(self, ray, camPos, objDepth):
        t = (objDepth - camPos[2]) / ray[2]
        return camPos + ray * t
        

def main(args=None):
	rclpy.init(args=args)
	yolo_node = DFCConverter()
	rclpy.spin(yolo_node)
	yolo_node.destroy_node()
	rclpy.shutdown()

if __name__ == '__main__':
	main()
