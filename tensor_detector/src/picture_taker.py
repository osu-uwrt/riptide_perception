#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger
from cv_bridge import CvBridge
import cv2
import os
from datetime import datetime


class ImageCaptureNode(Node):
    def __init__(self):
        super().__init__('image_capture_node')
        
        # Parameters
        self.declare_parameter('image_topic', '/talos/dfc/zed_node/rgb/image_raw_color')
        self.declare_parameter('save_directory', '~/cal_images')
        
        self.image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.save_directory = self.get_parameter('save_directory').get_parameter_value().string_value
        
        # Create save directory if it doesn't exist
        os.makedirs(self.save_directory, exist_ok=True)
        
        # Initialize CV bridge for image conversion
        self.bridge = CvBridge()
        
        # Store the latest image
        self.latest_image = None
        
        # Create subscriber for image topic
        self.image_subscriber = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            10
        )
        
        # Create service for capturing images
        self.capture_service = self.create_service(
            Trigger,
            'capture_image',
            self.capture_image_callback
        )
        


    def image_callback(self, msg):
        try:
            # Convert ROS image message to OpenCV format
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            pass

    def capture_image_callback(self, request, response):
        if self.latest_image is None:
            response.success = False
            response.message = "No image available"
            return response
        
        try:
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"captured_image_{timestamp}.png"
            filepath = os.path.join(self.save_directory, filename)
            
            # Save the image as PNG with no compression
            success = cv2.imwrite(filepath, self.latest_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            
            if success:
                response.success = True
                response.message = f"Image saved as {filename}"
            else:
                response.success = False
                response.message = "Failed to save image"
                
        except Exception as e:
            response.success = False
            response.message = f"Error: {str(e)}"
        
        return response


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = ImageCaptureNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()