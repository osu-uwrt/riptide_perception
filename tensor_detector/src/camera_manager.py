#!/usr/bin/env python3

# Standard library imports
import time

# Third-party imports
import rclpy

# ROS2 imports
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import SetBool

class CameraManager(Node):
    def __init__(self):
        super().__init__('camera_manager')
        
        # Declare parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                # Camera configuration
                ('available_cameras', ['ffc', 'dfc']),
                ('default_camera', 'ffc'),
                
                # Service names
                ('camera_switch_service', 'set_camera_is_dfc'),
                
                # Topic names
                ('active_camera_topic', 'active_camera'),
                ('camera_switch_status_topic', 'camera_switch_status'),
                
                # Timing
                ('switch_delay', 0.1),
                ('status_publish_rate', 1.0),  # Hz
                
                # Queue sizes
                ('publisher_queue_size', 10),
            ]
        )
        
        # Load parameters
        self.load_parameters()
        
        # Initialize state
        self.active_camera = self.default_camera
        self.switch_in_progress = False
        self.last_switch_time = None
        
        # Create publishers
        self.create_publishers()
        
        # Create services
        self.create_services()
        
        # Create status timer
        self.create_status_timer()
        
        self.get_logger().info(f"Camera manager started with default camera: {self.active_camera}")

    def load_parameters(self):
        """Load all parameters from ROS parameter server"""
        self.available_cameras = self.get_parameter('available_cameras').get_parameter_value().string_array_value
        self.default_camera = self.get_parameter('default_camera').get_parameter_value().string_value
        
        # Service and topic names
        self.camera_switch_service_name = self.get_parameter('camera_switch_service').get_parameter_value().string_value
        self.active_camera_topic = self.get_parameter('active_camera_topic').get_parameter_value().string_value
        self.camera_switch_status_topic = self.get_parameter('camera_switch_status_topic').get_parameter_value().string_value
        
        # Timing
        self.switch_delay = self.get_parameter('switch_delay').get_parameter_value().double_value
        self.status_publish_rate = self.get_parameter('status_publish_rate').get_parameter_value().double_value
        
        # Queue sizes
        self.publisher_queue_size = self.get_parameter('publisher_queue_size').get_parameter_value().integer_value
        
        # Validate default camera
        if self.default_camera not in self.available_cameras:
            self.get_logger().error(f"Default camera '{self.default_camera}' not in available cameras {self.available_cameras}")
            self.default_camera = self.available_cameras[0] if self.available_cameras else 'ffc'
            self.get_logger().info(f"Using first available camera: {self.default_camera}")

    def create_publishers(self):
        """Create ROS2 publishers"""
        self.active_camera_publisher = self.create_publisher(
            String, self.active_camera_topic, self.publisher_queue_size
        )
        self.switch_status_publisher = self.create_publisher(
            String, self.camera_switch_status_topic, self.publisher_queue_size
        )

    def create_services(self):
        """Create ROS2 services"""
        self.camera_switch_service = self.create_service(
            SetBool, self.camera_switch_service_name, self.switch_camera_callback
        )
        self.get_logger().info(f"Camera switch service created: {self.camera_switch_service_name}")

    def create_status_timer(self):
        """Create timer for publishing status"""
        timer_period = 1.0 / self.status_publish_rate
        self.status_timer = self.create_timer(timer_period, self.publish_status)

    def switch_camera_callback(self, request, response):
        """Handle camera switching requests"""
        if self.switch_in_progress:
            response.success = False
            response.message = "Camera switch already in progress"
            self.get_logger().warning("Camera switch request denied - switch in progress")
            return response

        # Determine new camera based on request
        new_camera = 'dfc' if request.data else 'ffc'
        
        # Validate camera is available
        if new_camera not in self.available_cameras:
            response.success = False
            response.message = f"Camera '{new_camera}' not available. Available: {self.available_cameras}"
            self.get_logger().error(f"Switch to unavailable camera requested: {new_camera}")
            return response

        old_camera = self.active_camera

        # Check if already using requested camera
        if new_camera == old_camera:
            response.success = True
            response.message = f"Camera already set to {new_camera}"
            self.get_logger().info(f"Camera switch request for current camera: {new_camera}")
            return response

        # Perform the switch
        self.get_logger().info(f"Switching camera from {old_camera} to {new_camera}")
        self.switch_in_progress = True
        
        # Schedule the actual switch with delay
        self.delayed_timer = self.create_timer(self.switch_delay, 
                                             lambda: self.execute_camera_switch(new_camera, old_camera))
        
        response.success = True
        response.message = f"Camera switch initiated from {old_camera} to {new_camera}"
        return response

    def execute_camera_switch(self, new_camera, old_camera):
        """Execute the actual camera switch"""
        try:
            self.active_camera = new_camera
            self.last_switch_time = self.get_clock().now()
            
            # Publish the new active camera immediately
            self.publish_active_camera()
            
            self.get_logger().info(f"Camera successfully switched from {old_camera} to {new_camera}")
            
        except Exception as e:
            self.get_logger().error(f"Error during camera switch: {e}")
            # Revert to old camera on error
            self.active_camera = old_camera
            
        finally:
            self.switch_in_progress = False
            self.delayed_timer.cancel()

    def publish_active_camera(self):
        """Publish the currently active camera"""
        msg = String()
        msg.data = self.active_camera
        self.active_camera_publisher.publish(msg)

    def publish_status(self):
        """Publish camera manager status"""
        # Always publish active camera
        self.publish_active_camera()
        
        # Publish switch status
        status_msg = String()
        if self.switch_in_progress:
            status_msg.data = "switching"
        else:
            status_msg.data = "ready"
        
        self.switch_status_publisher.publish(status_msg)

    def get_camera_info(self):
        """Get current camera information (useful for debugging)"""
        return {
            'active_camera': self.active_camera,
            'available_cameras': self.available_cameras,
            'switch_in_progress': self.switch_in_progress,
            'last_switch_time': self.last_switch_time.to_msg() if self.last_switch_time else None
        }


def main(args=None):
    rclpy.init(args=args)
    node = CameraManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()