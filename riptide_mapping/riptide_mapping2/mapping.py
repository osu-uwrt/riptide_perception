import rclpy
from rclpy.node import Node

import tf2_ros

class MappingNode(Node):

    objects = { 
        "gate", 
        "earth_glyph", 
        "buoy",
        "buoy_glyph_1",
        "buoy_glyph_2",
        "torpedo",
        "torpedo_upper_hole",
        "torpedo_lower_hole",
        "table",
        "prequal_gate",
        "prequal_pole"
    }

    def __init__(self):
        # Init the ROS Node
        super().__init__('riptide_mapping2')

        # Create the buffer to send 
        self.tf_buffer = tf2_ros.buffer.Buffer()
        self.tf_brod = self.tf_brod = tf2_ros.transform_broadcaster.TransformBroadcaster(self)

        # Create parameters on the ROS node with config data as default
        for object in self.objects:
            self.declare_parameters(
                namespace='',
                parameters=[
                    ("init_data.{}.parent".format(object), self.config["init_data.{}.parrent".format(object)]),
                    ("init_data.{}.pose.x".format(object), self.config["init_data.{}.pose.x".format(object)]),
                    ("init_data.{}.pose.y".format(object), self.config["init_data.{}.pose.y".format(object)]),
                    ("init_data.{}.pose.z".format(object), self.config["init_data.{}.pose.z".format(object)]),
                    ("init_data.{}.pose.yaw".format(object), self.config["init_data.{}.pose.yaw".format(object)]),
                    ("init_data.{}.covar.x".format(object), self.config["init_data.{}.covar.x".format(object)]),
                    ("init_data.{}.covar.y".format(object), self.config["init_data.{}.covar.y".format(object)]),
                    ("init_data.{}.covar.z".format(object), self.config["init_data.{}.covar.z".format(object)]),
                    ("init_data.{}.covar.yaw".format(object), self.config["init_data.{}.covar.yaw".format(object)]),
                ])



def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(MappingNode())
    rclpy.shutdown()

if __name__ == '__main__':
    main()
