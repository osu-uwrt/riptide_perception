import typing

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_system_default

from geometry_msgs.msg import PoseWithCovarianceStamped

import tf2_ros

class MappingNode(Node):

    # List of object frames that are being tracked
    objects : dict[str, dict[str, typing.Any]] = {
        "gate": None, 
        "earth_glyph": None, 
        "buoy": None,
        "buoy_glyph_1": None,
        "buoy_glyph_2": None,
        "torpedo": None,
        "torpedo_upper_hole": None,
        "torpedo_lower_hole": None,
        "table": None,
        "prequal_gate": None,
        "prequal_pole": None
    }

    def __init__(self):
        # Init the ROS Node
        super().__init__('riptide_mapping2')

        # Create the buffer to send 
        self.tf_buffer: tf2_ros.buffer.Buffer = tf2_ros.buffer.Buffer()
        self.tf_brod: tf2_ros.transform_broadcaster.TransformBroadcaster = tf2_ros.transform_broadcaster.TransformBroadcaster(self)

        # Create parameters on the ROS node with config data as default
        for object in self.objects.keys():
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
            
            # Create Estimate Objects in translational and rotational systems and publishers for ROS
            self.objects[object] = {
                "translational": None,
                "rotational": None,
                "publisher": self.create_publisher(PoseWithCovarianceStamped, "mapping/{}".format(object), qos_profile_system_default)
            }


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(MappingNode())
    rclpy.shutdown()

if __name__ == '__main__':
    main()
