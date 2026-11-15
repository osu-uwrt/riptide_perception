#!/usr/bin/env python3
"""Replace invalid depth values with 0 so FoundationPose can use ZED depth.

The ZED marks pixels it can't measure as NaN, too-far pixels as +inf and
too-close pixels as -inf. Isaac ROS FoundationPose expects invalid depth to be
0 (NVIDIA's quickstart data uses that convention), and NaN/inf under the mask
turn its pose estimates into NaN. This node republishes a 32FC1 depth image
with every non-finite value set to 0. The header is copied unchanged, because
FoundationPose pairs depth with RGB, camera_info and the mask on exactly equal
stamps.

Topics are relative and remapped at run time:
    ros2 run tensor_detector depth_sanitizer.py --ros-args \
        -r depth:=<zed depth> -r depth_finite:=<sanitized depth>
"""

import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image


def sanitize_depth(depth):
    """Return (copy of depth with non-finite values set to 0, count replaced)."""
    invalid = ~np.isfinite(depth)
    cleaned = depth.copy()
    cleaned[invalid] = 0.0
    return cleaned, int(np.count_nonzero(invalid))


class DepthSanitizer(Node):
    def __init__(self):
        super().__init__("depth_sanitizer")
        self.bridge = CvBridge()

        # Depth 1 on both sides: only the newest frame matters, and a backlog of
        # 9 MB images would only add latency. Reliable matches the ZED publisher
        # and FoundationPose's default depth QoS.
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
        self.publisher = self.create_publisher(Image, "depth_finite", qos)
        self.subscription = self.create_subscription(
            Image, "depth", self.depth_callback, qos
        )

    def depth_callback(self, msg: Image):
        if msg.encoding != "32FC1":
            self.get_logger().warning(
                f"Expected 32FC1 depth, got '{msg.encoding}'; dropping frames",
                throttle_duration_sec=5.0,
            )
            return

        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except CvBridgeError as e:
            self.get_logger().warning(f"CvBridge Error: {e}", throttle_duration_sec=5.0)
            return

        cleaned, replaced = sanitize_depth(depth)

        out = self.bridge.cv2_to_imgmsg(cleaned, encoding="32FC1", header=msg.header)
        self.publisher.publish(out)

        # The invalid fraction is worth watching: underwater it shows how much
        # of the image the stereo matcher couldn't measure.
        self.get_logger().info(
            f"Replaced {replaced} of {cleaned.size} pixels "
            f"({100.0 * replaced / cleaned.size:.1f}%) with 0",
            throttle_duration_sec=10.0,
        )


def main(args=None):
    rclpy.init(args=args)
    node = None

    try:
        node = DepthSanitizer()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
