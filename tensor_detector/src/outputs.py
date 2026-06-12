#!/usr/bin/env python3
"""ROS output data stuf: detections, hypotheses, markers"""
from scipy.spatial.transform import Rotation as R

from vision_msgs.msg import (Detection3D, ObjectHypothesisWithPose,
                             ObjectHypothesis)
from visualization_msgs.msg import Marker


def new_detection(frame_id, stamp):
    detection = Detection3D()
    detection.header.frame_id = frame_id
    detection.header.stamp = stamp
    return detection


def make_hypothesis(class_id, centroid, quat, conf):
    """Build an ObjectHypothesisWithPose. class_id is the final published name."""
    hypothesis_with_pose = ObjectHypothesisWithPose()
    hypothesis = ObjectHypothesis()
    hypothesis.class_id = class_id
    # conf may be a numpy scalar (from a YOLO box) or a plain float (synthesized).
    hypothesis.score = float(conf.item()) if hasattr(conf, "item") else float(conf)

    hypothesis_with_pose.hypothesis = hypothesis
    hypothesis_with_pose.pose.pose.position.x = float(centroid[0])
    hypothesis_with_pose.pose.pose.position.y = float(centroid[1])
    hypothesis_with_pose.pose.pose.position.z = float(centroid[2])
    hypothesis_with_pose.pose.pose.orientation.x = float(quat[0])
    hypothesis_with_pose.pose.pose.orientation.y = float(quat[1])
    hypothesis_with_pose.pose.pose.orientation.z = float(quat[2])
    hypothesis_with_pose.pose.pose.orientation.w = float(quat[3])
    return hypothesis_with_pose


class MarkerBuilder:
    """Builds the (cube + arrow) marker pair for a pose"""

    def __init__(self, config):
        self.cfg = config
        self._counter = 0

    def reset(self):
        self._counter = 0

    def _next_id(self):
        self._counter += 1
        return self._counter

    def build(self, frame_id, stamp, quat, centroid, color, scale_x, scale_y, flat=False):
        """Return [plane_marker, arrow_marker]. Caller appends to its buffer."""
        lifetime = int(self.cfg.publish_interval * 2.0 * 1e9)
        markers = []

        plane_marker = Marker()
        plane_marker.header.frame_id = frame_id
        plane_marker.header.stamp = stamp
        plane_marker.ns = "detection_markers"
        plane_marker.id = self._next_id()
        plane_marker.type = Marker.CUBE
        plane_marker.action = Marker.ADD
        plane_marker.lifetime.nanosec = lifetime
        plane_marker.pose.position.x = float(centroid[0])
        plane_marker.pose.position.y = float(centroid[1])
        plane_marker.pose.position.z = float(centroid[2])
        plane_marker.pose.orientation.x = float(quat[0])
        plane_marker.pose.orientation.y = float(quat[1])
        plane_marker.pose.orientation.z = float(quat[2])
        plane_marker.pose.orientation.w = float(quat[3])
        plane_marker.scale.x = float(scale_x)
        plane_marker.scale.y = float(scale_y)
        plane_marker.scale.z = 0.01 if flat else 0.05
        plane_marker.color.r = color[0]
        plane_marker.color.g = color[1]
        plane_marker.color.b = color[2]
        plane_marker.color.a = 0.8
        markers.append(plane_marker)

        arrow_marker = Marker()
        arrow_marker.header.frame_id = frame_id
        arrow_marker.header.stamp = stamp
        arrow_marker.ns = "orientation_markers"
        arrow_marker.id = self._next_id()
        arrow_marker.type = Marker.ARROW
        arrow_marker.action = Marker.ADD
        arrow_marker.lifetime.nanosec = lifetime
        arrow_marker.pose.position.x = float(centroid[0])
        arrow_marker.pose.position.y = float(centroid[1])
        arrow_marker.pose.position.z = float(centroid[2])

        additional_rotation = R.from_euler('y', -90, degrees=True).as_quat()
        arrow_quat = (R.from_quat(quat) * R.from_quat(additional_rotation)).as_quat()
        arrow_marker.pose.orientation.x = arrow_quat[0]
        arrow_marker.pose.orientation.y = arrow_quat[1]
        arrow_marker.pose.orientation.z = arrow_quat[2]
        arrow_marker.pose.orientation.w = arrow_quat[3]
        arrow_marker.scale.x = 1.0
        arrow_marker.scale.y = 0.05
        arrow_marker.scale.z = 0.05
        arrow_marker.color.r = color[0]
        arrow_marker.color.g = color[1]
        arrow_marker.color.b = color[2]
        arrow_marker.color.a = 0.8
        markers.append(arrow_marker)

        return markers
