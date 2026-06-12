#!/usr/bin/env python3
"""Point Clouds: back-project feature pixels to 3D, filter outliers,
accumulate, and drain into a PointCloud message"""
import cv2
import numpy as np

from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import Point32

import geometry

MAX_POINTS = 10000


class PointCloudBuilder:
    def __init__(self, min_points, logger):
        self.min_points = min_points
        self.log = logger
        self.accumulated_points = []

    def extract(self, frame, feature_points, mask):
        """Back-project masked feature pixels to 3D, filter, accumulate, return.

        Returns the filtered Nx3 array, or None if too few points survive.
        """
        points_3d = []
        for x, y in feature_points:
            xi = int(x)
            yi = int(y)
            if yi >= frame.depth.shape[0] or xi >= frame.depth.shape[1]:
                continue
            if mask[yi, xi] != 255:
                continue
            z = frame.depth[yi, xi]
            if np.isnan(z) or z == 0:
                continue
            points_3d.append(geometry.pixel_to_3d(xi, yi, z,
                                                  frame.fx, frame.fy, frame.cx, frame.cy))

        self._overlay(frame, points_3d)

        points_3d = np.array(points_3d)
        points_3d = geometry.radius_outlier_removal(
            points_3d, min_neighbors=min(10, int(len(points_3d) * 0.8)))
        points_3d = geometry.statistical_outlier_removal(
            points_3d, k=min(10, int(len(points_3d) * 0.8)))

        if points_3d is not None:
            self.accumulated_points.extend(points_3d)
            if len(self.accumulated_points) > MAX_POINTS:
                self.accumulated_points = self.accumulated_points[-MAX_POINTS:]
            if len(points_3d) < self.min_points:
                return None

        return points_3d

    def take_cloud(self, frame_id, stamp):
        """Build and clear the accumulated cloud (None if empty)."""
        if not self.accumulated_points:
            return None
        cloud = PointCloud()
        cloud.header.frame_id = frame_id
        cloud.header.stamp = stamp
        for point in self.accumulated_points:
            cloud.points.append(Point32(x=float(point[0]),
                                        y=float(point[1]),
                                        z=float(point[2])))
        self.accumulated_points.clear()
        return cloud

    def _overlay(self, frame, points):
        if len(points) == 0:
            return
        for point in points:
            try:
                if len(point) < 3:
                    continue
                if point[2] <= 0 or np.isnan(point[2]) or np.isinf(point[2]):
                    continue
                if (np.isnan(point[0]) or np.isinf(point[0])
                        or np.isnan(point[1]) or np.isinf(point[1])):
                    continue
                x2d = int(point[0] * frame.fx / point[2] + frame.cx)
                y2d = int(point[1] * frame.fy / point[2] + frame.cy)
                if 0 <= x2d < frame.image.shape[1] and 0 <= y2d < frame.image.shape[0]:
                    cv2.circle(frame.image, (x2d, y2d), radius=3, color=(0, 255, 0), thickness=-1)
            except (ZeroDivisionError, OverflowError, ValueError):
                continue
            except Exception as e:
                self.log.warning(f"Unexpected error in overlay_points_on_image: {e}")
                continue
