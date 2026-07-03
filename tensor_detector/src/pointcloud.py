#!/usr/bin/env python3
"""Point Clouds: back-project feature pixels to 3D, filter outliers,
accumulate, and drain into a PointCloud2 message with per-point RGB."""
import struct
from enum import Enum

import cv2
import numpy as np

from sensor_msgs.msg import PointCloud2, PointField

import geometry
from colors import COLOR_MAP

MAX_POINTS = 10000
DEFAULT_COLOR = (1.0, 1.0, 1.0)


class CloudColorMode(Enum):
    CLASS   = "class"    # Flat color from COLOR_MAP keyed by class name
    PIXEL   = "pixel"    # Sample the RGB image at each feature pixel


def _pack_rgb(r, g, b):
    """Pack three 0-1 floats into the uint32 that RViz expects for rgb fields."""
    ri = int(np.clip(r * 255, 0, 255))
    gi = int(np.clip(g * 255, 0, 255))
    bi = int(np.clip(b * 255, 0, 255))
    packed = (ri << 16) | (gi << 8) | bi
    return struct.unpack('f', struct.pack('I', packed))[0]


class PointCloudBuilder:
    def __init__(self, min_points, logger, color_mode=CloudColorMode.CLASS,
                 depth_mad_scale=3.0, depth_min_spread=0.05):
        self.min_points = min_points
        self.log = logger
        self.color_mode = color_mode
        # Robust depth-gate tuning. depth_mad_scale is how many robust stddevs
        # (MAD-derived) from the median depth a point may sit before it's culled;
        # <= 0 disables the gate. depth_min_spread (meters) floors the spread so a
        # near-planar patch (MAD ~ 0) isn't pruned down to nothing.
        self.depth_mad_scale = depth_mad_scale
        self.depth_min_spread = depth_min_spread
        # Each entry is (x, y, z, rgb_packed_float)
        self.accumulated_points = []

    def extract(self, frame, feature_points, mask, class_name=None):
        """Back-project masked feature pixels to 3D, filter, accumulate, return.

        Color per point is determined by self.color_mode:
          CLASS  – flat color from COLOR_MAP for class_name
          PIXEL  – sampled from frame.image at the source pixel (bgr8 -> rgb)

        Returns the filtered Nx3 array, or None if too few points survive.
        """
        class_color = COLOR_MAP.get(class_name, DEFAULT_COLOR)
        class_rgb_packed = _pack_rgb(*class_color)

        # Build (x, y, z, rgb) together so outlier removal keeps them in sync
        xyzrgb = []
        for x, y in feature_points:
            xi = int(x)
            yi = int(y)
            if yi >= frame.depth.shape[0] or xi >= frame.depth.shape[1]:
                continue
            if mask[yi, xi] != 255:
                continue
            z = frame.depth[yi, xi]
            if np.isnan(z) or np.isinf(z) or z == 0:
                continue

            pt = geometry.pixel_to_3d(xi, yi, z, frame.fx, frame.fy, frame.cx, frame.cy)

            if self.color_mode == CloudColorMode.PIXEL:
                b, g, r = frame.image[yi, xi]  # frame.image is bgr8
                rgb = _pack_rgb(r / 255.0, g / 255.0, b / 255.0)
            else:
                rgb = class_rgb_packed

            xyzrgb.append((*pt, rgb))

        points_3d = np.array([p[:3] for p in xyzrgb]) if xyzrgb else np.array([])

        if getattr(frame, "want_overlay_points", True):
            self._overlay(frame, points_3d, class_color)

        if len(points_3d) == 0:
            return None

        # Outlier removal operates on xyz; apply the same index mask to rgb
        rgb_arr = np.array([p[3] for p in xyzrgb])

        # Drop any NaN/infinite points so they don't poison the distance math
        finite = np.isfinite(points_3d).all(axis=1)
        points_3d = points_3d[finite]
        rgb_arr = rgb_arr[finite]

        if len(points_3d) == 0:
            return None

        # Depth gate first: edge pixels that straddle the object boundary read
        # the background depth and land far behind the object, poisoning the
        # plane fit and the radius/statistical passes below.
        points_3d, rgb_arr = self._depth_filter(points_3d, rgb_arr)
        if len(points_3d) == 0:
            return None

        indices = self._outlier_indices(points_3d)
        points_3d = points_3d[indices]
        rgb_arr   = rgb_arr[indices]

        # Only accumulate for the viz cloud when someone actually wants it
        want_cloud = getattr(frame, "want_cloud", True)
        if want_cloud and len(points_3d) > 0:
            self.accumulated_points.extend(
                (p[0], p[1], p[2], float(c)) for p, c in zip(points_3d, rgb_arr))
            if len(self.accumulated_points) > MAX_POINTS:
                self.accumulated_points = self.accumulated_points[-MAX_POINTS:]

        if len(points_3d) < self.min_points:
            return None

        return points_3d

    def take_cloud(self, frame_id, stamp):
        """Build and clear the accumulated cloud as PointCloud2 (None if empty)."""
        if not self.accumulated_points:
            return None

        fields = [
            PointField(name='x',   offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y',   offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z',   offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        point_step = 16  # 4 floats x 4 bytes

        arr = np.asarray(self.accumulated_points, dtype=np.float32)
        data = arr.tobytes()

        cloud = PointCloud2()
        cloud.header.frame_id = frame_id
        cloud.header.stamp = stamp
        cloud.height = 1
        cloud.width = len(self.accumulated_points)
        cloud.fields = fields
        cloud.is_bigendian = False
        cloud.point_step = point_step
        cloud.row_step = point_step * cloud.width
        cloud.data = data
        cloud.is_dense = True

        self.accumulated_points.clear()
        return cloud

    # --- internals ---

    def _depth_filter(self, points_3d, rgb_arr):
        """Cull points whose depth (z) is a robust outlier from the bulk.

        Uses a median/MAD gate so it adapts to the object's distance and scale
        rather than relying on a fixed depth window. Two-sided, so a stray point
        in front of the object is dropped too, though the far-behind background
        bleed at detection edges is the case this exists for.
        """
        if self.depth_mad_scale <= 0 or len(points_3d) < 3:
            return points_3d, rgb_arr

        z = points_3d[:, 2]
        med = np.median(z)
        mad = np.median(np.abs(z - med))
        # 1.4826 rescales MAD to a Gaussian-equivalent stddev; floor it so a flat,
        # head-on patch (mad ~ 0) doesn't collapse the gate onto the median.
        spread = max(mad * 1.4826, self.depth_min_spread)
        keep = np.abs(z - med) <= self.depth_mad_scale * spread

        dropped = len(z) - int(np.count_nonzero(keep))
        if dropped:
            self.log.debug(
                f"depth filter dropped {dropped}/{len(z)} points "
                f"(median {med:.2f}m, spread {spread:.2f}m)")
        return points_3d[keep], rgb_arr[keep]

    def _outlier_indices(self, points_3d):
        """Return the surviving index array after both outlier passes."""
        n = len(points_3d)
        if n == 0:
            return np.array([], dtype=int)

        # radius pass
        k = min(10, max(1, int(n * 0.8)))
        radius_idx = np.where(geometry.radius_outlier_mask(points_3d, min_neighbors=k))[0]
        if len(radius_idx) == 0:
            return radius_idx

        # statistical pass on the survivors, then map back through radius_idx
        survivors = points_3d[radius_idx]
        k2 = min(10, max(1, int(len(survivors) * 0.8)))
        stat_mask = geometry.statistical_outlier_mask(survivors, k=k2)
        return radius_idx[stat_mask]
    
    def _overlay(self, frame, points, color):
        if len(points) == 0:
            return
        bgr = (int(color[2] * 255), int(color[1] * 255), int(color[0] * 255))
        for point in points:
            try:
                if point[2] <= 0 or np.isnan(point[2]) or np.isinf(point[2]):
                    continue
                if np.isnan(point[0]) or np.isinf(point[0]) or np.isnan(point[1]) or np.isinf(point[1]):
                    continue
                x2d = int(point[0] * frame.fx / point[2] + frame.cx)
                y2d = int(point[1] * frame.fy / point[2] + frame.cy)
                if 0 <= x2d < frame.image.shape[1] and 0 <= y2d < frame.image.shape[0]:
                    cv2.circle(frame.image, (x2d, y2d), radius=3, color=bgr, thickness=-1)
            except (ZeroDivisionError, OverflowError, ValueError):
                continue
            except Exception as e:
                self.log.warning(f"Unexpected error in overlay_points_on_image: {e}")
                continue