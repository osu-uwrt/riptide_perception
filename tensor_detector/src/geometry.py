#!/usr/bin/env python3
"""Pure geometry helpers for yolo_orientation."""
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree 


# The camera-frame normal a flat, front-facing surface would have
# Don't touch this unless you know what you're doing
DEFAULT_NORMAL = np.array([0.0, 0.0, 1.0])


def pixel_to_3d(u, v, z, fx, fy, cx, cy):
    """Back-project a pixel (u, v) at depth z into a 3D camera-frame point."""
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return [x, y, z]


def ray_plane_intersection(u, v, K, normal, p0):
    """Intersect the camera ray through pixel (u, v) with the plane (normal, p0).

    Returns (point_3d, t). point_3d is None when the ray is parallel to the
    plane (denominator == 0). Scaling of the ray direction does not affect the
    intersection point, so the result is independent of normalization.
    """
    d = np.linalg.inv(K) @ np.array([u, v, 1.0])
    d = d / np.linalg.norm(d)
    denominator = np.dot(normal, d)
    if denominator == 0:
        return None, None
    t = np.dot(normal, p0) / denominator
    return t * d, t


def fit_plane(points_3d):
    """Least-squares plane fit via SVD.

    Returns (normal, d, centroid); (None, None, None) when there are no points.
    """
    if len(points_3d) == 0:
        return None, None, None
    centroid = np.mean(points_3d, axis=0)
    _, _, vh = np.linalg.svd(points_3d - centroid, full_matrices=False)
    normal = vh[-1]
    normal = normal / np.linalg.norm(normal)
    d = -np.dot(normal, centroid)
    return normal, d, centroid


def rotation_from_normal(normal, default_normal=DEFAULT_NORMAL):
    """Rotation that maps default_normal onto normal (scipy Rotation)."""
    axis = np.cross(default_normal, normal)
    axis_length = np.linalg.norm(axis)
    if axis_length == 0:
        # Normal is parallel/anti-parallel, rotate 180 deg about an arbitrary axis (goofy ahh rotation)
        axis = np.array([1, 0, 0])
        angle = np.pi
    else:
        axis = axis / axis_length
        angle = np.arccos(np.dot(default_normal, normal))
    return R.from_rotvec(axis * angle)


def normal_to_quaternion(normal, default_normal=DEFAULT_NORMAL):
    """(quat, euler_xyz_degrees) for a plane with the given normal.

    euler is None when no rotation is needed (normal == default_normal).
    """
    if np.allclose(normal, default_normal):
        return [0.0, 0.0, 0.0, 1.0], None
    rotation = rotation_from_normal(normal, default_normal)
    return rotation.as_quat(), rotation.as_euler('xyz', degrees=True)


def inplane_basis(normal):
    """Two orthonormal vectors (u, v) spanning the plane with the given normal.

    The absolute orientation/sign of u and v is arbitrary; callers that need a
    consistent 'up' must resolve it themselves. Stable for any normal direction.
    """
    normal = np.asarray(normal, dtype=float)
    normal = normal / np.linalg.norm(normal)
    # Reference axis least aligned with the normal, to avoid a degenerate cross.
    ref = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, ref)
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)
    v = v / np.linalg.norm(v)
    return u, v


def is_inside_bbox(inner_bbox, outer_bbox):
    inner_x_min, inner_y_min, inner_x_max, inner_y_max = inner_bbox
    outer_x_min, outer_y_min, outer_x_max, outer_y_max = outer_bbox
    return (inner_x_min >= outer_x_min and inner_x_max <= outer_x_max and
            inner_y_min >= outer_y_min and inner_y_max <= outer_y_max)


def radius_outlier_mask(points_3d, radius=1.0, min_neighbors=10):
    """Boolean mask: True for points with more than min_neighbors others within radius."""
    n = len(points_3d)
    if n == 0:
        return np.zeros(0, dtype=bool)
    tree = cKDTree(points_3d)
    # count includes the point itself, matching the original > min_neighbors semantics
    counts = tree.query_ball_point(points_3d, r=radius, return_length=True)
    return counts > min_neighbors


def statistical_outlier_mask(points_3d, k=10, std_ratio=1.0):
    """Boolean mask: True for inliers, based on mean distance to k nearest neighbors."""
    n = len(points_3d)
    if n == 0:
        return np.zeros(0, dtype=bool)
    k_eff = min(k, n - 1)
    if k_eff < 1:
        return np.ones(n, dtype=bool)
    tree = cKDTree(points_3d)
    # k_eff + 1 because the nearest neighbor is the point itself (distance 0)
    dists, _ = tree.query(points_3d, k=k_eff + 1)
    mean_distances = dists[:, 1:].mean(axis=1)  # drop the self column
    threshold = mean_distances.mean() + std_ratio * mean_distances.std()
    return mean_distances < threshold