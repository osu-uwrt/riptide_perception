#!/usr/bin/env python3

# Course map estimator for riptide_mapping2. No ROS in here so it can be tested on its own.
#
# The prior map (config init_data) is a dead-reckoned layout of the course, and the odometry frame drifts away from it.
# Every map-parented object is modeled as
#     position = prior + offset + deviation
# offset:    where the course sits in the odometry frame. Shared by every map object, random-walks with distance traveled.
# deviation: how far the dead-reckoned layout is off for that one object. Constant, zero mean.
# Both live in one Kalman filter, so a sighting of any map object refines the offset and through it every other object,
# and a sighting from long ago (more drift since) counts for less than a fresh one.
#
# Objects parented to another mapping frame (holes on the torpedo board, gate halves, ...) keep a 3-state filter in that
# frame. Yaw is a scalar filter per object; the FOG holds heading so the offset has no yaw.

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

MAP_FRAME = "map"


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


@dataclass
class EstimatorSettings:
    # Where the course sits in odometry at the start (std, m)
    offset_std_xy: float = 1.0
    offset_std_z: float = 0.3
    # Dead-reckoning drift of the offset: variance added per meter traveled and per second
    drift_var_per_m_xy: float = 0.002
    drift_var_per_m_z: float = 0.0002
    drift_var_per_s_xy: float = 1e-5
    drift_var_per_s_z: float = 1e-6
    # How far a map object can be from where the dead-reckoned layout puts it (std, m)
    layout_std_xy: float = 0.25
    layout_std_z: float = 0.15
    # Prior std of an object parented to another mapping frame, in that frame (m)
    child_std_xy: float = 0.5
    child_std_z: float = 0.3
    # Prior std of every object's yaw (deg)
    yaw_std_deg: float = 30.0
    # Once an object has gate_min_updates accepted detections, detections further than this squared Mahalanobis
    # distance are rejected (3 dof / 1 dof at 99.9 %). Catches wrong-class detections of a nearby lookalike.
    gate_chi2: float = 16.27
    yaw_gate_chi2: float = 10.83
    gate_min_updates: int = 10
    # This many position rejections in a row means the estimate locked onto the wrong thing: reacquire it
    gate_max_rejects: int = 30


@dataclass
class NoiseSettings:
    base_std: float = 0.05            # m, any detection
    lateral_std_per_m: float = 0.02   # m per m of range, across the camera ray
    range_std_per_m: float = 0.05     # m per m of range, along the ray (stereo depth)
    range_std_per_m2: float = 0.01    # m per m^2 of range, along the ray
    timing_std: float = 0.03          # s, error between the detection stamp and the pose used for it
    yaw_std_deg: float = 10.0         # deg, detected yaw
    correlation_time: float = 0.3     # s, detections closer together than this share most of their error
    max_inflation: float = 30.0       # cap on the correlation inflation (several detections in one frame)


@dataclass
class ObjectSpec:
    parent: str
    position: np.ndarray    # prior, in the parent frame
    yaw: float              # prior, rad
    lock_yaw: bool = False  # keep the prior yaw, ignore detected yaw


@dataclass
class Track:
    updates: int = 0      # accepted detections since the last reset
    rejects: int = 0      # position rejections in a row
    last_stamp: Optional[float] = None


def detection_covariance(position_camera, rotation_parent_camera, angular_speed: float, linear_speed: float,
                         noise: NoiseSettings) -> np.ndarray:
    # Covariance (parent frame) of one detection at position_camera. Depth error grows along the camera ray, bearing error
    # across it. A stamp error turns vehicle rotation into a sideways smear and vehicle translation into a shift
    p = np.asarray(position_camera, dtype=np.float64)
    r = float(np.linalg.norm(p))

    lateral_var = (noise.base_std + noise.lateral_std_per_m * r) ** 2 + (angular_speed * noise.timing_std * r) ** 2
    along_var = (noise.base_std + noise.range_std_per_m * r + noise.range_std_per_m2 * r * r) ** 2

    cov = lateral_var * np.eye(3)
    if r > 1e-6:
        ray = p / r
        cov += (along_var - lateral_var) * np.outer(ray, ray)
    cov += (linear_speed * noise.timing_std) ** 2 * np.eye(3)

    rot = np.asarray(rotation_parent_camera, dtype=np.float64)
    return rot @ cov @ rot.T


def slew_vector(current: np.ndarray, target: np.ndarray, max_speed: float, dt: float) -> np.ndarray:
    # Move current toward target at no more than max_speed (<= 0: jump)
    if max_speed <= 0:
        return np.array(target, dtype=np.float64)
    delta = np.asarray(target, dtype=np.float64) - current
    dist = float(np.linalg.norm(delta))
    step = max_speed * max(dt, 0.0)
    if dist <= step:
        return np.array(target, dtype=np.float64)
    return current + delta * (step / dist)


def slew_angle(current: float, target: float, max_rate: float, dt: float) -> float:
    if max_rate <= 0:
        return target
    delta = wrap_angle(target - current)
    step = max_rate * max(dt, 0.0)
    if abs(delta) <= step:
        return target
    return wrap_angle(current + math.copysign(step, delta))


class MapEstimator:

    def __init__(self, specs: Dict[str, ObjectSpec], settings: Optional[EstimatorSettings] = None,
                 noise: Optional[NoiseSettings] = None):
        self.settings = settings or EstimatorSettings()
        self.noise = noise or NoiseSettings()
        self.specs = {name: ObjectSpec(spec.parent, np.array(spec.position, dtype=np.float64), float(spec.yaw),
                                       bool(spec.lock_yaw)) for name, spec in specs.items()}

        # Joint state: [offset, deviation of each map object]
        self.map_names = [name for name, spec in self.specs.items() if spec.parent == MAP_FRAME]
        self._slot = {name: 3 + 3 * i for i, name in enumerate(self.map_names)}
        size = 3 + 3 * len(self.map_names)
        self.x = np.zeros(size)
        self.P = np.zeros((size, size))
        self.P[0:3, 0:3] = self._offset_prior()
        for name in self.map_names:
            s = self._slot[name]
            self.P[s:s + 3, s:s + 3] = self._layout_prior()

        self.child_mean: Dict[str, np.ndarray] = {}
        self.child_P: Dict[str, np.ndarray] = {}
        for name, spec in self.specs.items():
            if spec.parent != MAP_FRAME:
                self.child_mean[name] = spec.position.copy()
                self.child_P[name] = self._child_prior()

        self.yaw_mean = {name: wrap_angle(spec.yaw) for name, spec in self.specs.items()}
        self.yaw_var = {name: self._yaw_prior() for name in self.specs}
        self.tracks = {name: Track() for name in self.specs}

    # ----- priors -----

    def _offset_prior(self) -> np.ndarray:
        s = self.settings
        return np.diag([s.offset_std_xy ** 2, s.offset_std_xy ** 2, s.offset_std_z ** 2])

    def _layout_prior(self) -> np.ndarray:
        s = self.settings
        return np.diag([s.layout_std_xy ** 2, s.layout_std_xy ** 2, s.layout_std_z ** 2])

    def _child_prior(self) -> np.ndarray:
        s = self.settings
        return np.diag([s.child_std_xy ** 2, s.child_std_xy ** 2, s.child_std_z ** 2])

    def _yaw_prior(self) -> float:
        return math.radians(self.settings.yaw_std_deg) ** 2

    # ----- queries -----

    def is_map_object(self, name: str) -> bool:
        return name in self._slot

    def position(self, name: str) -> Tuple[np.ndarray, np.ndarray]:
        # Mean and covariance in the parent frame (map objects: in map, offset included)
        if name in self._slot:
            s = self._slot[name]
            mean = self.specs[name].position + self.x[0:3] + self.x[s:s + 3]
            P = self.P
            cov = P[0:3, 0:3] + P[s:s + 3, s:s + 3] + P[0:3, s:s + 3] + P[s:s + 3, 0:3]
            return mean, 0.5 * (cov + cov.T)
        return self.child_mean[name].copy(), self.child_P[name].copy()

    def frame_translation(self, name: str) -> np.ndarray:
        # Translation of the object's frame from its TF parent (map objects hang off the offset frame)
        if name in self._slot:
            s = self._slot[name]
            return self.specs[name].position + self.x[s:s + 3]
        return self.child_mean[name].copy()

    def offset(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.x[0:3].copy(), self.P[0:3, 0:3].copy()

    def yaw(self, name: str) -> Tuple[float, float]:
        if self.specs[name].lock_yaw:
            return self.yaw_mean[name], 0.0
        return self.yaw_mean[name], self.yaw_var[name]

    # ----- filter -----

    def predict(self, distance: float, dt: float):
        # Dead-reckoning drift since the last call
        s = self.settings
        distance = max(float(distance), 0.0)
        dt = max(float(dt), 0.0)
        xy = s.drift_var_per_m_xy * distance + s.drift_var_per_s_xy * dt
        z = s.drift_var_per_m_z * distance + s.drift_var_per_s_z * dt
        self.P[0:3, 0:3] += np.diag([xy, xy, z])

    def inflation(self, name: str, stamp: float) -> float:
        # Back-to-back detections of one object share most of their error, so each counts for less
        last = self.tracks[name].last_stamp
        if last is None:
            return 1.0
        dt = max(stamp - last, 1e-3)
        return min(max(1.0, self.noise.correlation_time / dt), self.noise.max_inflation)

    def observe(self, name: str, stamp: float, position=None, position_cov=None, yaw: Optional[float] = None,
                yaw_var: Optional[float] = None) -> Tuple[bool, str]:
        # Fuse one detection (position in the parent frame). Returns (accepted, reason)
        track = self.tracks[name]
        inflation = self.inflation(name, stamp)

        if position is not None:
            z = np.asarray(position, dtype=np.float64)
            R = np.asarray(position_cov, dtype=np.float64) * inflation
            if not (np.all(np.isfinite(z)) and np.all(np.isfinite(R))):
                return False, "non-finite detection"
            accepted, reason = self._update_position(name, z, R)
            if not accepted:
                return False, reason

        if yaw is not None and yaw_var is not None and not self.specs[name].lock_yaw and math.isfinite(yaw):
            self._update_yaw(name, float(yaw), float(yaw_var) * inflation)

        track.updates += 1
        track.rejects = 0
        track.last_stamp = stamp
        return True, ""

    def _update_position(self, name: str, z: np.ndarray, R: np.ndarray) -> Tuple[bool, str]:
        s = self.settings
        track = self.tracks[name]
        mean, cov = self.position(name)
        nu = z - mean
        S = cov + R
        d2 = float(nu @ np.linalg.solve(S, nu))

        if track.updates >= s.gate_min_updates and d2 > s.gate_chi2:
            track.rejects += 1
            if track.rejects >= s.gate_max_rejects:
                self.reset(name)
                return False, f"{s.gate_max_rejects} rejections in a row (last {math.sqrt(d2):.1f} sigma), reacquiring"
            return False, f"{math.sqrt(d2):.1f} sigma from the estimate"

        if name in self._slot:
            slot = self._slot[name]
            H = np.zeros((3, self.x.size))
            H[:, 0:3] = np.eye(3)
            H[:, slot:slot + 3] = np.eye(3)
            K = self.P @ H.T @ np.linalg.inv(S)
            self.x = self.x + K @ nu
            IKH = np.eye(self.x.size) - K @ H
            self.P = IKH @ self.P @ IKH.T + K @ R @ K.T
            self.P = 0.5 * (self.P + self.P.T)
        else:
            P = self.child_P[name]
            K = P @ np.linalg.inv(S)
            self.child_mean[name] = self.child_mean[name] + K @ nu
            IK = np.eye(3) - K
            P = IK @ P @ IK.T + K @ R @ K.T
            self.child_P[name] = 0.5 * (P + P.T)
        return True, ""

    def _update_yaw(self, name: str, yaw: float, var: float):
        nu = wrap_angle(yaw - self.yaw_mean[name])
        S = self.yaw_var[name] + var
        if self.tracks[name].updates >= self.settings.gate_min_updates and nu * nu / S > self.settings.yaw_gate_chi2:
            return
        K = self.yaw_var[name] / S
        self.yaw_mean[name] = wrap_angle(self.yaw_mean[name] + K * nu)
        self.yaw_var[name] = (1.0 - K) * self.yaw_var[name]

    # ----- edits from services -----

    def reset(self, name: str):
        # Forget what detections taught this object (back to its prior covariance), keep where it is
        if name in self._slot:
            s = self._slot[name]
            self.P[s:s + 3, :] = 0.0
            self.P[:, s:s + 3] = 0.0
            self.P[s:s + 3, s:s + 3] = self._layout_prior()
        else:
            self.child_P[name] = self._child_prior()
        self.yaw_var[name] = self._yaw_prior()
        self.tracks[name] = Track()

    def seed(self, name: str, position, yaw: Optional[float] = None):
        # Put the object at position (parent frame) with its prior covariance
        self.reset(name)
        position = np.asarray(position, dtype=np.float64)
        if name in self._slot:
            s = self._slot[name]
            self.x[s:s + 3] = position - self.specs[name].position - self.x[0:3]
        else:
            self.child_mean[name] = position.copy()
        if yaw is not None:
            self.yaw_mean[name] = wrap_angle(yaw)

    def shift(self, name: str, delta):
        # Translate the object in its parent frame, covariance unchanged
        delta = np.asarray(delta, dtype=np.float64)
        if name in self._slot:
            s = self._slot[name]
            self.x[s:s + 3] += delta
        else:
            self.child_mean[name] = self.child_mean[name] + delta

    def rotate_position(self, name: str, dyaw: float):
        # Rotate the object's position about its parent frame's z axis (for when the parent frame yaws)
        c, s_ = math.cos(dyaw), math.sin(dyaw)
        rot = np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])
        if name in self._slot:
            mean, _ = self.position(name)
            self.shift(name, rot @ mean - mean)
        else:
            self.child_mean[name] = rot @ self.child_mean[name]
            self.child_P[name] = rot @ self.child_P[name] @ rot.T

    def add_yaw(self, name: str, dyaw: float):
        self.yaw_mean[name] = wrap_angle(self.yaw_mean[name] + dyaw)

    def set_prior(self, name: str, position, yaw: float):
        # The prior map changed under us: move the object with it and forget its detections
        self.specs[name].position = np.asarray(position, dtype=np.float64)
        self.specs[name].yaw = float(yaw)
        if name in self._slot:
            self.x[self._slot[name]:self._slot[name] + 3] = 0.0
        else:
            self.child_mean[name] = self.specs[name].position.copy()
        self.yaw_mean[name] = wrap_angle(yaw)
        self.reset(name)
