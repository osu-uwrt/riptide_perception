#!/usr/bin/env python3

# Course map estimator for riptide_mapping2. No ROS in here so it can be tested on its own.
#
# The prior map (config init_data) is a dead-reckoned layout of the course, and the odometry frame drifts away from it.
# Every map-parented object is modeled as
#     position = R(course yaw) * prior + offset + deviation
# offset:      where the course sits in the odometry frame. Random-walks with distance traveled (dead-reckoning drift).
# deviation:   how far the dead-reckoned layout is off for that one object. Heavy-tailed: mostly small, sometimes meters.
# course yaw:  start-heading misalignment between the layout and odometry (small, the AprilTag calibration sets heading).
#
# Each map object is tracked from its detections (a 3-state filter in the odometry frame). Once a track has enough
# detections it becomes a fix: where that object was, at the odometry distance it was last seen at. Retargeting the
# object later freezes the fix and starts a new one, so revisits measure drift. A small robust least-squares solve over
# all fixes then gives the offset history, every seen object's deviation and the course yaw:
# - one sighting can't tell "this prior is off" from "the vehicle drifted", so it is split by their uncertainties;
# - a prior that disagrees with the other fixes is released (Cauchy weight) instead of dragging the map;
# - fixes that agree on a shift are drift, and the offset follows them.
# Unseen objects are predicted at R * prior + offset now.
#
# Objects parented to another mapping frame (holes on the torpedo board, gate halves, ...) keep a 3-state filter in that
# frame. Yaw is a scalar filter per object.

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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
    # How far a map object can be from where the dead-reckoned layout puts it (std, m)...
    layout_std_xy: float = 0.25
    layout_std_z: float = 0.15
    # ...with a heavy tail: a deviation this many stds out counts half as much as a Gaussian would (Cauchy scale),
    # so a badly placed prior is released instead of dragging the map. <= 0: Gaussian
    layout_robust_scale: float = 2.0
    # Start-heading misalignment between the layout and odometry (std, deg; 0: not estimated)
    course_yaw_std_deg: float = 2.0
    # Fixes of one object closer together than this (m traveled) are merged into one
    fix_merge_distance: float = 1.0
    # Prior std of an object parented to another mapping frame, in that frame (m)
    child_std_xy: float = 0.5
    child_std_z: float = 0.3
    # Prior std of every object's yaw (deg)
    yaw_std_deg: float = 30.0
    # Once an object has gate_min_updates accepted detections, detections further than this squared Mahalanobis
    # distance are rejected (3 dof / 1 dof at 99.9 %). Catches wrong-class detections of a nearby lookalike.
    gate_chi2: float = 16.27
    yaw_gate_chi2: float = 10.83
    gate_min_updates: int = 10   # also the detections a map object's track needs to become a fix
    # A target is confirmed (its real covariance published) after this many detections while targeted...
    confirm_detections: int = 10
    # ...or, when it was already known well, after this many that land where it was (widened by the drift since it was
    # last seen); one that lands elsewhere means it moved, and it is found again from scratch
    verify_detections: int = 3
    # This many position rejections in a row means the estimate locked onto the wrong thing: reacquire it
    gate_max_rejects: int = 30
    # An object's first detections since a reset are held back until acquire_detections of them agree (within
    # acquire_radius_m of each other, or 3 sigma of their noise, over the last acquire_window_s); the track then starts
    # from those and the rest are dropped, so a flicker or a stray false detection never moves the estimate or biases
    # it (the old mapping's trimmed buffer did this). A cluster further than acquire_prior_chi2 (squared Mahalanobis)
    # from where the map has the object needs acquire_far_detections over at least acquire_far_s instead: a lookalike
    # off to the side has to keep agreeing for longer before it takes over the prior. acquire_detections <= 1: off
    acquire_detections: int = 4
    acquire_radius_m: float = 0.3
    acquire_window_s: float = 4.0
    acquire_prior_chi2: float = 16.27
    acquire_far_detections: int = 12
    acquire_far_s: float = 2.0


# Defaults from what the trees were tuned against: the torpedo holes' 0.0005 passed with the old mapping only if
# their detections scattered under ~2 cm at 1.5 m. Measure the real scatter to tune these.
@dataclass
class NoiseSettings:
    base_std: float = 0.01            # m, any detection
    lateral_std_per_m: float = 0.003  # m per m of range, across the camera ray
    range_std_per_m: float = 0.008    # m per m of range, along the ray (stereo depth)
    range_std_per_m2: float = 0.003   # m per m^2 of range, along the ray
    timing_std: float = 0.03          # s, error between the detection stamp and the pose used for it
    yaw_std_deg: float = 3.0          # deg, detected yaw
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
    established: bool = False  # confirm_detections since the last full reset, targeted or not: the estimate is good
    confirmed: bool = False    # usable by a target check: confirmed while targeted, until the next reset
    verifying: int = -1        # targeted again while established: detections that agreed so far (-1: not verifying)


@dataclass
class Fix:
    # Where a map object was found (odometry frame), and the odometry distance and time it was last seen at
    name: str
    mean: np.ndarray
    cov: np.ndarray
    distance: float
    time: float


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

        # Odometry path length and time so far (predict()), the clocks fixes and drift are measured on
        self.distance = 0.0
        self.time = 0.0

        # Tracks: map objects in the odometry frame, the rest in their parent frame
        self.track_mean: Dict[str, np.ndarray] = {}
        self.track_cov: Dict[str, np.ndarray] = {}
        for name, spec in self.specs.items():
            self.track_mean[name] = spec.position.copy()
            self.track_cov[name] = self._layout_prior() if self.is_map_object(name) else self._child_prior()
        self.acquiring = set()  # map objects whose track is their estimate (detected or seeded, not a fix yet)
        self.live: Dict[str, Tuple[float, float]] = {}  # map objects whose track is a fix: (distance, time) last seen
        self.seen: Dict[str, Tuple[float, float]] = {}  # (distance, time) of each map track's last detection
        self.frozen: List[Fix] = []
        self._solution = None
        self._structure = None  # fixes the last full (multi-start) solve ran on
        self._warm = None

        # Detections held back until enough agree: (stamp, position, cov, yaw, yaw_var, targeted)
        self.pending: Dict[str, List[tuple]] = {}

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

    def _drift(self, distance: float, dt: float) -> np.ndarray:
        s = self.settings
        distance, dt = max(distance, 0.0), max(dt, 0.0)
        xy = s.drift_var_per_m_xy * distance + s.drift_var_per_s_xy * dt
        z = s.drift_var_per_m_z * distance + s.drift_var_per_s_z * dt
        return np.diag([xy, xy, z])

    # ----- queries -----

    def is_map_object(self, name: str) -> bool:
        return self.specs[name].parent == MAP_FRAME

    def position(self, name: str) -> Tuple[np.ndarray, np.ndarray]:
        # Mean and covariance in the parent frame (map objects: in the odometry frame, offset included)
        if not self.is_map_object(name) or name in self.acquiring:
            return self.track_mean[name].copy(), self.track_cov[name].copy()
        return self._predicted(name)

    def frame_translation(self, name: str) -> np.ndarray:
        # Translation of the object's frame from its TF parent (map objects hang off the offset frame)
        if not self.is_map_object(name):
            return self.track_mean[name].copy()
        return self.position(name)[0] - self.offset()[0]

    def offset(self) -> Tuple[np.ndarray, np.ndarray]:
        # Where the course sits in the odometry frame now
        sol = self._solve()
        if sol is None:
            return np.zeros(3), self._offset_prior() + self._drift(self.distance, self.time)
        o = sol["o"]
        cov = sol["cov"][o:o + 3, o:o + 3] + self._drift(self.distance - sol["distance"], self.time - sol["time"])
        return sol["x"][o:o + 3].copy(), 0.5 * (cov + cov.T)

    def course_yaw(self) -> Tuple[float, float]:
        sol = self._solve()
        if sol is None:
            return 0.0, math.radians(self.settings.course_yaw_std_deg) ** 2
        return float(sol["x"][0]), float(sol["cov"][0, 0])

    def yaw(self, name: str) -> Tuple[float, float]:
        if self.specs[name].lock_yaw:
            return self.yaw_mean[name], 0.0
        return self.yaw_mean[name], self.yaw_var[name]

    def confirmed(self, name: str) -> bool:
        # Whether fresh detections while targeted have confirmed the object since its last reset
        return self.tracks[name].confirmed

    def released(self, name: str) -> bool:
        # Whether the fixes say this object's prior is off (its layout weight is mostly gone)
        sol = self._solve()
        return sol is not None and sol["weights"].get(name, 1.0) < 0.5

    def _predicted(self, name: str) -> Tuple[np.ndarray, np.ndarray]:
        # The map's estimate of a map object: R(course yaw) * prior + offset now + its deviation
        prior = self.specs[name].position
        sol = self._solve()
        if sol is None:
            cov = self._offset_prior() + self._drift(self.distance, self.time) + self._layout_prior()
            return prior.copy(), cov
        x, P = sol["x"], sol["cov"]
        rotated, g = _rotate(x[0], prior)
        o = sol["o"]
        idx = [0, o, o + 1, o + 2]
        J = np.zeros((3, 7 if name in sol["d"] else 4))
        J[:, 0] = g
        J[:, 1:4] = np.eye(3)
        mean = rotated + x[o:o + 3]
        if name in sol["d"]:
            d = sol["d"][name]
            idx += [d, d + 1, d + 2]
            J[:, 4:7] = np.eye(3)
            mean = mean + x[d:d + 3]
        cov = J @ P[np.ix_(idx, idx)] @ J.T + self._drift(self.distance - sol["distance"], self.time - sol["time"])
        if name not in sol["d"]:
            cov = cov + self._layout_prior()
        return mean, 0.5 * (cov + cov.T)

    # ----- filter -----

    def predict(self, distance: float, dt: float):
        # The vehicle traveled `distance` (odometry path length) over dt since the last call
        self.distance += max(float(distance), 0.0)
        self.time += max(float(dt), 0.0)

    def inflation(self, name: str, stamp: float) -> float:
        # Back-to-back detections of one object share most of their error, so each counts for less
        last = self.tracks[name].last_stamp
        if last is None:
            return 1.0
        dt = max(stamp - last, 1e-3)
        return min(max(1.0, self.noise.correlation_time / dt), self.noise.max_inflation)

    def observe(self, name: str, stamp: float, position=None, position_cov=None, yaw: Optional[float] = None,
                yaw_var: Optional[float] = None, targeted: bool = True, agreed: bool = False) -> Tuple[bool, str]:
        # Fuse one detection (position in the parent frame). Only a targeted object's detections confirm it; the
        # rest (background objects) still improve its estimate. agreed: already agreed with others (the binary
        # classifier clusters before it names a target), so it needn't wait for more. Returns (accepted, reason)
        inflation = self.inflation(name, stamp)
        moved = ""

        if position is not None:
            z = np.asarray(position, dtype=np.float64)
            R = np.asarray(position_cov, dtype=np.float64) * inflation
            if not (np.all(np.isfinite(z)) and np.all(np.isfinite(R))):
                return False, "non-finite detection"
            started = False  # the map track is started (or aged by the drift since it was last seen) once per detection
            if self.tracks[name].verifying >= 0:
                if self.is_map_object(name):
                    self._start_or_age_track(name)
                    started = True
                nu = z - self.track_mean[name]
                d2 = float(nu @ np.linalg.solve(self.track_cov[name] + R, nu))
                if d2 > self.settings.gate_chi2:
                    # Not where it was: find it again from scratch (this detection is the first of the new search)
                    moved = f"moved {np.linalg.norm(nu[0:2]) * 100:.0f} cm since it was last seen, finding it again"
                    self.reset(name)
                    started = False
            if not agreed and self._acquiring_by_agreement(name):
                accepted, reason = self._agree(name, stamp, z, np.asarray(position_cov, dtype=np.float64), yaw, yaw_var,
                                               targeted)
                return accepted, "; ".join(r for r in (moved, reason) if r)
            if self.is_map_object(name) and not started:
                self._start_or_age_track(name)
            accepted, reason = self._update_position(name, z, R)
            if not accepted:
                return False, reason
        track = self.tracks[name]

        if yaw is not None and yaw_var is not None and not self.specs[name].lock_yaw and math.isfinite(yaw):
            self._update_yaw(name, float(yaw), float(yaw_var) * inflation)

        track.updates += 1
        track.rejects = 0
        track.last_stamp = stamp
        if position is not None:
            if track.updates >= self.settings.confirm_detections:
                track.established = True
            if targeted and track.verifying >= 0:
                track.verifying += 1
                if track.verifying >= self.settings.verify_detections:
                    track.verifying = -1
                    track.confirmed = True
            if targeted and track.verifying < 0 and track.established:
                track.confirmed = True
        if position is not None:
            self.seen[name] = (self.distance, self.time)
        if position is not None and self.is_map_object(name):
            if name in self.live or track.updates >= self.settings.gate_min_updates:
                # Enough detections: the track is a fix the map is solved from
                self.acquiring.discard(name)
                self.live[name] = (self.distance, self.time)
                self._solution = None
        return True, moved

    def _acquiring_by_agreement(self, name: str) -> bool:
        # No detection accepted since the last full reset (and not checking a known estimate): wait for agreement
        track = self.tracks[name]
        return self.settings.acquire_detections > 1 and track.updates == 0 and track.verifying < 0

    def _acquisition_prior(self, name: str) -> Tuple[np.ndarray, np.ndarray]:
        # Where the map expects the object (its layout and course-yaw uncertainty included)
        if self.is_map_object(name) and name not in self.acquiring and name not in self.live:
            return self._predicted(name)
        return self.track_mean[name], self.track_cov[name]

    def _agree(self, name: str, stamp: float, z: np.ndarray, R: np.ndarray, yaw, yaw_var, targeted: bool) -> Tuple[bool, str]:
        # Hold the detection until enough agree; then start the track from the agreeing ones only
        s = self.settings
        held = [d for d in self.pending.get(name, []) if stamp - d[0] <= s.acquire_window_s]
        held.append((stamp, z, R, yaw, yaw_var, targeted))
        self.pending[name] = held

        def agree(a, b):
            gap = float(np.linalg.norm(a[1] - b[1]))
            sigma = math.sqrt(max(float(np.linalg.eigvalsh(a[2] + b[2])[-1]), 0.0))
            return gap <= max(s.acquire_radius_m, 3.0 * sigma)

        members = max(([b for b in held if agree(a, b)] for a in held), key=len)
        mean = np.mean([m[1] for m in members], axis=0)
        prior_mean, prior_cov = self._acquisition_prior(name)
        nu = mean - prior_mean
        spread = np.mean([m[2] for m in members], axis=0) / len(members)
        far = float(nu @ np.linalg.solve(prior_cov + spread, nu)) > s.acquire_prior_chi2
        needed = s.acquire_far_detections if far else s.acquire_detections
        span = max(m[0] for m in members) - min(m[0] for m in members)
        if len(members) < needed or (far and span < s.acquire_far_s):
            return False, (f"acquiring: {len(members)}/{needed} detections agree" +
                           (f" over {span:.1f}/{s.acquire_far_s:.1f} s, away from the map's estimate" if far else ""))

        del self.pending[name]
        for m in sorted(members, key=lambda m: m[0]):
            self.observe(name, m[0], m[1], m[2], m[3], m[4], m[5], agreed=True)
        dropped = len(held) - len(members)
        reason = f"{len(members)} detections agree" + (f", {dropped} that didn't dropped" if dropped else "")
        if far:
            reason += f", {np.linalg.norm(nu[0:2]):.1f} m from where the map had it: taking them over the prior"
        return True, reason

    def _start_or_age_track(self, name: str):
        if name in self.acquiring or name in self.live:
            # Odometry drifted since its last detection, so older detections count for less
            distance, time = self.seen.get(name, (self.distance, self.time))
            self.track_cov[name] = self.track_cov[name] + self._drift(self.distance - distance, self.time - time)
            return
        # First detection since a reset: start from where the map has it. Verifying: as uncertain as the map says (its
        # drift since included); else uncertain enough to need fresh detections
        mean, cov = self._predicted(name)
        self.track_mean[name] = mean
        self.track_cov[name] = cov if self.tracks[name].verifying >= 0 else cov + self._layout_prior()
        self.acquiring.add(name)

    def _update_position(self, name: str, z: np.ndarray, R: np.ndarray) -> Tuple[bool, str]:
        s = self.settings
        track = self.tracks[name]
        mean, P = self.track_mean[name], self.track_cov[name]
        nu = z - mean
        S = P + R
        d2 = float(nu @ np.linalg.solve(S, nu))

        if track.updates >= s.gate_min_updates and d2 > s.gate_chi2:
            track.rejects += 1
            if track.rejects >= s.gate_max_rejects:
                self.live.pop(name, None)  # it was locked onto the wrong thing: drop that fix, don't freeze it
                self._solution = None
                self.reset(name)
                return False, f"{s.gate_max_rejects} rejections in a row (last {math.sqrt(d2):.1f} sigma), reacquiring"
            return False, f"{math.sqrt(d2):.1f} sigma from the estimate"

        K = P @ np.linalg.inv(S)
        self.track_mean[name] = mean + K @ nu
        IK = np.eye(3) - K
        P = IK @ P @ IK.T + K @ R @ K.T
        self.track_cov[name] = 0.5 * (P + P.T)
        return True, ""

    def _update_yaw(self, name: str, yaw: float, var: float):
        nu = wrap_angle(yaw - self.yaw_mean[name])
        S = self.yaw_var[name] + var
        if self.tracks[name].updates >= self.settings.gate_min_updates and nu * nu / S > self.settings.yaw_gate_chi2:
            return
        K = self.yaw_var[name] / S
        self.yaw_mean[name] = wrap_angle(self.yaw_mean[name] + K * nu)
        self.yaw_var[name] = (1.0 - K) * self.yaw_var[name]

    # ----- the map solve -----

    def _fixes(self) -> List[Fix]:
        live = [Fix(name, self.track_mean[name], self.track_cov[name], d, t) for name, (d, t) in self.live.items()]
        return self.frozen + live

    def _freeze(self, name: str):
        # The live fix stops changing; merged into the object's last frozen fix when barely any travel apart
        distance, time = self.live.pop(name)
        fix = Fix(name, self.track_mean[name].copy(), self.track_cov[name].copy(), distance, time)
        for other in reversed(self.frozen):
            if other.name == name and abs(other.distance - distance) < self.settings.fix_merge_distance:
                A, B = np.linalg.inv(other.cov), np.linalg.inv(fix.cov)
                other.cov = np.linalg.inv(A + B)
                other.mean = other.cov @ (A @ other.mean + B @ fix.mean)
                other.distance, other.time = max(other.distance, distance), max(other.time, time)
                return
        self.frozen.append(fix)

    def _solve(self):
        if self._solution is not None:
            return self._solution
        fixes = self._fixes()
        if not fixes:
            return None
        s = self.settings
        order = sorted(range(len(fixes)), key=lambda k: (fixes[k].distance, fixes[k].time))
        names = sorted({f.name for f in fixes})
        n = 1 + 3 * len(fixes) + 3 * len(names)
        node = {k: 1 + 3 * rank for rank, k in enumerate(order)}
        dev = {name: 1 + 3 * len(fixes) + 3 * j for j, name in enumerate(names)}
        layout_inv = np.linalg.inv(self._layout_prior())
        yaw_var = math.radians(s.course_yaw_std_deg) ** 2
        scale = s.layout_robust_scale
        fix_inv = [np.linalg.inv(f.cov) for f in fixes]

        # Offset chain: the first node from the start prior, then drift between consecutive nodes
        chain = []
        for rank, k in enumerate(order):
            f = fixes[k]
            if rank == 0:
                chain.append((None, node[k], np.linalg.inv(self._offset_prior() + self._drift(f.distance, f.time))))
            else:
                p = fixes[order[rank - 1]]
                Q = self._drift(f.distance - p.distance, f.time - p.time) + 1e-6 * np.eye(3)
                chain.append((node[order[rank - 1]], node[k], np.linalg.inv(Q)))

        def weights(x):
            w = {}
            for name in names:
                d = x[dev[name]:dev[name] + 3]
                r2 = float(d @ layout_inv @ d)
                w[name] = 1.0 / (1.0 + r2 / scale ** 2) if scale > 0 else 1.0
            return w

        def normal(x, w):
            A = np.zeros((n, n))
            b = np.zeros(n)
            A[0, 0] += 1.0 / yaw_var if yaw_var > 0 else 1e12
            for k, f in enumerate(fixes):
                rotated, g = _rotate(x[0], self.specs[f.name].position)
                y = f.mean - rotated + g * x[0]
                idx = [0] + list(range(node[k], node[k] + 3)) + list(range(dev[f.name], dev[f.name] + 3))
                J = np.zeros((3, 7))
                J[:, 0] = g
                J[:, 1:4] = np.eye(3)
                J[:, 4:7] = np.eye(3)
                A[np.ix_(idx, idx)] += J.T @ fix_inv[k] @ J
                b[idx] += J.T @ fix_inv[k] @ y
            for a, c, Qi in chain:
                A[c:c + 3, c:c + 3] += Qi
                if a is not None:
                    A[a:a + 3, a:a + 3] += Qi
                    A[a:a + 3, c:c + 3] -= Qi
                    A[c:c + 3, a:a + 3] -= Qi
            for name in names:
                d = dev[name]
                A[d:d + 3, d:d + 3] += w[name] * layout_inv
            return A, b

        def cost(x):
            total = x[0] ** 2 / yaw_var if yaw_var > 0 else 0.0
            for k, f in enumerate(fixes):
                rotated, _ = _rotate(x[0], self.specs[f.name].position)
                r = f.mean - rotated - x[node[k]:node[k] + 3] - x[dev[f.name]:dev[f.name] + 3]
                total += float(r @ fix_inv[k] @ r)
            for a, c, Qi in chain:
                r = x[c:c + 3] - (x[a:a + 3] if a is not None else 0.0)
                total += float(r @ Qi @ r)
            for name in names:
                d = x[dev[name]:dev[name] + 3]
                r2 = float(d @ layout_inv @ d)
                total += scale ** 2 * math.log1p(r2 / scale ** 2) if scale > 0 else r2
            return total

        def run(x, iterations, first_weights=None):
            w = first_weights or weights(x)
            for _ in range(iterations):
                A, b = normal(x, w)
                x_new = np.linalg.solve(A, b)
                done = float(np.max(np.abs(x_new - x))) < 1e-7
                x = x_new
                w = weights(x)
                if done:
                    break
            return x

        # Same fixes as last time (a live track moved): continue from the last answer. New or frozen fixes: also try
        # starting from every way of blaming one prior, and keep the best explanation
        structure = tuple((f.name, k < len(self.frozen)) for k, f in enumerate(fixes))
        if structure == self._structure and self._warm is not None and self._warm.size == n:
            x = run(self._warm, 8)
        else:
            starts = [run(np.zeros(n), 30)]
            if scale > 0 and len(names) > 1:
                for name in names:
                    w = {other: (0.0 if other == name else 1.0) for other in names}
                    starts.append(run(np.zeros(n), 30, first_weights=w))
            if self._warm is not None and self._warm.size == n:
                starts.append(run(self._warm, 30))
            x = min(starts, key=cost)
            self._structure = structure

        w = weights(x)
        A, _ = normal(x, w)
        last = fixes[order[-1]]
        self._warm = x
        self._solution = {"x": x, "cov": np.linalg.inv(A), "o": node[order[-1]], "d": dev, "weights": w,
                          "distance": last.distance, "time": last.time}
        return self._solution

    # ----- edits from services -----

    def reset(self, name: str, verify: bool = False):
        # Forget what detections taught this track (a cov check on it needs fresh detections). A map object's fix
        # is kept: it is frozen into the map, and its new track starts from where the map has it.
        # verify (a new target): an object known well keeps its estimate, widened by the drift since it was last seen,
        # and verify_detections that agree with it confirm it (observe() starts over if one does not)
        verify = verify and self.tracks[name].established
        if self.is_map_object(name):
            if name in self.live:
                self._freeze(name)
                self._solution = None
            self.acquiring.discard(name)
            self.seen.pop(name, None)
        elif verify:
            distance, time = self.seen.get(name, (self.distance, self.time))
            self.track_cov[name] = self.track_cov[name] + self._drift(self.distance - distance, self.time - time)
        else:
            self.track_cov[name] = self._child_prior()
        if not verify:
            self.yaw_var[name] = self._yaw_prior()
        self.tracks[name] = Track(established=verify, verifying=0 if verify else -1)
        self.pending.pop(name, None)

    def seed(self, name: str, position, yaw: Optional[float] = None):
        # Put the object at position (parent frame) with its prior covariance
        self.reset(name)
        self.track_mean[name] = np.asarray(position, dtype=np.float64).copy()
        if self.is_map_object(name):
            self.track_cov[name] = self._layout_prior()
            self.acquiring.add(name)
            self.seen[name] = (self.distance, self.time)
        if yaw is not None:
            self.yaw_mean[name] = wrap_angle(yaw)

    def shift(self, name: str, delta):
        # Translate the object in its parent frame, covariance unchanged
        delta = np.asarray(delta, dtype=np.float64)
        self.track_mean[name] = self.track_mean[name] + delta
        if self.is_map_object(name):
            # Moving it means its prior and everything seen of it move too
            self.specs[name].position = self.specs[name].position + delta
            for fix in self.frozen:
                if fix.name == name:
                    fix.mean = fix.mean + delta
            self._solution = None

    def rotate_position(self, name: str, dyaw: float):
        # Rotate the object's position about its parent frame's z axis (for when the parent frame yaws)
        c, s_ = math.cos(dyaw), math.sin(dyaw)
        rot = np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])
        if self.is_map_object(name):
            mean, _ = self.position(name)
            self.shift(name, rot @ mean - mean)
        else:
            self.track_mean[name] = rot @ self.track_mean[name]
            self.track_cov[name] = rot @ self.track_cov[name] @ rot.T

    def add_yaw(self, name: str, dyaw: float):
        self.yaw_mean[name] = wrap_angle(self.yaw_mean[name] + dyaw)

    def set_prior(self, name: str, position, yaw: float):
        # The prior map changed under us: move the object with it and forget its detections
        self.specs[name].position = np.asarray(position, dtype=np.float64)
        self.specs[name].yaw = float(yaw)
        self.reset(name)
        self.frozen = [fix for fix in self.frozen if fix.name != name]
        self._solution = None
        self._structure = None
        if not self.is_map_object(name):
            self.track_mean[name] = self.specs[name].position.copy()
        self.yaw_mean[name] = wrap_angle(yaw)


def _rotate(yaw: float, point: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # point rotated by yaw about z, and its derivative in yaw
    c, s = math.cos(yaw), math.sin(yaw)
    x, y, z = point
    return np.array([c * x - s * y, s * x + c * y, z]), np.array([-s * x - c * y, c * x - s * y, 0.0])
