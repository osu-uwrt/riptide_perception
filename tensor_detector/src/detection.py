#!/usr/bin/env python3
"""Detection processing
Turns YOLO results into Detection3D messages and markers
"""

# TODO: Make this completely class agnostic, pass class specific logic to a delegator, then to yolo class specific python classes
# Most of the refactor outside of this file is pretty much done, but this is harder to separate
# And move stuff to config files as well

import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from vision_msgs.msg import Detection3DArray
from tf_transformations import quaternion_from_euler, quaternion_multiply
from rclpy.time import Time

import geometry
import outputs
from outputs import MarkerBuilder
from pointcloud import PointCloudBuilder


# Colors used for rviz markers, keyed by the published class name.
COLOR_MAP = {
    # Slaloms (slalom_name set by service)
    'slalom_close': (1.0, 0.0, 0.0),
    'slalom_middle': (1.0, 1.0, 0.0),
    'slalom_far': (0.0, 1.0, 0.0),

    # Slalom publishes under slalom_name, this is the default (not sure if we are using the service anymore)
    'slalom_front': (0.0, 1.0, 0.4),

    # Gate combinations
    'gate_repair': (0.0, 1.0, 0.0),
    'gate_rescue': (0.0, 0.5, 1.0),

    # Torpedo stuff 
    'torpedo': (0.0, 1.0, 1.0),
    'fire_hole_large': (1.0, 0.4, 0.0),
    'fire_hole_small': (1.0, 0.7, 0.0),
    'blood_hole_large': (0.6, 0.0, 0.0),
    'blood_hole_small': (1.0, 0.0, 0.4),

    # FFC generic planar classes (when not merged into a gate pair)
    'buoy': (0.2, 0.8, 1.0),
    'compass': (0.9, 0.6, 0.1),
    'hammer_and_wrench': (0.4, 0.4, 0.45),
    'sos': (1.0, 0.1, 0.3),

    # DFC generic planar classes
    # fire/blood also fall here on dfc, as the torpedo task is disabled
    'bandage': (0.95, 0.85, 0.70),
    'blood': (0.80, 0.0, 0.0),
    'fire': (1.0, 0.5, 0.0),
    'helmet': (1.0, 1.0, 0.0),
    'nut_and_bolt': (0.5, 0.5, 0.5),
    'pill': (0.7, 0.2, 0.9),
    'plug': (0.0, 0.4, 1.0),
    'warning': (1.0, 0.0, 1.0),
}

# Published-name relabels (no class is currently relabeled on publish but good to have)
RELABEL_MAP = {}

# Classes intercepted by the Torpedo task when it is enabled (ffc)
TORPEDO_CLASSES = {'fire', 'blood', 'circle'}

# Pairs that, when both are seen in a frame, are merged into one combined detection (union bbox -> centroid) (Gate)
# When only one member is present, it runs through the normal planar path (Octagon)
GATE_PAIRS = [
    (('compass', 'hammer_and_wrench'), 'gate_repair'),
    (('buoy', 'sos'), 'gate_rescue'),
]
GATE_PAIR_CLASSES = {c for pair, _ in GATE_PAIRS for c in pair} # This way we dont have to type them twice (python moment)

SLALOM_CLASS = "slalom" #magic 🪄

@dataclass
class SurfaceFit:
    """Result of fitting a plane to one detection's surface patch."""
    points: Any          # Nx3 inlier points
    normal: Any          # unit normal, flipped to face the camera
    centroid: Any        # 3D centroid (bbox center x/y at the fitted plane depth)
    quat: Any            # orientation from the normal
    center2d: tuple      # (px, py) bbox center in image coords
    conf: Any            # detection confidence of the source box


@dataclass
class ProcessorConfig:
    class_detect_shrink: float = 0.15
    min_points: int = 5
    map_min_area: int = 50
    slalom_history_size: int = 10
    use_incoming_timestamp: bool = True
    publish_interval: float = 0.1   # also drives marker lifetime
    gftt_quality_level: float = 0.02   # goodFeaturesToTrack corner-score threshold
    gftt_min_distance: float = 1.0     # goodFeaturesToTrack min px between corners

@dataclass
class Frame:
    """Everything the processor needs for a single image. Built by the node."""
    image: Any            # cv_image (bgr8); feature points get overlaid on it
    gray: Any             # grayscale of image
    depth: Any            # depth image (passthrough)
    fx: float             # camera intrinsics (same values as K below, isolated for easy access)
    fy: float
    cx: float
    cy: float
    K: Any                # 3x3 intrinsic matrix [[fx,0,cx],[0,fy,cy],[0,0,1]] (useful for numpy lin alg math)
    frame_id: str         # Tf frame
    timestamp: Any        # incoming header stamp
    class_id_map: dict
    conf: float
    want_markers: bool = True
    want_cloud: bool = True


class DetectionProcessor:
    def __init__(self, config, logger, now_stamp, tf_buffer):
        self.cfg = config
        self.log = logger
        self._now_stamp = now_stamp        # callable lambda (ƛ🍾) -> builtin_interfaces/Time msg 
        self.tf_buffer = tf_buffer
        self._default_normal = geometry.DEFAULT_NORMAL

        # Output things
        self.markers = MarkerBuilder(config)
        self.cloud = PointCloudBuilder(config.min_points, logger)

        # Persistant states (stuff that lives across frames)
        self.plane_normal = None           # last fitted plane normal (debug/use)
        self.slalom_history = []
        self.slalom_name = 'slalom_front'
        self._markers = []

        # Per-frame states
        self._frame = None
        self._mask = None
        self.slalom_red_detections = []

        # Torpedo stuff
        self.torpedo_fire_box = None
        self.torpedo_blood_box = None
        self.torpedo_circles = []

        # When enabled, the torpedo classes are intercepted and
        # resolved by process_torpedo_task instead of the default plane-fit path.
        # (So blood/fire work for bins)
        self.torpedo_enabled = False

        # Gate pairs: class_name -> list of boxes seen this frame
        self.gate_pair_boxes = {}


    ### API
    def set_slalom_name(self, name):
        self.slalom_name = name

    def set_torpedo_enabled(self, enabled):
        """Turn the Task04 fire/blood torpedo logic on (ffc) or off (dfc)."""
        self.torpedo_enabled = bool(enabled)

    def process(self, results, frame):
        """Run one image through the pipeline.

        Returns (detections, markers_to_publish). The node handles marker
        throttling, the point cloud, the annotated image, and publishing.
        """
        self._frame = frame
        self._reset_frame_state()
        self.markers.reset()

        detections = Detection3DArray()
        detections.header.frame_id = frame.frame_id
        detections.header.stamp = self._stamp(frame)

        if self._mask is None or self._mask.shape[:2] != frame.image.shape[:2]:
            self._mask = np.zeros(frame.image.shape[:2], dtype=np.uint8)

        # Build the full-frame segmentation mask
        self._mask.fill(0)

        for result in results:
            if result is not None and result.masks is not None:
                for contour in result.masks.xy:
                    contour = np.array(contour, dtype=np.int32)
                    cv2.fillPoly(self._mask, [contour], 255)

        # Route each detection
        # Class specifics are handled, otherwise generic plane/fit
        for result in results:
            if result is None:
                continue
            if result.boxes is not None:
                for box in result.boxes.cpu().numpy():
                    if box.conf[0] <= frame.conf: # Technically redundant but extra safety check for now
                        continue
                    # int-cast here so routing keys match create_detection3d_message
                    # and the int-keyed YAML class map (was a raw numpy float).
                    class_id = int(box.cls[0])
                    if class_id not in frame.class_id_map:
                        continue

                    conf = box.conf[0]
                    name = frame.class_id_map[class_id]

                    if self.torpedo_enabled and name in TORPEDO_CLASSES:
                        self._torpedo_add(name, box)
                    elif name in GATE_PAIR_CLASSES:
                        self.gate_pair_boxes.setdefault(name, []).append(box)
                    elif name == SLALOM_CLASS:
                        detection_temp = self.create_detection3d_message(box, frame, conf)
                        if detection_temp and detection_temp.results:
                            self._stash_slalom(box, detection_temp)
                    else:
                        detection = self.create_detection3d_message(box, frame, conf)
                        if detection:
                            detections.detections.append(detection)

        # Resolve grouped detections that need the whole frame first
        self._resolve_gate_pairs(frame, detections)

        # Torpedo fire/blood (ffc)
        if self.torpedo_enabled:
            self.process_torpedo_task(frame, detections)

        self._process_slalom(frame, detections)

        # Flush markers built so far; markers added after this (slalom) carry over.
        markers_out = self._markers
        self._markers = []

        return detections, markers_out

    def take_point_cloud(self, frame_id, stamp):
        """Build and clear the accumulated point cloud (None if empty)."""
        return self.cloud.take_cloud(frame_id, stamp)

    ### Helpers
    def _reset_frame_state(self):
        self.slalom_red_detections = []
        self.torpedo_fire_box = None
        self.torpedo_blood_box = None
        self.torpedo_circles = []
        self.gate_pair_boxes = {}

    def _stamp(self, frame):
        if self.cfg.use_incoming_timestamp:
            return frame.timestamp
        return self._now_stamp()

    def _stash_slalom(self, box, detection_temp):
        result_temp = detection_temp.results[0]
        centroid = [
            result_temp.pose.pose.position.x,
            result_temp.pose.pose.position.y,
            result_temp.pose.pose.position.z,
        ]
        quat = [
            result_temp.pose.pose.orientation.x,
            result_temp.pose.pose.orientation.y,
            result_temp.pose.pose.orientation.z,
            result_temp.pose.pose.orientation.w,
        ]
        x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
        self.slalom_red_detections.append({
            'centroid': centroid,
            'quat': quat,
            'conf': box.conf[0],
            'bbox_width': x_max - x_min,
            'bbox_height': y_max - y_min,
        })

    ### Core dispatch
    def create_detection3d_message(self, box, frame, conf):
        """Default per-detection path: slalom (depth-sampled centroid) or the
        generic planar surface fit. Returns a Detection3D or None."""
        x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min

        class_id = int(box.cls[0])
        bbox_center_x = (x_min + x_max) / 2
        bbox_center_y = (y_min + y_max) / 2
        class_name = frame.class_id_map.get(class_id, "Unknown")

        # Slalom: depth-sampled centroid, default-facing orientation (mapping overrides it now)
        if class_name == SLALOM_CLASS:
            depth_value = frame.depth[int(bbox_center_y), int(bbox_center_x)]
            if (np.isnan(depth_value) or math.isinf(bbox_center_x)
                    or math.isinf(bbox_center_y) or math.isinf(depth_value)):
                return None
            centroid = geometry.pixel_to_3d(bbox_center_x, bbox_center_y,
                                            float(depth_value),
                                            frame.fx, frame.fy, frame.cx, frame.cy)
            quat, _ = geometry.normal_to_quaternion(-self._default_normal,
                                                    self._default_normal)
            detection = self._new_detection(frame)
            detection.results.append(self._make_hypothesis(class_name, centroid, quat, conf))
            return detection

        # Generic planar surface fit (everything else)
        fit = self._fit_bbox((x_min, y_min, x_max, y_max), frame, conf)
        if fit is None:
            return None

        self.plane_normal = fit.normal
        self._build_marker(frame, fit.quat, fit.centroid, class_name,
                           bbox_width, bbox_height)
        detection = self._new_detection(frame)
        detection.results.append(self._make_hypothesis(class_name, fit.centroid, fit.quat, conf))
        return detection

    ### Shared surface fitting
    def _extract_bbox_points(self, bbox, frame):
        """Shrink + mask one bbox, run goodFeatures, back-project to 3D"""
        x_min, y_min, x_max, y_max = map(int, bbox)
        shrink_x = (x_max - x_min) * self.cfg.class_detect_shrink
        shrink_y = (y_max - y_min) * self.cfg.class_detect_shrink
        x0 = int(x_min + shrink_x)
        x1 = int(x_max - shrink_x)
        y0 = int(y_min + shrink_y)
        y1 = int(y_max - shrink_y)

        mask_roi = self._mask[y0:y1, x0:x1]
        cropped_gray_image = frame.gray[y0:y1, x0:x1]
        masked_gray_image = cv2.bitwise_and(cropped_gray_image, cropped_gray_image, mask=mask_roi)

        quality = self.cfg.gftt_quality_level
        if quality <= 0.0:
            self.log.warning(
                f"gftt_quality_level={quality} invalid (must be > 0); using 0.02",
                throttle_duration_sec=5.0)
            quality = 0.02
        min_dist = max(0.0, self.cfg.gftt_min_distance)

        good_features = cv2.goodFeaturesToTrack(
            masked_gray_image, maxCorners=0, qualityLevel=quality, minDistance=min_dist)
        if good_features is None:
            return None

        good_features[:, 0, 0] += x0
        good_features[:, 0, 1] += y0
        feature_points = [pt[0] for pt in good_features]
        return self._get_3d_points(frame, feature_points)

    def _plane_from_points(self, points_3d, frame, center_bbox, conf):
        """Pooled 3D points -> SVD plane -> camera-facing surface"""
        if points_3d is None or len(points_3d) < self.cfg.min_points:
            return None

        normal, _, centroid3 = geometry.fit_plane(points_3d)
        if normal is None:
            return None

        bbox_center_x = (center_bbox[0] + center_bbox[2]) / 2
        bbox_center_y = (center_bbox[1] + center_bbox[3]) / 2
        centroid = geometry.pixel_to_3d(bbox_center_x, bbox_center_y, centroid3[2],
                                        frame.fx, frame.fy, frame.cx, frame.cy)
        if normal[2] > 0:
            normal = -normal
        quat, _ = geometry.normal_to_quaternion(normal, self._default_normal)

        return SurfaceFit(points=points_3d, normal=normal, centroid=centroid,
                          quat=quat, center2d=(bbox_center_x, bbox_center_y), conf=conf)

    def _fit_bbox(self, bbox, frame, conf):
        """Generic planar surface fit over a single bbox: sample -> SVD."""
        points_3d = self._extract_bbox_points(bbox, frame)
        return self._plane_from_points(points_3d, frame, bbox, conf)

    def _fit_boxes(self, boxes, frame, conf, center_bbox):
        """Sample each box independently, pool the points, then one SVD fit"""
        pooled = []
        for box in boxes:
            pts = self._extract_bbox_points(box.xyxy[0], frame)
            if pts is not None and len(pts) > 0:
                pooled.append(np.asarray(pts))
        if not pooled:
            return None
        return self._plane_from_points(np.vstack(pooled), frame, center_bbox, conf)

    def _fit_symbol(self, box, frame):
        """Run the generic surface pipeline on one symbol box (fire/blood)."""
        return self._fit_bbox(box.xyxy[0], frame, box.conf[0])

    ### Torpedo
    def _torpedo_add(self, name, box):
        if name == "fire":
            self.torpedo_fire_box = box
        elif name == "blood":
            self.torpedo_blood_box = box
        elif name == "circle":
            self.torpedo_circles.append(box)

    def process_torpedo_task(self, frame, detections):
        """Resolve the fire/blood torpedo board: publish the torpedo center and,
        when fully visible, the four labeled openings"""
        fire = self._fit_symbol(self.torpedo_fire_box, frame) if self.torpedo_fire_box is not None else None
        blood = self._fit_symbol(self.torpedo_blood_box, frame) if self.torpedo_blood_box is not None else None

        present = [f for f in (fire, blood) if f is not None]
        if not present:
            return  # no reliable symbol plane this frame; publish nothing

        # Board plane normal (pool both patches when available)
        if fire is not None and blood is not None:
            pooled = np.vstack([fire.points, blood.points])
            normal, _, _ = geometry.fit_plane(pooled)
            if normal is None:
                normal = present[0].normal
        else:
            normal = present[0].normal
        normal = np.asarray(normal, dtype=float)
        if normal[2] > 0:
            normal = -normal
        self.plane_normal = normal
        board_quat, _ = geometry.normal_to_quaternion(normal, self._default_normal)

        # Plane anchor: midpoint of visible symbol centroids (any on-plane point).
        anchor = np.mean(np.array([f.centroid for f in present]), axis=0)

        # Torpedo center via in-plane extent of all visible shapes
        centers_2d = [f.center2d for f in present]
        for cbox in self.torpedo_circles:
            centers_2d.append(self._box_center(cbox))

        projected = []
        for (px, py) in centers_2d:
            pt = self._project_center(px, py, frame, normal, anchor)
            if pt is not None:
                projected.append(pt)
        if not projected:
            return

        u, v = geometry.inplane_basis(normal)
        origin = projected[0]
        us = [float(np.dot(p - origin, u)) for p in projected]
        vs = [float(np.dot(p - origin, v)) for p in projected]
        torpedo_center = (origin
                          + ((min(us) + max(us)) / 2.0) * u
                          + ((min(vs) + max(vs)) / 2.0) * v)

        torpedo_conf = min(self._box_conf(b) for b in
                           (self.torpedo_fire_box, self.torpedo_blood_box) if b is not None)
        self._emit_task(frame, detections, "torpedo", torpedo_center, board_quat,
                        torpedo_conf, 0.3, 0.3)

        # Hole instance separation (needs both symbols + 4 circles)
        if fire is None or blood is None or len(self.torpedo_circles) != 4:
            return

        fpx, fpy = fire.center2d
        bpx, bpy = blood.center2d
        fire_pt = self._project_center(fpx, fpy, frame, normal, anchor)
        blood_pt = self._project_center(bpx, bpy, frame, normal, anchor)
        if fire_pt is None or blood_pt is None:
            return
        fire_pt = np.asarray(fire_pt)
        blood_pt = np.asarray(blood_pt)

        # Project the holes onto the fire->blood axis and sort
        # The two nearest fire go to the fire side, the two nearest blood to the blood side
        # This always splits them 2-2 and (should) work at any roll
        sep = blood_pt - fire_pt
        norm_sep = float(np.linalg.norm(sep))
        if norm_sep < 1e-6:
            return
        sep = sep / norm_sep

        holes = []
        for cbox in self.torpedo_circles:
            cx, cy = self._box_center(cbox)
            pt = self._project_center(cx, cy, frame, normal, anchor)
            if pt is None:
                return  # can't resolve all four then publish no holes
            pt = np.asarray(pt)
            holes.append({'box': cbox, 'pt': pt,
                          's': float(np.dot(pt - fire_pt, sep))})

        holes.sort(key=lambda h: h['s'])
        fire_pair, blood_pair = holes[:2], holes[2:]

        self._emit_hole_pair(frame, detections, "fire", fire, fire_pair, board_quat)
        self._emit_hole_pair(frame, detections, "blood", blood, blood_pair, board_quat)

    def _emit_hole_pair(self, frame, detections, symbol_name, symbol_fit, pair, quat):
        # The opening closest (in 3D) to its symbol is the large one
        d0 = float(np.linalg.norm(np.asarray(pair[0]['pt']) - np.asarray(symbol_fit.centroid)))
        d1 = float(np.linalg.norm(np.asarray(pair[1]['pt']) - np.asarray(symbol_fit.centroid)))
        large, small = (pair[0], pair[1]) if d0 <= d1 else (pair[1], pair[0])
        for hole, size_name in ((large, "large"), (small, "small")):
            cw, ch = self._box_size(hole['box'])
            self._emit_task(frame, detections, f"{symbol_name}_hole_{size_name}",
                            hole['pt'], quat, self._box_conf(hole['box']),
                            cw / 150.0, ch / 150.0)

    def _emit_task(self, frame, detections, class_id, centroid, quat, conf, scale_x, scale_y):
        if frame.want_markers:
            color = COLOR_MAP.get(class_id, (1.0, 1.0, 1.0))
            self._markers.extend(self.markers.build(
                frame.frame_id, self._stamp(frame), quat, centroid, color, scale_x, scale_y))
        detection = self._new_detection(frame)
        detection.results.append(self._make_hypothesis(class_id, centroid, quat, conf))
        detections.detections.append(detection)

    def _project_center(self, px, py, frame, normal, anchor):
        pt, _ = geometry.ray_plane_intersection(px, py, frame.K, normal, anchor)
        return pt

    def _box_center(self, box):
        x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
        return (x_min + x_max) / 2, (y_min + y_max) / 2

    def _box_size(self, box):
        x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
        return (x_max - x_min), (y_max - y_min)

    def _box_conf(self, box):
        return box.conf[0]

    ### Gate pairs
    def _union_bbox(self, boxes):
        """Axis-aligned union of several boxes"""
        xs0, ys0, xs1, ys1 = [], [], [], []
        for box in boxes:
            x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
            xs0.append(x_min); ys0.append(y_min); xs1.append(x_max); ys1.append(y_max)
        return (min(xs0), min(ys0), max(xs1), max(ys1))

    def _resolve_gate_pairs(self, frame, detections):
        """If both were seen this frame, publish ONE combined detection 
        If only one is present, publish each present box
        normally (its own class, standard planar path)."""
        for (a, b), gate_name in GATE_PAIRS:
            boxes_a = self.gate_pair_boxes.get(a, [])
            boxes_b = self.gate_pair_boxes.get(b, [])
            if boxes_a and boxes_b:
                self._emit_combined_gate(frame, detections, boxes_a + boxes_b, gate_name)
            else:
                for box in boxes_a + boxes_b:
                    det = self.create_detection3d_message(box, frame, box.conf[0])
                    if det:
                        detections.detections.append(det)

    def _emit_combined_gate(self, frame, detections, boxes, gate_name):
        """Union the member bboxes, fit one plane over the combined region, and
        publish it under gate_name (conf = min of the members)"""
        union = self._union_bbox(boxes)
        conf = min(self._box_conf(b) for b in boxes)

        fit = self._fit_boxes(boxes, frame, conf, center_bbox=union)
        if fit is None:
            return
        self.plane_normal = fit.normal # Stored for debug stuff
        bbox_width = union[2] - union[0]
        bbox_height = union[3] - union[1]
        self._build_marker(frame, fit.quat, fit.centroid, gate_name, bbox_width, bbox_height)
        detection = self._new_detection(frame)
        detection.results.append(self._make_hypothesis(gate_name, fit.centroid, fit.quat, conf))
        detections.detections.append(detection)

    ### Slalom
    def _process_slalom(self, frame, detections_array):
        """Report the closest slalom_red from history, oriented by tf parent"""
        if len(self.slalom_red_detections) > 0:
            closest_detection = min(self.slalom_red_detections, key=lambda x: x['centroid'][2])
            self.slalom_history.append({
                'centroid': closest_detection['centroid'],
                'quat': closest_detection['quat'],
                'conf': closest_detection['conf'],
            })
            if len(self.slalom_history) > self.cfg.slalom_history_size:
                self.slalom_history.pop(0)

            if self.slalom_history:
                closest_in_history = min(self.slalom_history, key=lambda x: x['centroid'][2])

                parent_frame = "slalom_parent_frame"
                try:
                    parent_quat_tf = self.tf_buffer.lookup_transform(
                        frame.frame_id, parent_frame, Time()).transform.rotation
                    detection_quat = [parent_quat_tf.x, parent_quat_tf.y,
                                      parent_quat_tf.z, parent_quat_tf.w]
                except Exception:
                    self.log.warning(f'Pubbing slalom detection with rotation since {parent_frame} not found')
                    detection_quat = closest_in_history['quat']

                z_to_x_quat = quaternion_from_euler(0.0, -1.57079632679, 0.0)
                corrected_quat = quaternion_multiply(detection_quat, z_to_x_quat)

                detection = self._new_detection(frame)
                detection.results.append(self._make_hypothesis(
                    self.slalom_name,
                    closest_in_history['centroid'],
                    corrected_quat,
                    closest_in_history['conf']))

                self._build_marker(
                    frame, corrected_quat, closest_in_history['centroid'],
                    self.slalom_name,
                    closest_detection['bbox_width'],
                    closest_detection['bbox_height'])
                detections_array.detections.append(detection)

            self.slalom_red_detections = []
        else:
            self.slalom_red_detections = []

    ### 3D / cloud
    def _get_3d_points(self, frame, feature_points):
        return self.cloud.extract(frame, feature_points, self._mask)

    ### Messages
    def _new_detection(self, frame):
        return outputs.new_detection(frame.frame_id, self._stamp(frame))

    def _make_hypothesis(self, class_name, centroid, quat, conf):
        # Apply the publish-name relabel (anchors publish under a different name)
        published = RELABEL_MAP.get(class_name, class_name)
        return outputs.make_hypothesis(published, centroid, quat, conf)

    def _build_marker(self, frame, quat, centroid, class_name, bbox_width, bbox_height):
        if not frame.want_markers:
            return
        color = COLOR_MAP.get(class_name, (1.0, 1.0, 1.0))
        flat = (class_name == "buoy")
        self._markers.extend(self.markers.build(
            frame.frame_id, self._stamp(frame), quat, centroid, color,
            float(bbox_width) / 150.0, float(bbox_height) / 150.0, flat))