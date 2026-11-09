#!/usr/bin/env python3
"""Pure helpers that turn detector output into ROS messages:
2D detections and the FoundationPose target mask.

No node, subscriptions or Ultralytics imports here, so it runs without a GPU.
"""

import numpy as np
import cv2

from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from sensor_msgs.msg import Image


def select_target(class_names, scores, target_name):
    """Index of the highest-scoring detection named target_name.

    Returns None when nothing matches; on a tie the first one found wins.
    """
    best = None

    for index, name in enumerate(class_names):
        if name == target_name:
            if best is None:
                best = index
            else:
                if scores[best] < scores[index]:
                    best = index

    return best


def make_target_mask(polygon, height, width, header, min_pixels):
    """mono8 Image of one instance polygon (255 inside, 0 outside), with the source image header.

    Returns None when the polygon has fewer than 3 points or fills fewer than min_pixels pixels.
    No mask means FoundationPose makes no attempt on that frame.
    """
    if len(polygon) < 3:
        return None

    canvas = np.zeros((height, width), dtype=np.uint8)

    cv2.fillPoly(canvas, [np.round(polygon).astype(np.int32)], 255)

    if np.count_nonzero(canvas) < min_pixels:
        return None

    return Image(
        header=header,
        height=height,
        width=width,
        encoding="mono8",
        step=width,
        data=canvas.tobytes(),
    )


def make_detections_2d(boxes_xyxy, scores, class_ids, names, header):
    """Detection2DArray with one detection per input box, in input order (index i matches polygon i).

    class_id is the class name from names. No detections gives an empty array, not None.
    """
    # The array and every detection share the source header object; nothing modifies it after this
    detections = Detection2DArray()
    detections.header = header

    # Same order as the inputs, so index i here is polygon i for the mask
    for box, score, class_id in zip(boxes_xyxy, scores, class_ids):
        x1, y1, x2, y2 = box

        detection = Detection2D()
        detection.header = header

        # vision_msgs boxes are center + size; rclpy rejects numpy float32, so cast to float
        detection.bbox.center.position.x = float((x1 + x2) / 2)
        detection.bbox.center.position.y = float((y1 + y2) / 2)
        detection.bbox.size_x = float(x2 - x1)
        detection.bbox.size_y = float(y2 - y1)

        hypothesis = ObjectHypothesisWithPose()
        hypothesis.hypothesis.class_id = names[int(class_id)]
        hypothesis.hypothesis.score = float(score)
        detection.results.append(hypothesis)

        detections.detections.append(detection)

    return detections
