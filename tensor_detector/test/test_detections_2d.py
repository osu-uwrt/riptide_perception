import os
import sys

import numpy as np

from builtin_interfaces.msg import Time
from std_msgs.msg import Header

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from detections_2d import make_detections_2d, make_target_mask, select_target  # noqa: E402

HEIGHT = 30
WIDTH = 40
NAMES = {0: "person", 39: "bottle"}


def make_header():
    # Distinctive values so a header built from the clock or a string would not match
    return Header(stamp=Time(sec=1234, nanosec=5678), frame_id="ffc_left_camera_optical_frame")


def square(x, y, side):
    # Corner pixels of a side x side square; fillPoly includes the edges,
    # so it fills side ** 2 pixels
    return np.array([[x, y], [x + side - 1, y], [x + side - 1, y + side - 1], [x, y + side - 1]],
                    dtype=np.float32)


def pixels(mask):
    return np.frombuffer(bytes(mask.data), dtype=np.uint8).reshape(mask.height, mask.width)


def test_select_target_picks_highest_scoring_match():
    assert select_target(["bottle", "bottle", "bottle"], [0.3, 0.9, 0.5], "bottle") == 1


def test_select_target_ignores_other_classes():
    assert select_target(["person", "bottle", "person"], [0.99, 0.4, 0.95], "bottle") == 1


def test_select_target_absent_or_empty_is_none():
    assert select_target(["person", "person"], [0.9, 0.8], "bottle") is None
    assert select_target([], [], "bottle") is None


def test_select_target_tie_keeps_first():
    assert select_target(["bottle", "person", "bottle"], [0.7, 0.9, 0.7], "bottle") == 0


def test_mask_format_and_header():
    header = make_header()
    mask = make_target_mask(square(10, 5, 10), HEIGHT, WIDTH, header, min_pixels=1)

    assert mask.height == HEIGHT
    assert mask.width == WIDTH
    assert mask.encoding == "mono8"
    assert mask.step == WIDTH
    assert len(mask.data) == HEIGHT * WIDTH
    assert mask.header == header


def test_mask_fills_only_the_polygon():
    mask = pixels(make_target_mask(square(10, 5, 10), HEIGHT, WIDTH, make_header(),
                                   min_pixels=1))

    assert set(np.unique(mask)) == {0, 255}
    assert np.count_nonzero(mask) == 100
    # Rows are y, columns are x
    assert mask[9, 14] == 255
    assert mask[5, 10] == 255 and mask[14, 19] == 255
    assert mask[4, 10] == 0 and mask[5, 20] == 0
    assert mask[25, 35] == 0


def test_mask_degenerate_polygon_is_none():
    header = make_header()

    assert make_target_mask(np.zeros((0, 2), dtype=np.float32), HEIGHT, WIDTH, header, 1) is None
    assert make_target_mask(np.array([[1.0, 1.0], [5.0, 5.0]]), HEIGHT, WIDTH, header, 1) is None


def test_mask_below_min_pixels_is_none():
    assert make_target_mask(square(0, 0, 2), HEIGHT, WIDTH, make_header(), min_pixels=10) is None


def test_mask_partly_outside_image_is_clipped():
    mask = pixels(make_target_mask(square(-5, -5, 10), HEIGHT, WIDTH, make_header(),
                                   min_pixels=1))

    # Only x and y in 0..4 lie inside the image
    assert np.count_nonzero(mask) == 25
    assert mask[0, 0] == 255 and mask[4, 4] == 255 and mask[5, 5] == 0


def test_mask_fully_outside_image_is_none():
    mask = make_target_mask(square(100, 100, 10), HEIGHT, WIDTH, make_header(), min_pixels=1)

    assert mask is None


def test_detections_bbox_is_center_and_size():
    detections = make_detections_2d([[10.0, 20.0, 30.0, 60.0]], [0.8], [39], NAMES, make_header())
    bbox = detections.detections[0].bbox

    assert bbox.center.position.x == 20.0
    assert bbox.center.position.y == 40.0
    assert bbox.size_x == 20.0
    assert bbox.size_y == 40.0


def test_detections_accept_numpy_inputs():
    # The types Ultralytics hands over; rclpy rejects numpy scalars unless they are cast
    boxes = np.array([[10, 20, 30, 60]], dtype=np.float32)
    scores = np.array([0.8], dtype=np.float32)
    class_ids = np.array([39], dtype=np.int64)

    detections = make_detections_2d(boxes, scores, class_ids, NAMES, make_header())
    results = detections.detections[0].results

    assert len(results) == 1
    assert results[0].hypothesis.class_id == "bottle"
    assert isinstance(results[0].hypothesis.score, float)
    assert abs(results[0].hypothesis.score - 0.8) < 1e-6


def test_detections_keep_input_order():
    boxes = [[0.0, 0.0, 2.0, 2.0], [10.0, 10.0, 20.0, 20.0], [5.0, 5.0, 7.0, 7.0]]
    detections = make_detections_2d(boxes, [0.5, 0.9, 0.7], [39, 0, 39], NAMES, make_header())

    class_ids = [d.results[0].hypothesis.class_id for d in detections.detections]

    assert class_ids == ["bottle", "person", "bottle"]
    assert [d.bbox.center.position.x for d in detections.detections] == [1.0, 15.0, 6.0]


def test_detections_copy_source_header():
    header = make_header()
    boxes = [[0.0, 0.0, 2.0, 2.0], [4.0, 4.0, 8.0, 8.0]]
    detections = make_detections_2d(boxes, [0.5, 0.6], [39, 0], NAMES, header)

    assert detections.header == header
    assert all(d.header == header for d in detections.detections)


def test_detections_empty_input_is_empty_array():
    header = make_header()
    detections = make_detections_2d([], [], [], NAMES, header)

    assert detections is not None
    assert len(detections.detections) == 0
    assert detections.header == header
