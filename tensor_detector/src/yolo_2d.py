#!/usr/bin/env python3

import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.time import Time
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from pathlib import Path
from sensor_msgs.msg import CompressedImage, Image
from vision_msgs.msg import Detection2DArray
from std_msgs.msg import Empty
from cv_bridge import CvBridge, CvBridgeError

from detections_2d import select_target, make_target_mask, make_detections_2d
from yolo_model import YoloModel


class YOLONode(Node):
    def __init__(self):
        super().__init__("yolo_2d")
        self.declare_parameters(
            namespace="",
            parameters=[
                ("target_class", "", ParameterDescriptor(read_only=True)),
                ("confidence", 0.25, ParameterDescriptor(read_only=True)),
                ("model_path", "", ParameterDescriptor(read_only=True)),
                ("minimum_mask_pixels", 300, ParameterDescriptor(read_only=True)),
                ("iou_threshold", 0.7, ParameterDescriptor(read_only=True)),
                ("input_type", "compressed", ParameterDescriptor(read_only=True)),
                # Gated: inference runs only when estimate_now is received
                ("gated", False, ParameterDescriptor(read_only=True)),
                ("max_frame_age_sec", 0.5, ParameterDescriptor(read_only=True)),
            ],
        )

        self.confidence = self.get_parameter("confidence").value
        if self.confidence < 0.01 or self.confidence > 1.0:
            raise ValueError(f"confidence = {self.confidence}, must be in [0.01, 1.0]")

        self.minimum_mask_pixels = self.get_parameter("minimum_mask_pixels").value
        if self.minimum_mask_pixels < 1:
            raise ValueError(
                f"minimum_mask_pixels = {self.minimum_mask_pixels}, must be in [1, ..."
            )

        self.iou_threshold = self.get_parameter("iou_threshold").value
        if self.iou_threshold < 0 or self.iou_threshold > 1:
            raise ValueError(f"iou_threshold = {self.iou_threshold}, must be in [0, 1]")

        self.gated = self.get_parameter("gated").value

        # Validated in both modes, so a bad value always fails at startup
        self.max_frame_age_sec = self.get_parameter("max_frame_age_sec").value
        if self.max_frame_age_sec <= 0:
            raise ValueError(
                f"max_frame_age_sec = {self.max_frame_age_sec}, must be > 0"
            )

        self.input_type = self.get_parameter("input_type").value
        if self.input_type not in {"raw", "compressed"}:
            raise ValueError(
                f"input_type = {self.input_type}, must be 'raw' or 'compressed'"
            )

        self.model_path = self.get_parameter("model_path").value
        if not self.model_path:
            raise ValueError("model_path is empty")
        model_path = Path(self.model_path)
        if not model_path.is_file():
            raise ValueError(f"model_path '{self.model_path}' does not exist")

        self.target_class = self.get_parameter("target_class").value
        if not self.target_class:
            raise ValueError("target_class is empty")

        try:
            self.model = YoloModel(self.model_path, logger=self.get_logger())
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model: {e}") from e

        if self.model.model.task != "segment":
            raise ValueError(
                f"YOLO model task is not 'segment', it is {self.model.model.task}"
            )

        if self.target_class not in self.model.model.names.values():
            raise ValueError(
                f"Target class '{self.target_class}' not found in YOLO model. "
                f"Available classes: {list(self.model.model.names.values())}"
            )

        self.bridge = CvBridge()

        self.mask_publisher = self.create_publisher(
            Image, "foundationpose/segmentation", 1
        )
        self.detections_2d_publisher = self.create_publisher(
            Detection2DArray, "detections_2d", 10
        )

        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)

        if self.input_type == "compressed":
            self.image_subscription = self.create_subscription(
                CompressedImage, "image/compressed", self.image_callback, qos
            )
            self.decode = self.bridge.compressed_imgmsg_to_cv2
        else:
            self.image_subscription = self.create_subscription(
                Image, "image", self.image_callback, qos
            )
            self.decode = self.bridge.imgmsg_to_cv2

        # Latest received frame, still encoded. None until the first arrives
        self.latest_frame = None
        # Stamp (ns) of the last frame process() ran on, to skip repeat triggers.
        # Plain int: rclpy Time raises TypeError when compared with None
        self.last_processed_stamp_ns = None

        if self.gated:
            # Reliable: a dropped trigger is a missed (re)acquisition.
            # Volatile: triggers sent before this node is up are lost; the
            # publisher's timeout covers that.
            trigger_qos = QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
            )
            self.trigger_subscription = self.create_subscription(
                Empty, "estimate_now", self.trigger_callback, trigger_qos
            )
            self.get_logger().info("Gated: inference runs only on estimate_now")

    def image_callback(self, msg: CompressedImage | Image):
        if not self.gated:
            self.process(msg)
            return

        # Gated: just keep the newest frame. No decode, so idle frames are cheap
        self.latest_frame = msg

    def trigger_callback(self, msg: Empty):
        frame = self.latest_frame

        if frame is None:
            self.get_logger().warning(
                "estimate_now received, but no frame yet", throttle_duration_sec=1.0
            )
            return

        stamp = Time.from_msg(frame.header.stamp)

        # Same frame as last time: running it again gives the same result
        if stamp.nanoseconds == self.last_processed_stamp_ns:
            self.get_logger().info(
                "estimate_now received, but no new frame since the last one; skipped"
            )
            return

        # Stale frame (e.g. camera stalled). Uses the node clock, so bag
        # replay needs use_sim_time:=true
        age_sec = (self.get_clock().now() - stamp).nanoseconds / 1e9
        if age_sec > self.max_frame_age_sec:
            self.get_logger().warning(
                f"estimate_now received, but latest frame is {age_sec:.2f} s old "
                f"(max {self.max_frame_age_sec} s); skipped",
                throttle_duration_sec=1.0,
            )
            return

        self.last_processed_stamp_ns = stamp.nanoseconds
        self.process(frame)

    def process(self, msg: CompressedImage | Image):
        """Full pipeline on one frame: decode -> infer -> detections_2d -> mask."""

        header = msg.header

        try:
            cv_image = self.decode(msg, "bgr8")
        except CvBridgeError as e:
            self.get_logger().warning(f"CvBridge Error: {e}", throttle_duration_sec=1.0)
            return

        results = self.model.infer(
            cv_image, conf=self.confidence, iou=self.iou_threshold
        )
        result = results[0]

        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()

        detections = make_detections_2d(
            boxes_xyxy=boxes,
            scores=scores,
            class_ids=class_ids,
            names=result.names,
            header=header,
        )

        self.detections_2d_publisher.publish(detections)

        class_names = [result.names[int(c)] for c in class_ids]
        target = select_target(
            class_names=class_names,
            scores=scores,
            target_name=self.target_class,
        )

        if target is not None:
            mask = make_target_mask(
                polygon=result.masks.xy[target],
                height=cv_image.shape[0],
                width=cv_image.shape[1],
                header=header,
                min_pixels=self.minimum_mask_pixels,
            )

            if mask:
                self.mask_publisher.publish(mask)


def main(args=None):
    rclpy.init(args=args)
    yolo_node = None

    try:
        yolo_node = YOLONode()
        rclpy.spin(yolo_node)
    except KeyboardInterrupt:
        pass
    finally:
        if yolo_node is not None:
            yolo_node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
