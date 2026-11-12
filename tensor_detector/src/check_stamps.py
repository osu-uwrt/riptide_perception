#!/usr/bin/env python3
"""Check that the camera topics FoundationPose pairs carry identical stamps.

FoundationPose matches RGB, depth, camera_info and the mask on exactly equal
header stamps (sync_threshold 0). This node records the stamps of raw RGB
(the reference), compressed RGB, depth and camera_info for `duration` seconds,
then reports for each one how many reference frames it matched exactly.

Topics are relative and remapped at run time:
    ros2 run tensor_detector check_stamps.py --ros-args \
        -r image:=<raw rgb> -r image/compressed:=<raw rgb>/compressed \
        -r depth:=<depth> -r camera_info:=<rgb camera_info>

Exits 0 on PASS, 1 on FAIL.
"""

import sys
import time

import rclpy
from functools import partial
from bisect import bisect_left
from statistics import median
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from sensor_msgs.msg import CompressedImage, Image, CameraInfo


def compare_stamps(reference, other):
    """Compare two collections of integer-nanosecond stamps.

    Only reference stamps inside the time range both topics covered are
    compared, so subscriptions that started or stopped at slightly different
    moments don't count as mismatches.

    Returns a dict with:
        compared: reference stamps inside the shared window
        matched: how many of those appear exactly in other
        median_gap_ns, max_gap_ns: for unmatched stamps, distance to the
            nearest stamp in other (None if everything matched)
    Returns None if either collection is empty.
    """
    if not reference or not other:
        return None

    other_sorted = sorted(other)
    other_set = set(other_sorted)

    start = max(min(reference), other_sorted[0])
    end = min(max(reference), other_sorted[-1])
    window = [s for s in reference if start <= s <= end]

    gaps = []
    for stamp in window:
        if stamp in other_set:
            continue
        i = bisect_left(other_sorted, stamp)
        neighbors = other_sorted[max(i - 1, 0) : i + 1]
        gaps.append(min(abs(stamp - n) for n in neighbors))

    return {
        "compared": len(window),
        "matched": len(window) - len(gaps),
        "median_gap_ns": median(gaps) if gaps else None,
        "max_gap_ns": max(gaps) if gaps else None,
    }


def format_gap(ns):
    """Milliseconds, or microseconds when under 1 ms so tiny gaps don't print as 0."""
    if ns >= 1_000_000:
        return f"{ns / 1e6:.3f} ms"
    return f"{ns / 1e3:.1f} µs"


class CheckStampsNode(Node):
    def __init__(self):
        super().__init__("check_stamps")
        self.declare_parameters(
            namespace="",
            parameters=[
                ("duration", 5.0, ParameterDescriptor(read_only=True)),
            ],
        )

        self.duration = self.get_parameter("duration").value
        if self.duration <= 0:
            raise ValueError(f"duration = {self.duration}, must be > 0")

        # Integer-nanosecond header stamps per input, in arrival order.
        self.stamps = {
            "image": [],
            "image/compressed": [],
            "depth": [],
            "camera_info": [],
        }

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        self.compressed_image_subscription = self.create_subscription(
            CompressedImage,
            "image/compressed",
            partial(self.record, "image/compressed"),
            qos,
        )
        self.image_subscription = self.create_subscription(
            Image, "image", partial(self.record, "image"), qos
        )

        self.depth_subscription = self.create_subscription(
            Image, "depth", partial(self.record, "depth"), qos
        )
        self.camera_info_subscription = self.create_subscription(
            CameraInfo, "camera_info", partial(self.record, "camera_info"), qos
        )

    def record(self, key, msg):
        stamp = msg.header.stamp
        self.stamps[key].append(stamp.sec * 1_000_000_000 + stamp.nanosec)

    def report(self):
        """Print the per-topic results. Returns True if every topic passed."""
        subscriptions = {
            "image": self.image_subscription,
            "image/compressed": self.compressed_image_subscription,
            "depth": self.depth_subscription,
            "camera_info": self.camera_info_subscription,
        }

        print(f"\nStamps recorded over {self.duration:.1f} s")
        for key, stamps in self.stamps.items():
            topic = subscriptions[key].topic_name
            print(
                f"  {key:<17} {topic}: {len(stamps)} msgs "
                f"({len(stamps) / self.duration:.1f} Hz)"
            )

        reference = self.stamps["image"]
        if not reference:
            print("\nFAIL: no raw RGB (image) messages, so nothing to compare against.")
            print("Check the topic name and that the camera is publishing.")
            return False

        print("\nExact matches against image:")
        failed = []
        for key in ("image/compressed", "depth", "camera_info"):
            result = compare_stamps(reference, self.stamps[key])

            if result is None:
                print(
                    f"  {key:<17} 0 msgs: wrong topic name, incompatible QoS, "
                    "or not published"
                )
                failed.append(key)
                continue

            if result["compared"] == 0:
                print(f"  {key:<17} no overlap in time with image")
                failed.append(key)
                continue

            compared = result["compared"]
            matched = result["matched"]
            line = f"  {key:<17} {matched}/{compared} ({100 * matched / compared:.1f}%)"
            if matched < compared:
                line += (
                    f", unmatched gap median {format_gap(result['median_gap_ns'])}, "
                    f"max {format_gap(result['max_gap_ns'])}"
                )
                failed.append(key)
            print(line)

        if failed:
            print(f"\nFAIL: {', '.join(failed)}")
            print(
                "Gaps of about one frame period (67 ms at 15 fps) are dropped "
                "frames; small consistent gaps mean the stamps really differ."
            )
            return False

        print("\nPASS: every compared frame matched exactly.")
        return True


def main(args=None):
    rclpy.init(args=args)
    node = None
    passed = False

    try:
        node = CheckStampsNode()
        end = time.monotonic() + node.duration
        while rclpy.ok() and time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.1)
        passed = node.report()
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
