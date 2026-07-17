#!/usr/bin/env python3
"""
Sync object poses from the mapping config.yaml into dummy_detections.yaml.

config.yaml expresses poses per-axis with yaw in DEGREES, parented on
"<object>_frame" frames. dummy_detections.yaml expresses poses as
[x, y, z, roll, pitch, yaw] with yaw in RADIANS, parented on dummy object
names. This script resolves each config object's pose in the map frame by
walking its parent chain, re-expresses it relative to the dummy object's
parent, and rewrites only the "pose:" (and "class_id:" where config defines
a class) lines of dummy_detections.yaml so comments and all dummy-specific
fields (noise, score, min_dist, ...) are preserved.

Usage:
    sync_dummy_detections.py [--namespace /talos/riptide_mapping2] [--dry-run]
"""

import argparse
import math
import re
import sys
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
CONFIG_YAML = CONFIG_DIR / "config.yaml"
DUMMY_YAML = CONFIG_DIR / "dummy_detections.yaml"

# dummy_detections object name -> config.yaml init_data object name
NAME_MAP = {
    "gate": "gate",
    "gate_repair": "gate_repair",
    "gate_rescue": "gate_rescue",
    "slalom_front": "slalom_front",
    "slalom_middle": "slalom_middle",
    "slalom_back": "slalom_back",
    "torpedo": "torpedo",
    "torpedo_blood_hole_large": "blood_hole_large",
    "torpedo_blood_hole_small": "blood_hole_small",
    "torpedo_fire_hole_large": "fire_hole_large",
    "torpedo_fire_hole_small": "fire_hole_small",
    "bin": "bin",
    "bin_vinyl1": "bin_vinyl1",
    "bin_vinyl2": "bin_vinyl2",
    "bin_vinyl3": "bin_vinyl3",
    "bin_vinyl4": "bin_vinyl4",
    "bin_magnet_1": "magnet1",
    "bin_magnet_2": "magnet2",
    "table": "table",
    "table_helmet": "helmet",
    "table_warning": "warning",
    "table_pill": "pill",
    "table_plug": "plug",
    "table_nut_and_bolt": "nut_and_bolt",
    "table_bandage": "bandage",
    "table_compass": "compass",
    "table_sos": "sos",
    "table_buoy": "buoy",
    "table_hammer_and_wrench": "hammer_and_wrench",
}


def compose(parent, child):
    """Compose two (x, y, z, yaw_rad) poses: parent * child."""
    px, py, pz, pyaw = parent
    x, y, z, yaw = child
    c, s = math.cos(pyaw), math.sin(pyaw)
    return (px + x * c - y * s, py + x * s + y * c, pz + z, pyaw + yaw)


def relative(parent, child):
    """Express global pose 'child' in the frame of global pose 'parent'."""
    px, py, pz, pyaw = parent
    x, y, z, yaw = child
    dx, dy = x - px, y - py
    c, s = math.cos(-pyaw), math.sin(-pyaw)
    return (dx * c - dy * s, dx * s + dy * c, z - pz, yaw - pyaw)


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def global_pose(name, init_data, cache):
    """Resolve an init_data object's pose in the map frame (yaw in radians)."""
    if name in cache:
        return cache[name]
    obj = init_data[name]
    p = obj["pose"]
    local = (p["x"], p["y"], p["z"], math.radians(p["yaw"]))
    parent = obj.get("parent", "map")
    if parent == "map":
        result = local
    else:
        parent_obj = re.sub(r"_frame$", "", parent)
        if parent_obj not in init_data:
            raise KeyError(f"{name}: parent frame '{parent}' has no init_data entry")
        result = compose(global_pose(parent_obj, init_data, cache), local)
    cache[name] = result
    return result


def fmt(value):
    """Format a float compactly, e.g. 0.0, -1.372, 4.608705."""
    text = f"{value:.6f}".rstrip("0")
    if text.endswith("."):
        text += "0"
    if text == "-0.0":
        text = "0.0"
    return text


def parse_dummy_objects(lines):
    """Map each detection_data object to its parent and field line numbers."""
    objects = {}
    in_detection_data = False
    current = None
    for i, line in enumerate(lines):
        if re.match(r"^    detection_data:", line):
            in_detection_data = True
            continue
        if in_detection_data and re.match(r"^\S", line):
            in_detection_data = False
        if not in_detection_data:
            continue
        m = re.match(r"^      (\w+):", line)
        if m:
            current = m.group(1)
            objects[current] = {"parent": "map", "pose_line": None, "class_id_line": None}
            continue
        if current is None:
            continue
        m = re.match(r"^        parent:\s*(\S+)", line)
        if m:
            objects[current]["parent"] = m.group(1)
        elif re.match(r"^        pose:\s*\[", line):
            objects[current]["pose_line"] = i
        elif re.match(r"^        class_id:\s*\S+", line):
            objects[current]["class_id_line"] = i
    return objects


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", default="/talos/riptide_mapping2",
                        help="config.yaml node namespace to pull init_data from")
    parser.add_argument("--dry-run", action="store_true",
                        help="print changes without writing dummy_detections.yaml")
    args = parser.parse_args()

    with open(CONFIG_YAML) as f:
        config = yaml.safe_load(f)
    try:
        init_data = config[args.namespace]["ros__parameters"]["init_data"]
    except KeyError:
        sys.exit(f"error: no init_data under '{args.namespace}' in {CONFIG_YAML}")

    lines = DUMMY_YAML.read_text().splitlines(keepends=True)
    objects = parse_dummy_objects(lines)

    cache = {}
    changed = 0
    for dummy_name, info in objects.items():
        config_name = NAME_MAP.get(dummy_name)
        if config_name is None or config_name not in init_data:
            print(f"skip {dummy_name}: no matching object in config init_data")
            continue
        if info["pose_line"] is None:
            print(f"skip {dummy_name}: no pose line found")
            continue

        child_global = global_pose(config_name, init_data, cache)
        dummy_parent = info["parent"]
        if dummy_parent == "map":
            rel = child_global
        else:
            parent_config = NAME_MAP.get(dummy_parent)
            if parent_config is None or parent_config not in init_data:
                print(f"skip {dummy_name}: dummy parent '{dummy_parent}' not in config")
                continue
            rel = relative(global_pose(parent_config, init_data, cache), child_global)

        x, y, z, yaw = rel
        pose_str = f"[{fmt(x)}, {fmt(y)}, {fmt(z)}, 0.0, 0.0, {fmt(wrap(yaw))}]"
        i = info["pose_line"]
        old = lines[i]
        comment = re.search(r"\s#.*$", old.rstrip("\n"))
        new = f"        pose: {pose_str}" + (comment.group(0) if comment else "") + "\n"
        if new != old:
            print(f"{dummy_name}: pose {old.strip()} -> {new.strip()}")
            lines[i] = new
            changed += 1

        config_class = init_data[config_name].get("class")
        j = info["class_id_line"]
        if config_class and j is not None:
            old = lines[j]
            new = re.sub(r"(class_id:\s*)\S+", rf"\g<1>{config_class}", old)
            if new != old:
                print(f"{dummy_name}: {old.strip()} -> {new.strip()}")
                lines[j] = new
                changed += 1

    if changed == 0:
        print("dummy_detections.yaml already in sync")
    elif args.dry_run:
        print(f"\ndry run: {changed} line(s) would change")
    else:
        DUMMY_YAML.write_text("".join(lines))
        print(f"\nwrote {DUMMY_YAML} ({changed} line(s) changed)")


if __name__ == "__main__":
    main()
