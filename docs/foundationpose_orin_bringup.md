# FoundationPose bring-up on the Orin

Procedure for the robot-side checks of M1 (`yolo_2d` on the real camera) and the first M2 bring-up (ZED → format converter → FoundationPose). The design is in [foundationpose_architecture.md](foundationpose_architecture.md). Topics, parameters and formats are in [foundationpose_isaac_reference.md](foundationpose_isaac_reference.md), cited below as "ref".

How to use it:
- Run the stages in order. Each one ends with a stop condition, because later stages depend on it. If one fails, fix it before moving on.
- The `→` lines are blanks to fill in during the session. Afterwards, move the answers into the design doc (see [stage 7](#7-afterwards)) so this file stays a reusable procedure.

## Before the session

- [ ] Branch `foundationpose-integration` pushed (`git push -u origin foundationpose-integration`)
- [ ] YOLO weights `yolo26s-seg.pt` (not in git) on a USB stick, or confirm the Orin can download it
- [ ] `meshes/owala.obj` + `owala.mtl` (not in git)
- [ ] Optional: FoundationPose ONNX models downloaded ahead of time (ref: [Models and TensorRT engines](foundationpose_isaac_reference.md#models-and-tensorrt-engines))

## 0. Set paths once per terminal

Fill in after stages 1–2, then paste into every new terminal:

```bash
export ROBOT=tempest            # from ros2 topic list
export CAM=ffc
export YOLO_MODEL=~/models/yolo26s-seg.pt
export MESH=~/meshes/owala.obj
export MODELS_DIR=${ISAAC_ROS_WS}/isaac_ros_assets/models/foundationpose
```

## 1. Platform inventory (design doc open question 1)

```bash
cat /etc/nv_tegra_release                                    # L4T / JetPack
dpkg -l | grep -E "nvinfer|tensorrt" | head -3               # TensorRT
dpkg -l | grep -i isaac-ros-foundationpose                   # Isaac installed? (may only be inside the Isaac container)
python3 -c "import ultralytics; print(ultralytics.__version__)"
ls ${ISAAC_ROS_WS}/isaac_ros_assets/models/foundationpose    # models / engines already there?
```
→ L4T: ______ TensorRT: ______ Isaac FP installed: ______ Ultralytics: ______ Engines exist: ______

**Find out where things run.** Isaac usually lives in a Docker container (`isaac_ros_common/scripts/run_dev.sh`). With `launch_foundationpose:=true`, `yolo_2d` starts in that same environment, so Ultralytics must be installed there too. If it's only on the host, run `yolo_2d` (Isaac off) on the host and the Isaac part in the container. Then confirm both see each other's topics, with the same `ROS_DOMAIN_ID` and host networking.
→ Where Isaac runs: ______ Ultralytics available there: ______

## 2. ZED topics (start the ZED with the team's usual launch first)

```bash
ros2 topic list | grep zed                          # namespace + wrapper version (5.1 rename)
ros2 topic list | grep compressed                   # NITROS check: no output = no /compressed topic
```
→ Namespace: `/______/______/zed_node/` Wrapper names: old (`rgb/image_rect_color`) / new (`rgb/color/rect/image`) `/compressed` exists: ______

- **New names?** Every launch command below needs `zed_rgb_topic:=rgb/color/rect/image zed_camera_info_topic:=rgb/color/rect/camera_info`.
- **No `/compressed`?** NITROS is on (ref: [NITROS and the compressed topic](foundationpose_isaac_reference.md#nitros-and-the-compressed-topic)). Add `input_type:=raw` to every launch command below. `yolo_2d` then reads the raw RGB topic, and `raw_image_topic` already defaults to it. The alternative is setting `debug.disable_nitros: true` in the ZED config, but that config is shared by everything that uses the camera, so decide on it together. With `input_type:=raw`, skip the compressed line in the stage 3 timestamp check: `yolo_2d` reads the same message Isaac does.

Set `RGB=/$ROBOT/$CAM/zed_node/<rgb topic>` and `DEPTH=/$ROBOT/$CAM/zed_node/depth/depth_registered` for the next steps.

## 3. Formats, QoS, timestamps

```bash
ros2 topic echo --once $RGB --field encoding        # expect bgra8
ros2 topic echo --once $RGB --field width           # expect 1920
ros2 topic echo --once $RGB --field height          # expect 1200
ros2 topic echo --once $DEPTH --field encoding      # expect 32FC1
ros2 topic info -v $RGB | grep -A3 "Reliability"    # reliable or best effort?
ros2 topic info -v $DEPTH | grep -A3 "Reliability"
ros2 param get /$ROBOT/$CAM/zed_node debug.use_pub_timestamps   # expect False
```
→ Encodings: ______ / ______ Size: ______ QoS RGB / depth: ______ / ______ `use_pub_timestamps`: ______

Best effort anywhere → add `zed_qos:=SENSOR_DATA` to the Isaac launch.

**Timestamp parity (design doc open question 2).** Run these in three terminals at the same moment and compare `sec` / `nanosec` (or record a short bag and compare there):
```bash
ros2 topic echo $RGB --field header.stamp
ros2 topic echo $RGB/compressed --field header.stamp
ros2 topic echo $DEPTH --field header.stamp
```
→ Identical stamps across all three: ______ **Stop if not.** With `sync_threshold` 0, FoundationPose silently produces nothing.

## 4. yolo_2d on the real camera (M1 robot checks)

```bash
ros2 launch tensor_detector foundationpose.launch.py \
  robot:=$ROBOT camera:=$CAM \
  target_class:=bottle yolo_model_path:=$YOLO_MODEL
```
With the Owala in view:
```bash
ros2 topic hz /detections_2d                       # processed fps
ros2 topic hz /foundationpose/segmentation         # masks only while the bottle is visible
ros2 topic echo --once /foundationpose/segmentation --field width   # 1920
ros2 topic info -v /foundationpose/segmentation    # 1 publisher
```
Latency: compare the stamp in `/detections_2d` with the current time. For a quick check, `ros2 topic delay /detections_2d` works if the header stamps use the same clock as the Orin.
→ fps: ______ latency: ______ mask size: ______ GPU memory with ZEDs + YOLO (`tegrastats`): ______

**Stop here if no masks appear.** Isaac can't do anything without them.

## 5. FoundationPose engines (only if missing in stage 1)

Takes 10–15 min and needs ≥ 7.5 GB free GPU memory. **Stop the ZED and yolo_2d first.** Commands: ref, [Models and TensorRT engines](foundationpose_isaac_reference.md#models-and-tensorrt-engines) (download the ONNX models first if they're missing). Run this inside the Isaac container.

Start it early and do other stages while it builds.

## 6. Full pipeline

ZED running again, then:
```bash
ros2 launch tensor_detector foundationpose.launch.py \
  robot:=$ROBOT camera:=$CAM \
  target_class:=bottle yolo_model_path:=$YOLO_MODEL \
  launch_foundationpose:=true models_dir:=$MODELS_DIR \
  mesh_file_path:=$MESH \
  object_config_path:=$(ros2 pkg prefix tensor_detector)/share/tensor_detector/config/foundationpose_owala.yaml \
  debug:=true
```
Add `zed_rgb_topic:=... zed_camera_info_topic:=...` (stage 2) and `zed_qos:=SENSOR_DATA` (stage 3) if needed.

In the log, check:
- Both components load, with no `Failed to load` errors.
- The engines load. The mesh loads: expect a gray-texture warning, not `Found texture path`, because the mesh has no texture.

Then check the wiring:
```bash
ros2 param get /foundationpose symmetry_axes       # ['z_full'], proves the YAML matched
ros2 topic info /foundationpose/rgb8               # converter pub + FP sub
ros2 topic info /foundationpose/segmentation       # yolo_2d pub + FP sub
ros2 topic info $DEPTH                             # FP among the subscribers
ros2 topic hz /foundationpose/output               # with the bottle in view
```
→ Output rate: ______ Estimation latency: ______ GPU memory (`tegrastats`): ______

**If nothing comes out and nothing errors,** check in this order:
1. Timestamps (stage 3).
2. Image sizes: RGB, depth and mask must all be 1920×1200.
3. QoS on each hop.
4. Whether masks are arriving at all.

Look in `/tmp/foundationpose` (the debug output) for what the node did get.

## 7. Afterwards

- Record the → answers in [foundationpose_architecture.md](foundationpose_architecture.md): the Milestones (M1 and M2 pending items) and open questions 1–2. Tick the matching items in the ref [Checklist](foundationpose_isaac_reference.md#checklist).
- Update the launch file's defaults if the robot's ZED wrapper names or namespace differ from the current defaults.
- Update this procedure with anything that was wrong or missing.
