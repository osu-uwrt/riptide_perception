# FoundationPose perception pipeline: design decisions

Status: agreed design for the perception overhaul. M1 is implemented and verified on a development VM (no GPU or camera); its robot checks are pending (see Milestones). All paths are relative to `riptide_perception/`. Topic names are relative to each node's namespace and can be remapped at launch.

Each statement is labeled as one of:
- **Requirement**: a project requirement.
- **Verified**: checked in source or primary documentation (see References).
- **Decided**: an agreed design choice that can still be revised if new evidence appears.
- **Open**: needs an experiment or a decision.

## Scope

- **Requirement:** reliable camera-relative full-6DoF object observations from ZED → detection/segmentation → FoundationPose. Position plus heading is not enough.
- **Requirement:** this work covers perception only. `riptide_mapping` is not modified. The pipeline's output is designed so mapping *could* consume it later without changes to the contract.
- **Decided:** build the FoundationPose pipeline first. After that, add a geometry-based pose path for this year's simple props (plane fit plus in-plane axes from object features) that publishes the same output topic. That path can replace expensive inference wherever geometry is enough.
- **Decided:** the legacy detector (`tensor_detector/src/yolo_orientation.py` and its `detected_objects` output, services, and camera switching) stays untouched. Autonomy trees call its services (`set_camera_is_dfc`, `set_slalom_type`, `set_table_pair_enabled`). The new pipeline is additive and runs instead of the legacy detector, never alongside it, because both run YOLO on the same GPU.
- **Decided:** start with one camera, one target object, and estimation before tracking.

## Verified facts the design relies on

- **Camera:** two ZED X Minis (ffc, dfc) in one component container. ffc runs HD1200 at 15 fps with `NEURAL_PLUS` depth, `max_depth` 4.0, and a custom OpenCV calibration file (`riptide_core/riptide_hardware/cfg/ffc_config.yaml`).
- **Camera timestamps:** the ZED wrapper publishes RGB and depth from the same grab with the same timestamp, unless `use_pub_timestamps` is enabled.
- **Isaac version:** on ROS 2 Humble, Isaac ROS is pinned to `release-3.2`. Later releases target newer ROS distributions.
- **Isaac inputs:** FoundationPose expects `rgb8` color, `32FC1` depth, a `mono8` mask, and camera info. It synchronizes these internally (`FoundationPoseSynchronization`, `sync_threshold` default 0, i.e. exact timestamp match). ZED depth defaults to `32FC1` meters (`openni_depth_mode: false`). ZED color is `bgra8`, so it needs format conversion.
- **Isaac performance:** the 3.2 benchmark reports FoundationPose *estimation* at 1.54 fps and 780 ms latency at 720p on AGX Orin. Tracking is reported to be much faster. Current Isaac documentation cites about 7 GB peak GPU memory. **Measured on this Orin** (`MODE_30W`, FP32, quickstart bag at 640×480): ≈ 3.6 s of engine time per estimate with the GPU otherwise idle, and ~7.5 s end to end with `yolo_2d` running on every frame (details: Isaac reference, Models and TensorRT engines).
- **Isaac parameters:** `symmetry_axes`, `min_depth`/`max_depth`, `mesh_file_path`, `discard_old_messages`, `pose_estimation_timeout_ms`, and a debug TF frame name (`tf_frame_name`, default `fp_object`).
- **Legacy detector behavior we deliberately do not reuse:**
  - It merges all instance contours into one union mask.
  - It pairs RGB with the nearest depth frame already received (up to 1 s away) instead of the exact partner.
  - Its plane-fit orientation fixes only the face normal. Rotation about the normal comes from a camera axis, not from the object.

## Data flow

```
ZED ffc ── rgb (bgra8) ──► ImageFormatConverter (Isaac, → rgb8) ─┐
        ── depth (32FC1, same stamp) ────────────────────────────┤
        ── camera_info ───────────────────────────────────────────┼─► FoundationPose (exact sync inside)
        ── rgb ──► yolo_2d ── foundationpose/segmentation ────────┘          │ mesh-frame Detection3DArray
                      └─────── detections_2d                                  ▼
                                                     pose adapter ──► object_pose_detections
```

- **Decided:** RGB, depth, and camera info go straight from the ZED (through format conversion) into Isaac. We do **not** write our own RGB/depth synchronizer or relay node. The ZED already gives exact pairs and Isaac already syncs exactly. Relaying large images through Python adds copies and latency without adding any guarantee.
- **Decided:** `yolo_2d` reads the ZED's compressed color stream (`rgb/image_rect_color/compressed`) by default, which keeps full-resolution frames cheap to move into Python. Open question 2 tracks confirming that its stamps match the raw RGB/depth Isaac receives.
- **Decided:** `input_type: raw` switches `yolo_2d` to the raw RGB topic (`sensor_msgs/Image`) instead. This is the fallback for when the ZED wrapper runs with NITROS and publishes no `/compressed` topic (see the Isaac reference). It costs about 9 MB per HD1200 frame copied into Python, against a few hundred KB compressed, so compressed stays the default. It also makes open question 2 trivial: `yolo_2d` then reads the exact message Isaac's format converter receives.
- **Decided:** the only input we produce for Isaac is the mask. It inherits the source RGB header, so Isaac's exact sync pairs it with the matching RGB/depth.
- **Decided:** publishing a mask is how we request a pose estimate. If no mask is published for a frame, Isaac makes no attempt on it. Later, throttling masks will be the rate-control mechanism.

## Interface contracts

### `detections_2d`: `vision_msgs/msg/Detection2DArray` (Decided, M1)

- Every instance from one processed frame, before any task-specific grouping.
- Pixel coordinates are in the original image (Ultralytics' `boxes.xyxy` and `masks.xy` already undo letterboxing). `bbox` holds center and size (`vision_msgs` 4.1.1: `bbox.center.position.x/y`, `size_x`, `size_y`).
- The array header and each detection's header are copied from the source image, giving the capture stamp and the optical `frame_id`. We never use the current time or a `frame_id` built from a string.
- One hypothesis per detection: `class_id` is the class *name*, and `score` is detector confidence as a Python `float`.
- An empty array means the frame was processed and nothing was found. A skipped frame produces no message.
- Detection IDs are not persistent track identities.

### `foundationpose/segmentation`: `sensor_msgs/msg/Image` (Decided, M1)

- Remapped to Isaac's `pose_estimation/segmentation`.
- `mono8`, 0 is background and 255 is the object. Same width and height as the RGB image Isaac receives.
- Header **identical** to the source RGB header.
- Contains only the selected instance. It is rasterized from that instance's **polygon**, never from a union of detections.
- **Not published** when the target is absent or its polygon is empty.
- Target selection: the highest-confidence detection whose class equals the `target_class` parameter.
- A frame whose target mask covers fewer than `minimum_mask_pixels` pixels gets no mask, so FoundationPose makes no attempt on it.

### `object_pose_detections`: `vision_msgs/msg/Detection3DArray` (Decided contract, M3)

- One array per camera frame. Header is the capture stamp and the camera optical frame from the source image, not the time inference finished.
- `results[0].hypothesis.class_id` is the Riptide object name. `score` is detector confidence, which is not pose quality.
- `results[0].pose.pose` is `T_camera_object` in a documented per-object frame. The adapter applies a fixed mesh→object transform per object.
- `results[0].pose.covariance` is 6×6: position in m² and rotation in rad², both about the axes of the header frame. Rotations made unobservable by symmetry get large variance.
- `id` is left empty. Persistent identity belongs downstream.
- A failure publishes nothing. We never publish an identity or zero pose as a placeholder.
- This is the topic the later geometry-based path also publishes.

### QoS (Decided, M1)

- `yolo_2d` subscribes **best effort, keep-last depth 1**. A best-effort subscriber connects to both reliable and best-effort publishers. Depth 1 means a slow inference always picks up the newest frame instead of working through a queue of stale ones.
- Both `yolo_2d` publishers are **reliable**: `foundationpose/segmentation` with depth 1 (a late mask is rarely pairable, and each one is about 2.3 MB at HD1200) and `detections_2d` with depth 10. A reliable publisher can feed both reliable and best-effort subscribers.

### Removed or deferred interfaces

- **Decided, removed:** the custom `riptide_perception_msgs` package. Nothing depends on it, Isaac consumes a plain `mono8` `Image`, and full-resolution per-instance masks are about 2.3 MB each at HD1200. It can be recovered from commit `25cceea` if simultaneous multi-instance output is needed later.
- **Decided, not adopted (for now):** `yolo_ros` and `yolo_msgs`.
  - `main` (5.0.0) is C++/ONNX Runtime, and its prebuilt GPU binaries are x64-only.
  - `4.7.1` (`eb11e81`) is the last Python/Ultralytics release. It mostly duplicates our wrapper and needs raw images.
  - Both versions represent masks as boundary polygons, which we would rasterize anyway.
  - Revisit this if we want built-in tracking IDs or lifecycle management.
- **Deferred:** the `set_target` service, target generations, target catalog, diagnostics status topic, and multiple simultaneous targets. M1 uses static parameters.

## Nodes and files

| File | Milestone | Responsibility |
| --- | --- | --- |
| `tensor_detector/src/detections_2d.py` | M1 | Pure functions with no ROS node and no Ultralytics import. They build the `Detection2DArray`, select the target index, and rasterize one polygon into the `mono8` mask with the source header. Unit-testable without a GPU. |
| `tensor_detector/test/test_detections_2d.py` | M1 | Registered with `ament_cmake_pytest`. Header copying, bbox center/size, mask size, values, and single instance, target selection, absent target, empty or out-of-bounds polygons. |
| `tensor_detector/src/yolo_2d.py` | M1 | Thin node: one fixed camera, one image subscription chosen at startup by `input_type` (`compressed`: `CompressedImage` on relative topic `image/compressed`; `raw`: `Image` on relative topic `image`; both remapped at launch, best effort, depth 1), `YoloModel`, the builders above, and two publishers. No depth, TF, services, camera switching, or task logic. Class names come from the model (`model.names`). Parameters, all read-only and validated at startup (an invalid value stops the node): `model_path` (required, full path to the file), `target_class` (required, must be one of the model's classes), `confidence` (0.25, [0.01, 1]), `iou_threshold` (0.7, [0, 1]), `minimum_mask_pixels` (300, ≥ 1), `input_type` (`compressed`, one of `compressed` / `raw`). The model must be a segmentation model. A frame that fails to decode is skipped with a throttled warning and produces no output. |
| Isaac launch for one object | M2 | ZED → format converter → FoundationPose for one mesh, with the mask remapped from `yolo_2d`. Isaac's debug TF frame is disabled or renamed so it cannot collide with mapping's `<object>_frame`. |
| Pose adapter node | M3 | Converts Isaac's mesh-frame output into `object_pose_detections`: mesh→object transform, class name, covariance model, symmetry canonicalization. |

`tensor_detector/CMakeLists.txt` already installs every `src/*.py`, so new modules need no build changes. CMake collects that list when it configures, so a brand-new file may need `colcon build --cmake-force-configure` once.

## Bring-up test object (Decided)

- Use an off-the-shelf `yolo26s-seg` (COCO class `bottle`) and a water-bottle mesh. This tests sync, masks, FoundationPose, latency, and tracking before retraining.
- The model is selected by parameter. Team weights and `yolo_orientation.yaml` are not overwritten. Keep downloaded `.pt` files out of the repository, since `.gitignore` does not cover them.
- **This validates the pipeline plumbing, not underwater accuracy.**
- Bottle-specific checks:
  - The mesh must be in meters (check its bounding box against a ruler).
  - Use an opaque bottle, because clear plastic defeats stereo depth and render-and-compare.
  - Rotation about the long axis is ambiguous; this exercises `symmetry_axes`.
- Capture and replay happen on the robot. The ZED X Mini uses GMSL, and SVO2 playback needs the ZED SDK and an NVIDIA GPU. Record a few SVO2 sessions (static, slow motion, occluded, leaving and re-entering the frame) for repeatable replay.

## Milestones

1. **M1:** `detections_2d.py` and its tests (VM), then `yolo_2d.py` (robot).
   - Done: `detections_2d.py`, 15 unit tests, and `yolo_2d.py`. On the VM (CPU inference, still images through `image_publisher`), detections arrive every processed frame, empty when nothing is in view. The mask appears only when the target is present, both outputs carry the input header, invalid parameters stop the node at startup, and there is no backlog at 15 Hz input.
   - Pending on the robot: the ZED topic's QoS (`ros2 topic info -v`), open questions 1 and 2, and real latency and fps on the Orin.
2. **M2:** Isaac FoundationPose graph for the bottle on the Orin; measure latency, fps, and GPU memory with both ZEDs running.
   - Done: `tensor_detector/launch/foundationpose.launch.py` and `tensor_detector/config/foundationpose_owala.yaml`. ZED topics are absolute, built from `robot`, `camera` and per-topic arguments (wrapper < 5.1 names by default). `launch_foundationpose:=false` (the default) runs `yolo_2d` alone; `true` adds one container with the format converter and FoundationPose. Per-object values (`symmetry_axes`, `tf_frame_name`) come from `object_config_path`. Verified on the VM: the `yolo_2d`-only launch, and that the object YAML's parameters match the FoundationPose node.
   - Done on the Orin without a camera (2026-11-13): native Isaac ROS 3.2 install (plus CV-CUDA), engines, the full launch under `rmw_zenoh_cpp` with correct wiring, and a first pose on NVIDIA's quickstart bag that lands on the object. Raw-image transport across processes is reliable under Zenoh, so `input_type: raw` is viable.
   - Pending on the robot (procedure: [foundationpose_orin_bringup.md](foundationpose_orin_bringup.md)): everything that needs the camera. ZED topic names and the `/compressed` topic (NITROS), QoS, timestamp parity, engine build, the first pose output, then latency, fps and GPU memory. The test mesh (`meshes/owala_textured/`) has a flat three-color texture (body, lid band, lid) sampled from a photo of the bottle, not a scanned texture.
3. **M3:** pose adapter and the `object_pose_detections` contract.
4. **M4:** tracking/selector, rate control via mask throttling, behavior under backlog.
5. **Later:** retrained model with whole-object classes, real prop meshes, and the geometry-based pose path.

## Open questions and experiments

1. **Platform inventory on the robot:** Orin model, JetPack/L4T, ZED SDK and wrapper versions, TensorRT version, and whether the installed Ultralytics version loads and exports YOLO26.
2. **Timestamp parity:** confirm identical `header.stamp` on raw RGB, depth, and the RGB topic `yolo_2d` subscribes to (compressed if used). Confirm the effective `use_pub_timestamps` is false.
3. **Isaac sync behavior:** whether RGB/depth frames that never get a mask are dropped cleanly or back up the queue, and how long a mask may lag its frame (`discard_old_messages`, queue depth).
4. **Compute budget:** FoundationPose alongside two `NEURAL_PLUS` depth pipelines and YOLO, at full and reduced resolution. Set pass criteria before running. First data, camera-free at `MODE_30W`: ≈ 3.6 s per estimate on an idle GPU, ~7.5 s with `yolo_2d` running on every frame. GPU contention roughly doubles estimation time, and the power mode (612 MHz, 4 TPCs) is the main gap from NVIDIA's 780 ms at MAXN. Whether the robot can run a higher power mode is a team question.
5. **Mask type:** instance polygon (default) versus bounding box, measured by pose accuracy.
6. **Detector classes for retraining:** FoundationPose needs masks that cover the object its mesh describes. Current classes are mostly symbols (`fire`, `blood`, `compass`), not whole props, so the retrained model needs whole-object classes. Those same classes serve the geometry-based path.
7. **Object-frame convention for real props:** must be agreed with mapping and autonomy contributors before M3 is used downstream. Inferred from mapping code, not confirmed: published object frames have +X along the outward face normal and +Z up.
8. **Tracking correlation:** repeated tracker outputs are not independent observations. Measure their autocorrelation before any downstream fusion uses them.
9. **Mask resolution vs Isaac input resolution:** the mask matches the full-resolution compressed image. If M2 runs FoundationPose at reduced resolution (see 4), the mask must match Isaac's input instead: subscribe `yolo_2d` to the same resized stream, or resize the mask.
10. **Segmentation check for exported engines:** `yolo_2d` rejects non-segmentation models by checking the model's task. For a `.pt` file the task comes from the file. For a TensorRT `.engine`, `YoloModel` supplies the task, so a detection-only engine would pass the check and never produce masks.
11. **`minimum_mask_pixels` tuning:** the default of 300 comes from the bottle's approximate projected size at `max_depth` 4 m. Tune it on the robot, and set it per object for real props.

## References

- Isaac ROS FoundationPose (release-3.2): https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_pose_estimation/tree/release-3.2/isaac_ros_foundationpose
- Isaac ROS FoundationPose API docs: https://nvidia-isaac-ros.github.io/v/release-3.2/repositories_and_packages/isaac_ros_pose_estimation/isaac_ros_foundationpose/index.html
- ZED RGB/depth publishing: https://github.com/stereolabs/zed-ros2-wrapper/blob/master/zed_components/src/zed_camera/src/zed_camera_component_video_depth.cpp
- yolo_ros: https://github.com/mgonzs13/yolo_ros (tags `4.7.1`, `main` at 5.0.0)
