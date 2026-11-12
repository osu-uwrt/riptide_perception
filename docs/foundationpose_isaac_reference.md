# Isaac ROS FoundationPose integration reference

Reference for the M2 integration: ZED → format converter → Isaac ROS FoundationPose, with the segmentation mask from `yolo_2d`. Design decisions live in `foundationpose_architecture.md`. This file collects the facts the launch configuration depends on: topics, parameters, formats, file layout and known pitfalls.

**Sources, checked 2026-11-11:**
- Isaac ROS `release-3.2` source: `isaac_ros_pose_estimation` (FoundationPose nodes, GXF graph, mesh loader) and `isaac_ros_image_pipeline` (format converter), plus the release-3.2 FoundationPose docs page.
- ZED wrapper: the workspace's `zed_ros2_wrapper` checkout (tag `v5.4.0`). **The robot may run a different version.** See [ZED facts](#zed-facts).
- Where the docs and the source disagree, the source wins. Discrepancies are marked.

## Contents

1. [File structure](#file-structure)
2. [Graph and topic wiring](#graph-and-topic-wiring)
3. [FoundationPoseNode reference](#foundationposenode-reference)
4. [ImageFormatConverterNode reference](#imageformatconverternode-reference)
5. [Models and TensorRT engines](#models-and-tensorrt-engines)
6. [Mesh requirements](#mesh-requirements)
7. [ZED facts](#zed-facts)
8. [Tracking (M4)](#tracking-m4)
9. [Multiple objects](#multiple-objects)
10. [Checklist](#checklist)
11. [Links](#links)

---

## File structure

### In this repo

```
riptide_perception/
├── docs/
│   ├── foundationpose_architecture.md
│   ├── foundationpose_isaac_reference.md      ← this file
│   └── foundationpose_orin_bringup.md         ← robot bring-up procedure
└── tensor_detector/
    ├── launch/
    │   ├── tensorrt.launch.py                 (legacy, untouched)
    │   └── foundationpose.launch.py           ← M2 launch file
    ├── config/
    │   └── foundationpose_<object>.yaml       ← per-object params (symmetry_axes, tf_frame_name), passed as object_config_path
    └── src/
        ├── detections_2d.py
        └── yolo_2d.py
```

- `tensor_detector/CMakeLists.txt` already installs `launch/` and `config/`. No build changes are needed.
- `<exec_depend>` entries for `isaac_ros_foundationpose` and `isaac_ros_image_proc` in `tensor_detector/package.xml` make rosdep fail on machines without Isaac ROS. The alternatives are to leave them out, or to put the launch file in a separate package that only the robot builds.

### On the robot (not in git)

Paths follow NVIDIA's convention so their commands work unchanged. Every path is a launch argument, so the exact location is not fixed.

```
${ISAAC_ROS_WS}/isaac_ros_assets/
├── models/foundationpose/
│   ├── refine_model.onnx            (downloaded; same for every object)
│   ├── score_model.onnx             (downloaded; same for every object)
│   ├── refine_trt_engine.plan       (built on the Orin with trtexec)
│   └── score_trt_engine.plan        (built on the Orin with trtexec)
└── meshes/<object>/
    ├── <object>.obj                 ← mesh_file_path
    ├── <object>.mtl                 (must reference the texture with map_Kd)
    └── <object>_texture.png
```

- The ONNX models and engines are **object-independent**. Only the mesh changes per object.
- Engines are tied to the GPU and TensorRT version. Build them on the target machine. They can't be copied from another one.
- The YOLO weights live wherever `yolo_2d`'s `model_path` points. They are not committed.

---

## Graph and topic wiring

```
/<robot>/ffc/zed_node/<RGB topic>  (bgra8) ──► ImageFormatConverterNode ──► rgb8 ──► pose_estimation/image ─────┐
/<robot>/ffc/zed_node/depth/depth_registered (32FC1) ─────────────────────────────► pose_estimation/depth_image ┤
/<robot>/ffc/zed_node/<RGB camera_info> ───────────────────────────────────────────► pose_estimation/camera_info ┼─► FoundationPoseNode
yolo_2d ── foundationpose/segmentation (mono8) ─────────────────────────────────────► pose_estimation/segmentation┘        │
                                                                                                                         ▼
                                                                         pose_estimation/output (vision_msgs/Detection3DArray)
                                                                         pose_estimation/pose_matrix_output (TensorList, for tracking)
```

`<RGB topic>` depends on the ZED wrapper version (see [ZED facts](#zed-facts)):

| Wrapper | Rectified RGB | RGB camera_info | Compressed (`yolo_2d` default input) |
| --- | --- | --- | --- |
| < 5.1 (used by the legacy node) | `rgb/image_rect_color` | `rgb/camera_info` | `rgb/image_rect_color/compressed` |
| ≥ 5.1 (workspace checkout is 5.4) | `rgb/color/rect/image` | `rgb/color/rect/camera_info` | `rgb/color/rect/image/compressed` |

Depth is `depth/depth_registered` in both. `riptide_hardware/launch/zed.launch.py` places the ffc under `/<robot>/ffc/zed_node/`. Its `robot` launch argument defaults to `tempest`.

**Remaps** (left: the node's built-in name, right: this pipeline's topic):

| Node | Built-in topic | Remapped to |
| --- | --- | --- |
| ImageFormatConverterNode | `image_raw` (input) | ZED rectified RGB |
| ImageFormatConverterNode | `image` (output) | a pipeline topic, e.g. `foundationpose/rgb8` |
| FoundationPoseNode | `pose_estimation/image` | the converter output |
| FoundationPoseNode | `pose_estimation/depth_image` | ZED `depth/depth_registered` |
| FoundationPoseNode | `pose_estimation/camera_info` | ZED RGB camera_info |
| FoundationPoseNode | `pose_estimation/segmentation` | `foundationpose/segmentation` |
| FoundationPoseNode | `pose_estimation/output` | unchanged, or e.g. `foundationpose/output` (M3 adapter input) |
| yolo_2d | `image/compressed` | ZED compressed RGB (launch argument `image_topic`; used when `input_type:=compressed`, the default) |
| yolo_2d | `image` | ZED rectified RGB (launch argument `raw_image_topic`; used when `input_type:=raw`) |

All three images and the mask must have the **same width and height**. At HD1200 that is 1920×1200.

---

## FoundationPoseNode reference

- Package `isaac_ros_foundationpose`, plugin `nvidia::isaac_ros::foundationpose::FoundationPoseNode`.
- A composable NITROS node, loaded into a `ComposableNodeContainer` (`component_container_mt` in NVIDIA's examples).

### Topics (`src/foundationpose_node.cpp` L55–77)

| Direction | Topic | Type | Required format |
| --- | --- | --- | --- |
| in | `pose_estimation/image` | `sensor_msgs/Image` | `rgb8` |
| in | `pose_estimation/depth_image` | `sensor_msgs/Image` | `32FC1`, meters |
| in | `pose_estimation/camera_info` | `sensor_msgs/CameraInfo` | |
| in | `pose_estimation/segmentation` | `sensor_msgs/Image` | `mono8` |
| out | `pose_estimation/output` | `vision_msgs/Detection3DArray` | `frame_id` copied from the RGB image |
| out | `pose_estimation/pose_matrix_output` | `isaac_ros_tensor_list_interfaces/TensorList` | used by tracking |

The four inputs are synchronized internally (`FoundationPoseSynchronization`). By default they need an **exact** timestamp match.

### Parameters (source defaults, `foundationpose_node.cpp` L154–202)

| Parameter | Type | Default | Value for this pipeline | Notes |
| --- | --- | --- | --- | --- |
| `mesh_file_path` | string | `textured_simple.obj` | absolute path to the object's `.obj` | **Changeable at runtime** (reloads the mesh). |
| `texture_path` | string | — | not needed | In the docs and NVIDIA's launch files, but **not declared by the 3.2 node**. The texture comes from the `.mtl` (see [Mesh requirements](#mesh-requirements)). |
| `min_depth` | float | 0.1 | 0.1 | Points closer than this are dropped. |
| `max_depth` | float | 4.0 | 4.0 (matches ffc `max_depth`) | The docs say 0.4 and the source says 4.0. Set it explicitly. |
| `refine_iterations` | int | 1 | 1 | More iterations are more accurate and slower. |
| `symmetry_axes` | string[] | `[]` | bottle: `['<long axis>_full']` | Format `axis_angle`, e.g. `x_180`, or `axis_full` (sampled every 30°). Axes are in the **mesh** frame. Changeable at runtime. |
| `symmetry_planes` | string[] | `[]` | — | Deprecated in favor of `symmetry_axes`. |
| `fixed_axis_angles` | string[] | `[]` | — | e.g. `x_30`. Pins an orientation angle. Changeable at runtime. |
| `fixed_translations` | string[] | `[]` | — | e.g. `z_0.5`. Overrides the estimated translation. Changeable at runtime. |
| `refine_model_file_path` | string | `/tmp/refine_model.onnx` | assets path | |
| `refine_engine_file_path` | string | `/tmp/refine_trt_engine.plan` | assets path | Per the docs, if the engine is missing it is generated from the ONNX at startup (slow). Prebuilding with trtexec is preferred. |
| `score_model_file_path` | string | `/tmp/score_model.onnx` | assets path | |
| `score_engine_file_path` | string | `/tmp/score_trt_engine.plan` | assets path | |
| `refine_input_tensor_names` | string[] | `[]` | `['input_tensor1', 'input_tensor2']` | Fixed by the model, as in NVIDIA's example. |
| `refine_input_binding_names` | string[] | `[]` | `['input1', 'input2']` | Fixed by the model. |
| `refine_output_tensor_names` | string[] | `[]` | `['output_tensor1', 'output_tensor2']` | Fixed by the model. |
| `refine_output_binding_names` | string[] | `[]` | `['output1', 'output2']` | Fixed by the model. |
| `score_input_tensor_names` | string[] | `[]` | `['input_tensor1', 'input_tensor2']` | Fixed by the model. |
| `score_input_binding_names` | string[] | `[]` | `['input1', 'input2']` | Fixed by the model. |
| `score_output_tensor_names` | string[] | `[]` | `['output_tensor']` | Fixed by the model. |
| `score_output_binding_names` | string[] | `[]` | `['output1']` | Fixed by the model. |
| `sync_threshold` | int | 0 | 0 (exact) | Nanoseconds. Raise only if stamps provably differ. |
| `discard_old_messages` | bool | false | false initially | Design doc open question 3. |
| `discard_msg_older_than_ms` | int | 1000 | | Only used when `discard_old_messages` is true. |
| `pose_estimation_timeout_ms` | int | 5000 | | |
| `tf_frame_name` | string | `fp_object` | e.g. `fp_<object>` | Debug TF frame. Must not collide with mapping's `<object>_frame`. Changeable at runtime. |
| `configuration_file` | string | `foundationpose_model_config.yaml` | default | Model internals (160×160 crops, 252 hypotheses). |
| `debug` / `debug_dir` | bool / string | false / `/tmp/foundationpose` | true during bring-up | Writes intermediate results to disk. |

### QoS parameters (source L207–218)

Each input's QoS is a parameter. The value is a profile name, and `<name>_depth` sets the queue depth.

| Parameter | Default |
| --- | --- |
| `color_qos`, `color_qos_depth` | `DEFAULT` (reliable, keep last 10) |
| `depth_qos`, `depth_qos_depth` | `DEFAULT` |
| `color_info_qos`, `color_info_qos_depth` | `DEFAULT` |
| `segmentation_qos`, `segmentation_qos_depth` | `DEFAULT` |

Profiles (`isaac_ros_common/src/qos.cpp`): `SYSTEM_DEFAULT`, `DEFAULT`, `PARAMETER_EVENTS`, `SERVICES_DEFAULT`, `PARAMETERS`, `SENSOR_DATA` (best effort).

- A reliable subscriber **does not connect** to a best-effort publisher. Any ZED topic published best effort needs its input set to `SENSOR_DATA`.
- `yolo_2d`'s mask publisher is reliable, so `DEFAULT` works for `segmentation_qos`.

---

## ImageFormatConverterNode reference

- Package `isaac_ros_image_proc`, plugin `nvidia::isaac_ros::image_proc::ImageFormatConverterNode`.
- Topics: input `image_raw`, output `image`.

| Parameter | Type | Default | Value for this pipeline |
| --- | --- | --- | --- |
| `encoding_desired` | string | `""` | `rgb8` |
| `image_width` | int | 1280 | 1920 (HD1200) |
| `image_height` | int | 720 | 1200 (HD1200) |
| `num_blocks` | int | 40 | default |
| `input_qos` / `output_qos` | string | `DEFAULT` | `SENSOR_DATA` on the input if the ZED publishes best effort |

- `bgra8` input is supported.
- `ResizeNode` (same package) with `encoding_desired: 'rgb8'` converts and resizes in one node. NVIDIA's RealSense example uses it this way. This is relevant if FoundationPose runs at reduced resolution (design doc open question 9).

---

## Models and TensorRT engines

Commands from the release-3.2 quickstart. They run inside the Isaac ROS dev container on the Orin, except the download, which can run anywhere.

**Download the ONNX models.** They are portable between machines.
```bash
mkdir -p ${ISAAC_ROS_WS}/isaac_ros_assets/models/foundationpose && \
cd ${ISAAC_ROS_WS}/isaac_ros_assets/models/foundationpose && \
wget 'https://api.ngc.nvidia.com/v2/models/nvidia/isaac/foundationpose/versions/1.0.0_onnx/files/refine_model.onnx' -O refine_model.onnx && \
wget 'https://api.ngc.nvidia.com/v2/models/nvidia/isaac/foundationpose/versions/1.0.0_onnx/files/score_model.onnx' -O score_model.onnx
```

**Build the engines.** This takes ~10–15 min on AGX Orin and needs **≥ 7.5 GB free GPU memory**, so other GPU workloads (ZED, YOLO) should be stopped first.
```bash
/usr/src/tensorrt/bin/trtexec --onnx=${ISAAC_ROS_WS}/isaac_ros_assets/models/foundationpose/refine_model.onnx \
  --saveEngine=${ISAAC_ROS_WS}/isaac_ros_assets/models/foundationpose/refine_trt_engine.plan \
  --minShapes=input1:1x160x160x6,input2:1x160x160x6 \
  --optShapes=input1:1x160x160x6,input2:1x160x160x6 \
  --maxShapes=input1:42x160x160x6,input2:42x160x160x6

/usr/src/tensorrt/bin/trtexec --onnx=${ISAAC_ROS_WS}/isaac_ros_assets/models/foundationpose/score_model.onnx \
  --saveEngine=${ISAAC_ROS_WS}/isaac_ros_assets/models/foundationpose/score_trt_engine.plan \
  --minShapes=input1:1x160x160x6,input2:1x160x160x6 \
  --optShapes=input1:1x160x160x6,input2:1x160x160x6 \
  --maxShapes=input1:252x160x160x6,input2:252x160x160x6
```
On TensorRT 10.3+, the engines run in FP32 (NVIDIA disabled FP16 because of precision loss).

**Install** (inside the container started by `isaac_ros_common/scripts/run_dev.sh`):
```bash
sudo apt-get update && sudo apt-get install -y ros-humble-isaac-ros-foundationpose
```

**Optional install check with NVIDIA's data.** After downloading the quickstart assets (docs, "Download Quickstart Assets"):
```bash
ros2 launch isaac_ros_foundationpose isaac_ros_foundationpose.launch.py \
  mesh_file_path:=.../Mustard/textured_simple.obj texture_path:=.../Mustard/texture_map.png \
  refine_engine_file_path:=.../refine_trt_engine.plan score_engine_file_path:=.../score_trt_engine.plan
ros2 bag play -l ${ISAAC_ROS_WS}/isaac_ros_assets/isaac_ros_foundationpose/quickstart.bag/
```
The quickstart bag also needs NVIDIA's RT-DETR detector to produce masks.

---

## Mesh requirements

Verified in `gxf_isaac_foundationpose/gxf/foundationpose/mesh_storage.cpp` (release-3.2), unless noted otherwise.

- **Loaded with assimp. Only the first mesh in the file is used** (`scene->mMeshes[0]`). A model with several parts (cap, body, label) must be merged into one mesh, for example in MeshLab or Blender.
- **The texture comes from the `.mtl`**: the diffuse map (`map_Kd`), resolved relative to the `.obj`'s folder. If none is found, the mesh is rendered flat gray (128) with only a warning, which weakens render-and-compare. A successful load logs `[MeshStorage] Found texture path`.
- **The loader re-centers the mesh at its bounding-box center** (it subtracts the center of the min/max vertex box). NVIDIA's docs also state that "FoundationPose expects the origin frame to be at the center of the mesh". **Consequence for M3:** the output pose is of the bounding-box-centered frame, not the mesh file's original origin. The pose adapter's mesh→object transform must include that offset.
- **Units: meters** (design doc requirement). The bounding box should match the physical object.
- Texture coordinates (UVs) are required for texturing. A mesh without UVs gets the gray fallback.
- High polygon counts slow rendering. NVIDIA's docs link a MeshLab simplification tutorial.
- `symmetry_axes` refers to axes of the mesh frame, so the object's symmetry axis must be identified in the mesh itself.

---

## ZED facts

| Fact | Value in the workspace's v5.4.0 source | How to confirm on the robot |
| --- | --- | --- |
| Topic names | Renamed in **5.1.0** (breaking): `rgb/image_rect_color` → `rgb/color/rect/image` | `ros2 topic list \| grep ffc`. The legacy node and the design doc use the **pre-5.1** names. |
| Publisher QoS | `rclcpp::QoS(10)`: reliable, keep last 10 (`sl_types.hpp` `QOS_QUEUE_SIZE`) | `ros2 topic info -v <topic>` |
| Depth | `32FC1`, meters (`openni_depth_mode: false`) | `ros2 topic echo --once <depth> --field encoding` |
| RGB encoding | `bgra8` | same check on RGB |
| Timestamps | camera grab time unless `debug.use_pub_timestamps: true` (default false) | `ros2 param get /<robot>/ffc/zed_node debug.use_pub_timestamps` |
| Resolution | HD1200, `pub_resolution: NATIVE` → 1920×1200 | `--field width` / `height` |
| Frame rate | 15 fps (`grab_frame_rate`) | `ros2 topic hz` |

### NITROS and the compressed topic

- A ZED wrapper **built where Isaac ROS NITROS packages are installed** enables NITROS publishing by default (`debug.disable_nitros: false`).
- With NITROS enabled, the wrapper **does not create `image_transport` publishers**. There is then **no `/compressed` topic**, which is `yolo_2d`'s input (`zed_camera_component_video_depth.cpp`: image_transport publishers are only created `if (_nitrosDisabled)`).
- NITROS is also disabled automatically when the wrapper runs with intra-process communication.

Resolutions:
1. Set `debug.disable_nitros: true` in `ffc_config.yaml`. Topics stay standard, at the cost of zero-copy transport into Isaac.
2. Subscribe `yolo_2d` to the raw RGB topic instead: launch with `input_type:=raw`. `raw_image_topic` defaults to the ZED's rectified RGB. This keeps NITROS for Isaac, at the cost of about 9 MB per frame copied into Python.

---

## Tracking (M4)

Estimation (~780 ms per frame on AGX Orin) is the expensive step. Tracking starts from the previous pose and is much faster. NVIDIA's tracking configuration (`isaac_ros_foundationpose_tracking.launch.py`) adds two nodes to the container:

| Node | Plugin | Role |
| --- | --- | --- |
| Selector | `nvidia::isaac_ros::foundationpose::Selector` | Exact-syncs `image`, `depth_image`, `segmentation`, `camera_info`. In the estimation state it forwards them to `pose_estimation/*`, then switches to tracking and forwards to `tracking/*` with the last pose. Every `reset_period` ms (default 20000) it forces a new estimation. |
| FoundationPoseTrackingNode | `nvidia::isaac_ros::foundationpose::FoundationPoseTrackingNode` | Inputs `tracking/image`, `tracking/depth_image`, `tracking/camera_info`, `tracking/pose_input`. Outputs `tracking/output` (Detection3DArray) and `tracking/pose_matrix_output`. Its parameters are a subset of FoundationPoseNode's: mesh, depth limits, refine model and engine only, and `tf_frame_name`. All inputs use QoS depth 1. It loads its own refine engine. |

With the Selector, raw inputs go to the **Selector's** topics (`image`, `depth_image`, `segmentation`, `camera_info`), not to `pose_estimation/*`.

**Open design conflict:** the Selector's sync includes `segmentation` even in the tracking state. It forwards a frame to the tracker only if a mask arrived for that frame. This conflicts with the design doc's "a mask requests an estimate" contract and with throttling masks for rate control. Either `yolo_2d` publishes a mask every frame while tracking, or the pipeline needs its own selector.

---

## Multiple objects

**One `FoundationPoseNode` estimates one object.** Release-3.2 has no multi-object mode. The current docs and `main` show none either. Multiple objects are possible by running several nodes or by switching one node between objects, with the costs below.

### Per-object state vs. shared networks

The FoundationPose method splits into two parts:

| Part | What it is | Per object? |
| --- | --- | --- |
| Networks | The refine and score models (ONNX, or TensorRT engines). They are generic: they compare a rendered view of a mesh against the real image. | **No.** The same weights serve every object. There is no per-object training. |
| Object state | The mesh (vertices, texture, bounding-box center), symmetry settings, the pose hypotheses sampled for it, and the last pose (for tracking). | **Yes.** |

In NVIDIA's reference Python implementation these are separate objects. The networks and renderer are loaded once, and each object gets a lightweight estimator that holds its mesh and pose and references the shared networks. N objects are then a loop over N estimators. Network memory is paid once. Compute is paid N times, because each object runs its own estimation or tracking pass.

### Why the Isaac node is single-object

`FoundationPoseNode` is a ROS wrapper around a fixed **GXF graph** (`config/nitros_foundationpose_node.yaml`): a pipeline of compiled components wired together at load time.

```
sync → depth→pointcloud → sampling → render → refine_inference (TensorRT) → transform
     → render_score → score_inference (TensorRT) → decoder → pose output
```

From the release-3.2 graph file:
- **There is exactly one mesh slot.** A single `MeshStorage` component (`utils/mesh_storage`) holds the mesh. `sampling`, both `render` stages, `transform` and `decoder` all reference that one component.
- **The networks are inside the graph.** `refine_inference` and `score_inference` are `TensorRtInference` components within the graph, each loading its own engine. They are not a service several graphs can share. A second node is a second graph, with a second copy of both engines and their working buffers. No sharing is configured. Whether TensorRT or GXF shares anything implicitly is unverified.
- **The interfaces are single-object.** The mask is one binary `mono8` image (0 = background, 255 = object) with no instance IDs. The sync stage pairs one mask with each RGB/depth frame, and one pose comes out.

The TensorRT engines themselves are object-independent. The single-object limit comes from the graph's wiring, not from the compiled networks. Supporting several meshes in one graph would mean modifying the GXF graph and its C++ components.

### Options

| Option | How | Effort | Limitations |
| --- | --- | --- | --- |
| **A. One Isaac node per object, concurrently** | N `FoundationPoseNode`s in one container, each with its own `name`, `mesh_file_path`, `tf_frame_name` and mask topic (e.g. `foundationpose/<object>/segmentation`). `yolo_2d` publishes one mask per target class. RGB, depth and camera_info fan out to all nodes. | Low: launch configuration, plus per-class masks in `yolo_2d` | GPU **memory** grows by a full set of engines and buffers per object (NVIDIA cites ~7 GB peak for one node). With tracking, each object also has a tracking node with its own refine engine. GPU **time** grows per object: estimation is ~780 ms each on AGX Orin, and all nodes share the GPU with two `NEURAL_PLUS` ZED pipelines and YOLO. Feasibility depends on measurement. |
| **B. One Isaac node, object switched at runtime** | `mesh_file_path`, `symmetry_axes`, `fixed_*` and `tf_frame_name` have parameter callbacks and change without a restart, e.g. `ros2 param set /foundationpose_container/foundationpose mesh_file_path <path>`. Paired with changing `yolo_2d`'s `target_class` (the design doc's deferred `set_target` service). | Low | **One object at a time.** Each switch reloads the mesh and discards tracking state. Rapid cycling gives each object a fresh ~780 ms estimate per turn, so it is not concurrent tracking. Unverified: reload time, and whether in-flight frames are posed with the previous mesh (masks should stop during a switch). |
| **C. Custom node on NVIDIA's reference FoundationPose code** | Shared networks plus one estimator per object, masks from `yolo_2d` per class, publishing `object_pose_detections`. A variant runs the same ONNX models through TensorRT instead of PyTorch. | Medium | Memory is paid once. Compute is still paid per object, sequentially. PyTorch inference is slower than TensorRT (not measured on the Orin). The reference code's CUDA extensions (e.g. nvdiffrast) are difficult to build on Jetson. Synchronization, QoS and tracking logic become this project's responsibility. |
| **D. Multi-mesh fork of the Isaac graph** | Modify the GXF graph and C++ components so one graph holds several `MeshStorage`s and routes masks by instance. | High | C++/CUDA/GXF work against a pinned release. |

No option makes additional objects free. Every object needs its own estimation or tracking pass. The options differ in how memory scales and in how much of the pipeline this project maintains.

### Choosing between them

The limiting resource determines which options help:

- **Memory-limited** (concurrent nodes don't fit): sharing networks helps. That means option C, or D.
- **Compute-limited** (nodes fit, but fps or latency is insufficient): restructuring barely helps, because the per-object work remains. The effective levers are a tracking-dominated duty cycle (M4), lower per-object update rates through mask throttling, running only task-relevant objects, reduced input resolution, and the geometry-based pose path planned in the design doc for simple props.

Notes on memory and duty cycle:
- On Jetson, CPU and GPU share one memory pool. ZED depth pipelines, YOLO and the OS all draw from the same total.
- The ~7 GB figure is NVIDIA's peak, and runtime usage per node is unmeasured. The estimation node keeps its buffers allocated while idle, so a tracking-dominated duty cycle reduces compute, not memory.
- The Selector re-estimates every `reset_period` (default 20 s) whether or not the object was lost.

### Recommended path

1. Build M2 with one object. The `object_pose_detections` contract (a `Detection3DArray` with `class_id`) is already multi-object, so downstream consumers are unaffected by the choice.
2. Once one object works, add a second `FoundationPoseNode` with a different mesh and record GPU memory (`tegrastats`) and per-node fps, with both ZEDs and `yolo_2d` running. Repeat with tracking enabled.
   - Fits, with acceptable fps → option A.
   - Doesn't fit → option B, with C as the fallback if concurrent objects are required.
3. Establish how many props must be tracked **concurrently** in this year's tasks. If tasks involve one prop at a time, option B covers them.

---

## Checklist

- [ ] ZED topic names match the wrapper version on the robot (5.1 rename).
- [ ] The `/compressed` topic exists (see [NITROS and the compressed topic](#nitros-and-the-compressed-topic)).
- [ ] RGB, depth and mask are all 1920×1200, matching what Isaac receives.
- [ ] Stamps on compressed RGB, raw RGB and depth are identical (`sync_threshold` 0). A mismatch produces no output and **no error**.
- [ ] QoS is compatible on every hop (`ros2 topic info -v`).
- [ ] Engines were built on the Orin with the container's TensorRT, with ≥ 7.5 GB free GPU memory.
- [ ] The mesh is a single mesh, in meters, with a working `.mtl` texture (log shows `Found texture path`).
- [x] `max_depth` is set explicitly (docs/source default mismatch). Set to 4.0 in the launch file.
- [x] `tf_frame_name` is not `fp_object` and doesn't collide with mapping frames. `fp_owala` in the object YAML.
- [ ] `debug: true` for the first run, with output inspected in `debug_dir`.

---

## Links

- FoundationPose package source (release-3.2): https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_pose_estimation/tree/release-3.2/isaac_ros_foundationpose
  - `launch/isaac_ros_foundationpose.launch.py`: minimal example with an external mask, the closest to this pipeline.
  - `launch/isaac_ros_foundationpose_realsense.launch.py`: full camera pipeline with format conversion and mask resize.
  - `launch/isaac_ros_foundationpose_tracking.launch.py`: Selector and tracking.
  - `src/foundationpose_node.cpp`: parameters, topics, QoS.
  - `config/nitros_foundationpose_node.yaml`: the GXF graph.
- Mesh loader: https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_pose_estimation/blob/release-3.2/isaac_ros_gxf_extensions/gxf_isaac_foundationpose/gxf/foundationpose/mesh_storage.cpp
- Format converter: https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_image_pipeline/blob/release-3.2/isaac_ros_image_proc/src/image_format_converter_node.cpp
- Docs (quickstart, parameters, mesh tutorials): https://nvidia-isaac-ros.github.io/v/release-3.2/repositories_and_packages/isaac_ros_pose_estimation/isaac_ros_foundationpose/index.html
- ZED wrapper topic rename: `zed_ros2_wrapper/CHANGELOG.rst`, section v5.1.0.
- ROS 2 composition: https://docs.ros.org/en/humble/Tutorials/Intermediate/Composition.html
