# FoundationPose bring-up on the Orin

Procedure for the robot-side checks of M1 (`yolo_2d` on the real camera) and the first M2 bring-up (ZED → format converter → FoundationPose). The design is in [foundationpose_architecture.md](foundationpose_architecture.md). Topics, parameters and formats are in [foundationpose_isaac_reference.md](foundationpose_isaac_reference.md), cited below as "ref".

How to use it:
- Run the stages in order. Each one ends with a stop condition, because later stages depend on it. If one fails, fix it before moving on.
- The `→` lines are blanks to fill in during the session. Afterwards, move the answers into the design doc (see [stage 7](#7-afterwards)) so this file stays a reusable procedure.
- **Everything is installed natively on the Orin, without Docker.** Isaac ROS comes from NVIDIA's apt repository and runs next to the ROS Humble install from [riptide_setup](https://github.com/osu-uwrt/riptide_setup). NVIDIA's own docs assume their Docker container (`run_dev.sh`); skip those steps and use the native equivalents here. The `component_container_mt` in the launch file is a ROS process, not a Docker container.

**Which stages need the camera:**

| Stage | Camera needed? |
| --- | --- |
| 1, 1a, 1b, 1c: inventory, workspace, Isaac install, PyTorch | No |
| 2, 3, 4: ZED topics, formats, `yolo_2d` on the real camera | Yes |
| 5: TensorRT engines | No. Best done with no camera running, since it needs ≥ 7.5 GB free GPU memory |
| 6: full pipeline | Partly. Loading and wiring can be checked without data; output needs the camera (or a recording) |

## Before the session

- [x] Branch `foundationpose-integration` pushed
- [ ] Orin set up with [riptide_setup](https://github.com/osu-uwrt/riptide_setup) (`setup.bash`)
- [ ] YOLO weights `yolo26s-seg.pt` (not in git) on a USB stick, or confirm the Orin can download it
- [ ] Optional: FoundationPose ONNX models downloaded ahead of time (ref: [Models and TensorRT engines](foundationpose_isaac_reference.md#models-and-tensorrt-engines))

## 0. Set paths once per terminal

Fill in after stages 1–2, then paste into every new terminal. `ISAAC_ROS_WS` is set in `~/.bashrc` by stage 1b.

```bash
export ROBOT=tempest            # from ros2 topic list
export CAM=ffc
export YOLO_MODEL=~/models/yolo26s-seg.pt
export MESH=~/osu-uwrt/development/software/src/riptide_perception/meshes/owala_textured/owala.obj   # in the repo
export MODELS_DIR=${ISAAC_ROS_WS}/isaac_ros_assets/models/foundationpose
```

## 1. Platform inventory (design doc open question 1)

```bash
cat /etc/nv_tegra_release                                    # L4T: expect R36, REVISION 4.x
lsb_release -sr                                              # Ubuntu: expect 22.04
dpkg -l | grep -E "nvidia-jetpack |nvinfer|tensorrt|vpi" | head   # JetPack components installed?
sudo nvpmodel -q                                             # power mode
grep -m3 "define ZED_SDK_.*_VERSION" /usr/local/zed/include/sl/Camera.hpp   # ZED SDK version (no file = not installed)
cd ~/osu-uwrt/development/dependencies/src/zed_ros2_wrapper && git describe --tags --always; cd -   # wrapper version
dpkg -l | grep -i isaac-ros-foundationpose                   # Isaac installed?
python3 -c "import ultralytics; print(ultralytics.__version__)"
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```
→ L4T: R36 REVISION 4.7 (apt source `r36.4`), within release-3.2's JetPack 6.1/6.2 range. Ubuntu: 22.04 JetPack pkgs: runtime only after flashing, no `nvidia-jetpack`, no `nvcc`/`trtexec`. `nvidia-jetpack` 6.2.1+b38 available (also 6.2+b77, 6.1+b123). Dev parts installed individually without OpenCV (stage 1a): yes TensorRT: 10.3.0.30 VPI: 3.2.4 CUDA: 12.6.68 (`cuda-nvcc-12-6`) cuDNN: 9.3.0.75 Power mode: `MODE_30W` (id 2), not MAXN Orin module (`cat /proc/device-tree/model`): ______ ZED SDK: 5.5.1 (installed after riptide_setup; build for L4T 36.4) Wrapper: `v5.5.0-22-gf134486` (riptide_setup clones `master`; post-5.1 topic names) Isaac FP installed: 3.2.14 (stage 1b) Ultralytics: 8.4.211 (stage 1c) torch / CUDA: 2.10.0 / True

**Compatibility (checked 2026-11-12):**
- **Isaac ROS release-3.2** is the last release for ROS 2 Humble. It supports Jetson Orin on **JetPack 6.1 and 6.2**, which NVIDIA identifies as `R36 (release), REVISION: 4.x` in `/etc/nv_tegra_release`. Isaac ROS 4.x targets ROS 2 Jazzy and 5.x targets ROS 2 Lyrical (both Ubuntu 24.04, JetPack 7), so they don't fit this robot's Humble stack.
- **JetPack 6.2.2 (L4T 36.5)** exists, but the release-3.2 docs only list 6.1 and 6.2. If the Orin reports `REVISION: 5.x`, the release-3.2 packages are untested on it. Note it, try the install, and watch for CUDA/TensorRT/VPI version conflicts in the apt output.
- riptide_setup only supports Ubuntu 22.04. If `lsb_release` shows 24.04, the Orin was flashed with JetPack 7 and none of this applies; stop and talk to the team.

**Stop if L4T isn't R36.** Everything after this assumes JetPack 6.

## 1a. Workspace from riptide_setup

riptide_setup's `install_dev.bash` keeps going after a failed build, so check what it left behind:

```bash
find ~/osu-uwrt/development -name COLCON_IGNORE      # expect only titan_firmware
cd ~/osu-uwrt/development/software/src/riptide_perception && git branch --show-current
```

- **`COLCON_IGNORE` in `tensor_detector` or the ZED packages:** the script adds these when `nvidia-smi` isn't found (`package_disable.py zed` and `tensor_detector`). Some JetPack versions don't include `nvidia-smi`, so this can happen even though the Orin has a GPU. Delete those `COLCON_IGNORE` files.
- **Branch:** riptide_setup clones `riptide_perception` on `master`. Run `git fetch && git checkout foundationpose-integration`.
- **ZED SDK:** riptide_setup doesn't install it, and `zed_ros2_wrapper` won't build without it. Install the SDK build for the Orin's L4T from the [ZED SDK release page](https://www.stereolabs.com/developers/release) (it lists builds for JetPack 6.1/6.2 (L4T 36.4) and 6.2.2 (L4T 36.5)). To pick the SDK version, don't go by the README's "Prerequisites" line, which is out of date (it says v5.2 in wrapper 5.4 and 5.5). The wrapper checks the version when the node starts: `zed_components/src/include/sl_version.hpp` accepts SDK 4.2 up to the wrapper's own `WRAPPER_MAJOR.WRAPPER_MINOR`, and the node refuses to start outside that range. Use the newest SDK release that isn't newer than the wrapper. riptide_setup clones the wrapper from `master`, so its version may differ from the v5.4.0 the ref doc was written against.
- **JetPack developer packages missing** (from stage 1: no `nvidia-jetpack`, no `/usr/local/cuda/bin/nvcc`, no `/usr/src/tensorrt/bin/trtexec`). A minimal flash installs only the runtime libraries. The ZED wrapper needs the CUDA toolkit (its README's fix for `Specify CUDA_TOOLKIT_ROOT_DIR` errors), and stage 5 needs `trtexec`.
  - **Don't install the full `nvidia-jetpack` / `nvidia-jetpack-dev`.** They include NVIDIA's OpenCV 4.8 (`nvidia-opencv-dev`), which removes Ubuntu's OpenCV 4.5.4 `-dev` packages that ROS Humble (`cv_bridge`, image transport plugins) is built against. Seen on the Orin, 2026-11-12: a simulated install showed 19 `Remv libopencv-*-dev` lines.
  - Instead, list the parts with `apt-cache depends nvidia-jetpack-dev nvidia-jetpack-runtime`, and install the CUDA, cuDNN, TensorRT and VPI ones (e.g. `nvidia-cuda-dev nvidia-cudnn-dev nvidia-tensorrt-dev nvidia-vpi-dev`) without the OpenCV one. Simulate first (`apt-get install -s`). There should be no `Remv` lines, no OpenCV, and no `nvidia-l4t-*` upgrades (those change the kernel and bootloader).
  - Then add `/usr/local/cuda/bin` to `PATH` and `/usr/local/cuda/lib64` to `LD_LIBRARY_PATH` in `~/.bashrc`, and check `nvcc --version` and that `/usr/src/tensorrt/bin/trtexec` exists.
- **Build** `dependencies`, then `software`, from each workspace root (not `src/`). Fix any failure before stage 1b.

→ COLCON_IGNOREs: none on `tensor_detector` or ZED packages (only `titan_firmware`, the in-repo `riptide_teleop` / `matlab_msg_gen` ones, and colcon's own in `build`/`install`/`log`), so nothing to remove. Branch: `foundationpose-integration` checked out. Both workspaces build: yes, after installing the dev packages and ZED SDK 5.5.1 (`ros2 pkg prefix` finds `zed_components` and `tensor_detector`)

## 1b. Install Isaac ROS (native, no Docker)

Steps from the release-3.2 [Isaac Apt Repository](https://nvidia-isaac-ros.github.io/v/release-3.2/getting_started/isaac_apt_repository.html) and [Compute Setup](https://nvidia-isaac-ros.github.io/v/release-3.2/getting_started/hardware_setup/compute/index.html) pages, without the Docker parts.

1. **Power mode: optional, and not needed to install or build anything.** NVIDIA's Compute Setup recommends `sudo /usr/sbin/nvpmodel -m 0` (MAXN) and `sudo /usr/bin/jetson_clocks` for best performance. Leave the robot's mode alone unless the team decides otherwise: it's the robot's power and thermal budget, and `nvpmodel` persists across reboots. Measurements in later stages (fps, latency) depend on the mode, so record which mode they were taken in.
2. **Locale:** `locale` must show UTF-8. The ROS install from riptide_setup normally already did this.
3. **Add the Isaac apt repository.** This line covers all `release-3.x` packages:
   ```bash
   wget -qO - https://isaac.download.nvidia.com/isaac-ros/repos.key | sudo apt-key add -
   grep -qxF "deb https://isaac.download.nvidia.com/isaac-ros/release-3 $(lsb_release -cs) release-3.0" /etc/apt/sources.list || \
   echo "deb https://isaac.download.nvidia.com/isaac-ros/release-3 $(lsb_release -cs) release-3.0" | sudo tee -a /etc/apt/sources.list
   sudo apt-get update
   ```
4. **Simulate the install before running it.** The Isaac repository carries its own builds of some `ros-humble-*` packages, and this Orin runs the team's whole stack:
   ```bash
   sudo apt-get install -s ros-humble-isaac-ros-foundationpose ros-humble-isaac-ros-image-proc | grep -E "^(Inst|Remv)"
   ```
   Read the list. To see only upgrades, filter for `grep -E "^Inst \S+ \["`: apt prints the old version in brackets right after the package name, while new packages have none. New `isaac-ros`/`gxf`/`nitros` packages are expected. **Upgrades or removals of core ROS packages, CUDA, TensorRT or VPI are not:** stop and raise them with the team before installing.
5. **Install:** the same command without `-s`. Then, in a new terminal: `ros2 pkg prefix isaac_ros_foundationpose` and `ros2 pkg prefix isaac_ros_image_proc` should both print a path under `/opt/ros/humble`.
6. **Install CV-CUDA (not pulled in by apt).** The FoundationPose GXF extension links against `libcvcuda.so.0`. NVIDIA's container installs CV-CUDA by hand in `isaac_ros_common`'s `Dockerfile.base`, so a native install has to do the same with the same version (v0.5.0-beta for release-3.2). Without it, the `foundationpose` component fails with `loadExtensions Error: GXF_EXTENSION_FILE_NOT_FOUND`, and the line above it reads `libcvcuda.so.0: cannot open shared object file`.
   ```bash
   mkdir -p ~/cvcuda_debs && cd ~/cvcuda_debs
   wget https://github.com/CVCUDA/CV-CUDA/releases/download/v0.5.0-beta/nvcv-lib-0.5.0_beta_DP-cuda12-aarch64-linux.deb
   wget https://github.com/CVCUDA/CV-CUDA/releases/download/v0.5.0-beta/nvcv-dev-0.5.0_beta_DP-cuda12-aarch64-linux.deb
   sudo dpkg -i nvcv-lib-0.5.0_beta_DP-cuda12-aarch64-linux.deb nvcv-dev-0.5.0_beta_DP-cuda12-aarch64-linux.deb
   sudo ldconfig
   ldd /opt/ros/humble/share/gxf_isaac_foundationpose/gxf/lib/libgxf_isaac_foundationpose.so | grep "not found"   # no output
   ```
   If `ldd` still reports `libcvcuda`, add the folder `dpkg -L nvcv-lib | grep "\.so"` shows to `/etc/ld.so.conf.d/cvcuda.conf` and rerun `sudo ldconfig`.
7. **Set `ISAAC_ROS_WS`.** NVIDIA's container sets it; a native install has to set it in `~/.bashrc`. The launch file's `models_dir` default is built from it, and falls back to the container path `/workspaces/isaac_ros-dev` when it's unset. Use NVIDIA's non-SSD location so their commands work unchanged:
   ```bash
   mkdir -p ~/workspaces/isaac_ros-dev
   echo 'export ISAAC_ROS_WS=${HOME}/workspaces/isaac_ros-dev/' >> ~/.bashrc
   source ~/.bashrc
   ```
   If the Orin has a separate SSD mount, put it there instead (NVIDIA's convention is `/mnt/nova_ssd/workspaces/isaac_ros-dev`). The engines and quickstart assets take a few GB.
8. **Rebuild the `dependencies` workspace** so the ZED wrapper sees NITROS. In the `zed_components` CMake output, look for ` * ISAAC ROS Nitros transport is available`. With NITROS available, the wrapper has no `/compressed` topic by default (ref: [NITROS and the compressed topic](foundationpose_isaac_reference.md#nitros-and-the-compressed-topic)), so stage 2 will probably need `input_type:=raw`.

→ Repo added: yes. Simulated install changed core packages? No: 39 new packages, no removals, no upgrades; the only non-Isaac package is `ros-humble-magic-enum` (new). Installed: `isaac_ros_foundationpose` 3.2.14, `isaac_ros_image_proc` 3.2.10 (`ros2 pkg prefix` found both after the install finished). `ISAAC_ROS_WS`: `~/workspaces/isaac_ros-dev/` (root is already NVMe, 191 GB free; no separate SSD mount). ONNX models downloaded (refine 66 MB, score 61 MB). Wrapper found NITROS: yes (` * ISAAC ROS Nitros transport is available`; all 11 `dependencies` packages built, `zed_components` now installed). Expect no `/compressed` topic in stage 2 unless the team ZED config sets `debug.disable_nitros: true`.

## 1c. PyTorch and Ultralytics for `yolo_2d`

`pip install ultralytics` pulls a PyTorch from PyPI that doesn't use the Jetson GPU. The wheels come from the "Run on JetPack 6.1" section of the [Ultralytics Jetson guide](https://docs.ultralytics.com/guides/nvidia-jetson/) (JetPack 6, Python 3.10). NVIDIA's own wheels are on the [PyTorch for Jetson](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html) page.

This procedure differs from the guide in two ways:
- **Install the Jetson torch wheels before `ultralytics`.** pip then sees torch is already installed and doesn't pull the PyPI build, so nothing needs replacing afterwards.
- **Pin `"numpy<2"`.** Humble's `cv_bridge` and `rclpy` are built against NumPy 1, and NumPy 2 breaks them at import.

```bash
python3 --version                                   # 3.10.x
mkdir -p ~/wheels && cd ~/wheels
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/torch-2.10.0-cp310-cp310-linux_aarch64.whl
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/torchvision-0.25.0-cp310-cp310-linux_aarch64.whl
wget https://developer.download.nvidia.com/compute/cudss/0.7.1/local_installers/cudss-local-tegra-repo-ubuntu2204-0.7.1_0.7.1-1_arm64.deb

# cuDSS (needed by torch 2.10.0, per the guide)
sudo dpkg -i cudss-local-tegra-repo-ubuntu2204-0.7.1_0.7.1-1_arm64.deb
sudo cp /var/cudss-local-tegra-repo-ubuntu2204-0.7.1/cudss-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update && sudo apt-get -y install cudss

# torch first, then ultralytics
pip install --user ./torch-2.10.0-cp310-cp310-linux_aarch64.whl ./torchvision-0.25.0-cp310-cp310-linux_aarch64.whl "numpy<2"
pip install --user ultralytics "numpy<2"
```
The guide also reboots after installing Ultralytics. That isn't needed for these checks.

Check:
```bash
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # 2.10.0, True
python3 -c "import numpy, cv_bridge, ultralytics; print(numpy.__version__, ultralytics.__version__)"   # numpy 1.x
pip list 2>/dev/null | grep -iE "^(torch|torchvision|numpy|ultralytics|opencv)"
```
If `torch` shows a version other than 2.10.0, `ultralytics` replaced it with the PyPI build. Reinstall the wheel with `--force-reinstall --no-deps`.
Then test `yolo_2d` on a still image (no camera needed). Publish a photo of the bottle as a camera and run the launch file with Isaac off:
```bash
mkdir -p ~/models && cd ~/models && python3 -c "from ultralytics import YOLO; YOLO('yolo26s-seg.pt')"
sudo apt install ros-humble-image-publisher ros-humble-compressed-image-transport
ros2 run image_publisher image_publisher_node ~/bottle.jpg --ros-args -p publish_rate:=2.0
ros2 launch tensor_detector foundationpose.launch.py \
  target_class:=bottle yolo_model_path:=$HOME/models/yolo26s-seg.pt \
  image_topic:=/image_raw/compressed            # or: input_type:=raw raw_image_topic:=/image_raw
ros2 topic hz /detections_2d --window 20          # should match publish_rate
ros2 topic echo --once /foundationpose/segmentation --field width   # photo width
```
Ignore the first ~20 s: the first frame includes CUDA and model start-up. To time YOLO without ROS, loop `YOLO(...)(img)` over the photo in Python and print `results[0].speed`. If raw is much slower or more irregular than compressed, run `ros2 topic hz /image_raw`: a low rate there means messages are lost in transport, not in `yolo_2d`.

→ torch: 2.10.0 (Jetson wheel), torchvision 0.25.0, numpy 1.26.4, ultralytics 8.4.211, cuDSS 0.7.1. CUDA available: True `yolo_2d` on a still image works: yes with `input_type:=compressed` (steady 2.000 Hz at `publish_rate` 2, mask 768×998 matches the photo). fps: YOLO alone (yolo26s-seg `.pt`, 768×998 photo, `MODE_30W`) ~50 ms/frame after warm-up (inference ~36 ms), so ~20 fps; first frame ~3 s.

**Finding (2026-11-13): raw images over default Fast DDS drop frames.** With `RMW_IMPLEMENTATION` unset (Fast DDS) and `net.core.rmem_max` = 1 MB, the 2.3 MB `/image_raw` from `image_publisher` arrived irregularly: `yolo_2d` got 0.5 to 1.5 Hz, and even `ros2 topic hz /image_raw` saw 0.1 Hz with gaps up to 25 s. The compressed topic was a steady 2 Hz. The ZED's raw RGB and depth are ~9.2 MB each, so this affects `input_type:=raw` and every cross-process ZED → Isaac hop. The team's `remote_launch` runs the robot with `rmw_zenoh_cpp` (router config in the release repo), so repeat the raw test under Zenoh before tuning anything.

**Zenoh result (2026-11-13): raw transport is steady.** Same test with `RMW_IMPLEMENTATION=rmw_zenoh_cpp` in every terminal and `rmw_zenohd` running: `/image_raw` (2.3 MB) a steady 2 Hz, and `/detections_2d` with `input_type:=raw` 2.000 Hz (period 0.494–0.506 s, std dev ~3 ms). Every terminal needs the same `RMW_IMPLEMENTATION`: a node started without it runs on Fast DDS and is invisible to the Zenoh nodes (`ros2 topic hz` then reports "does not appear to be published yet"). Run `ros2 daemon stop` after switching. ZED-sized raw (~9.2 MB, the photo resized to 2200×1400): `/image_raw` and `/detections_2d` both a steady 2 Hz.

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

**Timestamp parity (design doc open question 2).** `check_stamps.py` records the stamps on raw RGB, compressed RGB, depth and camera_info for a few seconds, then reports how many raw RGB frames each one matched exactly:
```bash
ros2 run tensor_detector check_stamps.py --ros-args \
  -r image:=$RGB -r image/compressed:=$RGB/compressed \
  -r depth:=$DEPTH -r camera_info:=/$ROBOT/$CAM/zed_node/<rgb camera_info topic>
```
- **PASS** (exit code 0): every compared frame matched exactly on every topic.
- **FAIL:** read the gaps. A gap of about one frame period (67 ms at 15 fps) is a frame dropped in transit. That's harmless if it's rare. Small consistent gaps (µs to a few ms) on most frames mean the stamps really differ.
- **0 msgs:** wrong topic name, incompatible QoS, or not published. On `image/compressed`, that's the NITROS case from stage 2. With `input_type:=raw`, ignore the compressed line.
- Add `-p duration:=30.0` for a longer sample.

→ Compressed / depth / camera_info matched: ______ / ______ / ______ **Stop if depth or camera_info really differ.** With `sync_threshold` 0, FoundationPose silently produces nothing.

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

Takes 10–15 min and needs ≥ 7.5 GB free GPU memory. **Stop the ZED and yolo_2d first.** This stage doesn't need the camera, so it can run any time after stage 1b. Commands: ref, [Models and TensorRT engines](foundationpose_isaac_reference.md#models-and-tensorrt-engines) (download the ONNX models first if they're missing). `trtexec` is the host's, from the JetPack TensorRT install (`/usr/src/tensorrt/bin/trtexec`). The engines only work with the TensorRT version that built them, so rebuild them after any JetPack or TensorRT upgrade.

→ TensorRT version used: 10.3.0.30. Both engines built and `trtexec` reported PASSED (at `MODE_30W`). Build time:75sec and 430sec 

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
- The engines load. The mesh loads with `[MeshStorage] Found texture path`. A gray-texture warning instead means the `.png` isn't next to the `.mtl`.

Then check the wiring:
```bash
ros2 param get /foundationpose symmetry_axes       # ['z_full'], proves the YAML matched
ros2 topic info /foundationpose/rgb8               # converter pub + FP sub
ros2 topic info /foundationpose/segmentation       # yolo_2d pub + FP sub
ros2 topic info $DEPTH                             # FP among the subscribers
ros2 topic hz /foundationpose/output               # with the bottle in view
```
→ Output rate: ______ Estimation latency: ______ GPU memory (`tegrastats`): ______

**Before the camera is attached,** run the same launch command with the ZED off. The log checks (components, engines, mesh, texture) and the first three wiring checks still apply, since topics and parameters exist without data. For `$DEPTH`, the subscriber shows up even with no publisher. Only the output rate needs real input: the camera, or a recorded bag/SVO of the ZED topics.

→ Without camera: components load ______ engines load ______ texture found ______ wiring OK ______

Result (2026-11-13, ZED off, `rmw_zenoh_cpp`, `MODE_30W`, wrapper 5.5 topic names, `input_type:=raw`): components load **yes** (after installing CV-CUDA, stage 1b step 6). Engines load **yes**: `[foundationpose] Node was started` came ~50 s after `Initializing and running GXF graph`, which is mostly TensorRT engine loading, so allow about a minute before expecting output. Texture found: **not shown**, because no mesh or texture line appears at the default log level. Check the rendered mesh in `/tmp/foundationpose` during the camera session instead. Wiring OK: **yes**. `symmetry_axes` = `['z_full']`. `/foundationpose/rgb8` and `/foundationpose/segmentation` each have 1 publisher and 1 subscriber. ZED depth and `camera_info` each have 1 subscriber and 0 publishers (camera off). No errors in the log. Isaac ROS 3.2 loads and connects under Zenoh.

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
