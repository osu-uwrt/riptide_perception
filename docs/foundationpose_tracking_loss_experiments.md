# FoundationPose tracking-loss experiments

Status: plan, not started. Most of it needs the camera. Belongs to milestone M4 (tracking, selector, rate control) in [foundationpose_architecture.md](foundationpose_architecture.md).

## Goal

Decide, from data, how the pipeline detects that FoundationPose tracking has lost the object, and when it requests a new estimate. The output is:
1. A small set of loss signals.
2. How to combine them.
3. Thresholds tuned on recordings.
4. Evidence that the combination catches real losses quickly without false alarms during steady tracking.

## Why this is needed

- **Estimation is expensive. Tracking is cheap.** Measured on the Orin (2026-11-13, `MODE_30W`, FP32):
  - One estimate is ≈ 3.6 s of engine time on an idle GPU, and ~7.5 s end to end with `yolo_2d` running on every frame.
  - A single-pose refine pass, the core of one tracking step, is 9.7 ms.
  - Details: [Isaac reference, Models and TensorRT engines](foundationpose_isaac_reference.md#models-and-tensorrt-engines).

  The pipeline should therefore estimate rarely and track every frame.
- **The tracker never reports a loss.** Each tracking step refines the previous pose against the new frame and always outputs a pose. FoundationPose's own benchmark applies no re-initialization after loss [1]. Isaac's Selector only re-estimates on a timer (`reset_period`, default 20 s), so a lost object can produce up to 20 s of confident, wrong poses.
- **The stock Selector conflicts with gating YOLO.** It forwards a frame to the tracker only if a mask arrived for that frame (see [Isaac reference, Tracking (M4)](foundationpose_isaac_reference.md#tracking-m4)). Running YOLO only on request therefore needs a custom selector.

## Design decisions this plan builds on

- **Decided:** `yolo_2d` runs inference only on request. It subscribes to an "estimate now" topic. Any node may publish it: the selector (start-up, loss, periodic re-check), autonomy, mapping or a timer. Requests are not hard-coded into `yolo_2d`. Still to specify:
  - Request semantics. Proposed: "armed" until a mask is published, or until N frames or T seconds pass.
  - Message type: `std_msgs/Empty`, or a message carrying the target class.
  - QoS: reliable, depth 1.
  - What `/detections_2d` does while YOLO is idle (architecture interface contracts).
- **Decided:** tracking-aware logic (loss detection, triggers) lives in a separate selector node, not in `yolo_2d`.
- **Proposed:** a custom selector instead of NVIDIA's, because of the mask-per-frame conflict above.

## Related work

Every system found that builds on FoundationPose adds an external trigger, then re-estimates once the object is visible again. None of them is a built-in FoundationPose feature.

| Approach | Loss signal | Source |
| --- | --- | --- |
| 2D/3D consistency | A 2D segmenter runs alongside tracking. Re-register when a robust distance between the mask centroid and the pose's projected centroid exceeds a threshold | SuperPose [2] (SAM2) |
| Motion prediction | A Kalman filter predicts the pose. Re-register when the 2D tracker recovers the mask, and use the filter's estimate in the meantime | KARL [3]; FoundationPose++ [4] (2D tracker for x/y, depth at that pixel for z, Kalman filter for rotation) |
| State machine | Explicit tracking / uncertain / lost / recovering states with threshold-driven transitions | RGBTrack [5], RRTrack [6] |
| Reliability monitor and global re-detection | A separate tracker monitors pose reliability. Recovery candidates are verified before tracking resumes | OPRA [7], [8] |
| Filtering for fusion | Ensemble directional Kalman filter on FoundationPose outputs | EnDKF [9] |

Most of these are preprints or small open-source projects. Their claims have not been checked or reproduced here.

## Candidate loss signals

Cost is per tracked frame. All signals use data the pipeline already has.

| ID | Signal | Computation | Cost | Expected strength / weakness |
| --- | --- | --- | --- | --- |
| S1 | Pose jump | Translation and rotation change between consecutive tracked poses, against a physical limit (object + vehicle motion per frame) | Free | Catches divergence after fast motion. Misses slow drift onto a wrong surface |
| S2 | Off-image projection | Projected pose center (and projected box) outside the image or mostly clipped | Free | Catches the object leaving the frame. Nothing else |
| S3 | Depth disagreement | ZED depth at the projected center (median over a small window) against the pose's z, ignoring invalid depth | Free | Catches occlusion by a nearer object and drift off the object. Noisy in turbid water and on depth holes |
| S4 | Periodic 2D consistency | Every N frames the selector triggers one YOLO pass. Compare the mask and box with the tracked pose: centroid distance (robust, as in [2]) and IoU between YOLO's box and the projected box | One YOLO pass per N frames | The strongest independent check. Rate-limited by N, and needs the object to be detectable |
| S5 | Kalman innovation | Constant-velocity filter on the pose. Normalized innovation (measurement vs. prediction) | Negligible | Smooth version of S1. Also gives filtered output for fusion (architecture open question 8) |
| S6 | Tracker's own score | Whatever `score` contains in `tracking/output` | Free | Unknown. First check whether it carries any information |
| B0 | Baseline: stock Selector | NVIDIA's timer reset (`reset_period` 20 s, also try 5 s) | — | Reference that every other signal is compared against |

Combinations to evaluate: the free signals alone (S1+S2+S3), each of those with S4 at several N, and S5 in place of S1.

## Prerequisites

1. **Tracking runs on this Orin.** Add the Selector and `FoundationPoseTrackingNode` to the launch, following NVIDIA's `isaac_ros_foundationpose_tracking.launch.py` [10]. Check output rate, GPU memory (`tegrastats`) and per-frame tracking time with the ZED running.
2. **Mask lag tolerance** (architecture open question 3, second half). Find how long after its frame a mask may arrive and still be matched by Isaac's exact-stamp sync. Do this with live camera stamps: the quickstart bag has every stamp at 0. This bounds how late a triggered YOLO pass may be.
3. **Signal logger.** A node that subscribes to the tracked pose, depth, camera_info, and YOLO output when present. Each frame it logs S1–S6 with the frame stamp. It logs only and changes nothing in the pipeline.
4. **Event markers.** A topic (e.g. `/experiment/event`, `std_msgs/String`) recorded alongside the data. The operator publishes labels such as `occlusion_start` with `ros2 topic pub --once`.
5. **Offline YOLO reference.** YOLO on every frame of a recording, used as the visibility and location reference during analysis.

## Data collection (camera session)

### Recording format
Raw ZED RGB and depth are ~9.2 MB each per frame. At 15 fps that's ~275 MB/s, too much for `ros2 bag`. Recording options:
- **ZED SVO (preferred):** the ZED SDK's compressed recording. Replayed through the wrapper, it publishes the same topics as a live camera. Depth is recomputed on replay with the configured depth mode. Check the exact recording service and playback parameter names in the v5.5 wrapper README [11].
- **Short `ros2 bag` clips** of RGB, depth and camera_info, for cases where SVO replay differs from live behavior.
- **Always alongside, in a small `ros2 bag`:** `/tf`, `/tf_static`, `/experiment/event`, and the pipeline's own outputs when tracking is running live.

### Scenes
Use the Owala test bottle (`meshes/owala_textured/`). Record each scene for 20–60 s, and mark every event with a marker message.

| # | Scene | Events to mark | Tests |
| --- | --- | --- | --- |
| 1 | Bottle static, camera static | — | Baseline noise and false-alarm rate of every signal |
| 2 | Slow bottle or camera motion | `motion_start/stop` | Normal tracking, S1/S5 thresholds |
| 3 | Fast motion, increasing until tracking breaks | `fast_start/stop`, `visible_loss` | Divergence detection (S1, S4, S5) |
| 4 | Partial, then full occlusion by a hand or board | `occl_partial`, `occl_full`, `occl_end` | S3, S4. Behavior while hidden |
| 5 | Bottle leaves the frame and returns | `exit`, `enter` | S2. Re-estimate on return |
| 6 | Similar distractor (another bottle) enters, then the target is removed | `distractor_in`, `target_out` | Tracker locking onto the wrong object (S4 is the main check) |
| 7 | Range sweep 0.5–4 m (`max_depth` 4.0) | `range_<m>` | Depth-dependent noise and `minimum_mask_pixels` |
| 8 | Underwater / turbid, when available | Same as above | All thresholds under realistic depth noise |

Repeat scenes 3–6 at least three times each, so there is more than one event per type to evaluate.

## Analysis

1. **Labels.** From the markers plus the offline YOLO reference, mark each frame as *visible and tracked correctly*, *visible but tracking wrong*, or *not visible*. "Tracking wrong" = the projected pose center falls outside the YOLO mask, or the box IoU falls below a set threshold. Spot-check the labels visually in Foxglove.
2. **Run tracking offline** on each recording. Log S1–S6 per frame with the signal logger.
3. **Per signal and combination, measure:**
   - **Detection delay:** frames or seconds from the start of a loss (label) to the first alarm.
   - **Missed losses:** losses with no alarm before the object is visible again.
   - **False alarms per minute** during *visible and tracked correctly* frames.
   - **Cost:** added GPU time per frame (S4 at each N).
4. **Thresholds.** Choose them on part of the recordings and evaluate on the rest. Don't report the tuning recordings' numbers as results.
5. **Recovery.** For the chosen combination, close the loop on held-out recordings: alarm → trigger topic → YOLO → estimate → tracking resumes. Measure the time from loss to a correct pose again, and how many poses published during a loss were wrong.

## Pass criteria (to agree with the team before analysis)

Set values for the following before looking at the results:
- **Maximum detection delay** (e.g. ≤ N frames at 15 fps).
- **Maximum false-alarm rate** (each false alarm costs one ~3.6–7.5 s estimate).
- **Maximum number of wrong poses published per loss.**
- **GPU budget** for loss detection alongside two `NEURAL_PLUS` ZED pipelines (architecture open question 4).

## Outputs

- Chosen signals, combination and thresholds, written into the architecture doc as **Decided**.
- Selector behavior: states, triggers, timeout after a failed estimate (no output), retry or back-off.
- Recordings and event logs kept for regression testing of later changes.

## References

1. Wen et al., *FoundationPose: Unified 6D Pose Estimation and Tracking of Novel Objects*. https://arxiv.org/pdf/2312.08344
2. *SuperPose: Improved 6D Pose Estimation with Robust Tracking and Mask-Free Initialization*. https://arxiv.org/html/2409.19986v2
3. *KARL: Kalman-Filter Assisted Reinforcement Learner for Dynamic Object Tracking and Grasping*. https://arxiv.org/pdf/2506.15945
4. FoundationPose++ (teal024, GitHub). Project summary: https://www.sourcepulse.org/projects/27724760
5. *RGBTrack: Fast, Robust Depth-Free 6D Pose Estimation and Tracking*. https://arxiv.org/html/2506.17119v1
6. *RRTrack: Robust and Recoverable Object 6D* (preprint). https://www.alphaxiv.org/abs/2607.23669.md
7. OPRA, University of Bonn (RA-L 2026). https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/opra2026ral.pdf
8. *Robust 6-DoF Object Pose Tracking with Built-In Recovery under Occlusions and Rapid Object Motions* (preprint). https://www.alphaxiv.org/abs/2607.23468
9. *Pose Tracking with a Foundation Pose Model and an Ensemble Directional Kalman Filter* (EnDKF). https://www.newx.sg/paper/detail/ab25ab3e-49c1-11f1-b84c-00163e10baa7
10. Isaac ROS FoundationPose, release-3.2: docs https://nvidia-isaac-ros.github.io/v/release-3.2/repositories_and_packages/isaac_ros_pose_estimation/isaac_ros_foundationpose/index.html; tracking launch `isaac_ros_foundationpose/launch/isaac_ros_foundationpose_tracking.launch.py`. NVIDIA forum thread on tracking start-up delay: https://forums.developer.nvidia.com/t/isaac_ros_foundationpose-tracking-takes-approximately-20-seconds-to-show-pose-detection-results-upon-starting-rviz/324197
11. Stereolabs ZED ROS 2 wrapper (SVO recording and playback): https://github.com/stereolabs/zed-ros2-wrapper
