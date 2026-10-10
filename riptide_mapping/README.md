# riptide_mapping
This package will handle our mapping system, which takes data in from our 3D Vision System, and uses it to update our model for the current position of all 
our objects of interest, which is then published for use in our navigation and task code systems.

## How the map is estimated
`config/config.yaml` `init_data` is a dead-reckoned layout of the course. The odometry frame drifts away from it, so every
map-parented object is estimated as `prior + offset + deviation` in one Kalman filter (`riptide_mapping2/map_estimator.py`):
- **offset** (`map -> offset` TF): where the course sits in odometry. Shared by all map objects; its uncertainty grows with
  distance traveled (`estimator.drift_*`), so old sightings count for less than fresh ones.
- **deviation**: how far the layout is off for that one object (`estimator.layout_std_*`).

Seeing any map object refines the offset, and with it every other map object. Objects parented to another mapping frame
(torpedo holes, gate halves, table items) are filtered in that frame.

Detection noise grows with range, more along the camera ray than across it (`noise.*`). Detections taken while the vehicle
turns or moves get a larger error (`noise.timing_std`, from `odometry/filtered`). Back-to-back detections count for less
(`noise.correlation_time`). Once an object has `estimator.gate_min_updates` detections, detections far from its estimate are
rejected, which catches a lookalike that got the target's class. `estimator.gate_max_rejects` in a row reacquires the object.

**Targets:** `mapping_target` takes one object or a comma-separated list (`"torpedo, bin"`). Only those classes are fused; with
no target, only the closest object is fused. Setting a target resets its covariance to the prior (it keeps its position), so a
covariance check needs fresh detections.

**Covariance** on `mapping/<object>` is the uncertainty of the estimate (offset included for map objects), not the spread of
recent detections. It shrinks as detections accumulate.

**Published frames** slide toward the estimate at `output.max_speed` / `output.max_yaw_rate_deg`, so a correction never steps a
frame the controller is tracking. The distance a frame still has to go is added to its published covariance. Service-driven
moves (seeding, table centering, bin fit, reset, prior changes) jump.
