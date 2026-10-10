# riptide_mapping
This package will handle our mapping system, which takes data in from our 3D Vision System, and uses it to update our model for the current position of all 
our objects of interest, which is then published for use in our navigation and task code systems.

## How the map is estimated
`config/config.yaml` `init_data` is a dead-reckoned layout of the course. The odometry frame drifts away from it, and
the layout itself is sometimes off, so every map-parented object is estimated as
`R(course yaw) * prior + offset + deviation` (`riptide_mapping2/map_estimator.py`):
- **offset** (`map -> offset` TF): where the course sits in odometry. It random-walks with distance traveled
  (`estimator.drift_*`), so old sightings count for less than fresh ones.
- **deviation**: how far the layout is off for that one object (`estimator.layout_std_*`). Heavy-tailed
  (`estimator.layout_robust_scale`): a prior that disagrees with the other props is *released* and stops moving the map.
  The node logs when that happens.
- **course yaw**: start-heading misalignment between the layout and odometry (`estimator.course_yaw_std_deg`). The
  AprilTag calibration keeps it small; two or more props pin it down.

Each map object is tracked from its detections. After `estimator.gate_min_updates` detections the track becomes a
**fix**: where that object was, at the odometry distance it was last seen. Retargeting the object freezes the fix and starts
a new one, so a revisit (the gate on the way home) measures drift. A small robust least-squares solve over all fixes gives
the offset history, the deviations and the course yaw. One sighting alone can't tell "this prior is off" from "the vehicle
drifted", so it is split by their uncertainties. With two or more props, the props that agree win.

Objects parented to another mapping frame (torpedo holes, gate halves, table items) are filtered in that frame.

**Targets:** `mapping_target` takes one object or a comma-separated list (`"torpedo, bin"`). Only those classes are fused; with
no target, only the closest object is fused. Setting a target resets its covariance to the prior (it keeps its position), so a
covariance check needs fresh detections.

**Covariance** on `mapping/<object>` is the uncertainty of the estimate (offset included for map objects), not the spread of
recent detections. It shrinks as detections accumulate.

**Published frames** slide toward the estimate at `output.max_speed` / `output.max_yaw_rate_deg`, so a correction never steps a
frame the controller is tracking. The distance a frame still has to go is added to its published covariance. Service-driven
moves (seeding, table centering, bin fit, reset, prior changes) jump.
