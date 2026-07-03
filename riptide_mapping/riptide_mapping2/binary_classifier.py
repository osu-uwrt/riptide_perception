from collections import deque
from dataclasses import dataclass, field
import numpy

# fraction of farthest cluster points dropped before averaging the centroid
# not a param because we really shouldn't need to tweak this (would likely do more harm than good)
CLUSTER_TRIM_QUANTILE = 0.90

# State machine for binary classifier
# stopped -> aquire_first -> track_first -> [frozen] -> aquire_second -> track_second -> stopped
# frozen is optional and can be skipped to go from track_first -> aquire_second directly
# stop() returns to stopped from any running state
class BinaryClassifierState:
    STOPPED = "stopped"                  # not running
    ACQUIRE_FIRST = "acquire_first"      # trying to find the first cluster to track
    TRACK_FIRST = "track_first"          # tracking the first cluster
    FROZEN = "frozen"                    # buffer is frozen, first is still tracking
    ACQUIRE_SECOND = "acquire_second"    # trying to find the second cluster
    TRACK_SECOND = "track_second"        # tracking the second cluster

# Defaults, overridden by the launch yaml
DEFAULTS = {
    "buffer_size": 500,                  # max detections held in the rolling buffer used for clustering
    "buffer_ttl_sec": 15.0,              # detections older than this are dropped, so a target we've left behind stops contributing
    "min_cluster_detections": 6,         # how many nearby detections it takes to lock an instance
    "cluster_radius_m": 0.15,            # neighbors within this radius count as the same cluster when finding an instance
    "assignment_gate_m": 0.20,           # once an instance is locked, a new detection farther than this from its centroid is rejected as not belonging to it
    "exclusion_radius_m": 0.30,          # while hunting for the second instance, detections within this radius of the first are thrown out so the first can't bleed into the second
    "min_instance_separation_m": 0.40,   # a candidate second instance must be at least this far from the first to be accepted
    "max_cluster_variance_m2": 0.02,     # reject a cluster whose spread exceeds this. 0 disables the check
    "centroid_ema_alpha": 0.15,          # Exponential Moving Average (EMA) rate for a locked centroid chasing new detections. lower = steadier & slower, higher = faster & noisier
    "use_3d_distance": False,            # cluster/measure distance in 3D (x,y,z) vs 2D (x,y). Keep False for top-down flat targets where z is the noisy axis
}

@dataclass
class BinaryClassifierParams:
    buffer_size: int = DEFAULTS["buffer_size"]
    buffer_ttl_sec: float = DEFAULTS["buffer_ttl_sec"]
    min_cluster_detections: int = DEFAULTS["min_cluster_detections"]
    cluster_radius_m: float = DEFAULTS["cluster_radius_m"]
    assignment_gate_m: float = DEFAULTS["assignment_gate_m"]
    exclusion_radius_m: float = DEFAULTS["exclusion_radius_m"]
    min_instance_separation_m: float = DEFAULTS["min_instance_separation_m"]
    max_cluster_variance_m2: float = DEFAULTS["max_cluster_variance_m2"]
    centroid_ema_alpha: float = DEFAULTS["centroid_ema_alpha"]
    use_3d_distance: bool = DEFAULTS["use_3d_distance"]

@dataclass
class DetectionSample:
    x: float
    y: float
    z: float
    score: float
    stamp_sec: float

    def point(self):
        return numpy.array([self.x, self.y, self.z], dtype=numpy.float64)

@dataclass
class AssignmentResult:
    accepted: bool = False
    target_name: str = ""
    reason: str = ""
    distance_m: float = None

class BinaryClassifier:
    def __init__(self, node=None, param_namespace="binary_classifier"):
        self.node = node
        self.param_namespace = param_namespace

        self.params = BinaryClassifierParams()
        self.samples = deque(maxlen=max(1, self.params.buffer_size))

        self.state = BinaryClassifierState.STOPPED
        self.class_name = ""
        self.instance1_name = ""
        self.instance2_name = ""

        self.instance1_centroid = None
        self.instance2_centroid = None

        self.declare_params()
        self.update_params()

    @property
    def running(self):
        return self.state != BinaryClassifierState.STOPPED

    @property
    def first_locked(self):
        return self.instance1_centroid is not None

    @property
    def second_locked(self):
        return self.instance2_centroid is not None

    @property
    def buffer_frozen(self):
        return self.state == BinaryClassifierState.FROZEN

    def param_name(self, name):
        return "{}.{}".format(self.param_namespace, name)

    def declare_params(self):
        if self.node is None:
            return
        # defaults come from the shared DEFAULTS dict; the launch-provided yaml overrides them
        self.node.declare_parameters(
            namespace="",
            parameters=[(self.param_name(name), value) for name, value in DEFAULTS.items()],
        )

    def update_params(self):
        if self.node is None:
            return

        self.params = BinaryClassifierParams(
            buffer_size=int(self.node.get_parameter(self.param_name("buffer_size")).value),
            buffer_ttl_sec=float(self.node.get_parameter(self.param_name("buffer_ttl_sec")).value),
            min_cluster_detections=int(self.node.get_parameter(self.param_name("min_cluster_detections")).value),
            cluster_radius_m=float(self.node.get_parameter(self.param_name("cluster_radius_m")).value),
            assignment_gate_m=float(self.node.get_parameter(self.param_name("assignment_gate_m")).value),
            exclusion_radius_m=float(self.node.get_parameter(self.param_name("exclusion_radius_m")).value),
            min_instance_separation_m=float(self.node.get_parameter(self.param_name("min_instance_separation_m")).value),
            max_cluster_variance_m2=float(self.node.get_parameter(self.param_name("max_cluster_variance_m2")).value),
            centroid_ema_alpha=float(self.node.get_parameter(self.param_name("centroid_ema_alpha")).value),
            use_3d_distance=bool(self.node.get_parameter(self.param_name("use_3d_distance")).value),
        )

        # re-wrap the existing samples in case buffer_size changed
        self.samples = deque(self.samples, maxlen=max(1, self.params.buffer_size))

    def start(self, class_name, instance1_name, instance2_name):
        self.update_params()

        class_name = class_name.strip()
        instance1_name = instance1_name.strip()
        instance2_name = instance2_name.strip()

        if self.running:
            return False, "BinaryClassifier is already running"

        if class_name == "":
            return False, "class_name cannot be empty"

        if instance1_name == "" or instance2_name == "":
            return False, "instance names cannot be empty"

        if instance1_name == instance2_name:
            return False, "instance names must be different"

        self.samples.clear()

        self.class_name = class_name
        self.instance1_name = instance1_name
        self.instance2_name = instance2_name

        self.instance1_centroid = None
        self.instance2_centroid = None
        self.state = BinaryClassifierState.ACQUIRE_FIRST

        return True, "BinaryClassifier started for class {}".format(class_name)

    def freeze_buffer(self):
        # Freeze the buffer while autonomy goes off to do the first task
        # Nothing is added or pruned, so the second instance detections we already gathered survive even with a short TTL
        # Only valid once the first instance is locked
        if not self.running:
            return False, "BinaryClassifier is not running"

        if self.state == BinaryClassifierState.FROZEN:
            return True, "BinaryClassifier buffer already frozen"

        if self.state != BinaryClassifierState.TRACK_FIRST:
            return False, "Can only lock the buffer while tracking the first instance"

        self.state = BinaryClassifierState.FROZEN
        return True, "BinaryClassifier buffer frozen"

    def start_second(self):
        self.update_params()

        if not self.running:
            return False, "BinaryClassifier is not running"

        # autonomy decides when to go looking for the second instance, but the first has to be locked first
        # (i.e. instance1_centroid has to exist)
        if not self.first_locked:
            return False, "Cannot start second instance before first instance is locked"

        if self.state == BinaryClassifierState.ACQUIRE_SECOND or self.state == BinaryClassifierState.TRACK_SECOND:
            return True, "BinaryClassifier is already working on second instance"

        # leaving FROZEN (or TRACK_FIRST) unfreezes the buffer. Dump what we preserved to
        # try to lock instance 2 from history, don't wait for a fresh detection
        self.state = BinaryClassifierState.ACQUIRE_SECOND

        candidate = self.find_best_cluster(self.samples_outside_first_instance())
        if candidate is not None:
            separation = self.distance(candidate["centroid"], self.instance1_centroid)
            if separation >= self.params.min_instance_separation_m:
                self.instance2_centroid = candidate["centroid"]
                self.state = BinaryClassifierState.TRACK_SECOND

        return True, "BinaryClassifier switched to second instance"

    def stop(self):
        if not self.running:
            return False, "BinaryClassifier is not running"

        self.samples.clear()

        self.state = BinaryClassifierState.STOPPED
        self.class_name = ""
        self.instance1_name = ""
        self.instance2_name = ""

        self.instance1_centroid = None
        self.instance2_centroid = None

        return True, "BinaryClassifier stopped"

    def observe(self, sample, now_sec):
        if not self.running:
            return AssignmentResult(False, "", "classifier stopped")

        # If FROZEN, don't append/prune, so the detections gathered before autonomy executes the first task survive regardless of TTL.
        # Instance 1 still tracks because its centroid is a field, not a buffer entry
        if self.state == BinaryClassifierState.FROZEN:
            return self.observe_track_first(sample)

        # prune on both sides of the append so a fresh sample can't keep stale ones alive, and the new sample itself is held to the same TTL
        self.prune(now_sec)
        self.samples.append(sample)
        self.prune(now_sec)

        if self.state == BinaryClassifierState.ACQUIRE_FIRST:
            return self.observe_acquire_first(sample)

        if self.state == BinaryClassifierState.TRACK_FIRST:
            return self.observe_track_first(sample)

        if self.state == BinaryClassifierState.ACQUIRE_SECOND:
            return self.observe_acquire_second(sample)

        if self.state == BinaryClassifierState.TRACK_SECOND:
            return self.observe_track_second(sample)

        return AssignmentResult(False, "", "invalid classifier state")

    def observe_acquire_first(self, sample):
        # try to lock instance 1 from the buffer, once locked we fall through and start assigning
        if self.instance1_centroid is None:
            cluster = self.find_best_cluster(list(self.samples))
            if cluster is not None:
                self.instance1_centroid = cluster["centroid"]
                self.state = BinaryClassifierState.TRACK_FIRST

        if self.instance1_centroid is None:
            return AssignmentResult(False, "", "acquiring first instance")

        return self.assign_to_locked_instance(
            sample,
            self.instance1_centroid,
            self.instance1_name,
            1,
            "first instance locked, sample outside gate",
        )

    def observe_track_first(self, sample):
        return self.assign_to_locked_instance(
            sample,
            self.instance1_centroid,
            self.instance1_name,
            1,
            "tracking first instance, sample outside gate",
        )

    def observe_acquire_second(self, sample):
        # reject anything sitting on top of instance 1 so it can't leak into instance 2
        if self.instance1_centroid is not None:
            dist_to_first = self.distance(sample.point(), self.instance1_centroid)
            if dist_to_first < self.params.exclusion_radius_m:
                return AssignmentResult(False, "", "sample inside first instance exclusion radius", dist_to_first)

        if self.instance2_centroid is None:
            candidate_samples = self.samples_outside_first_instance()
            cluster = self.find_best_cluster(candidate_samples)

            if cluster is not None:
                # a valid second instance has to be physically far enough from the first
                separation = self.distance(cluster["centroid"], self.instance1_centroid)
                if separation < self.params.min_instance_separation_m:
                    return AssignmentResult(False, "", "second candidate too close to first instance", separation)

                self.instance2_centroid = cluster["centroid"]
                self.state = BinaryClassifierState.TRACK_SECOND

        if self.instance2_centroid is None:
            return AssignmentResult(False, "", "acquiring second instance")

        return self.assign_to_locked_instance(
            sample,
            self.instance2_centroid,
            self.instance2_name,
            2,
            "second instance locked, sample outside gate",
        )

    def observe_track_second(self, sample):
        # keep excluding instance 1's neighborhood even while tracking the second
        if self.instance1_centroid is not None:
            dist_to_first = self.distance(sample.point(), self.instance1_centroid)
            if dist_to_first < self.params.exclusion_radius_m:
                return AssignmentResult(False, "", "sample inside first instance exclusion radius", dist_to_first)

        return self.assign_to_locked_instance(
            sample,
            self.instance2_centroid,
            self.instance2_name,
            2,
            "tracking second instance, sample outside gate",
        )

    def assign_to_locked_instance(self, sample, centroid, target_name, instance_num, reject_reason):
        if centroid is None:
            return AssignmentResult(False, "", "instance is not locked")

        dist = self.distance(sample.point(), centroid)

        # too far from the locked centroid -> not this instance
        if dist > self.params.assignment_gate_m:
            return AssignmentResult(False, "", reject_reason, dist)

        self.update_centroid(instance_num, sample.point())
        return AssignmentResult(True, target_name, "assigned", dist)

    def update_centroid(self, instance_num, point):
        # slow EMA so a single noisy detection can't yank the lock around
        alpha = self.params.centroid_ema_alpha

        if instance_num == 1 and self.instance1_centroid is not None:
            self.instance1_centroid = (1.0 - alpha) * self.instance1_centroid + alpha * point

        if instance_num == 2 and self.instance2_centroid is not None:
            self.instance2_centroid = (1.0 - alpha) * self.instance2_centroid + alpha * point

    def samples_outside_first_instance(self):
        if self.instance1_centroid is None:
            return list(self.samples)

        out = []

        for sample in self.samples:
            dist = self.distance(sample.point(), self.instance1_centroid)

            # only keep samples clearly away from instance 1 as candidates for instance 2
            if dist >= self.params.exclusion_radius_m and dist >= self.params.min_instance_separation_m:
                out.append(sample)

        return out

    def find_best_cluster(self, samples):
        min_count = max(1, self.params.min_cluster_detections)

        if len(samples) < min_count:
            return None

        # cluster in the metric space (2D by default), but keep full 3D points for the final centroid
        metric_points = numpy.array([self.metric_point(sample.point()) for sample in samples], dtype=numpy.float64)
        full_points = numpy.array([sample.point() for sample in samples], dtype=numpy.float64)

        # greedy mode-seek: whichever sample has the most neighbors within cluster_radius wins
        best_indices = None
        best_count = 0

        for i in range(metric_points.shape[0]):
            dists = numpy.linalg.norm(metric_points - metric_points[i], axis=1)
            indices = numpy.where(dists <= self.params.cluster_radius_m)[0]

            if len(indices) > best_count:
                best_count = len(indices)
                best_indices = indices

        if best_indices is None or best_count < min_count:
            return None

        cluster_points = full_points[best_indices]

        # drop the farthest CLUSTER_TRIM_QUANTILE tail (vs the median center) before averaging so stragglers don't pull the centroid
        center = numpy.nanmedian(cluster_points, axis=0)
        dists_to_center = numpy.array([self.distance(point, center) for point in cluster_points], dtype=numpy.float64)

        # only trim if the cluster has a couple samples to spare above the minimum
        if len(dists_to_center) >= min_count + 2:
            cutoff = numpy.nanquantile(dists_to_center, CLUSTER_TRIM_QUANTILE)
            trimmed = cluster_points[dists_to_center <= cutoff]

            if len(trimmed) >= min_count:
                cluster_points = trimmed

        centroid = numpy.nanmean(cluster_points, axis=0)

        metric_cluster_points = numpy.array([self.metric_point(point) for point in cluster_points], dtype=numpy.float64)
        metric_centroid = self.metric_point(centroid)

        # reject a cluster that's too spread out to be a single object
        variance_m2 = float(numpy.nanmean(numpy.sum((metric_cluster_points - metric_centroid) ** 2, axis=1)))

        if self.params.max_cluster_variance_m2 > 0.0 and variance_m2 > self.params.max_cluster_variance_m2:
            return None

        return {
            "centroid": centroid,
            "count": len(cluster_points),
            "variance_m2": variance_m2,
        }

    def age_buffer(self, now_sec):
        # Time-driven TTL eviction
        if not self.running or self.state == BinaryClassifierState.FROZEN:
            return
        self.prune(now_sec)
        
    def prune(self, now_sec):
        # never called while FROZEN (age_buffer/observe gates it)
        
        if self.params.buffer_ttl_sec <= 0.0:
            return

        # drop samples older than the TTL so old views of a bin don't linger after we've moved on
        self.samples = deque(
            [sample for sample in self.samples if now_sec - sample.stamp_sec <= self.params.buffer_ttl_sec],
            maxlen=max(1, self.params.buffer_size),
        )
    
    def distance(self, point_a, point_b):
        return float(numpy.linalg.norm(self.metric_point(point_a) - self.metric_point(point_b)))

    def metric_point(self, point):
        # default to 2D: for top-down flat targets the z/depth axis is the noisiest, so leave it out of distances
        if self.params.use_3d_distance:
            return point[:3]

        return point[:2]