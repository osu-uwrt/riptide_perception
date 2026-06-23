#!/usr/bin/env python3

#TODO: delete this after comp

# Cursed ah bin geometry fit. Fits the entire bin prop given some assumptions:
# 1) CAD geometry is accurate
# 2) Bin yaw/z reckoning is fairly accurate (if z fit fails it will fallback to config, but still use fit for x/y/yaw)
# 3) All 4 vinyl's indexed from lowest (z) to highest are identified and classes are set in config
# 4) bin_target1 and bin_target2 are located and tf's are accurate
# 5) DO NOT MAKE bin_cad_geometry a direct child of map or things will break badly

import traceback
from math import atan2, cos, sin, pi, degrees, radians

import numpy as np

from rclpy.time import Time
from tf2_ros import TransformException
from geometry_msgs.msg import Point, Vector3

from location import Location
from riptide_msgs2.srv import SetString


def angle_diff(a, b):
    # a-b wrapped to (-pi, pi]
    return (a - b + pi) % (2.0 * pi) - pi


def fit_bin_xy_yaw(model_a, model_b, meas_1, meas_2, nominal_yaw):
    # solve x, y, yaw from two vinyl correspondences
    # model_a/b: vinyl xy in CAD frame. meas_1/2: centroids in map frame, any order
    # nominal_yaw breaks the 180 deg flip
    
    ma = np.asarray(model_a, dtype=np.float64)
    mb = np.asarray(model_b, dtype=np.float64)
    p1 = np.asarray(meas_1, dtype=np.float64)
    p2 = np.asarray(meas_2, dtype=np.float64)

    u = mb - ma
    model_baseline = float(np.linalg.norm(u))
    if model_baseline < 1e-6:
        return None

    best = None
    # There are two potential solutions differing by 180 deg
    for swapped, (qa, qb) in enumerate(((p1, p2), (p2, p1))):
        w = qb - qa
        meas_baseline = float(np.linalg.norm(w))
        if meas_baseline < 1e-6:
            continue

        # yaw aligning u onto w
        yaw = atan2(u[0] * w[1] - u[1] * w[0], u[0] * w[0] + u[1] * w[1])

        # translation from centroid match
        c_model = 0.5 * (ma + mb)
        c_meas = 0.5 * (qa + qb)
        R = np.array([[cos(yaw), -sin(yaw)],
                      [sin(yaw),  cos(yaw)]], dtype=np.float64)
        t = c_meas - R @ c_model

        ra = float(np.linalg.norm((R @ ma + t) - qa))
        rb = float(np.linalg.norm((R @ mb + t) - qb))

        cand = {
            "x": float(t[0]),
            "y": float(t[1]),
            "yaw": float(yaw),
            "swapped": bool(swapped),
            "residual_m": max(ra, rb),
            "baseline_err_m": abs(meas_baseline - model_baseline),
            "model_baseline_m": model_baseline,
            "_score": abs(angle_diff(yaw, nominal_yaw)),
        }
        if best is None or cand["_score"] < best["_score"]:
            best = cand

    if best is not None:
        best.pop("_score", None)
    return best


# vinyls lowest->highest z, set per mapping config pre-run
BIN_VINYLS = ["bin_vinyl1", "bin_vinyl2", "bin_vinyl3", "bin_vinyl4"]

# objects the node adds to mapping
EXTRA_OBJECTS = [
    "bin_cad_geometry",
    "magnet_target1", "magnet_target2",
    "magnet1", "magnet2",
    "bin_vinyl1", "bin_vinyl2", "bin_vinyl3", "bin_vinyl4",
]

# defaults, overridden by mapping launch config.yaml
BIN_FIT_PARAMS = [
    ("bin_fit.residual_tol_m", 0.10),       # max reprojection error
    ("bin_fit.baseline_tol_m", 0.10),       # |measured - model| baseline
    ("bin_fit.min_baseline_m", 0.20),       # model baseline floor
    ("bin_fit.cov_floor_m2", 0.0025),       # published variance floor
    ("bin_fit.z_consistency_tol_m", 0.10),  # max z disagreement to trust measured z
]


class BinGeometryFitter:
    EXTRA_OBJECTS = EXTRA_OBJECTS

    def __init__(self, node):
        self.node = node # We live inside of mapping so we can be aborted later
        node.declare_parameters(namespace="", parameters=BIN_FIT_PARAMS)
        node.create_service(SetString, "mapping/fit_bin_geometry", self.handle_fit)

    def _param(self, name):
        return self.node.get_parameter(name).value

    def seed_bin_cad_geometry(self, x, y, z, yaw_rad, residual_m):
        # bin_cad_geometry is a bin child, pose is relative to bin
        node = self.node
        xyz = Point(x=float(x), y=float(y), z=float(z))
        # Location wants rpy in degrees, fit yaw is radians
        rpy = Vector3(x=0.0, y=0.0, z=float(degrees(yaw_rad)))
        node.objects["bin_cad_geometry"]["location"] = Location(
            xyz, rpy,
            int(self._param("buffer_size")),
            tuple(self._param("quantile")),
        )

        # residual-gated cov, applied in publish_pose
        cov_floor = float(self._param("bin_fit.cov_floor_m2"))
        node.objects["bin_cad_geometry"]["fit_covar"] = max(cov_floor, float(residual_m) ** 2)

    def handle_fit(self, request, response):
        # called after stop_binary_classifier
        node = self.node
        log = node.get_logger()
        try:
            class_name = str(request.data).strip()
            if class_name == "":
                response.success = False
                response.message = "fit refused: class_name cannot be empty"
                log.warning(response.message)
                return response

            matching = [v for v in BIN_VINYLS
                        if str(self._param(f"init_data.{v}.class")).strip() == class_name]
            if len(matching) != 2:
                response.success = False
                response.message = (f"fit refused: expected exactly 2 vinyls of class '{class_name}' in config, "
                                    f"found {len(matching)} {matching}")
                log.warning(response.message)
                return response
            vinyl_a, vinyl_b = matching[0], matching[1]
            log.info(f"bin fit: class '{class_name}' -> vinyls {vinyl_a}, {vinyl_b}")

            # the 2 class vinyl's xy in CAD frame from config
            model_a = (float(self._param(f"init_data.{vinyl_a}.pose.x")),
                       float(self._param(f"init_data.{vinyl_a}.pose.y")))
            model_b = (float(self._param(f"init_data.{vinyl_b}.pose.x")),
                       float(self._param(f"init_data.{vinyl_b}.pose.y")))

            # the two seeded bin target TFs read in bin frame
            try:
                tf1 = node.tf_buffer.lookup_transform("bin_frame", "bin_target1_frame", Time())
                tf2 = node.tf_buffer.lookup_transform("bin_frame", "bin_target2_frame", Time())
            except TransformException as ex:
                response.success = False
                response.message = f"fit refused: could not look up bin target TFs in bin frame ({ex})"
                log.warning(response.message)
                return response

            p1_bin = (float(tf1.transform.translation.x), float(tf1.transform.translation.y), float(tf1.transform.translation.z))
            p2_bin = (float(tf2.transform.translation.x), float(tf2.transform.translation.y), float(tf2.transform.translation.z))
            meas_1 = (p1_bin[0], p1_bin[1])
            meas_2 = (p2_bin[0], p2_bin[1])

            # sanity check they were seeded
            log.info(f"bin fit: target1 in bin=({p1_bin[0]:.3f}, {p1_bin[1]:.3f}, {p1_bin[2]:.3f})  "
                     f"target2 in bin=({p2_bin[0]:.3f}, {p2_bin[1]:.3f}, {p2_bin[2]:.3f})")

            # nominal yaw only breaks the 180 deg flip
            nominal_yaw = radians(float(self._param("init_data.bin_cad_geometry.pose.yaw")))

            fit = fit_bin_xy_yaw(model_a, model_b, meas_1, meas_2, nominal_yaw)
            if fit is None:
                response.success = False
                response.message = "fit refused: degenerate baseline (model or measured points match exactly, targets may not be seeded)"
                log.warning(response.message)
                return response

            # gates: baseline floor, baseline match, residual
            if fit["model_baseline_m"] < float(self._param("bin_fit.min_baseline_m")):
                response.success = False
                response.message = f"fit refused: model baseline {fit['model_baseline_m']:.3f} m below floor"
                log.warning(response.message)
                return response
            if fit["baseline_err_m"] > float(self._param("bin_fit.baseline_tol_m")):
                response.success = False
                response.message = (f"fit refused: baseline mismatch {fit['baseline_err_m']:.3f} m (bad detection or wrong vinyl map)")
                log.warning(response.message)
                return response
            if fit["residual_m"] > float(self._param("bin_fit.residual_tol_m")):
                response.success = False
                response.message = f"fit refused: residual {fit['residual_m']:.3f} m over tolerance"
                log.warning(response.message)
                return response

            # z solve, only after xy/yaw passed
            # cad z = measured bin-frame z minus the model z of the matching vinyl
            bz1 = float(p1_bin[2])
            bz2 = float(p2_bin[2])
            vza = float(self._param(f"init_data.{vinyl_a}.pose.z"))
            vzb = float(self._param(f"init_data.{vinyl_b}.pose.z"))

            # handle the 180 swap
            if not fit["swapped"]:
                cad_z_from_a, cad_z_from_b = bz1 - vza, bz2 - vzb
            else:
                cad_z_from_a, cad_z_from_b = bz2 - vza, bz1 - vzb

            z_config = float(self._param("init_data.bin_cad_geometry.pose.z"))
            z_tol = float(self._param("bin_fit.z_consistency_tol_m"))
            z_disagreement = abs(cad_z_from_a - cad_z_from_b)

            # if agree then use mean, else keep config for z
            if z_disagreement <= z_tol:
                z = 0.5 * (cad_z_from_a + cad_z_from_b)
                z_source = f"measured (disagreement {z_disagreement:.3f} m)"
            else:
                z = z_config
                z_source = f"config fallback (disagreement {z_disagreement:.3f} m > {z_tol:.3f} m)"
                log.warning(f"bin fit: z estimates disagree by {z_disagreement:.3f} m. Using config offset {z_config:.3f}")

            # seed only bin_cad_geometry, bin stays put or we drag bin_targets with us
            self.seed_bin_cad_geometry(fit["x"], fit["y"], z, fit["yaw"], fit["residual_m"])
            node.publish_pose()

            response.success = True
            response.message = (f"bin_cad_geometry fit ok (in bin frame): x={fit['x']:.3f} y={fit['y']:.3f} "
                                f"z={z:.3f} ({z_source}) yaw={degrees(fit['yaw']):.1f}deg "
                                f"residual={fit['residual_m']:.3f}m corr=[{vinyl_a},{vinyl_b}]")

            log.info(response.message)
            return response

        except Exception as ex:
            log.error(f"fit_bin_geometry crashed: {ex}\n{traceback.format_exc()}")
            response.success = False
            response.message = f"fit crashed: {ex}"
            return response