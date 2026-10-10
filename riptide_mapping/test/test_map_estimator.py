import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "riptide_mapping2"))

from map_estimator import (EstimatorSettings, MapEstimator, NoiseSettings, ObjectSpec,  # noqa: E402
                           detection_covariance, slew_angle, slew_vector)


def make_estimator(**settings):
    specs = {
        "gate": ObjectSpec("map", np.array([5.0, 0.0, -1.0]), 0.0),
        "torpedo": ObjectSpec("map", np.array([15.0, 3.0, -1.5]), math.pi),
        "bin": ObjectSpec("map", np.array([10.0, -4.0, -2.0]), 0.0),
        "hole": ObjectSpec("torpedo_frame", np.array([0.0, 0.3, 0.2]), 0.0),
        "locked": ObjectSpec("map", np.array([0.0, 8.0, -1.0]), 0.5, lock_yaw=True),
    }
    return MapEstimator(specs, EstimatorSettings(**settings), NoiseSettings())


def iso(std):
    return np.eye(3) * std ** 2


def feed(est, name, position, n, std=0.05, start=0.0, period=1.0, yaw=None):
    # n detections spaced far enough apart that none of them are inflated as correlated
    for i in range(n):
        accepted, reason = est.observe(name, start + i * period, position, iso(std), yaw,
                                       None if yaw is None else math.radians(10) ** 2)
        assert accepted, reason


def test_child_converges_and_reset_keeps_position():
    est = make_estimator()
    truth = np.array([0.1, 0.45, 0.25])
    feed(est, "hole", truth, 40)

    mean, cov = est.position("hole")
    assert np.allclose(mean, truth, atol=1e-3)
    assert cov[0, 0] < 0.05 ** 2 / 20

    est.reset("hole")
    mean_after, cov_after = est.position("hole")
    assert np.allclose(mean_after, mean)
    assert cov_after[0, 0] == pytest.approx(est.settings.child_std_xy ** 2)


def test_sighting_one_map_object_moves_the_rest_of_the_map():
    est = make_estimator(course_yaw_std_deg=0.0)
    offset = np.array([0.8, -0.5, 0.1])
    feed(est, "gate", est.specs["gate"].position + offset, 200, std=0.02)

    gate_mean, gate_cov = est.position("gate")
    assert np.allclose(gate_mean, est.specs["gate"].position + offset, atol=1e-3)
    assert gate_cov[0, 0] < 1e-4

    # One object can't tell the offset from its own layout error, so it splits them by the priors
    s = est.settings
    share_xy = s.offset_std_xy ** 2 / (s.offset_std_xy ** 2 + s.layout_std_xy ** 2)
    share_z = s.offset_std_z ** 2 / (s.offset_std_z ** 2 + s.layout_std_z ** 2)
    torpedo_mean, torpedo_cov = est.position("torpedo")
    moved = torpedo_mean - est.specs["torpedo"].position
    # (the heavy-tailed layout prior leans slightly further toward the offset than a Gaussian split)
    assert moved[0:2] == pytest.approx(share_xy * offset[0:2], abs=1e-2)
    assert moved[2] == pytest.approx(share_z * offset[2], abs=1e-2)

    # ...and the unseen torpedo is still at least as uncertain as its layout
    assert torpedo_cov[0, 0] >= s.layout_std_xy ** 2
    assert torpedo_cov[0, 0] == pytest.approx(share_xy * s.layout_std_xy ** 2 + s.layout_std_xy ** 2, rel=5e-2)


def test_two_objects_pin_the_offset_below_the_layout_error():
    est = make_estimator()
    offset = np.array([0.5, 0.5, 0.0])
    feed(est, "gate", est.specs["gate"].position + offset, 100, std=0.02)
    feed(est, "bin", est.specs["bin"].position + offset, 100, std=0.02, start=200.0)

    _, P_oo = est.offset()
    # Two independent layout errors average: offset variance -> about layout_var / 2
    assert P_oo[0, 0] < 0.6 * est.settings.layout_std_xy ** 2


def test_drift_makes_old_sightings_count_for_less():
    est = make_estimator()
    feed(est, "gate", est.specs["gate"].position + np.array([1.0, 0.0, 0.0]), 100, std=0.02)
    gate_var_before = est.position("gate")[1][0, 0]

    est.predict(distance=50.0, dt=100.0)
    gate_var_after = est.position("gate")[1][0, 0]
    expected = 50.0 * est.settings.drift_var_per_m_xy + 100.0 * est.settings.drift_var_per_s_xy
    assert gate_var_after - gate_var_before == pytest.approx(expected, rel=1e-6)

    # A fresh sighting of the gate, 30 cm further, pulls the whole map most of the way
    torpedo_before = est.position("torpedo")[0]
    feed(est, "gate", est.specs["gate"].position + np.array([1.3, 0.0, 0.0]), 1, std=0.02, start=1000.0)
    torpedo_after = est.position("torpedo")[0]
    assert torpedo_after[0] - torpedo_before[0] > 0.2


def test_gate_rejects_a_lookalike_then_reacquires_a_moved_object():
    est = make_estimator(gate_max_rejects=5)
    position = est.specs["gate"].position
    feed(est, "gate", position, est.settings.gate_min_updates, std=0.05)

    lookalike = position + np.array([0.0, 3.0, 0.0])
    accepted, reason = est.observe("gate", 100.0, lookalike, iso(0.05))
    assert not accepted and "sigma" in reason
    assert np.allclose(est.position("gate")[0], position, atol=1e-2)

    # A good detection in between clears the rejection streak
    assert est.observe("gate", 101.0, position, iso(0.05))[0]
    for i in range(4):
        assert not est.observe("gate", 102.0 + i, lookalike, iso(0.05))[0]
    assert est.tracks["gate"].rejects == 4

    # The fifth in a row reacquires: back to the prior covariance, then it follows the detections
    accepted, reason = est.observe("gate", 110.0, lookalike, iso(0.05))
    assert not accepted and "reacquiring" in reason
    feed(est, "gate", lookalike, 20, std=0.05, start=120.0)
    assert np.allclose(est.position("gate")[0], lookalike, atol=0.02)


def test_yaw_wraps_and_locked_yaw_ignores_detections():
    est = make_estimator()
    est.yaw_mean["gate"] = math.radians(179.0)
    feed(est, "gate", est.specs["gate"].position, 20, yaw=math.radians(-177.0))
    yaw, var = est.yaw("gate")
    assert abs(math.degrees(yaw)) > 175.0
    assert var < math.radians(10) ** 2

    feed(est, "locked", est.specs["locked"].position, 5, yaw=2.0)
    assert est.yaw("locked") == (pytest.approx(0.5), 0.0)


def test_back_to_back_detections_are_inflated():
    est = make_estimator()
    assert est.inflation("gate", 0.0) == 1.0
    est.observe("gate", 0.0, est.specs["gate"].position, iso(0.05))
    assert est.inflation("gate", 0.1) == pytest.approx(est.noise.correlation_time / 0.1)
    assert est.inflation("gate", 0.0) == est.noise.max_inflation
    assert est.inflation("gate", 5.0) == 1.0


def test_detection_covariance_shape():
    noise = NoiseSettings()
    cov = detection_covariance((0.0, 0.0, 3.0), np.eye(3), 0.0, 0.0, noise)
    along = noise.base_std + noise.range_std_per_m * 3 + noise.range_std_per_m2 * 9
    lateral = noise.base_std + noise.lateral_std_per_m * 3
    assert cov[2, 2] == pytest.approx(along ** 2)
    assert cov[0, 0] == pytest.approx(lateral ** 2)

    # Rotating the camera rotates the long axis
    rot = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    rotated = detection_covariance((0.0, 0.0, 3.0), rot, 0.0, 0.0, noise)
    assert rotated[0, 0] == pytest.approx(along ** 2)

    # Turning smears across the ray, not along it
    turning = detection_covariance((0.0, 0.0, 3.0), np.eye(3), 1.0, 0.0, noise)
    assert turning[0, 0] == pytest.approx(lateral ** 2 + (1.0 * noise.timing_std * 3) ** 2)
    assert turning[2, 2] == pytest.approx(along ** 2)


def test_seed_shift_rotate_and_set_prior():
    est = make_estimator()
    est.observe("gate", 0.0, est.specs["gate"].position + np.array([0.4, 0.0, 0.0]), iso(0.05))

    est.seed("torpedo", (14.0, 2.0, -1.0), yaw=0.3)
    assert np.allclose(est.position("torpedo")[0], (14.0, 2.0, -1.0))
    assert est.yaw("torpedo")[0] == pytest.approx(0.3)

    est.shift("torpedo", (0.5, 0.0, 0.0))
    assert np.allclose(est.position("torpedo")[0], (14.5, 2.0, -1.0))

    est.rotate_position("hole", math.pi / 2)
    assert np.allclose(est.position("hole")[0], (-0.3, 0.0, 0.2))

    est.set_prior("bin", (11.0, -4.0, -2.0), 0.2)
    offset = est.offset()[0]
    assert np.allclose(est.position("bin")[0], np.array([11.0, -4.0, -2.0]) + offset)
    assert np.allclose(est.frame_translation("bin"), (11.0, -4.0, -2.0))


def test_slew():
    assert np.allclose(slew_vector(np.zeros(3), np.array([1.0, 0.0, 0.0]), 0.3, 1.0), (0.3, 0.0, 0.0))
    assert np.allclose(slew_vector(np.zeros(3), np.array([0.1, 0.0, 0.0]), 0.3, 1.0), (0.1, 0.0, 0.0))
    assert np.allclose(slew_vector(np.zeros(3), np.array([5.0, 0.0, 0.0]), 0.0, 1.0), (5.0, 0.0, 0.0))
    # Across +-pi the short way round
    assert slew_angle(math.radians(170), math.radians(-170), math.radians(5), 1.0) == pytest.approx(math.radians(175))
    assert slew_angle(0.0, 1.0, 0.0, 0.0) == 1.0


COURSE = {"gate": (5.0, 0.0, -1.0), "torpedo": (15.0, 3.0, -1.5), "bin": (10.0, -4.0, -2.0), "table": (20.0, -6.0, -1.5)}


def fly(order, layout_err=None, yaw_deg=0.0, drift_at=None, drift=(0.0, 0.0, 0.0), **settings):
    # Locks each prop in turn (15 m of travel between), the props sitting at R(yaw) * prior + offset + layout error
    # (+ drift once it happens, as odometry sees them). Returns the estimator and where each prop really is now
    est = MapEstimator({n: ObjectSpec("map", np.array(p), 0.0) for n, p in COURSE.items()},
                       EstimatorSettings(**settings), NoiseSettings())
    offset, layout_err, d = np.array([0.5, -0.3, 0.0]), layout_err or {}, np.zeros(3)
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))

    def truth(name):
        x, y, z = COURSE[name]
        return np.array([c * x - s * y, s * x + c * y, z]) + offset + np.asarray(layout_err.get(name, np.zeros(3))) + d

    t = 0.0
    for k, name in enumerate(order):
        if k > 0:
            est.predict(15.0, 30.0)
        if drift_at == k:
            d = np.asarray(drift, dtype=float)
        est.reset(name)
        for _ in range(60):
            assert est.observe(name, t, truth(name), iso(0.1))[0]
            t += 0.5
    return est, truth


def horizontal_error(est, truth, name):
    return float(np.linalg.norm(est.position(name)[0][:2] - truth(name)[:2]))


def test_a_bad_prior_is_released_instead_of_dragging_the_map():
    # The torpedo's prior is 1.5 m off; the gate (seen first) says where the course is
    est, truth = fly(("gate", "torpedo"), {"torpedo": (1.5, 0.0, 0.0)})
    assert est.released("torpedo") and not est.released("gate")
    assert horizontal_error(est, truth, "torpedo") < 0.02  # where it was seen
    assert horizontal_error(est, truth, "bin") < 0.25      # the unseen bin isn't dragged 1.5 m


def test_a_bad_first_prior_is_released_once_another_prop_disagrees():
    # Seen alone, the gate's bad prior moves the whole map (one sighting can't tell drift from a bad prior)...
    est, truth = fly(("gate",), {"gate": (1.5, 0.0, 0.0)})
    assert horizontal_error(est, truth, "bin") > 1.0
    # ...until the torpedo disagrees with it
    est, truth = fly(("gate", "torpedo"), {"gate": (1.5, 0.0, 0.0)})
    assert est.released("gate")
    assert horizontal_error(est, truth, "gate") < 0.1
    assert horizontal_error(est, truth, "bin") < 0.25


def test_props_pin_the_course_yaw():
    est, truth = fly(("gate", "torpedo", "bin"), yaw_deg=2.0)
    yaw, var = est.course_yaw()
    assert math.degrees(yaw) == pytest.approx(2.0, abs=1.0)
    assert horizontal_error(est, truth, "table") < 0.25  # 20 m out, 0.7 m off before
    without, truth = fly(("gate", "torpedo", "bin"), yaw_deg=2.0, course_yaw_std_deg=0.0)
    assert horizontal_error(est, truth, "table") < horizontal_error(without, truth, "table")


def test_retargeting_keeps_the_fix_and_a_revisit_measures_drift():
    est, truth = fly(("gate", "torpedo", "gate"), drift_at=1, drift=(0.0, 1.0, 0.0))
    # The first gate visit was frozen when the gate was retargeted; the torpedo and the new gate track are live fixes
    assert [fix.name for fix in est.frozen] == ["gate"] and set(est.live) == {"gate", "torpedo"}
    assert horizontal_error(est, truth, "gate") < 0.05
    # The offset followed the drift
    assert est.offset()[0][1] == pytest.approx(-0.3 + 1.0, abs=0.2)
