import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "riptide_mapping2"))

from map_estimator import (EstimatorSettings, MapEstimator, NoiseSettings, ObjectSpec,  # noqa: E402
                           detection_covariance, slew_angle, slew_vector)


def make_estimator(**settings):
    # Detections are taken one at a time unless a test is about acquiring by agreement
    settings.setdefault("acquire_detections", 1)
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


def feed(est, name, position, n, std=0.05, start=0.0, period=1.0, yaw=None, targeted=True):
    # n detections spaced far enough apart that none of them are inflated as correlated
    for i in range(n):
        accepted, reason = est.observe(name, start + i * period, position, iso(std), yaw,
                                       None if yaw is None else math.radians(10) ** 2, targeted)
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
    settings.setdefault("acquire_detections", 1)
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


def test_confirmed_needs_fresh_detections_after_each_reset():
    est = make_estimator()
    assert not est.confirmed("hole")
    feed(est, "hole", est.specs["hole"].position, 9)
    assert not est.confirmed("hole")
    feed(est, "hole", est.specs["hole"].position, 1, start=20.0)
    assert est.confirmed("hole")  # confirm_detections while targeted
    est.reset("hole")  # a full reset (reacquire): from scratch
    assert not est.confirmed("hole")
    est.seed("gate", (5.0, 0.0, -1.0))
    assert not est.confirmed("gate")


def test_known_target_is_verified_by_a_few_detections_where_it_was():
    est = make_estimator()
    spot = est.specs["hole"].position + np.array([0.05, 0.0, 0.0])
    feed(est, "hole", spot, 10)
    est.reset("hole", verify=True)  # targeted again
    assert not est.confirmed("hole")
    feed(est, "hole", spot, est.settings.verify_detections - 1, start=100.0)
    assert not est.confirmed("hole")
    feed(est, "hole", spot, 1, start=200.0)
    assert est.confirmed("hole")
    assert np.allclose(est.position("hole")[0], spot, atol=0.01)


def test_a_small_drift_since_it_was_seen_passes_and_is_absorbed():
    est = make_estimator()
    spot = est.specs["hole"].position.copy()
    feed(est, "hole", spot, 20, std=0.02)
    est.predict(distance=4.0, dt=20.0)  # a surface trip: a few cm of expected drift
    est.reset("hole", verify=True)
    drifted = spot + np.array([0.04, -0.02, 0.0])
    for i in range(est.settings.verify_detections):
        accepted, reason = est.observe("hole", 100.0 + i, drifted, iso(0.02))
        assert accepted and "moved" not in reason
    assert est.confirmed("hole")
    assert np.linalg.norm(est.position("hole")[0] - drifted) < 0.02  # it followed the drift


def test_a_moved_target_is_found_again_from_scratch():
    est = make_estimator()
    spot = est.specs["hole"].position.copy()
    feed(est, "hole", spot, 20, std=0.02)
    est.reset("hole", verify=True)
    moved = spot + np.array([0.4, 0.0, 0.0])
    accepted, reason = est.observe("hole", 100.0, moved, iso(0.02))
    assert accepted and "moved 40 cm" in reason
    assert not est.confirmed("hole")
    feed(est, "hole", moved, est.settings.confirm_detections - 2, start=101.0)
    assert not est.confirmed("hole")  # the full confirm_detections at the new spot
    feed(est, "hole", moved, 1, start=200.0)
    assert est.confirmed("hole")
    assert np.allclose(est.position("hole")[0], moved, atol=0.01)


def test_background_detections_improve_but_do_not_confirm():
    est = make_estimator()
    spot = est.specs["hole"].position + np.array([0.1, 0.1, 0.0])
    feed(est, "hole", spot, 15, targeted=False)
    assert not est.confirmed("hole")
    assert np.allclose(est.position("hole")[0], spot, atol=0.02)  # it knows where it is
    est.reset("hole", verify=True)  # now the target: a quick check, not a fresh search
    feed(est, "hole", spot, est.settings.verify_detections, start=100.0)
    assert est.confirmed("hole")


def test_a_known_map_object_is_verified_too():
    est = make_estimator()
    gate = est.specs["gate"].position + np.array([0.3, 0.0, 0.0])
    feed(est, "gate", gate, 15, std=0.02)
    est.predict(distance=10.0, dt=30.0)
    est.reset("gate", verify=True)
    feed(est, "gate", gate, est.settings.verify_detections, std=0.02, start=100.0)
    assert est.confirmed("gate")


# ----- acquiring by agreement -----

def acquiring_estimator(**settings):
    settings.setdefault("acquire_detections", 4)
    return make_estimator(**settings)


def test_a_stray_first_detection_neither_moves_nor_biases_the_estimate():
    est = acquiring_estimator()
    prior = est.specs["hole"].position.copy()
    truth = prior + np.array([0.15, -0.1, 0.0])
    accepted, reason = est.observe("hole", 0.0, prior + np.array([1.5, 1.0, 0.0]), iso(0.03))  # a false detection
    assert not accepted and "acquiring" in reason
    assert np.allclose(est.position("hole")[0], prior)  # held back: the estimate didn't move
    for i in range(3):
        accepted, reason = est.observe("hole", 0.1 * (i + 1), truth, iso(0.03))
        assert not accepted, reason
    accepted, reason = est.observe("hole", 0.4, truth, iso(0.03))
    assert accepted and "4 detections agree" in reason and "1 that didn't dropped" in reason
    assert np.allclose(est.position("hole")[0], truth, atol=0.02)  # the stray one didn't bias it
    assert est.tracks["hole"].updates == 4


def test_flickers_that_never_agree_never_start_a_track():
    est = acquiring_estimator()
    prior = est.specs["gate"].position.copy()
    for i in range(10):
        accepted, _ = est.observe("gate", 0.5 * i, prior + np.array([0.0, 0.8 * i, 0.0]), iso(0.03))
        assert not accepted
    assert est.tracks["gate"].updates == 0 and np.allclose(est.position("gate")[0], prior)


def test_old_detections_fall_out_of_the_window():
    est = acquiring_estimator(acquire_window_s=1.0)
    spot = est.specs["gate"].position + np.array([0.2, 0.0, 0.0])
    for i in range(6):
        accepted, _ = est.observe("gate", 2.0 * i, spot, iso(0.03))  # agreeing, but each alone in its window
        assert not accepted


def test_agreement_away_from_the_map_needs_more_detections_for_longer():
    est = acquiring_estimator(course_yaw_std_deg=0.0)
    # One prop seen: the bin is expected within ~0.36 m (offset and layout), so 2 m off it is far
    for i in range(30):
        est.observe("gate", float(i), est.specs["gate"].position, iso(0.02), agreed=True)
    est.predict(5.0, 5.0)
    far_spot = est.specs["bin"].position + np.array([2.0, 0.0, 0.0])
    results = [est.observe("bin", 100.0 + 0.1 * i, far_spot, iso(0.03)) for i in range(25)]
    taken = [i for i, (accepted, _) in enumerate(results) if accepted]
    assert taken, results[-1]
    first = taken[0]
    assert first + 1 >= est.settings.acquire_far_detections  # more than the usual 4...
    assert 0.1 * first >= est.settings.acquire_far_s - 1e-9  # ...agreeing for acquire_far_s
    assert "taking them over the prior" in results[first][1]
    assert np.allclose(est.position("bin")[0], far_spot, atol=0.05)


def test_a_near_cluster_is_taken_after_the_usual_few():
    est = acquiring_estimator()
    for i in range(30):
        est.observe("gate", float(i), est.specs["gate"].position, iso(0.02), agreed=True)
    near = est.specs["bin"].position + np.array([0.1, 0.05, 0.0])
    results = [est.observe("bin", 100.0 + 0.1 * i, near, iso(0.03)) for i in range(6)]
    assert [accepted for accepted, _ in results][:4] == [False, False, False, True]


def test_agreed_detections_skip_the_wait():
    est = acquiring_estimator()
    accepted, reason = est.observe("hole", 0.0, est.specs["hole"].position, iso(0.03), agreed=True)
    assert accepted, reason
    assert est.tracks["hole"].updates == 1


def test_a_verified_target_that_moved_is_found_again_by_agreement():
    est = acquiring_estimator()
    spot = est.specs["hole"].position + np.array([0.05, 0.0, 0.0])
    for i in range(12):
        est.observe("hole", 0.1 * i, spot, iso(0.02))
    assert est.tracks["hole"].established
    est.reset("hole", verify=True)
    moved = spot + np.array([0.6, 0.0, 0.0])
    accepted, reason = est.observe("hole", 5.0, moved, iso(0.02))
    assert not accepted and "moved" in reason and "acquiring" in reason
    for i in range(3):
        accepted, reason = est.observe("hole", 5.1 + 0.1 * i, moved, iso(0.02))
    assert accepted, reason
    assert np.allclose(est.position("hole")[0], moved, atol=0.03)
