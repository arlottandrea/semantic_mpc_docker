import numpy as np

from semantic_mpc_package.planning import GaussianProcessCoverage, RRTWaypointPlanner


def test_gp_spreads_observation_signal_to_neighbors():
    coverage = GaussianProcessCoverage(bounds=[[-5.0, 5.0], [-5.0, 5.0]], resolution=1.0)
    coverage.observe(0.0, 0.0, value=1.0)
    assert coverage.informative_score(0.2, 0.0) > 0.0
    assert coverage.informative_score(3.0, 0.0) == 0.0


def test_rrt_returns_waypoints_from_information_map():
    coverage = GaussianProcessCoverage(bounds=[[-5.0, 5.0], [-5.0, 5.0]], resolution=1.0)
    coverage.observe(2.0, 1.0, value=1.0)
    planner = RRTWaypointPlanner(bounds=[[-5.0, 5.0], [-5.0, 5.0]], resolution=1.0, max_nodes=50)
    waypoints = planner.plan(start_pose=[0.0, 0.0, 0.0], coverage=coverage, target_count=3)
    assert len(waypoints) == 3
    assert all(np.isfinite(node).all() for node in waypoints)
