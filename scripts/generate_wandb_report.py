#!/usr/bin/env python3
"""Download paired experiment runs from W&B and generate batch reports."""

import argparse
import json
import math
import os
import re
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


REPORT_METRICS = [
    "success_rate",
    "elapsed_time_s",
    "mean_velocity_mps",
    "final_entropy",
    "mean_tree_entropy_convergence_time_s",
    "mean_tree_entropy_reduction_rate_bits_s",
    "tree_entropy_completion_rate",
    "mean_controller_compute_time_ms",
    "entropy_reduction_per_meter",
    "entropy_reduction_per_second",
    "entropy_reduction_per_compute_ms",
    "time_to_50pct_tracking_s",
    "time_to_90pct_tracking_s",
    "worst_tree_entropy_final",
]
HISTORY_METRICS = [
    "step",
    "time_execution_s",
    "distance_m",
    "entropy",
    "pose/x",
    "pose/y",
    "pose/theta",
    "speed_mps",
    "nearest_tree_id",
    "nearest_tree_distance_m",
    "velocity_reduction_mps",
    "command/speed_mps",
    "mpc_step_duration_s",
    "policy_inference_time_ms",
    "controller_compute_time_ms",
    "action/forward",
    "action/lateral",
    "policy_action/forward",
    "policy_action/lateral",
    "rl_liveness_recovery_active",
    "rl_liveness_recovery_count",
    "measurement_age_s",
    "num_tracked",
    "new_targets_tracked",
    "total_targets",
    "selected_target_id",
    "active_target_count_current",
    "observation_episode_active",
    "worst_tree_entropy",
    "unresolved_tree_count",
    "mpc_warm_start_reused",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project",
        action="append",
        default=[],
        help="W&B entity/project; repeat for baseline, RL and NMPC projects.",
    )
    parser.add_argument(
        "--input-dir",
        help="Regenerate a report from previously exported runs.csv and history.csv.",
    )
    parser.add_argument("--output-dir", default="runs/reports/latest")
    parser.add_argument("--state", default="finished", help="W&B run state filter; empty means all.")
    parser.add_argument("--tag", action="append", default=[], help="Require W&B tag; repeatable.")
    return parser.parse_args()


def _safe_number(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return np.nan
    return result if np.isfinite(result) else np.nan


def _first_number(mapping, names):
    for name in names:
        value = _safe_number(mapping.get(name))
        if np.isfinite(value):
            return value
    return np.nan


def infer_algorithm(name, config, project=""):
    explicit = config.get("algorithm") or config.get("mode")
    if explicit:
        return str(explicit).lower()
    lowered = "{} {}".format(project, name).lower()
    for candidate in ("casadi_mpc", "greedy_ig", "mower", "linear", "greedy", "nmpc", "rl_agent", "rl"):
        if candidate in lowered:
            return "active_rl" if candidate in ("rl", "rl_agent") else candidate
    return "unknown"


def infer_run_index(name, config):
    value = config.get("run_index")
    if value is not None:
        try:
            return int(value)
        except (TypeError, ValueError):
            pass
    match = re.search(r"(?:run[_-]?)(\d+)", name, flags=re.IGNORECASE)
    return int(match.group(1)) if match else np.nan


def _tree_count(summary, config):
    count = _first_number(summary, ("total_targets", "num_trees", "tree_count"))
    if np.isfinite(count) and count > 0:
        return int(count)
    final_belief = summary.get("final_belief")
    if isinstance(final_belief, (list, tuple)) and final_belief:
        classes = int(_first_number(config, ("nclasses", "nn_output_dim")))
        classes = classes if classes > 0 else 1
        return max(1, int(len(final_belief) / classes))
    initial_entropy = _first_number(summary, ("initial_entropy", "entropy_initial"))
    return int(round(initial_entropy)) if np.isfinite(initial_entropy) and initial_entropy > 0 else np.nan


def _max_velocity(config):
    value = _first_number(config, ("max_velocity", "max_linear_velocity"))
    if np.isfinite(value):
        return value
    scale = config.get("action_scale_xy")
    if isinstance(scale, (list, tuple)) and len(scale) >= 2:
        return float(np.linalg.norm(np.asarray(scale[:2], dtype=float)))
    return np.nan


def calculate_run_metrics(metadata, history):
    summary = metadata["summary"]
    config = metadata["config"]
    elapsed = _first_number(summary, ("total_time_execution_s", "elapsed_time_s", "runtime"))
    total_distance = _first_number(summary, ("total_distance_m", "distance_m"))

    if history.empty:
        history = pd.DataFrame()
    else:
        history = enrich_history(history)
    time_values = pd.to_numeric(history.get("time_execution_s", pd.Series(dtype=float)), errors="coerce")
    distance_values = pd.to_numeric(history.get("distance_m", pd.Series(dtype=float)), errors="coerce")
    entropy_values = pd.to_numeric(history.get("entropy", pd.Series(dtype=float)), errors="coerce")
    positive_periods = time_values.diff()
    positive_periods = positive_periods[positive_periods > 0].dropna()
    observed_control_period = float(positive_periods.median()) if not positive_periods.empty else np.nan
    observed_control_jitter_p95 = (
        float(np.percentile(np.abs(positive_periods - observed_control_period), 95))
        if not positive_periods.empty
        else np.nan
    )
    if not np.isfinite(elapsed) and time_values.notna().any():
        elapsed = float(time_values.max())
    if {"pose/x", "pose/y"}.issubset(history.columns):
        x_values = pd.to_numeric(history["pose/x"], errors="coerce")
        y_values = pd.to_numeric(history["pose/y"], errors="coerce")
        valid_pose = pd.DataFrame({"x": x_values, "y": y_values}).dropna()
        if len(valid_pose) > 1:
            total_distance = float(
                np.sqrt(valid_pose["x"].diff() ** 2 + valid_pose["y"].diff() ** 2).sum()
            )
    if not np.isfinite(total_distance) and distance_values.notna().any():
        total_distance = float(distance_values.max() - distance_values.min())

    mean_velocity = total_distance / elapsed if elapsed and elapsed > 0 and np.isfinite(total_distance) else np.nan
    final_entropy = _first_number(summary, ("final_entropy", "entropy_final"))
    if not np.isfinite(final_entropy) and entropy_values.notna().any():
        final_entropy = float(entropy_values.dropna().iloc[-1])
    initial_entropy = _first_number(summary, ("initial_entropy", "entropy_initial"))
    entropy_reduction = _first_number(summary, ("entropy_reduction",))
    if not np.isfinite(entropy_reduction) and np.isfinite(initial_entropy) and np.isfinite(final_entropy):
        entropy_reduction = initial_entropy - final_entropy
    tree_count = _tree_count(summary, config)
    entropy_per_tree = (
        entropy_reduction / tree_count
        if np.isfinite(entropy_reduction) and np.isfinite(tree_count) and tree_count > 0
        else np.nan
    )
    entropy_per_meter = (
        entropy_reduction / total_distance
        if np.isfinite(entropy_reduction) and np.isfinite(total_distance) and total_distance > 0
        else _first_number(summary, ("entropy_reduction_per_meter",))
    )
    entropy_per_second = (
        entropy_reduction / elapsed
        if np.isfinite(entropy_reduction) and np.isfinite(elapsed) and elapsed > 0
        else _first_number(summary, ("entropy_reduction_per_second",))
    )
    max_velocity = _max_velocity(config)
    velocity_headroom = (
        max(0.0, max_velocity - mean_velocity)
        if np.isfinite(max_velocity) and np.isfinite(mean_velocity)
        else np.nan
    )
    per_tree_velocity_reduction = _first_number(
        summary, ("mean_nearest_tree_speed_reduction_mps", "mean_velocity_reduction_per_tree_mps", "avg_velocity_reduction_per_tree_mps")
    )
    observed_max_velocity = _safe_number(
        pd.to_numeric(history.get("speed_mps", pd.Series(dtype=float)), errors="coerce").max()
    )
    observed_max_acceleration = _safe_number(
        pd.to_numeric(history.get("acceleration_mps2", pd.Series(dtype=float)), errors="coerce").max()
    )
    observed_min_clearance = _safe_number(
        pd.to_numeric(
            history.get("nearest_tree_distance_m", pd.Series(dtype=float)), errors="coerce"
        ).min()
    )
    tree_records = summary.get("tree_entropy_records", [])
    if not isinstance(tree_records, list):
        tree_records = []
    milestone_times = {}
    if {"time_execution_s", "num_tracked", "total_targets"}.issubset(history.columns):
        tracked_frame = history[["time_execution_s", "num_tracked", "total_targets"]].apply(
            pd.to_numeric, errors="coerce"
        ).dropna()
        fraction = tracked_frame["num_tracked"] / tracked_frame["total_targets"].replace(0, np.nan)
        for milestone in (0.5, 0.9):
            reached = tracked_frame.loc[fraction >= milestone, "time_execution_s"]
            milestone_times[milestone] = float(reached.iloc[0]) if not reached.empty else np.nan
    mean_compute = _first_number(summary, ("mean_controller_compute_time_ms",))
    total_steps = _first_number(summary, ("total_steps", "total_commands"))
    entropy_per_compute = _first_number(summary, ("entropy_reduction_per_compute_ms",))
    if not np.isfinite(entropy_per_compute) and np.isfinite(mean_compute) and np.isfinite(total_steps) and mean_compute * total_steps > 0:
        entropy_per_compute = entropy_reduction / (mean_compute * total_steps)

    return {
        "run_id": metadata["run_id"],
        "run_name": metadata["run_name"],
        "project": metadata["project"],
        "algorithm": metadata["algorithm"],
        "run_index": metadata["run_index"],
        "trial_seed": config.get("trial_seed", np.nan),
        "state": metadata["state"],
        "success_rate": _first_number(summary, ("success",)),
        "termination_reason": str(summary.get("termination_reason", "unknown")),
        "elapsed_time_s": elapsed,
        "total_distance_m": total_distance,
        "mean_velocity_mps": mean_velocity,
        "observed_control_period_s": observed_control_period,
        "observed_control_jitter_p95_s": observed_control_jitter_p95,
        "max_velocity_mps": max_velocity,
        "observed_max_velocity_mps": observed_max_velocity,
        "observed_max_acceleration_mps2": observed_max_acceleration,
        "observed_min_obstacle_clearance_m": observed_min_clearance,
        "mean_velocity_headroom_mps": velocity_headroom,
        "mean_nearest_tree_speed_reduction_mps": per_tree_velocity_reduction,
        "initial_entropy": initial_entropy,
        "final_entropy": final_entropy,
        "entropy_reduction": entropy_reduction,
        "tree_count": tree_count,
        "entropy_reduction_per_tree": entropy_per_tree,
        "entropy_reduction_per_meter": entropy_per_meter,
        "entropy_reduction_per_second": entropy_per_second,
        "entropy_reduction_per_observation_episode": _first_number(summary, ("entropy_reduction_per_observation_episode",)),
        "entropy_reduction_per_compute_ms": entropy_per_compute,
        "time_to_50pct_tracking_s": _first_number(summary, ("time_to_50pct_tracking_s",)) if np.isfinite(_first_number(summary, ("time_to_50pct_tracking_s",))) else milestone_times.get(0.5, np.nan),
        "time_to_90pct_tracking_s": _first_number(summary, ("time_to_90pct_tracking_s",)) if np.isfinite(_first_number(summary, ("time_to_90pct_tracking_s",))) else milestone_times.get(0.9, np.nan),
        "worst_tree_entropy_final": _first_number(summary, ("worst_tree_entropy_final",)),
        "p90_tree_entropy_final": _first_number(summary, ("p90_tree_entropy_final",)),
        "p95_tree_entropy_final": _first_number(summary, ("p95_tree_entropy_final",)),
        "unresolved_tree_count_auc_tree_s": _first_number(summary, ("unresolved_tree_count_auc_tree_s",)),
        "mean_tree_entropy_convergence_time_s": _first_number(
            summary, ("mean_tree_entropy_convergence_time_s",)
        ),
        "median_tree_entropy_convergence_time_s": _first_number(
            summary, ("median_tree_entropy_convergence_time_s",)
        ),
        "p95_tree_entropy_convergence_time_s": _first_number(
            summary, ("p95_tree_entropy_convergence_time_s",)
        ),
        "mean_tree_entropy_reduction_rate_bits_s": _first_number(
            summary, ("mean_tree_entropy_reduction_rate_bits_s",)
        ),
        "tree_entropy_completion_rate": _first_number(summary, ("tree_entropy_completion_rate",)),
        "mean_controller_compute_time_ms": _first_number(
            summary, ("mean_controller_compute_time_ms",)
        ),
        "median_controller_compute_time_ms": _first_number(
            summary, ("median_controller_compute_time_ms",)
        ),
        "p95_controller_compute_time_ms": _first_number(
            summary, ("p95_controller_compute_time_ms",)
        ),
        "mean_policy_inference_time_ms": _first_number(
            summary, ("mean_policy_inference_time_ms",)
        ),
        "rl_liveness_recovery_count": _first_number(
            summary, ("rl_liveness_recovery_count",)
        ),
        "rl_liveness_recovery_steps": _first_number(
            summary, ("rl_liveness_recovery_steps",)
        ),
        "tree_entropy_records_json": json.dumps(tree_records, sort_keys=True, default=str),
        "config_json": json.dumps(config, sort_keys=True, default=str),
    }


def expand_tree_metrics(runs):
    rows = []
    for _, run in runs.iterrows():
        try:
            records = json.loads(run.get("tree_entropy_records_json", "[]"))
        except (TypeError, ValueError, json.JSONDecodeError):
            records = []
        if not isinstance(records, list):
            continue
        for record in records:
            if not isinstance(record, dict):
                continue
            rows.append(
                {
                    "run_id": run["run_id"],
                    "algorithm": run["algorithm"],
                    "run_index": run["run_index"],
                    "trial_seed": run["trial_seed"],
                    **record,
                }
            )
    return pd.DataFrame(rows)


def _history_frame(run, metadata):
    records = list(run.scan_history(page_size=1000))
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        return frame
    available_metrics = [name for name in HISTORY_METRICS if name in frame.columns]
    frame = frame[available_metrics].copy()
    frame.insert(0, "run_id", metadata["run_id"])
    frame.insert(1, "algorithm", metadata["algorithm"])
    frame.insert(2, "run_index", metadata["run_index"])
    frame.insert(3, "trial_seed", metadata["config"].get("trial_seed", np.nan))
    return enrich_history(frame)


def enrich_history(frame):
    """Derive measured translational speed without relying on controller commands."""
    if frame.empty or "time_execution_s" not in frame.columns:
        return frame
    frame = frame.copy()
    time_values = pd.to_numeric(frame["time_execution_s"], errors="coerce")
    delta_t = time_values.diff()
    speed = None
    if {"pose/x", "pose/y"}.issubset(frame.columns):
        x_values = pd.to_numeric(frame["pose/x"], errors="coerce")
        y_values = pd.to_numeric(frame["pose/y"], errors="coerce")
        velocity_x = x_values.diff() / delta_t
        velocity_y = y_values.diff() / delta_t
        speed = np.sqrt(velocity_x ** 2 + velocity_y ** 2)
        frame["velocity_x_mps"] = velocity_x.where(delta_t > 0)
        frame["velocity_y_mps"] = velocity_y.where(delta_t > 0)
        acceleration = np.sqrt(velocity_x.diff() ** 2 + velocity_y.diff() ** 2) / delta_t
        frame["acceleration_mps2"] = acceleration.where(delta_t > 0).replace(
            [np.inf, -np.inf], np.nan
        )
    elif "distance_m" in frame.columns:
        distance = pd.to_numeric(frame["distance_m"], errors="coerce")
        speed = distance.diff() / delta_t
    if speed is not None:
        speed = speed.where(delta_t > 0).replace([np.inf, -np.inf], np.nan).clip(lower=0.0)
        if "speed_mps" not in frame.columns:
            frame["speed_mps"] = speed
        else:
            existing = pd.to_numeric(frame["speed_mps"], errors="coerce")
            frame["speed_mps"] = existing.fillna(speed)
    return frame


def download_wandb(projects, state="finished", tags=None):
    import wandb

    api = wandb.Api()
    run_rows = []
    history_frames = []
    for project in projects:
        filters = {}
        if state:
            filters["state"] = state
        if tags:
            filters["tags"] = {"$all": tags}
        for run in api.runs(project, filters=filters):
            config = dict(run.config or {})
            summary = dict(run.summary or {})
            metadata = {
                "run_id": run.id,
                "run_name": run.name or run.id,
                "project": project,
                "state": run.state,
                "config": config,
                "summary": summary,
            }
            metadata["algorithm"] = infer_algorithm(metadata["run_name"], config, project)
            metadata["run_index"] = infer_run_index(metadata["run_name"], config)
            history = _history_frame(run, metadata)
            run_rows.append(calculate_run_metrics(metadata, history))
            if not history.empty:
                history_frames.append(history)
    runs = pd.DataFrame(run_rows)
    histories = pd.concat(history_frames, ignore_index=True, sort=False) if history_frames else pd.DataFrame()
    return runs, histories


def summarize_runs(runs):
    rows = []
    for algorithm, group in runs.groupby("algorithm", dropna=False):
        for metric in REPORT_METRICS + [
            "mean_nearest_tree_speed_reduction_mps",
            "rl_liveness_recovery_count",
        ]:
            values = pd.to_numeric(
                group.get(metric, pd.Series(index=group.index, dtype=float)), errors="coerce"
            ).dropna()
            count = int(values.count())
            mean = float(values.mean()) if count else np.nan
            std = float(values.std(ddof=1)) if count > 1 else np.nan
            ci95 = 1.96 * std / math.sqrt(count) if count > 1 else np.nan
            rows.append(
                {
                    "algorithm": algorithm,
                    "metric": metric,
                    "n": count,
                    "mean": mean,
                    "std": std,
                    "median": float(values.median()) if count else np.nan,
                    "min": float(values.min()) if count else np.nan,
                    "max": float(values.max()) if count else np.nan,
                    "ci95_low": mean - ci95 if np.isfinite(ci95) else np.nan,
                    "ci95_high": mean + ci95 if np.isfinite(ci95) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def paired_comparisons(runs):
    rows = []
    algorithms = sorted(runs["algorithm"].dropna().unique())
    for algorithm_a, algorithm_b in combinations(algorithms, 2):
        left = runs[runs["algorithm"] == algorithm_a].dropna(subset=["trial_seed"])
        right = runs[runs["algorithm"] == algorithm_b].dropna(subset=["trial_seed"])
        left = left.drop_duplicates("trial_seed", keep=False)
        right = right.drop_duplicates("trial_seed", keep=False)
        paired = left.merge(right, on="trial_seed", suffixes=("_a", "_b"), validate="one_to_one")
        for metric in REPORT_METRICS:
            column_a = metric + "_a"
            column_b = metric + "_b"
            if column_a not in paired or column_b not in paired:
                continue
            delta = (
                pd.to_numeric(paired[column_a], errors="coerce")
                - pd.to_numeric(paired[column_b], errors="coerce")
            ).dropna()
            count = int(delta.count())
            mean = float(delta.mean()) if count else np.nan
            std = float(delta.std(ddof=1)) if count > 1 else np.nan
            ci95 = 1.96 * std / math.sqrt(count) if count > 1 else np.nan
            rows.append(
                {
                    "algorithm_a": algorithm_a,
                    "algorithm_b": algorithm_b,
                    "metric": metric,
                    "delta_definition": "algorithm_a - algorithm_b",
                    "n_pairs": count,
                    "mean_delta": mean,
                    "median_delta": float(delta.median()) if count else np.nan,
                    "std_delta": std,
                    "ci95_low": mean - ci95 if np.isfinite(ci95) else np.nan,
                    "ci95_high": mean + ci95 if np.isfinite(ci95) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _config_value(row, names):
    try:
        config = json.loads(row["config_json"])
    except (TypeError, ValueError, json.JSONDecodeError):
        return np.nan
    return _first_number(config, names)


def fairness_audit(runs):
    definitions = {
        "observed_control_period_s": (),
        "control_period_s": ("control_period", "dt", "delta_t"),
        "measurement_period_s": ("measurement_period",),
        "max_velocity_mps": ("max_velocity", "max_linear_velocity"),
        "max_yaw_velocity_radps": ("max_yaw_velocity",),
        "max_linear_accel_mps2": ("max_lin_accel", "max_accel_xy"),
        "max_yaw_accel_radps2": ("max_yaw_accel", "max_accel_yaw"),
        "obstacle_clearance_m": ("obstacle_avoidance_distance", "safe_distance"),
        "observation_range_m": ("observation_range", "obs_range"),
        "active_target_count": ("active_target_count", "k_obs", "num_target_trees"),
        "active_obstacle_count": ("active_obstacle_count", "num_obstacle_trees"),
        "belief_tracking_threshold": ("belief_tracking_threshold",),
        "max_experiment_steps": ("max_experiment_steps", "sim_steps"),
    }
    rows = []
    algorithm_values = {}
    for algorithm, group in runs.groupby("algorithm"):
        representative = group.iloc[0]
        values = {}
        for constraint, names in definitions.items():
            value = (
                _safe_number(group[constraint].median())
                if not names and constraint in group.columns
                else _config_value(representative, names)
            )
            if constraint == "control_period_s" and not np.isfinite(value):
                frequency = _config_value(representative, ("step_frequency", "hz"))
                value = 1.0 / frequency if np.isfinite(frequency) and frequency > 0 else np.nan
            values[constraint] = value
        algorithm_values[algorithm] = values

    for constraint in definitions:
        present = [values[constraint] for values in algorithm_values.values() if np.isfinite(values[constraint])]
        if constraint == "observed_control_period_s" and present:
            period_tolerance = max(0.005, 0.05 * float(np.median(present)))
            values_match = max(present) - min(present) <= period_tolerance
        else:
            values_match = bool(present) and np.allclose(present, present[0])
        comparable = len(present) == len(algorithm_values) and values_match
        for algorithm, values in algorithm_values.items():
            rows.append(
                {
                    "constraint": constraint,
                    "algorithm": algorithm,
                    "value": values[constraint],
                    "all_algorithms_match": bool(comparable),
                    "status": "PASS" if comparable else "FAIL",
                }
            )
    termination_values = {}
    for algorithm, group in runs.groupby("algorithm"):
        try:
            config = json.loads(group.iloc[0]["config_json"])
        except (TypeError, ValueError, json.JSONDecodeError):
            config = {}
        termination_values[algorithm] = str(config.get("termination_criterion", "missing"))
    termination_match = len(set(termination_values.values())) == 1 and "missing" not in termination_values.values()
    for algorithm, value in termination_values.items():
        rows.append(
            {
                "constraint": "termination_criterion",
                "algorithm": algorithm,
                "value": value,
                "all_algorithms_match": bool(termination_match),
                "status": "PASS" if termination_match else "FAIL",
            }
        )
    compliance = (
        ("velocity_limit_compliance", "observed_max_velocity_mps", ("max_velocity", "max_linear_velocity"), "max"),
        ("acceleration_limit_compliance", "observed_max_acceleration_mps2", ("max_lin_accel", "max_accel_xy"), "max"),
        ("obstacle_clearance_compliance", "observed_min_obstacle_clearance_m", ("obstacle_avoidance_distance", "safe_distance"), "min"),
    )
    for constraint, observed_column, configured_names, direction in compliance:
        for algorithm, group in runs.groupby("algorithm"):
            configured = _config_value(group.iloc[0], configured_names)
            observed_values = pd.to_numeric(
                group.get(observed_column, pd.Series(index=group.index, dtype=float)), errors="coerce"
            ).dropna()
            observed = (
                float(observed_values.max() if direction == "max" else observed_values.min())
                if not observed_values.empty
                else np.nan
            )
            tolerance = 0.05 * abs(configured) if np.isfinite(configured) else 0.0
            passed = (
                observed <= configured + tolerance
                if direction == "max" and np.isfinite(observed) and np.isfinite(configured)
                else observed >= configured - tolerance
                if direction == "min" and np.isfinite(observed) and np.isfinite(configured)
                else False
            )
            rows.append(
                {
                    "constraint": constraint,
                    "algorithm": algorithm,
                    "value": observed,
                    "configured_limit": configured,
                    "all_algorithms_match": np.nan,
                    "status": "PASS" if passed else "FAIL",
                }
            )
    return pd.DataFrame(rows)


def pairing_audit(runs):
    algorithms = sorted(runs["algorithm"].dropna().unique())
    valid = runs.dropna(subset=["trial_seed"])
    rows = []
    for trial_seed, group in valid.groupby("trial_seed"):
        counts = group.groupby("algorithm").size().to_dict()
        complete = all(counts.get(algorithm, 0) == 1 for algorithm in algorithms)
        row = {"trial_seed": trial_seed, "status": "PASS" if complete else "FAIL"}
        row.update({"{}_count".format(algorithm): counts.get(algorithm, 0) for algorithm in algorithms})
        rows.append(row)
    return pd.DataFrame(rows)


def plot_metric_boxplots(runs, output_path):
    sns.set_theme(style="whitegrid")
    columns = 3
    rows = int(math.ceil(len(REPORT_METRICS) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(16, 4.5 * rows), squeeze=False)
    labels = {
        "success_rate": "Success rate",
        "elapsed_time_s": "Elapsed time [s]",
        "mean_velocity_mps": "Mean velocity [m/s]",
        "final_entropy": "Final entropy [bits]",
        "mean_tree_entropy_convergence_time_s": "Mean tree convergence time [s]",
        "mean_tree_entropy_reduction_rate_bits_s": "Mean tree entropy reduction [bit/s]",
        "tree_entropy_completion_rate": "Tree completion rate",
        "mean_controller_compute_time_ms": "Controller compute time [ms]",
    }
    for axis, metric in zip(axes.flat, REPORT_METRICS):
        data = (
            runs[["algorithm", metric]].dropna()
            if metric in runs.columns
            else pd.DataFrame(columns=["algorithm", metric])
        )
        if not data.empty:
            sns.boxplot(data=data, x="algorithm", y=metric, ax=axis, showfliers=False)
            sns.stripplot(data=data, x="algorithm", y=metric, ax=axis, color="black", alpha=0.45, size=3)
        axis.set_title(labels[metric])
        axis.set_xlabel("")
        axis.tick_params(axis="x", rotation=30)
    for axis in axes.flat[len(REPORT_METRICS) :]:
        axis.set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_tree_entropy_metrics(tree_metrics, output_path):
    if tree_metrics.empty:
        return
    data = tree_metrics.copy()
    data["convergence_time_s"] = pd.to_numeric(data.get("convergence_time_s"), errors="coerce")
    data["entropy_reduction_rate_bits_s"] = pd.to_numeric(
        data.get("entropy_reduction_rate_bits_s"), errors="coerce"
    )
    data["completed"] = data.get("completed", False).astype(str).str.lower().eq("true")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    completed = data[data["completed"]]
    for axis, metric, title in (
        (axes[0], "convergence_time_s", "Completed-tree convergence time [s]"),
        (axes[1], "entropy_reduction_rate_bits_s", "Completed-tree reduction rate [bit/s]"),
    ):
        values = completed[["algorithm", metric]].dropna()
        if not values.empty:
            sns.boxplot(data=values, x="algorithm", y=metric, ax=axis, showfliers=False)
            sns.stripplot(data=values, x="algorithm", y=metric, ax=axis, color="black", alpha=0.35, size=2)
        axis.set_title(title)
        axis.set_xlabel("")
        axis.tick_params(axis="x", rotation=30)
    completion = data.groupby("algorithm", as_index=False)["completed"].mean()
    sns.barplot(data=completion, x="algorithm", y="completed", ax=axes[2])
    axes[2].set_title("Tree completion rate (censoring retained)")
    axes[2].set_xlabel("")
    axes[2].set_ylabel("Fraction")
    axes[2].set_ylim(0.0, 1.0)
    axes[2].tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_profiles(histories, metric, ylabel, output_path):
    required = {"algorithm", "run_id", "time_execution_s", metric}
    if histories.empty or not required.issubset(histories.columns):
        return
    data = histories[list(required)].copy()
    data["time_execution_s"] = pd.to_numeric(data["time_execution_s"], errors="coerce")
    data[metric] = pd.to_numeric(data[metric], errors="coerce")
    data = data.dropna()
    if data.empty:
        return
    maxima = data.groupby("run_id")["time_execution_s"].transform("max")
    data = data[maxima > 0].copy()
    data["normalized_time"] = data["time_execution_s"] / maxima[maxima > 0]
    data["time_bin"] = (data["normalized_time"] * 20).round() / 20.0
    fig, axis = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=data, x="time_bin", y=metric, hue="algorithm", estimator="mean", errorbar=("ci", 95), ax=axis)
    axis.set_xlabel("Normalized execution time")
    axis.set_ylabel(ylabel)
    axis.set_xlim(0.0, 1.0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_excel(path, runs, tree_metrics, summary, comparisons, fairness, pairing, histories):
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        runs.to_excel(writer, sheet_name="runs", index=False)
        summary.to_excel(writer, sheet_name="summary", index=False)
        comparisons.to_excel(writer, sheet_name="paired_comparisons", index=False)
        fairness.to_excel(writer, sheet_name="fairness_audit", index=False)
        pairing.to_excel(writer, sheet_name="pairing_audit", index=False)
        if not tree_metrics.empty:
            tree_metrics.to_excel(writer, sheet_name="tree_metrics", index=False)
        if not histories.empty:
            max_rows = 1_000_000
            for index, start in enumerate(range(0, len(histories), max_rows), start=1):
                histories.iloc[start : start + max_rows].to_excel(
                    writer, sheet_name="history_{}".format(index), index=False
                )


def write_readme(path, runs, tree_metrics, fairness, pairing):
    failed = fairness.loc[fairness["status"] == "FAIL", "constraint"].drop_duplicates().tolist()
    missing_tree_velocity = int(runs["mean_nearest_tree_speed_reduction_mps"].isna().sum())
    failed_pairs = int((pairing["status"] == "FAIL").sum()) if not pairing.empty else len(runs)
    text = [
        "# Batch experiment report",
        "",
        "Runs: {}".format(len(runs)),
        "Algorithms: {}".format(", ".join(sorted(runs["algorithm"].dropna().unique()))),
        "",
        "## Fairness",
        "",
        "Mismatched or missing constraints: {}.".format(", ".join(failed) if failed else "none"),
        "See `fairness_audit.csv` before interpreting ranking plots.",
        "Incomplete or duplicated paired trial seeds: {}. See `pairing_audit.csv`.".format(failed_pairs),
        "",
        "## Metric availability",
        "",
        "Per-tree entropy convergence records: {}. Trees that did not reach the threshold are retained with `censored=true`.".format(
            len(tree_metrics)
        ),
        "Controller compute time is measured locally with a monotonic high-resolution clock and excludes the control-loop sleep.",
        "",
        "`mean_nearest_tree_speed_reduction_mps` is unavailable for {}/{} runs. It is a nearest-tree slowdown diagnostic, not an information-quality metric; `entropy_reduction_per_meter` and `entropy_reduction_per_second` are the primary efficiency metrics.".format(
            missing_tree_velocity, len(runs)
        ),
    ]
    path.write_text("\n".join(text) + "\n", encoding="utf-8")


def generate_report(runs, histories, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    if runs.empty:
        raise RuntimeError("No runs matched the requested projects/filters.")
    if not histories.empty and "run_id" in histories.columns:
        histories = pd.concat(
            [enrich_history(group) for _, group in histories.groupby("run_id", sort=False)],
            ignore_index=True,
            sort=False,
        )
    pairing = pairing_audit(runs)
    passing_seeds = set(pairing.loc[pairing["status"] == "PASS", "trial_seed"]) if not pairing.empty else set()
    analysis_runs = runs[runs["trial_seed"].isin(passing_seeds)] if passing_seeds else runs
    summary = summarize_runs(analysis_runs)
    comparisons = paired_comparisons(analysis_runs)
    fairness = fairness_audit(runs)
    tree_metrics = expand_tree_metrics(runs)
    runs.to_csv(output_dir / "runs.csv", index=False)
    histories.to_csv(output_dir / "history.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    comparisons.to_csv(output_dir / "paired_comparisons.csv", index=False)
    fairness.to_csv(output_dir / "fairness_audit.csv", index=False)
    pairing.to_csv(output_dir / "pairing_audit.csv", index=False)
    tree_metrics.to_csv(output_dir / "tree_metrics.csv", index=False)
    plot_metric_boxplots(analysis_runs, output_dir / "metric_boxplots.png")
    analysis_history = histories
    if not histories.empty and "run_id" in histories.columns:
        analysis_history = histories[histories["run_id"].isin(set(analysis_runs["run_id"]))]
    _plot_profiles(analysis_history, "entropy", "Entropy [bits]", output_dir / "entropy_profiles.png")
    if "speed_mps" in analysis_history.columns:
        _plot_profiles(analysis_history, "speed_mps", "Velocity [m/s]", output_dir / "velocity_profiles.png")
    analysis_tree_metrics = tree_metrics
    if not tree_metrics.empty:
        analysis_tree_metrics = tree_metrics[tree_metrics["run_id"].isin(set(analysis_runs["run_id"]))]
    plot_tree_entropy_metrics(analysis_tree_metrics, output_dir / "tree_entropy_metrics.png")
    write_excel(
        output_dir / "batch_report.xlsx",
        runs,
        analysis_tree_metrics,
        summary,
        comparisons,
        fairness,
        pairing,
        histories,
    )
    write_readme(output_dir / "REPORT.md", runs, tree_metrics, fairness, pairing)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    if args.input_dir:
        input_dir = Path(args.input_dir)
        runs = pd.read_csv(input_dir / "runs.csv")
        history_path = input_dir / "history.csv"
        histories = pd.read_csv(history_path) if history_path.exists() else pd.DataFrame()
    else:
        if not args.project:
            raise SystemExit("Provide at least one --project or use --input-dir.")
        runs, histories = download_wandb(args.project, state=args.state, tags=args.tag)
    generate_report(runs, histories, output_dir)
    print("Report written to {}".format(output_dir.resolve()))


if __name__ == "__main__":
    main()
