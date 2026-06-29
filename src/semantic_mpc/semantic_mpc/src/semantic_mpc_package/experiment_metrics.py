import os
import time

import numpy as np
import rospy


class VelocityTreeMetrics:
    """Time-weighted measured speed reduction associated with the nearest tree."""

    def __init__(self, max_velocity, tree_radius):
        self.max_velocity = float(max_velocity)
        self.tree_radius = float(tree_radius)
        self.last_time = None
        self.last_xy = None
        self.speed_mps = 0.0
        self.nearest_tree_id = None
        self.nearest_tree_distance_m = np.nan
        self.reduction_integral = {}
        self.duration = {}

    def update(self, timestamp, pose, tree_positions=None):
        xy = np.asarray(pose, dtype=float).flatten()[:2]
        timestamp = float(timestamp)
        dt = 0.0 if self.last_time is None else timestamp - self.last_time
        if self.last_xy is not None and dt > 1e-9:
            self.speed_mps = float(np.linalg.norm(xy - self.last_xy) / dt)

        self.nearest_tree_id = None
        self.nearest_tree_distance_m = np.nan
        if tree_positions is not None and len(tree_positions):
            trees = np.asarray(tree_positions, dtype=float)[:, :2]
            distances = np.linalg.norm(trees - xy, axis=1)
            tree_id = int(np.argmin(distances))
            tree_distance = float(distances[tree_id])
            self.nearest_tree_id = tree_id
            self.nearest_tree_distance_m = tree_distance
            if dt > 1e-9 and tree_distance <= self.tree_radius:
                reduction = max(0.0, self.max_velocity - self.speed_mps)
                self.reduction_integral[tree_id] = self.reduction_integral.get(tree_id, 0.0) + reduction * dt
                self.duration[tree_id] = self.duration.get(tree_id, 0.0) + dt

        self.last_time = timestamp
        self.last_xy = xy.copy()
        return {
            "speed_mps": self.speed_mps,
            "nearest_tree_id": self.nearest_tree_id,
            "nearest_tree_distance_m": self.nearest_tree_distance_m,
            "velocity_reduction_mps": max(0.0, self.max_velocity - self.speed_mps),
        }

    def summary(self):
        per_tree = [
            self.reduction_integral[tree_id] / duration
            for tree_id, duration in self.duration.items()
            if duration > 0.0
        ]
        return {
            "mean_velocity_reduction_per_tree_mps": float(np.mean(per_tree)) if per_tree else np.nan,
            "velocity_reduction_tree_count": len(per_tree),
        }


class PerTreeEntropyMetrics:
    """Track per-tree entropy convergence with explicit right-censoring."""

    def __init__(self, threshold=0.025, start_epsilon=1e-4):
        self.threshold = float(threshold)
        self.start_epsilon = float(start_epsilon)
        self.initial = None
        self.last_entropy = None
        self.last_time = None
        self.start_time = None
        self.start_entropy = None
        self.completion_time = None
        self.completion_entropy = None
        self.entropy_auc = None

    @staticmethod
    def entropies(belief):
        values = np.asarray(belief, dtype=float)
        if values.ndim == 1:
            probabilities = np.clip(values, 1e-9, 1.0 - 1e-9)
            return -(
                probabilities * np.log2(probabilities)
                + (1.0 - probabilities) * np.log2(1.0 - probabilities)
            )
        probabilities = np.clip(values, 1e-9, 1.0)
        probabilities /= np.sum(probabilities, axis=1, keepdims=True)
        return -np.sum(probabilities * np.log2(probabilities), axis=1)

    def update(self, timestamp, belief):
        entropy = self.entropies(belief)
        timestamp = float(timestamp)
        if self.initial is None:
            self.initial = entropy.copy()
            self.last_entropy = entropy.copy()
            self.last_time = timestamp
            self.start_time = np.full(len(entropy), np.nan)
            self.start_entropy = np.full(len(entropy), np.nan)
            self.completion_time = np.full(len(entropy), np.nan)
            self.completion_entropy = np.full(len(entropy), np.nan)
            self.entropy_auc = np.zeros(len(entropy), dtype=float)
            return entropy
        if len(entropy) != len(self.initial):
            raise ValueError("Tree belief count changed during a run")

        dt = max(0.0, timestamp - self.last_time)
        for tree_id in range(len(entropy)):
            started = np.isfinite(self.start_time[tree_id])
            completed = np.isfinite(self.completion_time[tree_id])
            just_started = False
            if not started and entropy[tree_id] <= self.initial[tree_id] - self.start_epsilon:
                self.start_time[tree_id] = timestamp
                self.start_entropy[tree_id] = entropy[tree_id]
                started = True
                just_started = True
            if started and not completed and not just_started:
                self.entropy_auc[tree_id] += 0.5 * (
                    self.last_entropy[tree_id] + entropy[tree_id]
                ) * dt
            if started and not completed and entropy[tree_id] <= self.threshold:
                self.completion_time[tree_id] = timestamp
                self.completion_entropy[tree_id] = entropy[tree_id]

        self.last_entropy = entropy.copy()
        self.last_time = timestamp
        return entropy

    def records(self, final_time=None):
        if self.initial is None:
            return []
        final_time = self.last_time if final_time is None else float(final_time)
        records = []
        for tree_id in range(len(self.initial)):
            started = np.isfinite(self.start_time[tree_id])
            completed = np.isfinite(self.completion_time[tree_id])
            active_duration = (
                (self.completion_time[tree_id] if completed else final_time) - self.start_time[tree_id]
                if started
                else np.nan
            )
            convergence_time = active_duration if completed else np.nan
            terminal_entropy = (
                self.completion_entropy[tree_id] if completed else self.last_entropy[tree_id]
            )
            reduction = self.start_entropy[tree_id] - terminal_entropy if started else 0.0
            entropy_auc = self.entropy_auc[tree_id]
            if started and not completed and final_time > self.last_time:
                entropy_auc += self.last_entropy[tree_id] * (final_time - self.last_time)
            records.append(
                {
                    "tree_id": tree_id,
                    "started": bool(started),
                    "completed": bool(completed),
                    "censored": bool(not completed),
                    "start_time_s": float(self.start_time[tree_id]) if started else np.nan,
                    "completion_time_s": float(self.completion_time[tree_id]) if completed else np.nan,
                    "active_duration_s": float(active_duration) if started else np.nan,
                    "convergence_time_s": float(convergence_time) if completed else np.nan,
                    "initial_entropy": float(self.initial[tree_id]),
                    "start_entropy": float(self.start_entropy[tree_id]) if started else np.nan,
                    "final_entropy": float(self.last_entropy[tree_id]),
                    "entropy_reduction": float(reduction),
                    "entropy_reduction_rate_bits_s": (
                        float(reduction / active_duration)
                        if started and active_duration > 0.0
                        else np.nan
                    ),
                    "entropy_auc_bit_s": float(entropy_auc),
                }
            )
        return records

    def summary(self, final_time=None):
        records = self.records(final_time=final_time)
        completed = [record for record in records if record["completed"]]
        convergence = [record["convergence_time_s"] for record in completed]
        rates = [record["entropy_reduction_rate_bits_s"] for record in completed]
        return {
            "mean_tree_entropy_convergence_time_s": float(np.mean(convergence)) if convergence else np.nan,
            "median_tree_entropy_convergence_time_s": float(np.median(convergence)) if convergence else np.nan,
            "p95_tree_entropy_convergence_time_s": float(np.percentile(convergence, 95)) if convergence else np.nan,
            "mean_tree_entropy_reduction_rate_bits_s": float(np.mean(rates)) if rates else np.nan,
            "tree_entropy_completion_rate": float(len(completed) / len(records)) if records else 0.0,
            "tree_entropy_completed_count": len(completed),
            "tree_entropy_censored_count": len(records) - len(completed),
            "tree_entropy_records": records,
        }


class RosWandbLogger:
    def __init__(self, mode, run_index=0, params=None, default_project="semantic_mpc"):
        self.mode = mode
        self.run_index = run_index
        self.params = params or {}
        self.start_time = None
        self.last_log_time = 0.0
        self.run = None
        self.wandb = None

        try:
            import wandb
        except ImportError:
            rospy.logwarn("wandb is not installed; metrics will stay in memory only.")
            return

        self.wandb = wandb
        project = self.params.get("wandb_project", default_project)
        entity = self.params.get("wandb_entity") or None
        name = self.params.get("wandb_name") or "{}_run_{:03d}".format(mode, run_index)
        run_dir = self.params.get("run_dir")
        if run_dir:
            os.makedirs(run_dir, exist_ok=True)
        self.run = wandb.init(
            project=project,
            entity=entity,
            mode=self.params.get("wandb_mode", "offline"),
            name=name,
            dir=run_dir,
            config=self.params,
            reinit=True,
        )

    def start(self):
        self.start_time = time.time()
        self.last_log_time = 0.0

    def elapsed(self):
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time

    def log(self, values):
        if self.run is not None:
            self.wandb.log(values)

    def should_log(self, force=False):
        t = self.elapsed()
        if not force and t - self.last_log_time < self.params.get("wandb_log_period", 1.0):
            return False
        self.last_log_time = t
        return True

    def finish(self, summary=None):
        summary = summary or {}
        if self.run is not None:
            if summary:
                self.run.summary.update(summary)
            self.run.finish()
        return summary


class WandbMetrics(RosWandbLogger):
    def __init__(self, mode, run_index=0, params=None):
        super(WandbMetrics, self).__init__(mode, run_index, params, default_project="semantic_mpc")
        self.total_distance = 0.0
        self.rows = []
        self.velocity_metrics = VelocityTreeMetrics(
            self.params.get("max_velocity", 0.0),
            self.params.get("tree_velocity_radius", 5.0),
        )
        self.entropy_metrics = PerTreeEntropyMetrics(
            self.params.get("tree_entropy_threshold", 0.025),
            self.params.get("tree_entropy_start_epsilon", 1e-4),
        )

    def add_distance(self, distance):
        self.total_distance += float(distance)

    def log_pose(self, pose, entropy, belief=None, force=False, tree_positions=None):
        t = self.elapsed()
        x, y, theta = [float(v) for v in np.asarray(pose).flatten()[:3]]
        entropy = float(entropy)
        velocity = self.velocity_metrics.update(t, pose, tree_positions=tree_positions)
        if belief is not None:
            self.entropy_metrics.update(t, belief)
        self.rows.append(
            [
                t,
                x,
                y,
                theta,
                entropy,
                self.total_distance,
                velocity["speed_mps"],
                velocity["nearest_tree_id"],
                velocity["nearest_tree_distance_m"],
                velocity["velocity_reduction_mps"],
            ]
        )

        if self.run is None:
            return
        if not self.should_log(force=force):
            return
        payload = {
            "time_execution_s": t,
            "distance_m": self.total_distance,
            "entropy": entropy,
            "pose/x": x,
            "pose/y": y,
            "pose/theta": theta,
            **velocity,
        }
        if belief is not None:
            payload["belief_mean"] = float(np.mean(np.asarray(belief, dtype=float)))
        self.wandb.log(payload)

    def finish(self, initial_entropy, final_entropy, final_belief=None, extra=None):
        total_time = self.elapsed()
        entropy_reduction = float(initial_entropy - final_entropy)
        summary = {
            "total_time_execution_s": float(total_time),
            "total_distance_m": float(self.total_distance),
            "initial_entropy": float(initial_entropy),
            "final_entropy": float(final_entropy),
            "entropy_reduction": entropy_reduction,
            "mean_velocity_mps": float(self.total_distance / total_time) if total_time > 0.0 else 0.0,
            **self.velocity_metrics.summary(),
            **self.entropy_metrics.summary(final_time=total_time),
        }
        if extra:
            summary.update(extra)

        if self.run is not None:
            table = self.wandb.Table(
                columns=[
                    "time",
                    "x",
                    "y",
                    "theta",
                    "entropy",
                    "distance",
                    "speed_mps",
                    "nearest_tree_id",
                    "nearest_tree_distance_m",
                    "velocity_reduction_mps",
                ],
                data=self.rows,
            )
            self.wandb.log({"trajectory": table, **summary})
            self.run.summary.update(summary)
            if final_belief is not None:
                self.run.summary["final_belief"] = np.asarray(final_belief, dtype=float).flatten().tolist()
            self.run.finish()

        return summary
