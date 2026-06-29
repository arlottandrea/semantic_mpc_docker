import os
import time

import numpy as np
import rospy


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

    def add_distance(self, distance):
        self.total_distance += float(distance)

    def log_pose(self, pose, entropy, belief=None, force=False):
        t = self.elapsed()
        x, y, theta = [float(v) for v in np.asarray(pose).flatten()[:3]]
        entropy = float(entropy)
        self.rows.append([t, x, y, theta, entropy, self.total_distance])

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
        }
        if extra:
            summary.update(extra)

        if self.run is not None:
            table = self.wandb.Table(
                columns=["time", "x", "y", "theta", "entropy", "distance"],
                data=self.rows,
            )
            self.wandb.log({"trajectory": table, **summary})
            self.run.summary.update(summary)
            if final_belief is not None:
                self.run.summary["final_belief"] = np.asarray(final_belief, dtype=float).flatten().tolist()
            self.run.finish()

        return summary
