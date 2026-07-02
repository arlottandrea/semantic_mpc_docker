import importlib.util
from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).parents[1] / "scripts" / "generate_wandb_report.py"
SPEC = importlib.util.spec_from_file_location("generate_wandb_report", SCRIPT)
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


def test_every_report_metric_has_a_plot_label():
    assert set(REPORT.REPORT_METRICS) == set(REPORT.REPORT_METRIC_LABELS)


def test_tree_count_defaults_to_one_class_when_class_count_is_missing():
    assert REPORT._tree_count({"final_belief": [0.2, 0.8]}, {}) == 2


def test_tree_count_defaults_to_one_class_when_class_count_is_invalid():
    assert REPORT._tree_count({"final_belief": [0.2, 0.8]}, {"nclasses": "nan"}) == 2


def test_tree_count_uses_valid_class_count():
    assert REPORT._tree_count({"final_belief": [0.2, 0.8, 0.1, 0.9]}, {"nclasses": 2}) == 2


def test_control_period_uses_explicit_control_dt_not_wandb_row_spacing():
    metadata = {
        "summary": {},
        "config": {},
        "run_id": "test",
        "run_name": "test",
        "project": "test",
        "algorithm": "test",
        "run_index": 0,
        "state": "finished",
    }
    history = pd.DataFrame({"time_execution_s": [0.0, 1.0, 2.0], "control_dt_s": [0.25, 0.25, 0.25]})

    metrics = REPORT.calculate_run_metrics(metadata, history)

    assert metrics["observed_control_period_s"] == 0.25
    assert metrics["observed_logging_period_s"] == 1.0
