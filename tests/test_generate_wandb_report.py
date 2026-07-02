import importlib.util
from pathlib import Path


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
