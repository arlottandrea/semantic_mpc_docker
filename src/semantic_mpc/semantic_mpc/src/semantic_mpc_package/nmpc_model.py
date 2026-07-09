import json
import os
import re

import l4casadi as l4c
import rospy
import torch

from semantic_mpc_package.perception_model import MultiLayerPerceptron

torch.jit.set_fusion_strategy([("STATIC", 0)])


def _checkpoint_epoch(path):
    match = re.match(r"best_model_epoch_(\d+)\.pth", os.path.basename(path))
    if not match:
        return -1
    return int(match.group(1))


def _checkpoint_metadata(path):
    metadata_path = os.path.join(os.path.dirname(path), "training_metadata.json")
    if not os.path.isfile(metadata_path):
        return {}
    with open(metadata_path, "r", encoding="utf-8") as metadata_file:
        return json.load(metadata_file)


def _is_compatible_checkpoint(path, params):
    metadata = _checkpoint_metadata(path)
    labels = metadata.get("output_labels")
    expected_labels = params.get("nn_output_labels")
    if labels is not None and expected_labels is not None:
        if list(labels) != list(expected_labels):
            return False
    expected_architecture = {
        "hidden_size": int(params.get("hidden_size", 64)),
        "hidden_layers": int(params.get("hidden_layers", 3)),
        "yaw_harmonics": int(params.get("nn_yaw_harmonics", 1)),
        "include_alignment_features": bool(
            params.get("nn_include_alignment_features", False)
        ),
    }
    legacy_defaults = {
        "hidden_size": 64,
        "hidden_layers": 3,
        "yaw_harmonics": 1,
        "include_alignment_features": False,
    }
    for key, expected in expected_architecture.items():
        if key not in metadata:
            if expected != legacy_defaults[key]:
                return False
            continue
        if metadata[key] != expected:
            return False
    return True


def get_latest_best_model(params):
    model_dir = os.environ.get(
        "NMPC_MODEL_DIR",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"),
    )
    model_files = []
    for root, _, filenames in os.walk(model_dir):
        for filename in filenames:
            if re.match(r"best_model_epoch_(\d+)\.pth", filename):
                path = os.path.join(root, filename)
                if _is_compatible_checkpoint(path, params):
                    model_files.append(path)
    if not model_files:
        raise FileNotFoundError("No compatible model files found in {}".format(model_dir))
    latest_model = max(model_files, key=_checkpoint_epoch)
    model_path = latest_model
    rospy.loginfo("Loading model: %s", model_path)
    return model_path


def load_l4casadi_model(params):
    model = MultiLayerPerceptron(
        input_dim=int(params["nn_input_dim"]), hidden_size=int(params["hidden_size"]),
        hidden_layers=int(params["hidden_layers"]), output_dim=int(params["nn_output_dim"]),
        threshold=float(params["nn_threshold"]), gate_slope=float(params["nn_gate_slope"]),
        yaw_harmonics=int(params.get("nn_yaw_harmonics", 1)),
        include_alignment_features=bool(params.get("nn_include_alignment_features", False)),
        output_temperature=float(params.get("nn_output_temperature", 1.0)),
    )
    model.load_state_dict(torch.load(
        get_latest_best_model(params), map_location=torch.device(params["model_device"])
    ))
    model.eval()
    return l4c.L4CasADi(model, batched=True, device=params["model_device"], name="perception")
