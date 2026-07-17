import json
import os
import re

import rospy


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


def get_latest_best_model(params, label=None):
    compatibility_params = dict(params)
    if label:
        compatibility_params["nn_output_labels"] = ["accuracy_{}".format(label)]
    suffix = "_{}".format(label) if label else ""
    explicit_model = params.get("nn_model_path{}".format(suffix), params.get("nn_model_path"))
    if explicit_model not in (None, "", []):
        model_path = os.path.expandvars(os.path.expanduser(str(explicit_model)))
        if not os.path.isfile(model_path):
            raise FileNotFoundError("Configured nn_model_path does not exist: {}".format(model_path))
        if not _is_compatible_checkpoint(model_path, compatibility_params):
            raise ValueError("Configured nn_model_path is not compatible: {}".format(model_path))
        rospy.loginfo("Loading configured NMPC model: %s", model_path)
        return model_path

    configured_dir = params.get("nn_model_dir{}".format(suffix), params.get("nn_model_dir"))
    model_dir = os.path.expandvars(os.path.expanduser(str(configured_dir))) if configured_dir else None
    if not model_dir:
        model_dir = os.environ.get(
            "NMPC_MODEL_DIR",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"),
        )
    model_files = []
    for root, _, filenames in os.walk(model_dir):
        for filename in filenames:
            if re.match(r"best_model_epoch_(\d+)\.pth", filename):
                path = os.path.join(root, filename)
                if _is_compatible_checkpoint(path, compatibility_params):
                    model_files.append(path)
    if not model_files:
        raise FileNotFoundError("No compatible model files found in {}".format(model_dir))
    latest_model = max(model_files, key=_checkpoint_epoch)
    model_path = latest_model
    rospy.loginfo("Loading latest compatible NMPC model from %s: %s", model_dir, model_path)
    return model_path


def load_l4casadi_model(params):
    import l4casadi as l4c
    import torch

    from semantic_mpc_package.perception_model import DualReliabilityMLP, MultiLayerPerceptron

    torch.jit.set_fusion_strategy([("STATIC", 0)])
    def make_scalar():
        return MultiLayerPerceptron(
        input_dim=int(params["nn_input_dim"]), hidden_size=int(params["hidden_size"]),
        hidden_layers=int(params["hidden_layers"]), output_dim=1,
        threshold=float(params["nn_threshold"]), gate_slope=float(params["nn_gate_slope"]),
        yaw_harmonics=int(params.get("nn_yaw_harmonics", 1)),
        include_alignment_features=bool(params.get("nn_include_alignment_features", False)),
        output_temperature=float(params.get("nn_output_temperature", 1.0)),
    )
    raw_model, ripe_model = make_scalar(), make_scalar()
    device = torch.device(params["model_device"])
    raw_model.load_state_dict(torch.load(get_latest_best_model(params, "raw"), map_location=device))
    ripe_model.load_state_dict(torch.load(get_latest_best_model(params, "ripe"), map_location=device))
    model = DualReliabilityMLP(raw_model, ripe_model)
    model.eval()
    return l4c.L4CasADi(model, batched=True, device=params["model_device"], name="perception")
