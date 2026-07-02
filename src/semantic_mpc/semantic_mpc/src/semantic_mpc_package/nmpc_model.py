import os
import re

import l4casadi as l4c
import rospy
import torch

from semantic_mpc_package.perception_model import MultiLayerPerceptron

torch.jit.set_fusion_strategy([("STATIC", 0)])


def get_latest_best_model(label):
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", label)
    model_files = [
        filename
        for filename in os.listdir(model_dir)
        if re.match(r"best_model_epoch_(\d+)\.pth", filename)
    ]
    if not model_files:
        raise FileNotFoundError("No model files found in {}".format(model_dir))
    latest_model = max(
        model_files,
        key=lambda filename: int(re.match(r"best_model_epoch_(\d+)\.pth", filename).group(1)),
    )
    model_path = os.path.join(model_dir, latest_model)
    rospy.loginfo("Loading model: %s", model_path)
    return model_path


def load_l4casadi_models(params):
    models = []
    for label in list(params["model_labels"]):
        model = MultiLayerPerceptron(
            input_dim=int(params["nn_input_dim"]),
            hidden_size=int(params["hidden_size"]),
            hidden_layers=int(params["hidden_layers"]),
            output_dim=int(params["nn_output_dim"]),
            threshold=float(params["nn_threshold"]),
            gate_slope=float(params["nn_gate_slope"]),
        )
        model.load_state_dict(
            torch.load(get_latest_best_model(label), map_location=torch.device(params["model_device"]))
        )
        model.eval()
        models.append(
            l4c.L4CasADi(
                model,
                batched=True,
                device=params["model_device"],
                name=label,
            )
        )
    return models
