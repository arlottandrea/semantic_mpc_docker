import os
import re

import l4casadi as l4c
import rospy
import torch
import torch.nn.functional as F

torch.jit.set_fusion_strategy([("STATIC", 0)])


class MultiLayerPerceptron(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_size=64,
        hidden_layers=3,
        output_dim=2,
        threshold=8.0,
        gate_slope=10.0,
    ):
        super().__init__()
        in_features = input_dim if input_dim != 3 else input_dim + 1
        self.input_layer = torch.nn.Linear(in_features, hidden_size)
        self.hidden_layers = torch.nn.ModuleList(
            [torch.nn.Linear(hidden_size, hidden_size) for _ in range(hidden_layers)]
        )
        self.out_layer = torch.nn.Linear(hidden_size, output_dim)
        self.register_buffer("threshold", torch.tensor(threshold))
        self.register_buffer("gate_slope", torch.tensor(gate_slope))

    def forward(self, x):
        if x.shape[-1] == 3:
            sin_cos = torch.cat([torch.sin(x[..., -1:]), torch.cos(x[..., -1:])], dim=-1)
            x = torch.cat([x[..., :-1], sin_cos], dim=-1)

        raw_2d = x[..., :2]
        norm2d = raw_2d.norm(dim=-1)
        gate = torch.sigmoid(self.gate_slope * (self.threshold - norm2d))

        h = torch.tanh(self.input_layer(x))
        for layer in self.hidden_layers:
            h = torch.tanh(layer(h))
        logits = self.out_layer(h)
        return F.softmax(logits * gate.unsqueeze(-1), dim=-1)


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
