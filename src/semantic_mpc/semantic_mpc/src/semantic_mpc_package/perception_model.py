"""PyTorch perception surrogate shared by training and the NMPC runtime."""

import torch
import torch.nn.functional as F


class MultiLayerPerceptron(torch.nn.Module):
    def __init__(self, input_dim=3, hidden_size=64, hidden_layers=3,
                 output_dim=2, threshold=8.0, gate_slope=10.0):
        super().__init__()
        in_features = input_dim + 1 if input_dim == 3 else input_dim
        self.input_layer = torch.nn.Linear(in_features, hidden_size)
        self.hidden_layers = torch.nn.ModuleList(
            [torch.nn.Linear(hidden_size, hidden_size) for _ in range(hidden_layers)]
        )
        self.out_layer = torch.nn.Linear(hidden_size, output_dim)
        self.register_buffer("threshold", torch.tensor(float(threshold)))
        self.register_buffer("gate_slope", torch.tensor(float(gate_slope)))

    def forward(self, x):
        if x.shape[-1] == 3:
            angle = x[..., -1:]
            x = torch.cat([x[..., :-1], torch.sin(angle), torch.cos(angle)], dim=-1)
        distance = x[..., :2].norm(dim=-1)
        gate = torch.sigmoid(self.gate_slope * (self.threshold - distance))
        h = torch.tanh(self.input_layer(x))
        for layer in self.hidden_layers:
            h = torch.tanh(layer(h))
        return F.softmax(self.out_layer(h) * gate.unsqueeze(-1), dim=-1)
