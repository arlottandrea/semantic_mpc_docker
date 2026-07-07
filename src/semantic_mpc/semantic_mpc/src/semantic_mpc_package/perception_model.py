"""PyTorch perception surrogate shared by training and the NMPC runtime."""

import torch
import torch.nn.functional as F


class ThreeClassFieldMLP(torch.nn.Module):
    """Differentiable ``pose -> [ripe, raw, nothing]`` surrogate."""

    def __init__(self, input_dim=3, hidden_size=64, hidden_layers=3):
        super().__init__()
        in_features = input_dim + 1 if input_dim == 3 else input_dim
        layers = [torch.nn.Linear(in_features, hidden_size), torch.nn.GELU()]
        for _ in range(hidden_layers - 1):
            layers.extend([torch.nn.Linear(hidden_size, hidden_size), torch.nn.GELU()])
        layers.append(torch.nn.Linear(hidden_size, 3))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        if x.shape[-1] == 3:
            yaw = x[..., -1:]
            x = torch.cat([x[..., :-1], torch.sin(yaw), torch.cos(yaw)], dim=-1)
        return torch.softmax(self.network(x), dim=-1)


class ReliabilityMLP(torch.nn.Module):
    """Differentiable pose-dependent ripe/raw accuracy in [0.5, 1]."""

    def __init__(self, input_dim=3, hidden_size=32, hidden_layers=2):
        super().__init__()
        in_features = input_dim + 1 if input_dim == 3 else input_dim
        layers = [torch.nn.Linear(in_features, hidden_size), torch.nn.GELU()]
        for _ in range(hidden_layers - 1):
            layers.extend([torch.nn.Linear(hidden_size, hidden_size), torch.nn.GELU()])
        layers.append(torch.nn.Linear(hidden_size, 1))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        if x.shape[-1] == 3:
            yaw = x[..., -1:]
            x = torch.cat([x[..., :-1], torch.sin(yaw), torch.cos(yaw)], dim=-1)
        return 0.5 + 0.5 * torch.sigmoid(self.network(x))


class ResidualBlock(torch.nn.Module):
    """Simple residual block with expansion, GELU and LayerNorm."""
    def __init__(self, dim, expansion=2, dropout=0.0):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim, dim * expansion)
        self.act = torch.nn.GELU()
        self.fc2 = torch.nn.Linear(dim * expansion, dim)
        self.dropout = torch.nn.Dropout(dropout) if dropout and dropout > 0.0 else torch.nn.Identity()
        self.norm = torch.nn.LayerNorm(dim)

    def forward(self, x):
        h = self.fc1(x)
        h = self.act(h)
        h = self.dropout(self.fc2(h))
        return self.norm(x + h)


class MultiLayerPerceptron(torch.nn.Module):
    """Pose-dependent 2x3 conditional sensor likelihood.

    - Inputs keep the sine/cos trick for angles: call with raw yaw radians
      and the forward will cat sin(yaw), cos(yaw) to the features.
    Flattened output rows correspond to true classes ``[raw, ripe]`` and
    columns to observations ``[nothing, raw, ripe]``. Each row sums to one.
    """
    def __init__(
        self,
        input_dim=3,
        hidden_size=128,
        hidden_layers=2,
        output_dim=6,
        threshold=8.0,
        gate_slope=10.0,
        dropout=0.0,
    ):
        super().__init__()
        if output_dim != 6:
            raise ValueError("conditional observation model requires six outputs")
        in_features = input_dim + 1 if input_dim == 3 else input_dim
        self.input_layer = torch.nn.Linear(in_features, hidden_size)
        self.blocks = torch.nn.ModuleList([ResidualBlock(hidden_size, expansion=2, dropout=dropout) for _ in range(hidden_layers)])
        self.norm = torch.nn.LayerNorm(hidden_size)
        self.out_layer = torch.nn.Linear(hidden_size, output_dim)
        self.register_buffer("threshold", torch.tensor(float(threshold)))
        self.register_buffer("gate_slope", torch.tensor(float(gate_slope)))

    def forward(self, x):
        # angle handling: expect last column to be yaw in radians when present
        if x.shape[-1] == 3:
            angle = x[..., -1:]
            x = torch.cat([x[..., :-1], torch.sin(angle), torch.cos(angle)], dim=-1)

        h = self.input_layer(x)
        for blk in self.blocks:
            h = blk(h)
        h = self.norm(h)
        logits = self.out_layer(h)

        # Outside the observation range both true classes emit ``nothing``.
        distance = x[..., :2].norm(dim=-1)
        gate = torch.sigmoid(self.gate_slope * (self.threshold - distance))
        learned = torch.softmax(logits.reshape(*logits.shape[:-1], 2, 3), dim=-1)
        gate = gate[..., None, None]
        nothing = torch.zeros_like(learned)
        nothing[..., 0] = 1.0
        likelihood = gate * learned + (1.0 - gate) * nothing
        return likelihood.flatten(start_dim=-2)
