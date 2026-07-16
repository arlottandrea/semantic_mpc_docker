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
    """Pose-dependent 2x2 conditional sensor likelihood.

    - Inputs keep the sine/cos trick for angles: call with raw yaw radians
      and the forward will cat sin(yaw), cos(yaw) to the features.
    Flattened output rows correspond to true classes ``[raw, ripe]`` and
    columns to observations ``[raw, ripe]``. Each row sums to one.
    """
    def __init__(
        self,
        input_dim=3,
        hidden_size=128,
        hidden_layers=2,
        output_dim=4,
        threshold=8.0,
        gate_slope=10.0,
        dropout=0.0,
        yaw_harmonics=1,
        include_alignment_features=False,
        output_temperature=1.0,
    ):
        super().__init__()
        if output_dim != 4:
            raise ValueError("binary conditional observation model requires four outputs")
        self.input_dim = int(input_dim)
        self.yaw_harmonics = max(1, int(yaw_harmonics))
        self.include_alignment_features = bool(include_alignment_features)
        self.output_temperature = max(float(output_temperature), 1e-3)
        if self.input_dim == 3:
            in_features = 2 + 2 * self.yaw_harmonics
            if self.include_alignment_features:
                in_features += 5
        else:
            in_features = self.input_dim
        self.input_layer = torch.nn.Linear(in_features, hidden_size)
        self.blocks = torch.nn.ModuleList([ResidualBlock(hidden_size, expansion=2, dropout=dropout) for _ in range(hidden_layers)])
        self.norm = torch.nn.LayerNorm(hidden_size)
        self.out_layer = torch.nn.Linear(hidden_size, output_dim)
        self.register_buffer("threshold", torch.tensor(float(threshold)))
        self.register_buffer("gate_slope", torch.tensor(float(gate_slope)))

    def encode_pose(self, x):
        if x.shape[-1] != 3:
            return x
        xy = x[..., :2]
        yaw = x[..., -1:]
        harmonic_features = []
        for harmonic in range(1, self.yaw_harmonics + 1):
            angle = harmonic * yaw
            harmonic_features.extend([torch.sin(angle), torch.cos(angle)])
        encoded = [xy, *harmonic_features]
        if self.include_alignment_features:
            distance = xy.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            direction_to_tree = -xy / distance
            heading = torch.cat([torch.cos(yaw), torch.sin(yaw)], dim=-1)
            facing = (heading * direction_to_tree).sum(dim=-1, keepdim=True)
            lateral = (
                heading[..., :1] * direction_to_tree[..., 1:2]
                - heading[..., 1:2] * direction_to_tree[..., :1]
            )
            encoded.extend([distance, direction_to_tree, facing, lateral])
        return torch.cat(encoded, dim=-1)

    def forward(self, x):
        pose_xy = x[..., :2]
        x = self.encode_pose(x)

        h = self.input_layer(x)
        for blk in self.blocks:
            h = blk(h)
        h = self.norm(h)
        logits = self.out_layer(h)

        # Outside the observation range both rows become uninformative.
        distance = pose_xy.norm(dim=-1)
        gate = torch.sigmoid(self.gate_slope * (self.threshold - distance))
        learned = torch.softmax(
            logits.reshape(*logits.shape[:-1], 2, 2) / self.output_temperature,
            dim=-1,
        )
        gate = gate[..., None, None]
        neutral = torch.full_like(learned, 0.5)
        likelihood = gate * learned + (1.0 - gate) * neutral
        return likelihood.flatten(start_dim=-2)
