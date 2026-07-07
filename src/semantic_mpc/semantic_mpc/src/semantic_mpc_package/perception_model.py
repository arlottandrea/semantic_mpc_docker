"""PyTorch perception surrogate shared by training and the NMPC runtime."""

import torch
import torch.nn.functional as F


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
    """Compact structured surrogate producing ``[visibility, raw, ripe]``.

    - Inputs keep the sine/cos trick for angles: call with raw yaw radians
      and the forward will cat sin(yaw), cos(yaw) to the features.
    ``visibility`` is in [0, 1]. The two class-conditional observation
    accuracies are in [0.5, 1], where 0.5 means neutral evidence.
    """
    def __init__(
        self,
        input_dim=3,
        hidden_size=128,
        hidden_layers=2,
        output_dim=3,
        threshold=8.0,
        gate_slope=10.0,
        dropout=0.0,
    ):
        super().__init__()
        if output_dim != 3:
            raise ValueError("structured perception model requires three outputs")
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

        # Outside the observation range the model emits ``nothing`` with
        # neutral ripe/raw evidence.
        distance = x[..., :2].norm(dim=-1)
        gate = torch.sigmoid(self.gate_slope * (self.threshold - distance))
        probability = torch.sigmoid(logits)
        visibility = gate.unsqueeze(-1) * probability[..., :1]
        accuracy = 0.5 + 0.5 * gate.unsqueeze(-1) * probability[..., 1:3]
        return torch.cat([visibility, accuracy], dim=-1)
