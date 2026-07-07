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
    """Compact, fast residual MLP producing per-head probabilities in [0,1].

    - Inputs keep the sine/cos trick for angles: call with raw yaw radians
      and the forward will cat sin(yaw), cos(yaw) to the features.
    - Outputs are independent probabilities (sigmoid per head). This fits
      both scalar and multi-head targets.
    """
    def __init__(
        self,
        input_dim=3,
        hidden_size=128,
        hidden_layers=2,
        output_dim=2,
        threshold=8.0,
        gate_slope=10.0,
        dropout=0.0,
    ):
        super().__init__()
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

        # distance-based gating (keeps behavior similar to previous model)
        distance = x[..., :2].norm(dim=-1)
        gate = torch.sigmoid(self.gate_slope * (self.threshold - distance))
        # These heads represent observation accuracy, not an unconstrained
        # Bernoulli probability.  Accuracy 0.5 is neutral evidence; values
        # below 0.5 would be interpreted by Bayes as an informative inverted
        # classifier.  The visibility gate therefore interpolates between
        # neutral evidence and learned accuracy in [0.5, 1].
        confidence = torch.sigmoid(logits)
        return 0.5 + 0.5 * gate.unsqueeze(-1) * confidence
