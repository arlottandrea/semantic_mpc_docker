"""
Set Transformer attention modules: MAB, SAB, PMA.

Ported from layers/modules_torch_ray086.py in the original active classification codebase.
Based on the Set Transformer paper (Lee et al., 2019).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MAB(nn.Module):
    """Multi-head Attention Block with residual connections and optional LayerNorm."""

    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K, mask_1d=None):
        Q_norm, K_norm = Q, K
        Q_norm = self.fc_q(Q_norm)
        K_norm, V_norm = self.fc_k(K_norm), self.fc_v(K_norm)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q_norm.split(dim_split, 2), 0)
        K_ = torch.cat(K_norm.split(dim_split, 2), 0)
        V_ = torch.cat(V_norm.split(dim_split, 2), 0)

        AW = Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V)
        A = torch.softmax(AW, 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)

        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O, A


class SAB(nn.Module):
    """Self-Attention Block: MAB where Q = K = input."""

    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X, mask_1d=None, output_attention=False):
        output, attention = self.mab(X, X, mask_1d)
        if output_attention:
            return output, attention
        else:
            return output


class PMA(nn.Module):
    """Pooling by Multi-head Attention: aggregates variable-length set to fixed-size output."""

    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X, mask_1d=None, output_attention=False):
        output, attention = self.mab(self.S.repeat(X.size(0), 1, 1), X, mask_1d)
        if output_attention:
            return output, attention
        else:
            return output
