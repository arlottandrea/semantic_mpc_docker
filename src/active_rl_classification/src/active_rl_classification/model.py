"""
Set-Transformer feature extractor for Stable-Baselines3.

Adapted from models/models_torch_ray086.py (SE_Attention_noParamSh_OA class)
in the original active classification codebase.

Architecture:
    1. location[3] -> FC(3, 128)
    2. SAB(128, 128, 4 heads, LayerNorm)
    3. cat(SAB_output[128], measurement[1], belief[1], tracked[1]) -> FC(131, 128)
    4. PMA(128, 4 heads, 1 seed) -> (batch, 1, 128) -> squeeze -> (batch, 128)
"""

import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from active_rl_classification.attention import SAB, PMA


class TreeClassFeatureExtractor(BaseFeaturesExtractor):
    """Set-transformer feature extractor for tree classification observations.

    Takes a Dict observation with all trees sorted by distance (padded to MAX_TARGETS)
    and produces a fixed 128-dim feature vector via attention pooling.

    Args:
        observation_space: The observation space (Dict).
        features_dim: Output feature dimension (default 128).
        dim_hidden: Hidden dimension for attention layers (default 128).
        num_heads: Number of attention heads (default 4).
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 128,
        dim_hidden: int = 128,
        num_heads: int = 4,
    ):
        super().__init__(observation_space, features_dim)

        # Location encoder: location[3] -> dim_hidden
        self._pv_encoder = nn.Sequential(
            nn.Linear(3, dim_hidden),
        )

        # Self-attention block
        self._SAB = SAB(dim_hidden, dim_hidden, num_heads, ln=True)

        # EBS encoder: (SAB_output[dim_hidden] + measurement[1] + belief[1] + tracked[1]) -> dim_hidden
        self._ebs_encoder = nn.Sequential(
            nn.Linear(dim_hidden + 3, dim_hidden),
        )

        # Pooling by multi-head attention: variable-length -> fixed 1 x dim_hidden
        self._PMA = PMA(dim_hidden, num_heads, 1, ln=True)

    def forward(self, observations):
        location = observations["location"]        # (batch, k_obs, 3)
        belief = observations["belief"]            # (batch, k_obs, 1)
        measurement = observations["measurement"]  # (batch, k_obs, 1)
        tracked = observations["tracked"]          # (batch, k_obs, 1)
        mask = observations["mask"]                # (batch, k_obs)

        target_ebs = torch.cat([measurement, belief, tracked], dim=-1)  # (batch, k_obs, 3)

        # Apply mask: zero out padded slots
        mask_3d = mask.unsqueeze(-1)  # (batch, k_obs, 1)
        location = location * mask_3d
        target_ebs = target_ebs * mask_3d

        # Encode location
        t_pv_enc = self._pv_encoder(location)  # (batch, k_obs, dim_hidden)

        # Self-attention
        latent_features_SAB = self._SAB(t_pv_enc)  # (batch, k_obs, dim_hidden)

        # Concatenate with sensor info and encode
        lf_with_ebs = torch.cat([latent_features_SAB, target_ebs], dim=-1)  # (batch, k_obs, dim_hidden+3)
        lf_encoded = self._ebs_encoder(lf_with_ebs)  # (batch, k_obs, dim_hidden)

        # Apply mask again after encoding
        lf_encoded = lf_encoded * mask_3d

        # Pool to fixed-size output
        pooled = self._PMA(lf_encoded)  # (batch, 1, dim_hidden)
        pooled = pooled.squeeze(1)       # (batch, dim_hidden)

        return pooled
