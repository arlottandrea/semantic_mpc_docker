"""Reliability-weighted ripe/raw active-sensing primitives."""

import numpy as np
import torch


EPS = 1e-8


def synthesize_three_class(ripe_score, raw_score, eps=EPS):
    """Combine independent ripe/raw scores into ``[ripe, raw, nothing]``."""
    ripe = np.clip(np.asarray(ripe_score, dtype=float), 0.0, None)
    raw = np.clip(np.asarray(raw_score, dtype=float), 0.0, None)
    scale = np.maximum(1.0, ripe + raw)
    distribution = np.stack([ripe / scale, raw / scale, 1.0 - (ripe + raw) / scale], axis=-1)
    distribution[..., 2] = np.maximum(0.0, distribution[..., 2])
    distribution /= np.maximum(eps, distribution.sum(axis=-1, keepdims=True))
    return distribution


def conditional_fruit_entropy(probabilities, eps=EPS):
    """Shannon entropy of ripe/raw after excluding the nothing channel."""
    values = np.asarray(probabilities, dtype=float)
    fruit = np.clip(values[..., :2], 0.0, None)
    mass = fruit.sum(axis=-1, keepdims=True)
    conditional = fruit / np.maximum(eps, mass)
    terms = np.where(conditional > eps, conditional * np.log2(np.maximum(eps, conditional)), 0.0)
    return np.where(mass[..., 0] > eps, -terms.sum(axis=-1), 0.0)


def binary_channel_capacity(reliability, eps=EPS):
    """Capacity of a binary symmetric channel with accuracy in [0.5, 1]."""
    reliability = np.clip(np.asarray(reliability, dtype=float), 0.5, 1.0)
    error = 1.0 - reliability
    entropy = np.where(
        (error > eps) & (error < 1.0 - eps),
        -(error * np.log2(np.maximum(eps, error))
          + (1.0 - error) * np.log2(np.maximum(eps, 1.0 - error))),
        0.0,
    )
    return 1.0 - entropy


def reliability_weighted_eig(probabilities, reliability, eps=EPS):
    return conditional_fruit_entropy(probabilities, eps) * binary_channel_capacity(reliability, eps)


def fuse_binary_belief(prior, likelihood, eps=EPS):
    """Bayesian ripe/raw fusion with normalized finite output."""
    prior = np.asarray(prior, dtype=float)
    likelihood = np.asarray(likelihood, dtype=float)
    if prior.shape != likelihood.shape or prior.shape[-1] != 2:
        raise ValueError("prior and likelihood must have matching final dimension 2")
    posterior = np.clip(prior, 0.0, None) * np.clip(likelihood, 0.0, None)
    normalization = posterior.sum(axis=-1, keepdims=True)
    fallback = np.full_like(posterior, 0.5)
    return np.where(normalization > eps, posterior / np.maximum(eps, normalization), fallback)


def torch_reliability_weighted_eig(probabilities, reliability, eps=EPS):
    """Differentiable, vectorized PyTorch implementation."""
    fruit = probabilities[..., :2].clamp_min(0.0)
    mass = fruit.sum(dim=-1, keepdim=True)
    conditional = fruit / mass.clamp_min(eps)
    fruit_entropy = -(conditional * conditional.clamp_min(eps).log2()).sum(dim=-1)
    fruit_entropy = torch.where(mass[..., 0] > eps, fruit_entropy, torch.zeros_like(fruit_entropy))
    reliability = reliability.clamp(0.5, 1.0)
    error = 1.0 - reliability
    binary_entropy = -(error * error.clamp_min(eps).log2()
                       + reliability * reliability.clamp_min(eps).log2())
    return fruit_entropy * (1.0 - binary_entropy)
