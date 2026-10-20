"""Actual anchor occupancy, bounded corrections and frozen-observation relations.

Hungarian is used in BOTH the training forward and inference. Sinkhorn provides
only a biased straight-through gradient; its barycentres are not the output.
"""
from __future__ import annotations

import copy
import numpy as np
import torch
from torch import nn
from scipy.optimize import linear_sum_assignment


def support_anchors(count, normals=None, boundaries=None):
    """Quasi-uniform area samples of disk intersected with observed halfplanes.

    Polar area CDF handles clipped circles without a Cartesian grid. The anchor
    support is a geometry prior. It does not estimate biological cell density.
    """
    normals = np.asarray(normals if normals is not None else [], dtype=float).reshape(-1, 2)
    boundaries = np.asarray(boundaries if boundaries is not None else [], dtype=float)
    if count == 0:
        return np.empty((0, 2), dtype=np.float32)
    if count == 1:
        return np.zeros((1, 2), dtype=np.float32)
    angles = (np.arange(8192) + .5) * 2 * np.pi / 8192
    directions = np.column_stack([np.cos(angles), np.sin(angles)])
    max_radius = np.ones(len(angles))
    for normal, boundary in zip(normals, boundaries):
        positive = directions @ normal > 1e-10
        max_radius[positive] = np.minimum(max_radius[positive], boundary / (directions[positive] @ normal))
    area = max_radius ** 2
    cdf = np.cumsum(area) / area.sum()
    golden = (np.sqrt(5) - 1) / 2
    angular_quantiles = ((np.arange(count) + .5) * golden) % 1
    angle_index = np.searchsorted(cdf, angular_quantiles)
    radial_quantiles = (np.arange(count) + .5) / count
    anchors = directions[angle_index] * (np.sqrt(radial_quantiles) * max_radius[angle_index])[:, None]
    return anchors.astype(np.float32)


def boundary_and_spacing(anchors, normals=None, boundaries=None):
    anchors = np.asarray(anchors)
    if not len(anchors):
        return np.empty(0), np.empty(0)
    distance = np.linalg.norm(anchors[:, None] - anchors[None], axis=2)
    np.fill_diagonal(distance, np.inf)
    nearest = distance.min(1) if len(anchors) > 1 else np.ones(1)
    margin = 1 - np.linalg.norm(anchors, axis=1)
    if normals is not None and len(normals):
        margin = np.minimum(margin, (np.asarray(boundaries)[None] - anchors @ np.asarray(normals).T).min(1))
    assert margin.min() > 0
    return margin.astype(np.float32), nearest.astype(np.float32)


def sinkhorn(logits, temperature=.2, iterations=40):
    """Square doubly stochastic relaxation, unit row and column capacities."""
    log_q = logits / temperature
    for _ in range(iterations):
        log_q = log_q - torch.logsumexp(log_q, dim=1, keepdim=True)
        log_q = log_q - torch.logsumexp(log_q, dim=0, keepdim=True)
    return log_q.exp()


def hard_occupancy(logits, relaxed):
    row, col = linear_sum_assignment(-logits.detach().cpu().numpy())
    hard = torch.zeros_like(relaxed)
    hard[row, col] = 1
    return hard - relaxed.detach() + relaxed, hard


def bounded_vector(raw):
    length = raw.norm(dim=1, keepdim=True)
    factor = torch.where(length < 1e-4, 1-length.square()/3,
                         torch.tanh(length)/length.clamp_min(1e-7))
    return factor * raw


def double_center(energy):
    return energy - energy.mean(1, keepdims=True) - energy.mean(0, keepdims=True) + energy.mean()


def build_groups(centers, membership, z_sc, z_sp, original_anchors, radius=.5):
    centers = np.asarray(centers, dtype=float)
    spacing = np.linalg.norm(centers[:, None] - centers[None], axis=2)
    np.fill_diagonal(spacing, np.inf)
    bandwidth = np.median(spacing.min(1))
    groups = []
    energy_contrasts = []
    for j in range(len(centers)):
        indices = np.flatnonzero(membership == j)
        if not len(indices):
            continue
        adjacent = np.flatnonzero(spacing[j] < 2 * radius - 1e-8)
        vectors = (centers[adjacent] - centers[j]) / radius
        lengths = np.linalg.norm(vectors, axis=1)
        normals = vectors / lengths[:, None]
        boundaries = lengths / 2
        anchors = support_anchors(len(indices), normals, boundaries)
        margin, nearest = boundary_and_spacing(anchors, normals, boundaries)
        # Interpolate the original 32 anchor logits to adaptive physical slots.
        d2 = np.sum((anchors[:, None] - original_anchors[None]) ** 2, axis=2)
        interpolation = np.exp(-d2 / (2 * .2 ** 2))
        interpolation /= interpolation.sum(1, keepdims=True)
        neighbors = np.r_[j, np.argsort(spacing[j])[:min(6, len(centers)-1)]]
        affinities = z_sc[indices] @ z_sp[neighbors].T
        energies = []
        for direction_sign in [1, -1]:
            positions = centers[j] + direction_sign * radius * anchors
            log_w = -np.sum((positions[:, None] - centers[neighbors][None]) ** 2, axis=2) / (2 * bandwidth ** 2)
            weights = np.exp(log_w - log_w.max(1, keepdims=True))
            weights /= weights.sum(1, keepdims=True)
            energies.append(double_center(-affinities @ weights.T))
        energy_contrasts.append(energies[0].reshape(-1))
        groups.append(dict(spot=j, indices=indices, anchors=anchors, margin=margin,
                           nearest=nearest, normals=normals, boundaries=boundaries,
                           interpolation=interpolation, energy_raw=energies[0], reverse_raw=energies[1]))
    # One dataset-wide scale. A floor prevents amplifying nearly flat RNA evidence.
    energy_scale = max(.02, float(np.std(np.concatenate(energy_contrasts))))
    for group in groups:
        group['energy'] = group['energy_raw'] / energy_scale
        group['reverse_energy'] = group['reverse_raw'] / energy_scale
        for key in ['anchors', 'margin', 'nearest', 'interpolation', 'energy', 'reverse_energy']:
            group[key] = torch.as_tensor(group[key], dtype=torch.float32)
        group['indices_t'] = torch.as_tensor(group['indices'], dtype=torch.long)
    return groups, dict(neighbor_bandwidth=float(bandwidth), neighbor_energy_scale=energy_scale,
                        double_centered_energy_sd=float(np.std(np.concatenate(energy_contrasts))),
                        support='unit disk intersected with observed-center nearest-center halfplanes')


class OccupiedAnchorDecoder(nn.Module):
    """Original learned backbone/anchor scores, actual slots and scaled residual."""

    def __init__(self, original_decoder, residual_fraction=.35):
        super().__init__()
        self.backbone = copy.deepcopy(original_decoder.backbone)
        # Stable occupancy during fine-tuning, rather than dropout-driven reorders.
        for module in self.backbone.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0
        self.anchor_head = copy.deepcopy(original_decoder.anchor_head)
        self.residual_head = copy.deepcopy(original_decoder.residual_head)
        self.anchor_context = nn.Linear(2, self.residual_head.in_features, bias=False)
        nn.init.zeros_(self.anchor_context.weight)
        self.residual_fraction = residual_fraction

    def forward(self, inputs, centers, groups, radius=.5, kappa=.3, temperature=.2,
                soft_readout=False, reverse_neighbors=False, residual_fraction=None,
                free_residual=False):
        eta = self.residual_fraction if residual_fraction is None else residual_fraction
        if not 0 <= eta < .5:
            raise ValueError('residual_fraction must be in [0, .5)')
        features = self.backbone(inputs)
        native_logits = self.anchor_head(features)
        result = torch.zeros((len(inputs), 2), dtype=inputs.dtype, device=inputs.device)
        details = []
        for group in groups:
            idx = group['indices_t']
            energy = group['reverse_energy'] if reverse_neighbors else group['energy']
            logits = native_logits[idx] @ group['interpolation'].T - kappa * energy
            relaxed = sinkhorn(logits, temperature)
            occupied, hard = hard_occupancy(logits, relaxed)
            allocation = relaxed if soft_readout else occupied
            base = allocation @ group['anchors']
            raw_residual = self.residual_head(features[idx] + self.anchor_context(base))
            if free_residual:
                bound = torch.full_like(group['margin'], .15)
            else:
                bound = torch.minimum(eta * group['nearest'], .98 * group['margin'])
            cell_bound = allocation @ bound
            residual = cell_bound[:, None] * bounded_vector(raw_residual)
            offset = base + residual
            if free_residual:
                offset = offset / offset.norm(dim=1, keepdim=True).clamp_min(1)
            result[idx] = centers[group['spot']] + radius * offset
            details.append(dict(group=group, relaxed=relaxed, hard=hard,
                                base=base, residual=residual, bounds=cell_bound,
                                coordinates=result[idx], energy=energy, logits=logits))
        return result, details


def relation_objective(coordinates, details, latent, edge_sc, radius=.5,
                       kappa=.3, global_weight=.1, residual_weight=.01):
    local_terms = []
    neighbor_terms = []
    residual_terms = []
    for d in details:
        group = d['group']; idx = group['indices_t']; count = len(idx)
        if count > 1:
            pair = torch.triu_indices(count, count, offset=1)
            spatial = (d['coordinates'][pair[0]] - d['coordinates'][pair[1]]).norm(dim=1)
            observed = (latent[idx[pair[0]]] - latent[idx[pair[1]]]).norm(dim=1)
            scaffold = (group['anchors'][pair[0]] - group['anchors'][pair[1]]).norm(dim=1).mean() * radius
            target = observed / observed.mean().clamp_min(.02)
            local_terms.append(((spatial / scaffold.clamp_min(1e-7) - target) ** 2).mean())
        coupling = d['hard'] - d['relaxed'].detach() + d['relaxed']
        neighbor_terms.append((coupling * d['energy']).sum() / count)
        residual_terms.append((d['residual'].norm(dim=1) / d['bounds'].clamp_min(1e-6)).square().mean())
    zero = coordinates.new_tensor(0.)
    local = torch.stack(local_terms).mean() if local_terms else zero
    neighbor = torch.stack(neighbor_terms).mean()
    correction = torch.stack(residual_terms).mean()
    src, dst = edge_sc
    physical = (coordinates[src] - coordinates[dst]).norm(dim=1)
    transcription = (latent[src] - latent[dst]).norm(dim=1)
    global_relation = ((physical / physical.mean().clamp_min(.02) - transcription / transcription.mean().clamp_min(.02)) ** 2).mean()
    total = local + global_weight * global_relation + kappa * neighbor + residual_weight * correction
    return dict(total=total, local=local, global_relation=global_relation,
                neighbor=neighbor, residual_prior=correction)
