"""Loss functions for SpaLMC."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def reconstruct_spot_expression(
    assignment: torch.Tensor,
    x_sc: torch.Tensor,
    candidate_spots: torch.Tensor | None = None,
    n_spots: int | None = None,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aggregate assigned cells to reconstruct spot expression."""

    if assignment.ndim != 2:
        raise ValueError("`assignment` must have shape n_cells x n_spots or n_cells x k.")
    if candidate_spots is not None:
        if n_spots is None:
            n_spots = int(candidate_spots.max().item()) + 1
        flat_spots = candidate_spots.reshape(-1)
        flat_weights = assignment.reshape(-1)
        weighted_x = x_sc[:, None, :] * assignment[:, :, None]
        x_sum = x_sc.new_zeros((n_spots, x_sc.shape[1]))
        spot_mass = x_sc.new_zeros(n_spots)
        x_sum.index_add_(0, flat_spots, weighted_x.reshape(-1, x_sc.shape[1]))
        spot_mass.index_add_(0, flat_spots, flat_weights)
        return x_sum / spot_mass[:, None].clamp_min(eps), spot_mass

    spot_mass = assignment.sum(dim=0)
    x_hat = assignment.T @ x_sc
    x_hat = x_hat / spot_mass[:, None].clamp_min(eps)
    return x_hat, spot_mass


def expression_reconstruction_loss(
    x_sp_hat: torch.Tensor,
    x_sp: torch.Tensor,
    mode: str = "cosine",
    log1p: bool = False,
    mask: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compare reconstructed spot expression with observed spot expression."""

    pred = x_sp_hat
    target = x_sp
    if mask is not None:
        if mask.sum() == 0:
            return x_sp_hat.new_tensor(0.0)
        pred = pred[mask]
        target = target[mask]
    if log1p:
        pred = torch.log1p(pred.clamp_min(0.0))
        target = torch.log1p(target.clamp_min(0.0))
    if mode == "mse":
        return F.mse_loss(pred, target)
    if mode == "cosine":
        return (1.0 - F.cosine_similarity(pred, target, dim=1, eps=eps)).mean()
    if mode == "poisson_nll":
        return F.poisson_nll_loss(pred.clamp_min(eps), target.clamp_min(0.0), log_input=False)
    if mode == "negative_binomial":
        raise NotImplementedError("Negative-binomial expression loss is reserved for future work.")
    raise ValueError("`mode` must be one of {'mse', 'cosine', 'poisson_nll', 'negative_binomial'}.")


def predict_spot_composition(
    assignment: torch.Tensor,
    cell_type_onehot: torch.Tensor,
    candidate_spots: torch.Tensor | None = None,
    n_spots: int | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Predict spot cell-type composition from soft assignments."""

    if candidate_spots is not None:
        if n_spots is None:
            n_spots = int(candidate_spots.max().item()) + 1
        weighted = cell_type_onehot[:, None, :] * assignment[:, :, None]
        pred = cell_type_onehot.new_zeros((n_spots, cell_type_onehot.shape[1]))
        pred.index_add_(0, candidate_spots.reshape(-1), weighted.reshape(-1, cell_type_onehot.shape[1]))
    else:
        pred = assignment.T @ cell_type_onehot
    return pred / pred.sum(dim=1, keepdim=True).clamp_min(eps)


def celltype_composition_loss(
    pred_prior: torch.Tensor,
    spot_prior: torch.Tensor,
    mode: str = "kl",
    eps: float = 1e-8,
) -> torch.Tensor:
    """Loss between predicted and prior spot cell-type compositions."""

    pred = pred_prior.clamp_min(eps)
    target = spot_prior.clamp_min(eps)
    pred = pred / pred.sum(dim=1, keepdim=True).clamp_min(eps)
    target = target / target.sum(dim=1, keepdim=True).clamp_min(eps)
    if mode == "kl":
        return (pred * (pred.log() - target.log())).sum(dim=1).mean()
    if mode == "js":
        mid = 0.5 * (pred + target)
        kl_pm = (pred * (pred.log() - mid.clamp_min(eps).log())).sum(dim=1)
        kl_tm = (target * (target.log() - mid.clamp_min(eps).log())).sum(dim=1)
        return (0.5 * (kl_pm + kl_tm)).mean()
    if mode == "mse":
        return F.mse_loss(pred, target)
    raise ValueError("`mode` must be one of {'kl', 'js', 'mse'}.")


def manifold_preservation_loss(
    z_sc: torch.Tensor,
    coords: torch.Tensor,
    edge_index: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Preserve local scRNA manifold distances in reconstructed space."""

    if edge_index.numel() == 0:
        return z_sc.new_tensor(0.0)
    src, dst = edge_index[0], edge_index[1]
    d_latent = (z_sc[src] - z_sc[dst]).norm(dim=1)
    d_spatial = (coords[src] - coords[dst]).norm(dim=1)
    d_latent = d_latent / d_latent.mean().clamp_min(eps)
    d_spatial = d_spatial / d_spatial.mean().clamp_min(eps)
    return F.mse_loss(d_spatial, d_latent)


def spatial_smoothness_loss(
    pred_comp: torch.Tensor | None,
    spot_mass: torch.Tensor,
    x_sp: torch.Tensor,
    edge_index: torch.Tensor,
    sigma: float | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Smooth neighboring spot composition or assignment mass."""

    if edge_index.numel() == 0:
        return spot_mass.new_tensor(0.0)
    src, dst = edge_index[0], edge_index[1]
    expr_diff2 = (x_sp[src] - x_sp[dst]).square().mean(dim=1)
    if sigma is None:
        sigma = float(torch.sqrt(expr_diff2.detach().median().clamp_min(eps)).item())
    weights = torch.exp(-expr_diff2 / max(float(sigma) ** 2, eps))
    if pred_comp is not None:
        diff2 = (pred_comp[src] - pred_comp[dst]).square().sum(dim=1)
    else:
        mass = spot_mass / spot_mass.mean().clamp_min(eps)
        diff2 = (mass[src] - mass[dst]).square()
    return (weights * diff2).mean()


def offset_regularization_loss(
    offsets: torch.Tensor,
    assignment: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Penalize large relative offsets weighted by assignment probability."""

    offset2 = offsets.square().sum(dim=-1)
    return (assignment * offset2).sum() / assignment.sum().clamp_min(eps)


def assignment_entropy_loss(
    assignment: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Mean cell-wise assignment entropy."""

    m = assignment.clamp_min(eps)
    return -(m * m.log()).sum(dim=1).mean()


def spot_capacity_loss(
    spot_mass: torch.Tensor,
    target_spot_mass: torch.Tensor | None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Encourage each spot/local manifold to receive its expected cell mass."""

    if target_spot_mass is None:
        return spot_mass.new_tensor(0.0)
    target = target_spot_mass.to(device=spot_mass.device, dtype=spot_mass.dtype)
    target = target / target.sum().clamp_min(eps) * spot_mass.sum().detach().clamp_min(eps)
    pred = spot_mass / spot_mass.mean().clamp_min(eps)
    tgt = target / target.mean().clamp_min(eps)
    return F.mse_loss(torch.log1p(pred), torch.log1p(tgt))


def local_filling_loss(
    offsets: torch.Tensor,
    assignment: torch.Tensor,
    spot_mass: torch.Tensor,
    candidate_spots: torch.Tensor | None = None,
    spot_radius: torch.Tensor | float | None = None,
    target_fraction: float = 0.45,
    min_mass: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Encourage cells to fill a local spot manifold instead of collapsing.

    For a uniform disk, expected squared radius is roughly ``R^2 / 2``. We use
    a tunable fraction of ``R^2`` as a soft target so the offset regularizer can
    still keep cells inside the spot.
    """

    radius2 = offsets.square().sum(dim=-1)
    if candidate_spots is not None:
        weighted = assignment * radius2
        sum_r2 = spot_mass.new_zeros(spot_mass.shape[0])
        sum_r2.index_add_(0, candidate_spots.reshape(-1), weighted.reshape(-1))
    else:
        sum_r2 = (assignment * radius2).sum(dim=0)
    mean_r2 = sum_r2 / spot_mass.clamp_min(eps)
    if spot_radius is None:
        target = mean_r2.detach().mean().clamp_min(eps)
    else:
        radius = torch.as_tensor(spot_radius, dtype=mean_r2.dtype, device=mean_r2.device)
        target = (radius.square() * float(target_fraction)).clamp_min(eps)
    mask = spot_mass > min_mass
    if mask.sum() == 0:
        return mean_r2.new_tensor(0.0)
    return F.mse_loss(mean_r2[mask] / target, torch.ones_like(mean_r2[mask]))


def anchor_occupancy_loss(
    local_probs: torch.Tensor | None,
    assignment: torch.Tensor,
    spot_mass: torch.Tensor,
    disk_anchors: torch.Tensor | None,
    candidate_spots: torch.Tensor | None = None,
    min_mass: float = 2.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Match spot-wise local manifold occupancy to a disk support distribution.

    This is the key local-completion term: for each spot, many assigned cells
    should collectively occupy the local manifold anchors rather than all
    choosing the same location. The target is approximately uniform over the
    quasi-uniform disk anchors.
    """

    if local_probs is None or disk_anchors is None:
        return assignment.new_tensor(0.0)
    n_spots = spot_mass.shape[0]
    n_anchors = local_probs.shape[-1]
    weighted = assignment[..., None] * local_probs
    occ = assignment.new_zeros((n_spots, n_anchors))
    if candidate_spots is not None:
        occ.index_add_(0, candidate_spots.reshape(-1), weighted.reshape(-1, n_anchors))
    else:
        occ = weighted.sum(dim=0)
    mask = spot_mass > min_mass
    if mask.sum() == 0:
        return assignment.new_tensor(0.0)
    occ = occ[mask] / spot_mass[mask, None].clamp_min(eps)
    target = torch.ones(n_anchors, dtype=occ.dtype, device=occ.device) / float(n_anchors)
    return (occ * (occ.clamp_min(eps).log() - target.clamp_min(eps).log())).sum(dim=1).mean()


def total_spalmc_loss(
    outputs: dict[str, torch.Tensor],
    x_sc: torch.Tensor,
    x_sp: torch.Tensor,
    cell_type_onehot: torch.Tensor | None = None,
    spot_prior: torch.Tensor | None = None,
    edge_sc: torch.Tensor | None = None,
    edge_spot: torch.Tensor | None = None,
    lambda_expr: float = 1.0,
    lambda_comp: float = 1.0,
    lambda_mani: float = 0.1,
    lambda_spatial: float = 0.1,
    lambda_offset: float = 0.01,
    lambda_entropy: float = 0.01,
    lambda_capacity: float = 0.1,
    lambda_fill: float = 0.05,
    lambda_anchor: float = 0.1,
    target_spot_mass: torch.Tensor | None = None,
    spot_radius: torch.Tensor | float | None = None,
    fill_target_fraction: float = 0.45,
    expr_loss_mode: str = "cosine",
    expr_log1p: bool = False,
    comp_loss_mode: str = "kl",
    entropy_warmup_factor: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Compute all SpaLMC loss terms and a weighted total."""

    assignment = outputs["assignment"]
    candidate_spots = outputs.get("candidate_spots")
    x_sp_hat, spot_mass = reconstruct_spot_expression(
        assignment,
        x_sc,
        candidate_spots=candidate_spots,
        n_spots=x_sp.shape[0],
    )
    loss_expr = expression_reconstruction_loss(
        x_sp_hat,
        x_sp,
        mode=expr_loss_mode,
        log1p=expr_log1p,
        mask=spot_mass > 1e-6,
    )

    pred_comp = None
    if cell_type_onehot is not None and spot_prior is not None:
        pred_comp = predict_spot_composition(
            assignment,
            cell_type_onehot,
            candidate_spots=candidate_spots,
            n_spots=x_sp.shape[0],
        )
        loss_comp = celltype_composition_loss(pred_comp, spot_prior, mode=comp_loss_mode)
    else:
        loss_comp = assignment.new_tensor(0.0)

    if edge_sc is not None:
        loss_mani = manifold_preservation_loss(outputs["z_sc"], outputs["cell_coords"], edge_sc)
    else:
        loss_mani = assignment.new_tensor(0.0)

    if edge_spot is not None:
        loss_spatial = spatial_smoothness_loss(pred_comp, spot_mass, x_sp, edge_spot)
    else:
        loss_spatial = assignment.new_tensor(0.0)

    loss_offset = offset_regularization_loss(outputs["offsets"], assignment)
    loss_entropy = assignment_entropy_loss(assignment)
    loss_capacity = spot_capacity_loss(spot_mass, target_spot_mass)
    loss_fill = local_filling_loss(
        outputs["offsets"],
        assignment,
        spot_mass,
        candidate_spots=candidate_spots,
        spot_radius=spot_radius,
        target_fraction=fill_target_fraction,
    )
    loss_anchor = anchor_occupancy_loss(
        outputs.get("local_probs"),
        assignment,
        spot_mass,
        outputs.get("disk_anchors"),
        candidate_spots=candidate_spots,
    )

    total = (
        lambda_expr * loss_expr
        + lambda_comp * loss_comp
        + lambda_mani * loss_mani
        + lambda_spatial * loss_spatial
        + lambda_offset * loss_offset
        + (lambda_entropy * entropy_warmup_factor) * loss_entropy
        + lambda_capacity * loss_capacity
        + lambda_fill * loss_fill
        + lambda_anchor * loss_anchor
    )
    return {
        "loss_total": total,
        "loss_expr": loss_expr,
        "loss_comp": loss_comp,
        "loss_mani": loss_mani,
        "loss_spatial": loss_spatial,
        "loss_offset": loss_offset,
        "loss_entropy": loss_entropy,
        "loss_capacity": loss_capacity,
        "loss_fill": loss_fill,
        "loss_anchor": loss_anchor,
        "x_sp_hat": x_sp_hat,
        "spot_mass": spot_mass,
        "pred_comp": pred_comp if pred_comp is not None else assignment.new_empty((0, 0)),
    }
