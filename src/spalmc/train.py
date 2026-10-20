"""Training helpers for SpaLMC."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .losses import total_spalmc_loss


@dataclass
class LossWeights:
    """Loss weights used by the SpaLMC trainer."""

    lambda_expr: float = 1.0
    lambda_comp: float = 1.0
    lambda_mani: float = 0.1
    lambda_spatial: float = 0.1
    lambda_offset: float = 0.01
    lambda_entropy: float = 0.01
    lambda_capacity: float = 0.1
    lambda_fill: float = 0.05
    lambda_anchor: float = 0.1


class SpaLMCTrainer:
    """Full-batch trainer for the SpaLMC prototype."""

    def __init__(
        self,
        model: torch.nn.Module,
        weights: LossWeights,
        expr_loss_mode: str = "cosine",
        expr_log1p: bool = False,
        comp_loss_mode: str = "kl",
        entropy_warmup_epochs: int = 50,
    ) -> None:
        self.model = model
        self.weights = weights
        self.expr_loss_mode = expr_loss_mode
        self.expr_log1p = expr_log1p
        self.comp_loss_mode = comp_loss_mode
        self.entropy_warmup_epochs = max(1, int(entropy_warmup_epochs))

    def compute_loss(
        self,
        tensors: dict[str, torch.Tensor | None],
        epoch: int,
    ) -> dict[str, torch.Tensor]:
        """Forward model and compute losses."""

        outputs = self.model(
            tensors["x_sc"],
            tensors["x_sp"],
            tensors["spot_coords"],
            tensors["spatial_context"],
            candidate_spots=tensors.get("candidate_spots"),
        )
        warmup = min(1.0, float(epoch + 1) / float(self.entropy_warmup_epochs))
        return total_spalmc_loss(
            outputs,
            tensors["x_sc"],
            tensors["x_sp"],
            cell_type_onehot=tensors.get("cell_type_onehot"),
            spot_prior=tensors.get("spot_prior"),
            edge_sc=tensors.get("edge_sc"),
            edge_spot=tensors.get("edge_spot"),
            lambda_expr=self.weights.lambda_expr,
            lambda_comp=self.weights.lambda_comp,
            lambda_mani=self.weights.lambda_mani,
            lambda_spatial=self.weights.lambda_spatial,
            lambda_offset=self.weights.lambda_offset,
            lambda_entropy=self.weights.lambda_entropy,
            lambda_capacity=self.weights.lambda_capacity,
            lambda_fill=self.weights.lambda_fill,
            lambda_anchor=self.weights.lambda_anchor,
            target_spot_mass=tensors.get("target_spot_mass"),
            spot_radius=tensors.get("spot_radius"),
            fill_target_fraction=float(tensors.get("fill_target_fraction", 0.45)),
            expr_loss_mode=self.expr_loss_mode,
            expr_log1p=self.expr_log1p,
            comp_loss_mode=self.comp_loss_mode,
            entropy_warmup_factor=warmup,
        )
