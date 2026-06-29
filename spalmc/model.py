"""High-level SpaLMC API."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

import numpy as np
import pandas as pd
import torch

from .data import (
    align_genes,
    encode_celltypes,
    estimate_spot_radius,
    build_expression_candidate_spots,
    get_expression_matrix,
    infer_spot_cell_count_prior,
    infer_spatial_coords,
    normalize_expression,
    normalize_spatial_coords,
    prepare_spot_prior,
)
from .losses import reconstruct_spot_expression
from .modules import SpaLMCNet
from .train import LossWeights, SpaLMCTrainer
from .utils import build_knn_graph, set_seed, to_tensor


class SpaLMC:
    """Spot-wise Local Manifold Completion for single-cell spatial mapping."""

    def __init__(
        self,
        adata_sc,
        adata_sp,
        celltype_key: str | None = None,
        layer_key: str | None = None,
        spot_celltype_prior: np.ndarray | pd.DataFrame | None = None,
        latent_dim: int = 32,
        hidden_dim: int = 256,
        n_layers: int = 2,
        dropout: float = 0.1,
        temperature: float = 0.1,
        top_k_spots: int | None = None,
        spot_radius: float | None = None,
        lambda_expr: float = 1.0,
        lambda_comp: float = 1.0,
        lambda_mani: float = 0.1,
        lambda_spatial: float = 0.1,
        lambda_offset: float = 0.01,
        lambda_entropy: float = 0.01,
        lambda_capacity: float = 0.1,
        lambda_fill: float = 0.05,
        lambda_anchor: float = 0.1,
        device: str | torch.device | None = None,
        random_state: int = 0,
        use_hvg: bool = False,
        normalize: bool = True,
        log1p: bool = True,
        scale: bool = True,
        shared_encoder: bool = True,
        use_batch_norm: bool = True,
        learnable_radius: bool = False,
        offset_hidden_dim: int = 128,
        offset_n_layers: int = 2,
        k_sc: int = 10,
        k_spatial: int = 6,
        expr_loss_mode: str = "cosine",
        comp_loss_mode: str = "kl",
        entropy_warmup_epochs: int = 50,
        candidate_metric: str = "cosine",
        coordinate_mode: str = "hard",
        density_key: str | None = "auto",
        fill_target_fraction: float = 0.45,
        n_manifold_anchors: int = 32,
        local_temperature: float = 0.5,
        residual_scale: float = 0.15,
        min_cells_per_spot: int = 1,
        max_cells_per_spot: int | None = 25,
        coverage_assignment: bool = True,
        hard_assignment_mode: str = "capacity",
        capacity_slack: float = 1.25,
        spotwise_coordinate_allocation: bool = True,
        coordinate_allocator: str = "energy",
        max_ot_cells: int = 250,
        allocation_cost_blend: float = 0.35,
        allocation_refine_steps: int = 80,
        allocation_lr: float = 0.08,
        allocation_w_chart: float = 1.0,
        allocation_w_knn: float = 0.4,
        allocation_w_repulsion: float = 0.15,
        allocation_w_boundary: float = 2.0,
        allocation_w_type: float = 0.35,
        allocation_w_type_repulsion: float = 0.05,
    ) -> None:
        self.adata_sc = adata_sc
        self.adata_sp = adata_sp
        self.celltype_key = celltype_key
        self.layer_key = layer_key
        self.spot_celltype_prior = spot_celltype_prior
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.temperature = temperature
        self.top_k_spots = top_k_spots
        self.user_spot_radius = spot_radius
        self.random_state = random_state
        self.use_hvg = use_hvg
        self.normalize = normalize
        self.log1p = log1p
        self.scale = scale
        self.shared_encoder = shared_encoder
        self.use_batch_norm = use_batch_norm
        self.learnable_radius = learnable_radius
        self.offset_hidden_dim = offset_hidden_dim
        self.offset_n_layers = offset_n_layers
        self.k_sc = k_sc
        self.k_spatial = k_spatial
        self.expr_loss_mode = expr_loss_mode
        self.comp_loss_mode = comp_loss_mode
        self.entropy_warmup_epochs = entropy_warmup_epochs
        self.candidate_metric = candidate_metric
        self.coordinate_mode = coordinate_mode
        self.density_key = density_key
        self.fill_target_fraction = fill_target_fraction
        self.n_manifold_anchors = n_manifold_anchors
        self.local_temperature = local_temperature
        self.residual_scale = residual_scale
        self.min_cells_per_spot = int(min_cells_per_spot)
        self.max_cells_per_spot = None if max_cells_per_spot is None else int(max_cells_per_spot)
        self.coverage_assignment = coverage_assignment
        if hard_assignment_mode not in {"capacity", "argmax"}:
            raise ValueError("`hard_assignment_mode` must be 'capacity' or 'argmax'.")
        self.hard_assignment_mode = hard_assignment_mode
        self.capacity_slack = float(capacity_slack)
        self.spotwise_coordinate_allocation = spotwise_coordinate_allocation
        if coordinate_allocator not in {"energy", "ot", "rank"}:
            raise ValueError("`coordinate_allocator` must be 'energy', 'ot' or 'rank'.")
        self.coordinate_allocator = coordinate_allocator
        self.max_ot_cells = int(max_ot_cells)
        self.allocation_cost_blend = float(allocation_cost_blend)
        self.allocation_refine_steps = int(allocation_refine_steps)
        self.allocation_lr = float(allocation_lr)
        self.allocation_w_chart = float(allocation_w_chart)
        self.allocation_w_knn = float(allocation_w_knn)
        self.allocation_w_repulsion = float(allocation_w_repulsion)
        self.allocation_w_boundary = float(allocation_w_boundary)
        self.allocation_w_type = float(allocation_w_type)
        self.allocation_w_type_repulsion = float(allocation_w_type_repulsion)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.weights = LossWeights(
            lambda_expr=lambda_expr,
            lambda_comp=lambda_comp,
            lambda_mani=lambda_mani,
            lambda_spatial=lambda_spatial,
            lambda_offset=lambda_offset,
            lambda_entropy=lambda_entropy,
            lambda_capacity=lambda_capacity,
            lambda_fill=lambda_fill,
            lambda_anchor=lambda_anchor,
        )
        self.model: SpaLMCNet | None = None
        self.trainer: SpaLMCTrainer | None = None
        self.loss_history: pd.DataFrame = pd.DataFrame()
        self.prepared = False
        self.results_: dict[str, np.ndarray] = {}
        self._outputs_stale = True

        set_seed(random_state)

    def prepare_data(self) -> "SpaLMC":
        """Align genes, prepare tensors, graphs, priors and model modules."""

        spot_coords = infer_spatial_coords(self.adata_sp)
        training_genes = self.adata_sc.uns.get("training_genes", None)
        self.adata_sc_aligned, self.adata_sp_aligned = align_genes(
            self.adata_sc,
            self.adata_sp,
            genes=training_genes,
            use_hvg=self.use_hvg,
        )
        x_sc = get_expression_matrix(self.adata_sc_aligned, self.layer_key)
        x_sp = get_expression_matrix(self.adata_sp_aligned, self.layer_key)
        if x_sc.shape[1] != x_sp.shape[1]:
            raise ValueError("Aligned scRNA and spot matrices must have the same number of genes.")
        if self.normalize:
            x_sc = normalize_expression(x_sc, log1p=self.log1p, scale=self.scale)
            x_sp = normalize_expression(x_sp, log1p=self.log1p, scale=self.scale)

        self.gene_names_ = self.adata_sc_aligned.var_names.astype(str).tolist()
        self.cell_names_ = self.adata_sc_aligned.obs_names.astype(str).tolist()
        self.spot_names_ = self.adata_sp_aligned.obs_names.astype(str).tolist()
        self.x_sc_np_ = x_sc.astype(np.float32)
        self.x_sp_np_ = x_sp.astype(np.float32)
        self.spot_coords_np_ = np.asarray(spot_coords, dtype=np.float32)
        self.spatial_context_np_, self.spatial_norm_ = normalize_spatial_coords(self.spot_coords_np_)
        self.spot_radius_ = float(self.user_spot_radius or estimate_spot_radius(self.spot_coords_np_))
        self.target_spot_mass_np_, self.target_spot_mass_source_ = infer_spot_cell_count_prior(
            self.adata_sp_aligned,
            n_cells=self.x_sc_np_.shape[0],
            density_key=self.density_key,
        )
        self.effective_top_k_spots_ = self.top_k_spots
        if self.effective_top_k_spots_ is None:
            pair_count = self.x_sc_np_.shape[0] * self.x_sp_np_.shape[0]
            if pair_count > 2_000_000:
                self.effective_top_k_spots_ = min(64, self.x_sp_np_.shape[0])
                warnings.warn(
                    "Large cell-spot product detected; using sparse candidate assignment with "
                    f"`top_k_spots={self.effective_top_k_spots_}`. Set `top_k_spots=None` only "
                    "for small debugging datasets.",
                    RuntimeWarning,
                )

        cell_onehot, categories = encode_celltypes(self.adata_sc_aligned, self.celltype_key)
        self.celltype_categories_ = categories
        prior = prepare_spot_prior(
            self.spot_celltype_prior,
            categories,
            n_spots=self.x_sp_np_.shape[0],
            spot_names=self.spot_names_,
        )
        if self.weights.lambda_comp > 0 and (cell_onehot is None or prior is None):
            warnings.warn(
                "Composition loss requested but cell type labels or spot prior are missing; "
                "`loss_comp` will be skipped.",
                RuntimeWarning,
            )

        candidate_spots = None
        if self.effective_top_k_spots_ is not None and self.effective_top_k_spots_ < self.x_sp_np_.shape[0]:
            candidate_spots = build_expression_candidate_spots(
                self.x_sc_np_,
                self.x_sp_np_,
                top_k=self.effective_top_k_spots_,
                metric=self.candidate_metric,
            )
            self.candidate_spots_np_ = candidate_spots
        else:
            self.candidate_spots_np_ = None

        self.tensors_: dict[str, torch.Tensor | None] = {
            "x_sc": to_tensor(self.x_sc_np_, self.device),
            "x_sp": to_tensor(self.x_sp_np_, self.device),
            "spot_coords": to_tensor(self.spot_coords_np_, self.device),
            "spatial_context": to_tensor(self.spatial_context_np_, self.device),
            "cell_type_onehot": to_tensor(cell_onehot, self.device) if cell_onehot is not None else None,
            "spot_prior": to_tensor(prior, self.device) if prior is not None else None,
            "candidate_spots": torch.as_tensor(candidate_spots, dtype=torch.long, device=self.device)
            if candidate_spots is not None
            else None,
            "target_spot_mass": to_tensor(self.target_spot_mass_np_, self.device),
            "spot_radius": torch.as_tensor(self.spot_radius_, dtype=torch.float32, device=self.device),
            "fill_target_fraction": float(self.fill_target_fraction),
            "edge_sc": None,
            "edge_spot": build_knn_graph(self.spot_coords_np_, k=self.k_spatial, device=self.device),
        }

        self.model = SpaLMCNet(
            n_genes=self.x_sc_np_.shape[1],
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
            dropout=self.dropout,
            use_batch_norm=self.use_batch_norm,
            shared_encoder=self.shared_encoder,
            temperature=self.temperature,
            top_k_spots=None if self.tensors_["candidate_spots"] is not None else self.top_k_spots,
            spot_radius=self.spot_radius_,
            learnable_radius=self.learnable_radius,
            offset_hidden_dim=self.offset_hidden_dim,
            offset_n_layers=self.offset_n_layers,
            coordinate_mode=self.coordinate_mode,
            n_manifold_anchors=self.n_manifold_anchors,
            local_temperature=self.local_temperature,
            residual_scale=self.residual_scale,
        ).to(self.device)
        self.trainer = SpaLMCTrainer(
            self.model,
            self.weights,
            expr_loss_mode=self.expr_loss_mode,
            expr_log1p=False,
            comp_loss_mode=self.comp_loss_mode,
            entropy_warmup_epochs=self.entropy_warmup_epochs,
        )
        self._refresh_sc_graph_from_latent()
        self.prepared = True
        return self

    def _require_prepared(self) -> None:
        if not self.prepared or self.model is None or self.trainer is None:
            self.prepare_data()

    def _refresh_sc_graph_from_latent(self) -> None:
        """Build the scRNA kNN graph from the model's current latent space."""

        if self.model is None:
            return
        self.model.eval()
        with torch.no_grad():
            z_sc, _ = self.model.encoder(self.tensors_["x_sc"], self.tensors_["x_sp"])
        self.tensors_["edge_sc"] = build_knn_graph(z_sc, k=self.k_sc, device=self.device)

    def fit(
        self,
        max_epochs: int = 500,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int | None = None,
        verbose: bool = True,
        eval_every: int = 20,
        cache_outputs: bool = False,
        early_stopping: bool = True,
        patience: int = 80,
        min_delta: float = 1e-4,
    ) -> "SpaLMC":
        """Train SpaLMC with full-batch optimization.

        ``batch_size`` is accepted for API stability; the current prototype
        trains full-batch because expression reconstruction couples all cells
        and spots through the assignment matrix.

        ``cache_outputs=False`` avoids running the expensive final spot-wise
        coordinate allocation at the end of training. Call ``add_to_adata`` or
        ``transform`` when coordinates are actually needed.
        """

        self._require_prepared()
        if batch_size is not None:
            warnings.warn("Mini-batch training is not implemented yet; using full-batch training.", RuntimeWarning)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        rows: list[dict[str, float]] = []
        best_loss = float("inf")
        bad_epochs = 0

        for epoch in range(max_epochs):
            self.model.train()
            optimizer.zero_grad(set_to_none=True)
            loss_dict = self.trainer.compute_loss(self.tensors_, epoch)
            loss_dict["loss_total"].backward()
            optimizer.step()

            record = {
                key: float(value.detach().cpu())
                for key, value in loss_dict.items()
                if key.startswith("loss_")
            }
            record["epoch"] = epoch + 1
            rows.append(record)
            current = record["loss_total"]
            if current < best_loss - min_delta:
                best_loss = current
                bad_epochs = 0
            else:
                bad_epochs += 1
            if verbose and ((epoch + 1) % eval_every == 0 or epoch == 0 or epoch + 1 == max_epochs):
                msg = " | ".join(f"{k}={v:.4f}" for k, v in record.items() if k != "epoch")
                print(f"[SpaLMC] epoch {epoch + 1:04d}: {msg}")
            if early_stopping and bad_epochs >= patience:
                if verbose:
                    print(
                        f"[SpaLMC] early stopping at epoch {epoch + 1}; "
                        f"best loss_total={best_loss:.4f}"
                    )
                break

        self.loss_history = pd.DataFrame(rows)
        self._outputs_stale = True
        if cache_outputs:
            self._cache_outputs()
        return self

    def _forward_eval(self) -> dict[str, torch.Tensor]:
        self._require_prepared()
        self.model.eval()
        with torch.no_grad():
            return self.model(
                self.tensors_["x_sc"],
                self.tensors_["x_sp"],
                self.tensors_["spot_coords"],
                self.tensors_["spatial_context"],
                candidate_spots=self.tensors_.get("candidate_spots"),
            )

    def _cache_outputs(self) -> None:
        if self.results_ and not self._outputs_stale:
            return
        outputs = self._forward_eval()
        assignment = outputs["assignment"]
        local_hard = assignment.argmax(dim=1)
        if "candidate_spots" in outputs:
            candidates = outputs["candidate_spots"]
            hard_np, local_hard_np = self._coverage_hard_assignment(
                assignment.detach().cpu().numpy(),
                candidates.detach().cpu().numpy(),
            )
            hard = torch.as_tensor(hard_np, dtype=torch.long, device=self.device)
            local_hard = torch.as_tensor(local_hard_np, dtype=torch.long, device=self.device)
            if self.spotwise_coordinate_allocation:
                coords_np, offsets_np = self._allocate_spotwise_coordinates(
                    hard_np,
                    outputs["z_sc"].detach().cpu().numpy(),
                )
                coords = torch.as_tensor(coords_np, dtype=torch.float32, device=self.device)
                offsets = torch.as_tensor(offsets_np, dtype=torch.float32, device=self.device)
            else:
                offsets = outputs["offsets"][torch.arange(assignment.shape[0], device=self.device), local_hard]
                coords = self.tensors_["spot_coords"][hard] + offsets
            x_hat, _ = reconstruct_spot_expression(
                assignment,
                self.tensors_["x_sc"],
                candidate_spots=candidates,
                n_spots=self.tensors_["x_sp"].shape[0],
            )
            self.results_ = {
                "coords": coords.detach().cpu().numpy(),
                "assignment_sparse": assignment.detach().cpu().numpy(),
                "candidate_spots": candidates.detach().cpu().numpy(),
                "hard_spot": hard_np,
                "prob": assignment.max(dim=1).values.detach().cpu().numpy(),
                "relative_offset": offsets.detach().cpu().numpy(),
                "x_sp_hat": x_hat.detach().cpu().numpy(),
            }
            self._outputs_stale = False
            return

        hard = local_hard
        if self.coverage_assignment and self.min_cells_per_spot > 0:
            dense = assignment.detach().cpu().numpy()
            candidates_np = np.tile(np.arange(dense.shape[1]), (dense.shape[0], 1))
            hard_np, local_hard_np = self._coverage_hard_assignment(dense, candidates_np)
            hard = torch.as_tensor(hard_np, dtype=torch.long, device=self.device)
            local_hard = hard
        prob = assignment.max(dim=1).values
        if self.spotwise_coordinate_allocation:
            hard_np = hard.detach().cpu().numpy()
            coords_np, offsets_np = self._allocate_spotwise_coordinates(
                hard_np,
                outputs["z_sc"].detach().cpu().numpy(),
            )
            coords = torch.as_tensor(coords_np, dtype=torch.float32, device=self.device)
            offsets = torch.as_tensor(offsets_np, dtype=torch.float32, device=self.device)
        else:
            offsets = outputs["offsets"][torch.arange(assignment.shape[0], device=self.device), hard]
            coords = self.tensors_["spot_coords"][hard] + offsets
        x_hat, _ = reconstruct_spot_expression(assignment, self.tensors_["x_sc"])
        self.results_ = {
            "coords": coords.detach().cpu().numpy(),
            "assignment": assignment.detach().cpu().numpy(),
            "hard_spot": hard.detach().cpu().numpy(),
            "prob": prob.detach().cpu().numpy(),
            "relative_offset": offsets.detach().cpu().numpy(),
            "x_sp_hat": x_hat.detach().cpu().numpy(),
        }
        self._outputs_stale = False

    def _coverage_hard_assignment(
        self,
        probs: np.ndarray,
        candidates: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Choose hard spots while encouraging at least one cell per spot.

        This is a post-training decoding step. Soft assignments are still used
        for expression reconstruction, but final cell coordinates must live
        inside a concrete spot-local circle. The greedy repair fills empty spots
        using candidate cells when possible, guided by assignment probability.
        """

        if self.hard_assignment_mode == "capacity":
            return self._capacity_constrained_hard_assignment(probs, candidates)

        local = probs.argmax(axis=1).astype(np.int64)
        hard = candidates[np.arange(candidates.shape[0]), local].astype(np.int64)
        if not self.coverage_assignment or self.min_cells_per_spot <= 0:
            return hard, local
        n_cells, n_spots = probs.shape[0], len(self.spot_names_)
        if n_cells < self.min_cells_per_spot * n_spots:
            warnings.warn(
                "Cannot satisfy `min_cells_per_spot` because there are fewer cells than required.",
                RuntimeWarning,
            )
            return hard, local
        counts = np.bincount(hard, minlength=n_spots)
        flat_spots = candidates.reshape(-1)
        flat_cells = np.repeat(np.arange(candidates.shape[0]), candidates.shape[1])
        flat_local = np.tile(np.arange(candidates.shape[1]), candidates.shape[0])
        order_by_spot = np.argsort(flat_spots, kind="mergesort")
        sorted_spots = flat_spots[order_by_spot]
        starts = np.searchsorted(sorted_spots, np.arange(n_spots), side="left")
        ends = np.searchsorted(sorted_spots, np.arange(n_spots), side="right")

        for spot in np.where(counts < self.min_cells_per_spot)[0]:
            need = self.min_cells_per_spot - counts[spot]
            start, end = starts[spot], ends[spot]
            if start == end:
                continue
            idx = order_by_spot[start:end]
            rows = flat_cells[idx]
            local_cols = flat_local[idx]
            order = np.argsort(-probs[rows, local_cols], kind="mergesort")
            filled = 0
            for pos in order:
                cell = int(rows[pos])
                old = int(hard[cell])
                if old == spot:
                    continue
                if counts[old] <= self.min_cells_per_spot:
                    continue
                counts[old] -= 1
                hard[cell] = spot
                local[cell] = int(local_cols[pos])
                counts[spot] += 1
                filled += 1
                if filled >= need:
                    break
        empty = int((counts < self.min_cells_per_spot).sum())
        if empty > 0:
            warnings.warn(
                f"{empty} spots could not be filled because they were absent from candidate sets. "
                "Increase `top_k_spots` or use a broader candidate selection.",
                RuntimeWarning,
            )
        return hard, local

    def _capacity_constrained_hard_assignment(
        self,
        probs: np.ndarray,
        candidates: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Decode cell-to-spot assignment with target spot capacities.

        Soft probabilities remain the training object for expression
        reconstruction, but final hard placement should not let one spot absorb
        hundreds of cells unless its density prior supports that capacity.
        """

        n_cells, k = probs.shape
        n_spots = len(self.spot_names_)
        quotas = self._integer_spot_quotas(n_cells)
        max_caps = np.maximum(
            quotas,
            np.ceil(np.maximum(quotas, 1) * self.capacity_slack).astype(np.int64),
        )
        if self.max_cells_per_spot is not None:
            max_caps = np.minimum(max_caps, int(self.max_cells_per_spot))
            if max_caps.sum() < n_cells:
                warnings.warn(
                    "`max_cells_per_spot` is too small to place all cells under the hard cap; "
                    "some spots will exceed the cap in the overflow pass. Increase "
                    "`max_cells_per_spot` or reduce the number of mapped cells.",
                    RuntimeWarning,
                )
        assigned = np.zeros(n_cells, dtype=bool)
        hard = np.full(n_cells, -1, dtype=np.int64)
        local = np.full(n_cells, -1, dtype=np.int64)
        counts = np.zeros(n_spots, dtype=np.int64)

        flat_cell = np.repeat(np.arange(n_cells), k)
        flat_local = np.tile(np.arange(k), n_cells)
        flat_spot = candidates.reshape(-1)
        flat_prob = probs.reshape(-1)
        order = np.argsort(-flat_prob, kind="mergesort")

        for idx in order:
            cell = int(flat_cell[idx])
            if assigned[cell]:
                continue
            spot = int(flat_spot[idx])
            if counts[spot] >= max_caps[spot]:
                continue
            hard[cell] = spot
            local[cell] = int(flat_local[idx])
            counts[spot] += 1
            assigned[cell] = True
            if assigned.all():
                break

        # Second pass: fill any unassigned cells into the least-overfull
        # candidate spot, so every scRNA cell still enters exactly one spot.
        remaining = np.where(~assigned)[0]
        for cell in remaining:
            cand = candidates[cell]
            over = counts[cand] / np.maximum(max_caps[cand], 1)
            best_pos = int(np.lexsort((-probs[cell], over))[0])
            spot = int(cand[best_pos])
            hard[cell] = spot
            local[cell] = best_pos
            counts[spot] += 1

        return hard, local

    def _integer_spot_quotas(self, n_cells: int) -> np.ndarray:
        """Convert density-derived target masses into integer spot quotas."""

        target = getattr(self, "target_spot_mass_np_", None)
        if target is None:
            target = np.ones(len(self.spot_names_), dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        target = np.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
        target = np.maximum(target, 0.0)
        if target.sum() <= 0:
            target = np.ones_like(target)
        raw = target / target.sum() * float(n_cells)
        if self.max_cells_per_spot is not None:
            raw = np.minimum(raw, float(self.max_cells_per_spot))
        quotas = np.floor(raw).astype(np.int64)
        if n_cells >= len(quotas) * self.min_cells_per_spot:
            quotas = np.maximum(quotas, self.min_cells_per_spot)
        if self.max_cells_per_spot is not None:
            quotas = np.minimum(quotas, int(self.max_cells_per_spot))
        diff = int(n_cells - quotas.sum())
        if diff > 0:
            frac_order = np.argsort(-(raw - np.floor(raw)), kind="mergesort")
            for spot in frac_order:
                if diff == 0:
                    break
                if self.max_cells_per_spot is not None and quotas[spot] >= self.max_cells_per_spot:
                    continue
                quotas[spot] += 1
                diff -= 1
        elif diff < 0:
            removable = np.where(quotas > self.min_cells_per_spot)[0]
            order = removable[np.argsort(raw[removable], kind="mergesort")]
            for spot in order:
                if diff == 0:
                    break
                take = min(quotas[spot] - self.min_cells_per_spot, -diff)
                quotas[spot] -= take
                diff += int(take)
        return quotas

    def _allocate_spotwise_coordinates(
        self,
        hard_spot: np.ndarray,
        z_sc: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Collectively allocate cells inside each assigned spot circle.

        This is the final SpaLMC coordinate semantics: first every scRNA cell
        enters one spot, then all cells inside that spot are arranged together
        in a circular local manifold. The arrangement uses each spot's local
        latent cell structure to order cells and a quasi-uniform disk support to
        avoid arbitrary independent coordinate regression.
        """

        n_cells = hard_spot.shape[0]
        offsets = np.zeros((n_cells, 2), dtype=np.float32)
        type_codes = self._get_celltype_codes()
        for spot in range(len(self.spot_names_)):
            cells = np.flatnonzero(hard_spot == spot)
            m = cells.size
            if m == 0:
                continue
            if m == 1:
                offsets[cells[0]] = 0.0
                continue
            disk = self._fibonacci_disk(m).astype(np.float32) * float(self.spot_radius_)
            local_z = z_sc[cells]
            local_2d = self._local_latent_chart(local_z)
            target_2d = self._normalize_chart_to_disk(local_2d) * float(self.spot_radius_)
            if self.coordinate_allocator in {"energy", "ot"} and m <= self.max_ot_cells:
                init = self._optimal_disk_assignment(
                    target_2d,
                    disk,
                    local_z,
                    blend=self.allocation_cost_blend,
                )
                if self.coordinate_allocator == "energy":
                    offsets[cells] = self._energy_refine_disk_coordinates(
                        init,
                        target_2d,
                        local_z,
                        radius=float(self.spot_radius_),
                        steps=self.allocation_refine_steps,
                        lr=self.allocation_lr,
                        w_chart=self.allocation_w_chart,
                        w_knn=self.allocation_w_knn,
                        w_repulsion=self.allocation_w_repulsion,
                        w_boundary=self.allocation_w_boundary,
                        type_codes=type_codes[cells] if type_codes is not None else None,
                        w_type=self.allocation_w_type,
                        w_type_repulsion=self.allocation_w_type_repulsion,
                    )
                else:
                    offsets[cells] = init
            else:
                offsets[cells] = self._rank_disk_assignment(target_2d, disk)

        coords = self.spot_coords_np_[hard_spot] + offsets
        return coords.astype(np.float32), offsets.astype(np.float32)

    def _get_celltype_codes(self) -> np.ndarray | None:
        """Return integer cell type codes when labels are available."""

        if self.celltype_key is None or self.celltype_key not in self.adata_sc.obs:
            return None
        labels = pd.Categorical(self.adata_sc.obs[self.celltype_key])
        return labels.codes.astype(np.int64)

    @staticmethod
    def _fibonacci_disk(n: int) -> np.ndarray:
        idx = np.arange(n, dtype=np.float32) + 0.5
        radius = np.sqrt(idx / float(n))
        angle = idx * (np.pi * (3.0 - np.sqrt(5.0)))
        return np.column_stack([radius * np.cos(angle), radius * np.sin(angle)])

    @staticmethod
    def _local_latent_chart(local_z: np.ndarray) -> np.ndarray:
        """Estimate a local 2D chart from cells assigned to one spot."""

        z = np.asarray(local_z, dtype=np.float32)
        z = z - z.mean(axis=0, keepdims=True)
        if z.shape[0] < 3 or np.allclose(z, 0):
            return np.zeros((z.shape[0], 2), dtype=np.float32)
        try:
            _, _, vt = np.linalg.svd(z, full_matrices=False)
            y = z @ vt[:2].T
        except np.linalg.LinAlgError:
            y = z[:, :2] if z.shape[1] >= 2 else np.column_stack([z[:, 0], np.zeros(z.shape[0])])
        if y.shape[1] == 1:
            y = np.column_stack([y[:, 0], np.zeros(y.shape[0], dtype=np.float32)])
        return y.astype(np.float32)

    @staticmethod
    def _normalize_chart_to_disk(y: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """Map a local chart into the unit disk while preserving angular order."""

        y = np.asarray(y, dtype=np.float32)
        y = y - y.mean(axis=0, keepdims=True)
        cov_scale = np.sqrt((np.square(y).sum(axis=1)).mean())
        if cov_scale < eps:
            return np.zeros_like(y)
        y = y / cov_scale
        norms = np.linalg.norm(y, axis=1, keepdims=True)
        max_norm = np.max(norms)
        if max_norm < eps:
            return np.zeros_like(y)
        # Squash smoothly rather than hard clipping so outliers do not dominate.
        unit = y / np.clip(norms, eps, None)
        radius = np.tanh(norms / np.clip(max_norm, eps, None))
        return (unit * radius).astype(np.float32)

    @staticmethod
    def _rank_disk_assignment(target_2d: np.ndarray, disk: np.ndarray) -> np.ndarray:
        """Fallback assignment preserving radial and angular ranks."""

        cell_angle = np.arctan2(target_2d[:, 1], target_2d[:, 0])
        cell_radius = np.linalg.norm(target_2d, axis=1)
        disk_angle = np.arctan2(disk[:, 1], disk[:, 0])
        disk_radius = np.linalg.norm(disk, axis=1)
        cell_order = np.lexsort((cell_angle, cell_radius))
        disk_order = np.lexsort((disk_angle, disk_radius))
        out = np.zeros_like(disk, dtype=np.float32)
        out[cell_order] = disk[disk_order]
        return out

    @staticmethod
    def _optimal_disk_assignment(
        target_2d: np.ndarray,
        disk: np.ndarray,
        local_z: np.ndarray,
        blend: float = 0.35,
    ) -> np.ndarray:
        """Assign cells to disk points with a local-manifold-aware cost.

        The cost combines distance in the inferred 2D local chart and a radial
        ordering term. This makes coordinate allocation a spot-wise optimization
        problem instead of independent offset regression.
        """

        blend = float(np.clip(blend, 0.0, 1.0))
        chart_cost = ((target_2d[:, None, :] - disk[None, :, :]) ** 2).sum(axis=2)
        cell_r = np.linalg.norm(target_2d, axis=1, keepdims=True)
        disk_r = np.linalg.norm(disk, axis=1, keepdims=True).T
        radial_cost = (cell_r - disk_r) ** 2
        cost = (1.0 - blend) * chart_cost + blend * radial_cost
        try:
            from scipy.optimize import linear_sum_assignment

            row, col = linear_sum_assignment(cost)
            out = np.zeros_like(disk, dtype=np.float32)
            out[row] = disk[col]
            return out
        except Exception:
            return SpaLMC._rank_disk_assignment(target_2d, disk)

    @staticmethod
    def _energy_refine_disk_coordinates(
        init: np.ndarray,
        target_2d: np.ndarray,
        local_z: np.ndarray,
        radius: float,
        steps: int = 80,
        lr: float = 0.08,
        w_chart: float = 1.0,
        w_knn: float = 0.4,
        w_repulsion: float = 0.15,
        w_boundary: float = 2.0,
        type_codes: np.ndarray | None = None,
        w_type: float = 0.35,
        w_type_repulsion: float = 0.05,
    ) -> np.ndarray:
        """Refine spot-local coordinates with a multi-term energy model.

        Terms:
        - chart attraction: keep coordinates near the inferred local expression chart;
        - local kNN preservation: preserve relative distances among nearby cells;
        - repulsion: avoid unrealistic coordinate collapse and overlaps;
        - boundary: keep coordinates inside the circular spot support.
        - type coherence: encourage cells with the same cell type/state to form
          local neighborhoods, similar in spirit to SpaSlot's slot occupancy
          smoothness over a spatial support.
        """

        if init.shape[0] <= 2 or steps <= 0:
            return init.astype(np.float32)
        device = torch.device("cpu")
        pos = torch.tensor(init, dtype=torch.float32, device=device, requires_grad=True)
        target = torch.tensor(target_2d, dtype=torch.float32, device=device)
        z = torch.tensor(local_z, dtype=torch.float32, device=device)
        n = init.shape[0]
        k = min(6, n - 1)
        with torch.no_grad():
            latent_dist = torch.cdist(z, z)
            nn_idx = torch.topk(latent_dist, k=k + 1, largest=False).indices[:, 1:]
            src = torch.arange(n, device=device).repeat_interleave(k)
            dst = nn_idx.reshape(-1)
            target_dist = torch.norm(target[src] - target[dst], dim=1)
            target_dist = target_dist / target_dist.mean().clamp_min(1e-6)
            rep_sigma = max(float(radius) / max(np.sqrt(n), 2.0), 1e-6)
            type_pair = None
            type_centroid_repulsion = None
            if type_codes is not None and np.any(type_codes >= 0):
                t = torch.tensor(type_codes, dtype=torch.long, device=device)
                valid = t >= 0
                same = (t[:, None] == t[None, :]) & valid[:, None] & valid[None, :]
                same.fill_diagonal_(False)
                if same.any():
                    type_pair = same
                unique = torch.unique(t[valid])
                if unique.numel() > 1:
                    type_centroid_repulsion = unique

        opt = torch.optim.Adam([pos], lr=lr)
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            chart = ((pos - target) ** 2).mean()
            spatial_dist = torch.norm(pos[src] - pos[dst], dim=1)
            spatial_dist = spatial_dist / spatial_dist.mean().clamp_min(1e-6)
            knn = ((spatial_dist - target_dist) ** 2).mean()

            d = torch.cdist(pos, pos)
            eye = torch.eye(n, dtype=torch.bool, device=device)
            repulsion = torch.exp(-(d[~eye] ** 2) / (2.0 * rep_sigma**2)).mean()

            r = torch.norm(pos, dim=1)
            boundary = torch.relu(r - float(radius)).pow(2).mean()
            type_loss = pos.new_tensor(0.0)
            if type_pair is not None and w_type > 0:
                same_dist2 = torch.cdist(pos, pos).pow(2)[type_pair]
                type_loss = same_dist2.mean() / max(float(radius) ** 2, 1e-6)
            type_repulsion = pos.new_tensor(0.0)
            if type_centroid_repulsion is not None and w_type_repulsion > 0:
                t = torch.tensor(type_codes, dtype=torch.long, device=device)
                centroids = []
                for code in type_centroid_repulsion:
                    mask = t == code
                    if mask.sum() > 0:
                        centroids.append(pos[mask].mean(dim=0))
                if len(centroids) > 1:
                    c = torch.stack(centroids, dim=0)
                    cd = torch.cdist(c, c)
                    eye = torch.eye(c.shape[0], dtype=torch.bool, device=device)
                    type_repulsion = torch.exp(-(cd[~eye] ** 2) / (2.0 * (float(radius) * 0.35) ** 2)).mean()
            loss = (
                w_chart * chart
                + w_knn * knn
                + w_repulsion * repulsion
                + w_boundary * boundary
                + w_type * type_loss
                + w_type_repulsion * type_repulsion
            )
            loss.backward()
            opt.step()
            with torch.no_grad():
                r = torch.norm(pos, dim=1, keepdim=True)
                scale = torch.clamp(float(radius) / r.clamp_min(1e-6), max=1.0)
                pos.mul_(scale)
        return pos.detach().cpu().numpy().astype(np.float32)

    def transform(self) -> np.ndarray:
        """Return reconstructed single-cell spatial coordinates."""

        self._cache_outputs()
        return self.results_["coords"]

    def add_to_adata(self):
        """Write SpaLMC outputs into ``adata_sc`` and spot probabilities into ``adata_sp``."""

        self._cache_outputs()
        hard = self.results_["hard_spot"]
        self.adata_sc.obsm["spalmc_spatial"] = self.results_["coords"]
        self.adata_sc.obs["spalmc_spot_id"] = pd.Categorical([self.spot_names_[i] for i in hard])
        self.adata_sc.obs["spalmc_assignment_prob"] = self.results_["prob"]
        self.adata_sc.obsm["spalmc_relative_offset"] = self.results_["relative_offset"]
        probs = self.get_spot_celltype_probabilities(as_dataframe=True)
        if probs is not None:
            self.adata_sp.obsm["spalmc_celltype_prob"] = probs.to_numpy(dtype=np.float32)
            self.adata_sp.uns["spalmc_celltype_categories"] = probs.columns.astype(str).tolist()
            for col in probs.columns:
                self.adata_sp.obs[f"spalmc_prob_{col}"] = probs[col].to_numpy()
        return self.adata_sc

    def get_assignment(self, as_dataframe: bool = False) -> np.ndarray | pd.DataFrame:
        """Return the cell-to-spot assignment matrix."""

        self._cache_outputs()
        if "assignment" in self.results_:
            assignment = self.results_["assignment"]
        else:
            assignment = np.zeros((len(self.cell_names_), len(self.spot_names_)), dtype=np.float32)
            rows = np.arange(assignment.shape[0])[:, None]
            assignment[rows, self.results_["candidate_spots"]] = self.results_["assignment_sparse"]
            warnings.warn(
                "Materialized a dense assignment matrix from sparse top-k assignments. "
                "For large datasets this can consume substantial memory.",
                RuntimeWarning,
            )
        if as_dataframe:
            return pd.DataFrame(assignment, index=self.cell_names_, columns=self.spot_names_)
        return assignment

    def get_sparse_assignment(self) -> pd.DataFrame:
        """Return top-k sparse assignments as a long table."""

        self._cache_outputs()
        if "assignment_sparse" not in self.results_:
            dense = self.results_["assignment"]
            rows, cols = np.nonzero(dense)
            return pd.DataFrame(
                {
                    "cell": np.asarray(self.cell_names_)[rows],
                    "spot": np.asarray(self.spot_names_)[cols],
                    "probability": dense[rows, cols],
                }
            )
        candidates = self.results_["candidate_spots"]
        probs = self.results_["assignment_sparse"]
        n, k = candidates.shape
        rows = np.repeat(np.arange(n), k)
        cols = candidates.reshape(-1)
        return pd.DataFrame(
            {
                "cell": np.asarray(self.cell_names_)[rows],
                "spot": np.asarray(self.spot_names_)[cols],
                "probability": probs.reshape(-1),
            }
        )

    def get_spot_reconstruction(self, as_dataframe: bool = False) -> np.ndarray | pd.DataFrame:
        """Return reconstructed spot expression matrix."""

        self._cache_outputs()
        x_hat = self.results_["x_sp_hat"]
        if as_dataframe:
            return pd.DataFrame(x_hat, index=self.spot_names_, columns=self.gene_names_)
        return x_hat

    def get_spot_fill_summary(self) -> pd.DataFrame:
        """Summarize how many cells fill each spot/local manifold."""

        self._cache_outputs()
        hard = self.results_["hard_spot"]
        counts = np.bincount(hard, minlength=len(self.spot_names_)).astype(float)
        target = getattr(self, "target_spot_mass_np_", np.full(len(self.spot_names_), np.nan))
        out = pd.DataFrame(
            {
                "spot": self.spot_names_,
                "hard_cell_count": counts,
                "target_cell_mass": target,
            }
        )
        if "assignment_sparse" in self.results_:
            soft_mass = np.zeros(len(self.spot_names_), dtype=np.float32)
            np.add.at(
                soft_mass,
                self.results_["candidate_spots"].reshape(-1),
                self.results_["assignment_sparse"].reshape(-1),
            )
            out["soft_cell_mass"] = soft_mass
        else:
            out["soft_cell_mass"] = self.results_["assignment"].sum(axis=0)
        return out

    def get_spot_celltype_probabilities(self, as_dataframe: bool = True) -> pd.DataFrame | np.ndarray | None:
        """Return per-spot cell-type probabilities from final hard assignments."""

        self._cache_outputs()
        if self.celltype_key is None or self.celltype_key not in self.adata_sc.obs:
            return None
        hard = self.results_["hard_spot"]
        labels = pd.Categorical(self.adata_sc.obs[self.celltype_key])
        categories = labels.categories.astype(str).tolist()
        table = np.zeros((len(self.spot_names_), len(categories)), dtype=np.float32)
        codes = labels.codes
        valid = codes >= 0
        np.add.at(table, (hard[valid], codes[valid]), 1.0)
        denom = table.sum(axis=1, keepdims=True)
        probs = np.divide(table, np.clip(denom, 1e-8, None), out=np.zeros_like(table), where=denom > 0)
        if as_dataframe:
            return pd.DataFrame(probs, index=self.spot_names_, columns=categories)
        return probs

    def save_model(self, path: str | Path) -> None:
        """Save model weights and metadata."""

        self._require_prepared()
        payload: dict[str, Any] = {
            "state_dict": self.model.state_dict(),
            "gene_names": self.gene_names_,
            "spot_names": self.spot_names_,
            "celltype_categories": self.celltype_categories_,
            "spot_radius": self.spot_radius_,
            "target_spot_mass_source": self.target_spot_mass_source_,
            "config": {
                "latent_dim": self.latent_dim,
                "hidden_dim": self.hidden_dim,
                "n_layers": self.n_layers,
                "dropout": self.dropout,
                "temperature": self.temperature,
                "top_k_spots": self.top_k_spots,
                "shared_encoder": self.shared_encoder,
                "use_batch_norm": self.use_batch_norm,
                "learnable_radius": self.learnable_radius,
                "offset_hidden_dim": self.offset_hidden_dim,
                "offset_n_layers": self.offset_n_layers,
                "coordinate_mode": self.coordinate_mode,
                "density_key": self.density_key,
                "fill_target_fraction": self.fill_target_fraction,
                "n_manifold_anchors": self.n_manifold_anchors,
                "local_temperature": self.local_temperature,
                "residual_scale": self.residual_scale,
                "min_cells_per_spot": self.min_cells_per_spot,
                "max_cells_per_spot": self.max_cells_per_spot,
                "coverage_assignment": self.coverage_assignment,
                "hard_assignment_mode": self.hard_assignment_mode,
                "capacity_slack": self.capacity_slack,
                "spotwise_coordinate_allocation": self.spotwise_coordinate_allocation,
                "coordinate_allocator": self.coordinate_allocator,
                "max_ot_cells": self.max_ot_cells,
                "allocation_cost_blend": self.allocation_cost_blend,
                "allocation_refine_steps": self.allocation_refine_steps,
                "allocation_lr": self.allocation_lr,
                "allocation_w_chart": self.allocation_w_chart,
                "allocation_w_knn": self.allocation_w_knn,
                "allocation_w_repulsion": self.allocation_w_repulsion,
                "allocation_w_boundary": self.allocation_w_boundary,
                "allocation_w_type": self.allocation_w_type,
                "allocation_w_type_repulsion": self.allocation_w_type_repulsion,
            },
            "loss_history": self.loss_history,
        }
        torch.save(payload, Path(path))

    def load_model(self, path: str | Path, map_location: str | torch.device | None = None) -> "SpaLMC":
        """Load model weights into a prepared SpaLMC instance."""

        self._require_prepared()
        payload = torch.load(Path(path), map_location=map_location or self.device)
        self.model.load_state_dict(payload["state_dict"])
        if "loss_history" in payload:
            self.loss_history = payload["loss_history"]
        self._cache_outputs()
        return self
