"""Neural network modules for SpaLMC."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .utils import masked_softmax_topk, safe_normalize


def build_disk_anchors(n_anchors: int = 32) -> torch.Tensor:
    """Create quasi-uniform anchor points in the unit disk.

    The anchors form the support of each spot-local manifold. A Fibonacci disk
    is used instead of a grid so the support has no privileged axis.
    """

    if n_anchors < 4:
        raise ValueError("`n_anchors` must be at least 4.")
    idx = torch.arange(n_anchors, dtype=torch.float32) + 0.5
    radius = torch.sqrt(idx / float(n_anchors))
    angle = idx * (torch.pi * (3.0 - 5.0**0.5))
    return torch.stack([radius * torch.cos(angle), radius * torch.sin(angle)], dim=1)


class MLPEncoder(nn.Module):
    """Configurable MLP encoder for expression profiles."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dim: int = 256,
        n_layers: int = 2,
        dropout: float = 0.1,
        use_batch_norm: bool = True,
    ) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError("`n_layers` must be at least 1.")
        layers: list[nn.Module] = []
        dim = input_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            dim = hidden_dim
        layers.append(nn.Linear(dim, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode expression to a L2-normalized latent vector."""

        return safe_normalize(self.net(x), dim=-1)


class SharedTranscriptionalManifoldEncoder(nn.Module):
    """Map scRNA-seq cells and ST spots into a shared latent space."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dim: int = 256,
        n_layers: int = 2,
        dropout: float = 0.1,
        use_batch_norm: bool = True,
        shared_encoder: bool = True,
    ) -> None:
        super().__init__()
        self.shared_encoder = shared_encoder
        self.encoder_sc = MLPEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
            use_batch_norm=use_batch_norm,
        )
        self.encoder_sp = self.encoder_sc if shared_encoder else MLPEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
            use_batch_norm=use_batch_norm,
        )

    def forward(self, x_sc: torch.Tensor, x_sp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``z_sc`` and ``z_sp``."""

        return self.encoder_sc(x_sc), self.encoder_sp(x_sp)


class CellToSpotAssignment(nn.Module):
    """Soft assignment from cells to spots using latent cosine similarity."""

    def __init__(
        self,
        temperature: float = 0.1,
        top_k_spots: int | None = None,
        assignment_sharpness: float = 1.0,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("`temperature` must be positive.")
        self.temperature = temperature
        self.top_k_spots = top_k_spots
        self.assignment_sharpness = assignment_sharpness

    def forward(self, z_sc: torch.Tensor, z_sp: torch.Tensor) -> torch.Tensor:
        """Compute assignment matrix ``M`` with shape n_cells x n_spots."""

        scores = (z_sc @ z_sp.T) / self.temperature
        scores = scores * self.assignment_sharpness
        return masked_softmax_topk(scores, self.top_k_spots, dim=1)


class RelativeOffsetDecoder(nn.Module):
    """Decode spot-wise local-manifold coordinates for every cell-spot pair.

    Instead of directly regressing arbitrary 2D offsets, each spot contains a
    quasi-uniform disk of local manifold anchors. A cell chooses a soft mixture
    of those anchors conditioned on ``(z_cell, z_spot, spatial_context)`` and
    receives a small residual. This makes the geometry explicit: cells fill a
    bounded local manifold inside the spot rather than collapsing to unconstrained
    MLP-produced points.
    """

    def __init__(
        self,
        latent_dim: int = 32,
        spatial_context_dim: int = 2,
        hidden_dim: int = 128,
        n_layers: int = 2,
        dropout: float = 0.1,
        spot_radius: float = 1.0,
        learnable_radius: bool = False,
        n_manifold_anchors: int = 32,
        local_temperature: float = 0.5,
        residual_scale: float = 0.15,
    ) -> None:
        super().__init__()
        if local_temperature <= 0:
            raise ValueError("`local_temperature` must be positive.")
        input_dim = latent_dim * 2 + spatial_context_dim
        layers: list[nn.Module] = []
        dim = input_dim
        for _ in range(max(1, n_layers)):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.anchor_head = nn.Linear(dim, n_manifold_anchors)
        self.residual_head = nn.Linear(dim, 2)
        self.local_temperature = local_temperature
        self.residual_scale = residual_scale
        self.n_manifold_anchors = n_manifold_anchors
        self.register_buffer("disk_anchors", build_disk_anchors(n_manifold_anchors))
        radius = torch.tensor(float(spot_radius), dtype=torch.float32)
        if learnable_radius:
            self.log_radius = nn.Parameter(radius.clamp_min(1e-6).log())
        else:
            self.register_buffer("radius", radius)
            self.log_radius = None

    @property
    def spot_radius(self) -> torch.Tensor:
        """Return positive radius tensor."""

        if self.log_radius is not None:
            return torch.exp(self.log_radius)
        return self.radius

    def forward(
        self,
        z_sc: torch.Tensor,
        z_sp: torch.Tensor,
        spatial_context: torch.Tensor,
        candidate_spots: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode bounded offsets in the original spatial coordinate scale.

        If ``candidate_spots`` is provided, only offsets for those cell-spot
        pairs are decoded and the returned shape is ``n_cells x k x 2``.
        Otherwise the dense prototype path returns ``n_cells x n_spots x 2``.
        """

        n_cells, n_spots = z_sc.shape[0], z_sp.shape[0]
        if candidate_spots is not None:
            zc = z_sc[:, None, :].expand(-1, candidate_spots.shape[1], -1)
            zs = z_sp[candidate_spots]
            ctx = spatial_context[candidate_spots]
            inp = torch.cat([zc, zs, ctx], dim=-1)
            feat = self.backbone(inp.reshape(n_cells * candidate_spots.shape[1], -1))
            anchor_logits = self.anchor_head(feat).reshape(
                n_cells,
                candidate_spots.shape[1],
                self.n_manifold_anchors,
            )
            local_probs = torch.softmax(anchor_logits / self.local_temperature, dim=-1)
            base = local_probs @ self.disk_anchors.to(local_probs.device, local_probs.dtype)
            residual = self.residual_scale * torch.tanh(self.residual_head(feat)).reshape(
                n_cells,
                candidate_spots.shape[1],
                2,
            )
            unit_offset = self._project_to_unit_disk(base + residual)
            return self.spot_radius * unit_offset, local_probs

        zc = z_sc[:, None, :].expand(n_cells, n_spots, -1)
        zs = z_sp[None, :, :].expand(n_cells, n_spots, -1)
        ctx = spatial_context[None, :, :].expand(n_cells, n_spots, -1)
        inp = torch.cat([zc, zs, ctx], dim=-1)
        feat = self.backbone(inp.reshape(n_cells * n_spots, -1))
        anchor_logits = self.anchor_head(feat).reshape(n_cells, n_spots, self.n_manifold_anchors)
        local_probs = torch.softmax(anchor_logits / self.local_temperature, dim=-1)
        base = local_probs @ self.disk_anchors.to(local_probs.device, local_probs.dtype)
        residual = self.residual_scale * torch.tanh(self.residual_head(feat)).reshape(n_cells, n_spots, 2)
        unit_offset = self._project_to_unit_disk(base + residual)
        return self.spot_radius * unit_offset, local_probs

    @staticmethod
    def _project_to_unit_disk(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        norm = x.norm(dim=-1, keepdim=True)
        return x / norm.clamp_min(eps) * norm.clamp_max(1.0)


class SpaLMCNet(nn.Module):
    """End-to-end SpaLMC neural module."""

    def __init__(
        self,
        n_genes: int,
        latent_dim: int = 32,
        hidden_dim: int = 256,
        n_layers: int = 2,
        dropout: float = 0.1,
        use_batch_norm: bool = True,
        shared_encoder: bool = True,
        temperature: float = 0.1,
        top_k_spots: int | None = None,
        assignment_sharpness: float = 1.0,
        spot_radius: float = 1.0,
        learnable_radius: bool = False,
        offset_hidden_dim: int = 128,
        offset_n_layers: int = 2,
        coordinate_mode: str = "hard",
        n_manifold_anchors: int = 32,
        local_temperature: float = 0.5,
        residual_scale: float = 0.15,
    ) -> None:
        super().__init__()
        if coordinate_mode not in {"hard", "soft"}:
            raise ValueError("`coordinate_mode` must be 'hard' or 'soft'.")
        self.coordinate_mode = coordinate_mode
        self.encoder = SharedTranscriptionalManifoldEncoder(
            input_dim=n_genes,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
            use_batch_norm=use_batch_norm,
            shared_encoder=shared_encoder,
        )
        self.assignment = CellToSpotAssignment(
            temperature=temperature,
            top_k_spots=top_k_spots,
            assignment_sharpness=assignment_sharpness,
        )
        self.offset_decoder = RelativeOffsetDecoder(
            latent_dim=latent_dim,
            hidden_dim=offset_hidden_dim,
            n_layers=offset_n_layers,
            dropout=dropout,
            spot_radius=spot_radius,
            learnable_radius=learnable_radius,
            n_manifold_anchors=n_manifold_anchors,
            local_temperature=local_temperature,
            residual_scale=residual_scale,
        )

    def forward(
        self,
        x_sc: torch.Tensor,
        x_sp: torch.Tensor,
        spot_coords: torch.Tensor,
        spatial_context: torch.Tensor,
        candidate_spots: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run SpaLMC forward pass."""

        z_sc, z_sp = self.encoder(x_sc, x_sp)
        if candidate_spots is not None:
            cand_z = z_sp[candidate_spots]
            scores = (z_sc[:, None, :] * cand_z).sum(dim=-1) / self.assignment.temperature
            scores = scores * self.assignment.assignment_sharpness
            assignment = torch.softmax(scores, dim=1)
            offsets, local_probs = self.offset_decoder(z_sc, z_sp, spatial_context, candidate_spots=candidate_spots)
            pair_coords = spot_coords[candidate_spots] + offsets
            if self.coordinate_mode == "hard":
                local_hard = assignment.argmax(dim=1)
                cell_coords = pair_coords[torch.arange(z_sc.shape[0], device=z_sc.device), local_hard]
            else:
                cell_coords = (assignment[:, :, None] * pair_coords).sum(dim=1)
            return {
                "z_sc": z_sc,
                "z_sp": z_sp,
                "assignment": assignment,
                "candidate_spots": candidate_spots,
                "offsets": offsets,
                "local_probs": local_probs,
                "disk_anchors": self.offset_decoder.disk_anchors,
                "pair_coords": pair_coords,
                "cell_coords": cell_coords,
            }

        assignment = self.assignment(z_sc, z_sp)
        offsets, local_probs = self.offset_decoder(z_sc, z_sp, spatial_context)
        pair_coords = spot_coords[None, :, :] + offsets
        if self.coordinate_mode == "hard":
            hard = assignment.argmax(dim=1)
            cell_coords = pair_coords[torch.arange(z_sc.shape[0], device=z_sc.device), hard]
        else:
            cell_coords = (assignment[:, :, None] * pair_coords).sum(dim=1)
        return {
            "z_sc": z_sc,
            "z_sp": z_sp,
            "assignment": assignment,
            "offsets": offsets,
            "local_probs": local_probs,
            "disk_anchors": self.offset_decoder.disk_anchors,
            "pair_coords": pair_coords,
            "cell_coords": cell_coords,
        }
