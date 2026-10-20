"""Utility functions for SpaLMC.

The helpers in this module deliberately avoid all-pair graph construction
unless the caller asks for it. SpaLMC's first prototype is full-batch, but
single-cell and spot counts can grow quickly.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def set_seed(seed: int = 0) -> None:
    """Set random seeds for Python, NumPy and PyTorch."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_tensor(
    x: Any,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Convert an array-like object to a dense PyTorch tensor."""

    if hasattr(x, "toarray"):
        x = x.toarray()
    if isinstance(x, torch.Tensor):
        tensor = x.to(dtype=dtype)
    else:
        tensor = torch.as_tensor(np.asarray(x), dtype=dtype)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def safe_normalize(
    x: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """L2-normalize a tensor with an epsilon guard."""

    return x / x.norm(p=2, dim=dim, keepdim=True).clamp_min(eps)


def cosine_similarity_matrix(
    a: torch.Tensor,
    b: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return pairwise cosine similarities between rows of ``a`` and ``b``."""

    return safe_normalize(a, eps=eps) @ safe_normalize(b, eps=eps).T


def pairwise_distance(
    a: torch.Tensor,
    b: torch.Tensor | None = None,
    squared: bool = False,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Compute pairwise Euclidean distances."""

    b = a if b is None else b
    dist2 = torch.cdist(a, b, p=2).square()
    if squared:
        return dist2
    return torch.sqrt(dist2.clamp_min(eps))


def build_knn_graph(
    x: np.ndarray | torch.Tensor,
    k: int = 10,
    include_self: bool = False,
    device: str | torch.device | None = None,
) -> torch.LongTensor:
    """Build a directed kNN graph.

    Args:
        x: Matrix with shape ``n_samples x n_features``.
        k: Number of neighbors per sample.
        include_self: Whether to keep the self edge returned by kNN.
        device: Optional output device.

    Returns:
        ``edge_index`` with shape ``2 x n_edges``.
    """

    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().numpy()
    else:
        x_np = np.asarray(x)
    if x_np.ndim != 2:
        raise ValueError("`x` must have shape n_samples x n_features.")
    n = x_np.shape[0]
    if n < 2:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    n_neighbors = min(n, k + (0 if include_self else 1))
    try:
        from sklearn.neighbors import NearestNeighbors
    except ImportError as exc:
        raise ImportError("SpaLMC requires scikit-learn for kNN graph construction.") from exc

    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(x_np)
    indices = nn.kneighbors(x_np, return_distance=False)

    src, dst = [], []
    for i, neigh in enumerate(indices):
        kept = []
        for j in neigh:
            if not include_self and int(j) == i:
                continue
            kept.append(int(j))
            if len(kept) >= k:
                break
        src.extend([i] * len(kept))
        dst.extend(kept)

    return torch.as_tensor([src, dst], dtype=torch.long, device=device)


def masked_softmax_topk(
    scores: torch.Tensor,
    top_k: int | None,
    dim: int = 1,
) -> torch.Tensor:
    """Softmax after masking all but the top-k scores along ``dim``."""

    if top_k is None or top_k >= scores.shape[dim]:
        return F.softmax(scores, dim=dim)
    if top_k <= 0:
        raise ValueError("`top_k` must be positive when provided.")
    values, indices = torch.topk(scores, k=top_k, dim=dim)
    masked = torch.full_like(scores, torch.finfo(scores.dtype).min)
    masked.scatter_(dim, indices, values)
    return F.softmax(masked, dim=dim)
