"""AnnData preprocessing utilities for SpaLMC."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import warnings

import numpy as np
import pandas as pd


@dataclass
class SpatialNormalization:
    """Affine normalization parameters for spatial coordinates."""

    mean: np.ndarray
    scale: np.ndarray


def align_genes(
    adata_sc,
    adata_sp,
    genes: Iterable[str] | None = None,
    use_hvg: bool = False,
    hvg_key: str = "highly_variable",
) -> tuple:
    """Return AnnData views aligned to shared genes, optionally restricted to HVGs."""

    shared = pd.Index(adata_sc.var_names).intersection(pd.Index(adata_sp.var_names))
    if len(shared) == 0:
        raise ValueError("No shared genes found between `adata_sc` and `adata_sp`.")
    if genes is not None:
        requested = pd.Index([str(g) for g in genes])
        shared = requested.intersection(shared)
        if len(shared) == 0:
            raise ValueError("None of the requested training genes are present in both AnnData objects.")
    if use_hvg:
        mask = pd.Series(True, index=shared)
        if hvg_key in adata_sc.var:
            mask &= adata_sc.var.loc[shared, hvg_key].astype(bool)
        if hvg_key in adata_sp.var:
            mask &= adata_sp.var.loc[shared, hvg_key].astype(bool)
        shared = shared[mask.values]
        if len(shared) == 0:
            raise ValueError("HVG filtering removed all shared genes.")
    return adata_sc[:, shared].copy(), adata_sp[:, shared].copy()


def load_gene_list(genes: Iterable[str] | str | Path | None) -> list[str] | None:
    """Load training genes from a list-like object or a CSV/TXT file.

    Tangram marker CSV files often look like an index column plus one gene-name
    column; this helper picks the first non-index-like column by default.
    """

    if genes is None:
        return None
    if isinstance(genes, (str, Path)):
        path = Path(genes)
        if not path.exists():
            warnings.warn(f"Gene list file `{path}` was not found; falling back to overlap genes.", RuntimeWarning)
            return None
        if path.suffix.lower() in {".csv", ".tsv"}:
            sep = "\t" if path.suffix.lower() == ".tsv" else ","
            df = pd.read_csv(path, sep=sep)
            if df.empty:
                warnings.warn(f"Gene list file `{path}` is empty; falling back to overlap genes.", RuntimeWarning)
                return None
            columns = [c for c in df.columns if not str(c).lower().startswith("unnamed")]
            gene_col = None
            for col in columns:
                values = df[col].dropna().astype(str)
                if len(values) > 0 and not values.str.fullmatch(r"\d+").all():
                    gene_col = col
                    break
            if gene_col is None:
                gene_col = df.columns[-1]
            values = df[gene_col].dropna().astype(str).tolist()
        else:
            values = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    else:
        values = [str(g) for g in genes]
    seen = set()
    out = []
    for gene in values:
        gene = str(gene).strip()
        if gene and gene not in seen:
            seen.add(gene)
            out.append(gene)
    return out or None


def pp_adatas(
    adata_sc,
    adata_sp,
    genes: Iterable[str] | str | Path | None = None,
    celltype_key: str | None = None,
    spatial_key: str = "spatial",
    density_key: str | None = "auto",
    copy: bool = False,
    min_cells_per_gene: int = 1,
) -> tuple:
    """Preprocess AnnData objects for SpaLMC, inspired by Tangram's ``pp_adatas``.

    This function does lightweight bookkeeping rather than heavy normalization:
    it determines shared/training genes, stores them in ``.uns``, ensures spatial
    coordinates exist, prepares a density prior column, and records cell-type
    categories when available. The model will then consume ``uns['training_genes']``.

    Args:
        adata_sc: Single-cell AnnData.
        adata_sp: Spatial AnnData.
        genes: Optional marker/training gene list or path to CSV/TXT. If omitted
            or unusable, uses the full overlap between ``adata_sc.var_names`` and
            ``adata_sp.var_names``.
        celltype_key: Optional scRNA cell-type label key.
        spatial_key: Spatial coordinate key to ensure in ``adata_sp.obsm``.
        density_key: Density prior key, ``"auto"`` for common columns, or
            ``None`` for uniform.
        copy: Whether to return copies.
        min_cells_per_gene: Drop genes expressed in fewer cells/spots when
            ``var['n_cells']`` is available.

    Returns:
        ``(adata_sc_pp, adata_sp_pp)``.
    """

    ad_sc = adata_sc.copy() if copy else adata_sc
    ad_sp = adata_sp.copy() if copy else adata_sp

    infer_spatial_coords(ad_sp, spatial_key=spatial_key)
    overlap = pd.Index(ad_sc.var_names.astype(str)).intersection(pd.Index(ad_sp.var_names.astype(str)))
    if min_cells_per_gene > 0:
        if "n_cells" in ad_sc.var:
            overlap = overlap[ad_sc.var.loc[overlap, "n_cells"].to_numpy() >= min_cells_per_gene]
        if "n_cells" in ad_sp.var:
            overlap = overlap[ad_sp.var.loc[overlap, "n_cells"].to_numpy() >= min_cells_per_gene]
    if len(overlap) == 0:
        raise ValueError("No overlap genes remain after preprocessing.")

    loaded_genes = load_gene_list(genes)
    if loaded_genes is None:
        training = overlap
    else:
        requested = pd.Index([str(g) for g in loaded_genes])
        training = requested.intersection(overlap)
        missing = requested.difference(overlap)
        if len(missing) > 0:
            warnings.warn(f"{len(missing)} requested genes are not shared and were ignored.", RuntimeWarning)
    if len(training) == 0:
        warnings.warn(
            "No requested marker genes overlap both datasets; falling back to all overlap genes.",
            RuntimeWarning,
        )
        training = overlap

    training_list = training.astype(str).tolist()
    overlap_list = overlap.astype(str).tolist()
    for ad in (ad_sc, ad_sp):
        ad.uns["overlap_genes"] = overlap_list
        ad.uns["training_genes"] = training_list
        ad.var["spalmc_overlap_gene"] = ad.var_names.astype(str).isin(overlap_list)
        ad.var["spalmc_training_gene"] = ad.var_names.astype(str).isin(training_list)

    density_prior, source = infer_spot_cell_count_prior(ad_sp, n_cells=ad_sc.n_obs, density_key=density_key)
    # Store as a probability-like prior independent of the current scRNA cell count.
    density_prob = density_prior / np.clip(density_prior.sum(), 1e-8, None)
    ad_sp.obs["spalmc_density_prior"] = density_prob.astype(np.float32)
    ad_sp.uns["spalmc_density_source"] = source

    if celltype_key is not None and celltype_key in ad_sc.obs:
        cats = pd.Categorical(ad_sc.obs[celltype_key]).categories.astype(str).tolist()
        ad_sc.uns["spalmc_celltype_key"] = celltype_key
        ad_sc.uns["spalmc_celltype_categories"] = cats
    elif celltype_key is not None:
        warnings.warn(f"`celltype_key={celltype_key}` was not found in adata_sc.obs.", RuntimeWarning)

    return ad_sc, ad_sp


def get_expression_matrix(adata, layer_key: str | None = None) -> np.ndarray:
    """Extract a dense expression matrix from ``adata.X`` or a layer."""

    if layer_key is not None:
        if layer_key not in adata.layers:
            raise KeyError(f"Layer `{layer_key}` not found in AnnData.")
        x = adata.layers[layer_key]
    else:
        x = adata.X
    if hasattr(x, "toarray"):
        x = x.toarray()
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("Expression matrix must be two-dimensional.")
    return x


def normalize_expression(
    x: np.ndarray,
    log1p: bool = True,
    scale: bool = True,
    target_sum: float | None = 1e4,
    eps: float = 1e-8,
) -> np.ndarray:
    """Library-size normalize, log-transform and feature-scale expression."""

    x = np.asarray(x, dtype=np.float32)
    x = np.maximum(x, 0.0)
    out = x.copy()
    if target_sum is not None:
        lib = out.sum(axis=1, keepdims=True)
        out = out / np.clip(lib, eps, None) * float(target_sum)
    if log1p:
        out = np.log1p(out)
    if scale:
        mean = out.mean(axis=0, keepdims=True)
        std = out.std(axis=0, keepdims=True)
        out = (out - mean) / np.clip(std, eps, None)
    return out.astype(np.float32)


def encode_celltypes(
    adata_sc,
    celltype_key: str | None,
) -> tuple[np.ndarray | None, list[str] | None]:
    """One-hot encode cell types from ``adata_sc.obs``."""

    if celltype_key is None:
        return None, None
    if celltype_key not in adata_sc.obs:
        warnings.warn(
            f"`celltype_key={celltype_key}` was not found; composition loss will be skipped.",
            RuntimeWarning,
        )
        return None, None
    categories = pd.Categorical(adata_sc.obs[celltype_key]).categories.astype(str).tolist()
    codes = pd.Categorical(adata_sc.obs[celltype_key], categories=categories).codes
    onehot = np.zeros((adata_sc.n_obs, len(categories)), dtype=np.float32)
    valid = codes >= 0
    onehot[np.arange(adata_sc.n_obs)[valid], codes[valid]] = 1.0
    return onehot, categories


def prepare_spot_prior(
    spot_celltype_prior,
    celltype_categories: Iterable[str] | None,
    n_spots: int | None = None,
    spot_names: Iterable[str] | None = None,
    eps: float = 1e-8,
) -> np.ndarray | None:
    """Prepare and normalize a spot-celltype prior matrix."""

    if spot_celltype_prior is None or celltype_categories is None:
        return None
    categories = list(celltype_categories)
    if isinstance(spot_celltype_prior, pd.DataFrame):
        if spot_names is not None:
            spot_names = [str(s) for s in spot_names]
            prior_index = spot_celltype_prior.index.astype(str)
            if set(spot_names).issubset(set(prior_index)):
                spot_celltype_prior = spot_celltype_prior.copy()
                spot_celltype_prior.index = prior_index
                spot_celltype_prior = spot_celltype_prior.loc[spot_names]
        missing = [c for c in categories if c not in spot_celltype_prior.columns]
        if missing:
            raise ValueError(f"Spot prior is missing cell type columns: {missing}")
        prior = spot_celltype_prior.loc[:, categories].to_numpy(dtype=np.float32)
    else:
        prior = np.asarray(spot_celltype_prior, dtype=np.float32)
    if prior.ndim != 2:
        raise ValueError("`spot_celltype_prior` must have shape n_spots x n_celltypes.")
    if n_spots is not None and prior.shape[0] != n_spots:
        raise ValueError("`spot_celltype_prior` has a different number of spots.")
    if prior.shape[1] != len(categories):
        raise ValueError("`spot_celltype_prior` has a different number of cell types.")
    prior = np.maximum(prior, 0.0)
    prior = prior / np.clip(prior.sum(axis=1, keepdims=True), eps, None)
    return prior.astype(np.float32)


def normalize_spatial_coords(coords: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, SpatialNormalization]:
    """Standardize spatial coordinates and return normalization metadata."""

    coords = np.asarray(coords, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("Spatial coordinates must have shape n_spots x 2.")
    mean = coords.mean(axis=0)
    scale = coords.std(axis=0)
    scale = np.where(scale < eps, 1.0, scale)
    return ((coords - mean) / scale).astype(np.float32), SpatialNormalization(mean=mean, scale=scale)


def estimate_spot_radius(coords: np.ndarray) -> float:
    """Estimate radius as half of the median nearest-neighbor spot distance."""

    coords = np.asarray(coords, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("Spatial coordinates must have shape n_spots x 2.")
    if coords.shape[0] < 2:
        return 1.0
    try:
        from sklearn.neighbors import NearestNeighbors
    except ImportError as exc:
        raise ImportError("SpaLMC requires scikit-learn to estimate spot radius.") from exc
    nn = NearestNeighbors(n_neighbors=2)
    nn.fit(coords)
    distances = nn.kneighbors(coords, return_distance=True)[0][:, 1]
    return float(np.median(distances) / 2.0)


def infer_spatial_coords(adata_sp, spatial_key: str = "spatial") -> np.ndarray:
    """Infer spot coordinates from ``obsm[spatial_key]`` or obs columns x/y."""

    if spatial_key in adata_sp.obsm:
        return np.asarray(adata_sp.obsm[spatial_key], dtype=np.float32)
    lower_map = {str(c).lower(): c for c in adata_sp.obs.columns}
    if "x" in lower_map and "y" in lower_map:
        coords = adata_sp.obs[[lower_map["x"], lower_map["y"]]].to_numpy(dtype=np.float32)
        adata_sp.obsm[spatial_key] = coords
        return coords
    raise KeyError(
        f'`adata_sp.obsm["{spatial_key}"]` is required, or provide coordinate columns '
        '`adata_sp.obs["x"]` and `adata_sp.obs["y"]`.'
    )


def build_expression_candidate_spots(
    x_sc: np.ndarray,
    x_sp: np.ndarray,
    top_k: int,
    metric: str = "cosine",
    ensure_spot_coverage: bool = True,
) -> np.ndarray:
    """Preselect candidate spots for each cell using expression-space kNN."""

    if top_k <= 0:
        raise ValueError("`top_k` must be positive.")
    top_k = min(int(top_k), x_sp.shape[0])
    try:
        from sklearn.neighbors import NearestNeighbors
    except ImportError as exc:
        raise ImportError("SpaLMC requires scikit-learn for candidate spot selection.") from exc
    nn = NearestNeighbors(n_neighbors=top_k, metric=metric)
    nn.fit(x_sp)
    candidates = nn.kneighbors(x_sc, return_distance=False).astype(np.int64)
    if ensure_spot_coverage and x_sc.shape[0] >= x_sp.shape[0]:
        covered = np.zeros(x_sp.shape[0], dtype=bool)
        covered[np.unique(candidates)] = True
        missing = np.where(~covered)[0]
        if missing.size > 0:
            cell_nn = NearestNeighbors(n_neighbors=1, metric=metric)
            cell_nn.fit(x_sc)
            nearest_cells = cell_nn.kneighbors(x_sp[missing], return_distance=False)[:, 0]
            for spot, cell in zip(missing, nearest_cells):
                candidates[int(cell), -1] = int(spot)
    return candidates


def infer_spot_cell_count_prior(
    adata_sp,
    n_cells: int,
    density_key: str | None = "auto",
    eps: float = 1e-8,
) -> tuple[np.ndarray, str]:
    """Infer target cell mass per spot.

    The returned vector sums to ``n_cells`` because each scRNA-seq cell
    contributes one unit of assignment mass. If no density column is available,
    a uniform prior is used.
    """

    if density_key == "auto":
        candidates = [
            "rna_count_based_density",
            "uniform_density",
            "cell_count",
            "n_cells",
            "density",
        ]
        density_key = next((key for key in candidates if key in adata_sp.obs), None)
    if density_key is not None and density_key in adata_sp.obs:
        raw = adata_sp.obs[density_key].to_numpy(dtype=np.float32)
        source = str(density_key)
    else:
        raw = np.ones(adata_sp.n_obs, dtype=np.float32)
        source = "uniform"
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    raw = np.maximum(raw, 0.0)
    if raw.sum() <= eps:
        raw = np.ones_like(raw)
        source = "uniform"
    prior = raw / np.clip(raw.sum(), eps, None) * float(n_cells)
    return prior.astype(np.float32), source
