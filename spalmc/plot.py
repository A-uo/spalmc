"""Matplotlib visualizations for SpaLMC outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("SpaLMC plotting requires matplotlib.") from exc
    return plt


def _save_or_show(fig, save_path: str | Path | None, dpi: int = 150):
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig


def _get_spatial_image(
    adata_sp,
    library_id: str | None = None,
    img_key: str = "hires",
    scale_factor: float | None = None,
):
    """Return image and coordinate scale for manual spatial overlays."""

    spatial_uns = adata_sp.uns.get("spatial", None)
    if not spatial_uns:
        return None, 1.0
    if library_id is None:
        library_id = next(iter(spatial_uns.keys()))
    lib = spatial_uns.get(library_id, {})
    images = lib.get("images", {})
    image = images.get(img_key, None)
    if image is None and images:
        image = next(iter(images.values()))
    if scale_factor is None:
        scalefactors = lib.get("scalefactors", {})
        scale_factor = scalefactors.get(f"tissue_{img_key}_scalef", None)
        if scale_factor is None and img_key == "hires":
            scale_factor = scalefactors.get("tissue_hires_scalef", None)
        if scale_factor is None and img_key == "lowres":
            scale_factor = scalefactors.get("tissue_lowres_scalef", None)
    return image, float(scale_factor) if scale_factor is not None else 1.0


def _draw_spatial_background(
    ax,
    image,
    alpha_img: float = 1.0,
    bw: bool = False,
):
    """Draw a spatial image background on an axis."""

    if image is None:
        return
    if bw and image.ndim == 3:
        img = image[..., :3].mean(axis=2)
        ax.imshow(img, cmap="gray", origin="upper", alpha=alpha_img)
    else:
        ax.imshow(image, origin="upper", alpha=alpha_img)


def _finish_spatial_axis(ax, image=None, invert_y: bool = True):
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    if image is None and invert_y:
        ax.invert_yaxis()


def _cell_colors(adata_sc, celltype_key: str | None):
    if celltype_key is None or celltype_key not in adata_sc.obs:
        return None, None
    values = adata_sc.obs[celltype_key].astype("category")
    return values.cat.codes.to_numpy(), values.cat.categories.astype(str).tolist()


def plot_spatial_mapping(
    adata_sc,
    adata_sp,
    celltype_key: str | None = None,
    save_path: str | Path | None = None,
    dpi: int = 150,
    spot_size: float = 16,
    cell_size: float = 7,
    show_spots: bool = False,
    max_points: int | None = None,
    invert_y: bool = True,
    hide_axes: bool = True,
    alpha: float = 0.82,
    palette: dict[str, str] | None = None,
):
    """Plot reconstructed single-cell coordinates colored by cell type.

    The default style follows a single-cell dot distribution plot: cell types
    are drawn one-by-one with a right-side legend, equal aspect ratio, inverted
    y-axis and no axis ticks. Spot centers can be shown as a light background
    with ``show_spots=True``.
    """

    plt = _require_matplotlib()
    if "spatial" not in adata_sp.obsm:
        raise KeyError('`adata_sp.obsm["spatial"]` is required.')
    if "spalmc_spatial" not in adata_sc.obsm:
        raise KeyError('`adata_sc.obsm["spalmc_spatial"]` is required.')
    spots = np.asarray(adata_sp.obsm["spatial"])
    cells = np.asarray(adata_sc.obsm["spalmc_spatial"])
    obs = adata_sc.obs.copy()
    obs["_x"] = cells[:, 0]
    obs["_y"] = cells[:, 1]
    if max_points is not None and len(obs) > max_points:
        obs = obs.sample(max_points, random_state=0)

    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    if show_spots:
        ax.scatter(spots[:, 0], spots[:, 1], s=spot_size, c="lightgray", alpha=0.45, linewidths=0, label="_spots")
    if celltype_key is not None and celltype_key in obs:
        labels = obs[celltype_key].astype("category")
        categories = labels.cat.categories.astype(str).tolist()
        cmap = plt.get_cmap("tab20")
        for idx, cell_type in enumerate(categories):
            sub = obs[labels.astype(str) == cell_type]
            if len(sub) == 0:
                continue
            color = palette[cell_type] if palette is not None and cell_type in palette else cmap(idx % 20)
            ax.scatter(
                sub["_x"],
                sub["_y"],
                s=cell_size,
                alpha=alpha,
                color=color,
                label=cell_type,
                linewidths=0,
            )
        ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5))
    else:
        ax.scatter(obs["_x"], obs["_y"], s=cell_size, alpha=alpha, color="#4C78A8", linewidths=0)
    ax.set_aspect("equal", adjustable="box")
    if invert_y:
        ax.invert_yaxis()
    if hide_axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
    else:
        ax.set_xlabel("spatial x")
        ax.set_ylabel("spatial y")
    ax.set_title("SpaLMC reconstructed single-cell spatial map")
    fig.tight_layout()
    return _save_or_show(fig, save_path, dpi=dpi)


def plot_assignment_probability(
    adata_sc,
    save_path: str | Path | None = None,
    dpi: int = 150,
    bins: int = 40,
):
    """Plot the distribution of maximum assignment probabilities."""

    plt = _require_matplotlib()
    if "spalmc_assignment_prob" not in adata_sc.obs:
        raise KeyError('`adata_sc.obs["spalmc_assignment_prob"]` is required.')
    probs = np.asarray(adata_sc.obs["spalmc_assignment_prob"], dtype=float)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(probs, bins=bins, color="#4C78A8", alpha=0.9)
    ax.set_xlabel("max assignment probability")
    ax.set_ylabel("cell count")
    ax.set_title("SpaLMC assignment confidence")
    return _save_or_show(fig, save_path, dpi=dpi)


def plot_spot_cell_counts(
    adata_sc,
    adata_sp=None,
    save_path: str | Path | None = None,
    dpi: int = 150,
):
    """Plot hard-assigned cell counts per spot as a bar or spatial plot."""

    plt = _require_matplotlib()
    if "spalmc_spot_id" not in adata_sc.obs:
        raise KeyError('`adata_sc.obs["spalmc_spot_id"]` is required.')
    counts = adata_sc.obs["spalmc_spot_id"].astype(str).value_counts()
    if adata_sp is not None and "spatial" in adata_sp.obsm:
        spots = np.asarray(adata_sp.obsm["spatial"])
        vals = np.asarray([counts.get(str(s), 0) for s in adata_sp.obs_names], dtype=float)
        fig, ax = plt.subplots(figsize=(6, 5))
        sc = ax.scatter(spots[:, 0], spots[:, 1], c=vals, s=90, cmap="viridis", edgecolor="white", linewidth=0.5)
        fig.colorbar(sc, ax=ax, label="assigned cells")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("spatial x")
        ax.set_ylabel("spatial y")
    else:
        fig, ax = plt.subplots(figsize=(8, 4))
        counts.sort_index().plot(kind="bar", ax=ax, color="#59A14F")
        ax.set_xlabel("spot")
        ax.set_ylabel("assigned cells")
    ax.set_title("SpaLMC hard assignment counts")
    return _save_or_show(fig, save_path, dpi=dpi)


def plot_loss_history(
    model,
    save_path: str | Path | None = None,
    dpi: int = 150,
):
    """Plot SpaLMC training loss curves."""

    plt = _require_matplotlib()
    history = model.loss_history
    if history is None or len(history) == 0:
        raise ValueError("No loss history found. Run `fit` first.")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for col in history.columns:
        if col.startswith("loss_"):
            ax.plot(history["epoch"], history[col], label=col)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("SpaLMC training history")
    ax.legend(frameon=False, fontsize=8)
    return _save_or_show(fig, save_path, dpi=dpi)


def plot_local_spot_manifold(
    adata_sc,
    adata_sp,
    spot_id,
    celltype_key: str | None = None,
    save_path: str | Path | None = None,
    dpi: int = 150,
    cell_size: float = 22,
    show_radius: bool = True,
):
    """Plot cells and relative offsets assigned to one spot."""

    plt = _require_matplotlib()
    required = ["spalmc_spatial", "spalmc_relative_offset"]
    for key in required:
        if key not in adata_sc.obsm:
            raise KeyError(f'`adata_sc.obsm["{key}"]` is required.')
    if "spalmc_spot_id" not in adata_sc.obs:
        raise KeyError('`adata_sc.obs["spalmc_spot_id"]` is required.')

    spot_name = str(spot_id)
    if isinstance(spot_id, int):
        spot_name = str(adata_sp.obs_names[spot_id])
        center = np.asarray(adata_sp.obsm["spatial"])[spot_id]
    else:
        idx = adata_sp.obs_names.astype(str).get_loc(spot_name)
        center = np.asarray(adata_sp.obsm["spatial"])[idx]

    mask = adata_sc.obs["spalmc_spot_id"].astype(str).to_numpy() == spot_name
    coords = np.asarray(adata_sc.obsm["spalmc_spatial"])[mask]
    colors, labels = _cell_colors(adata_sc[mask], celltype_key)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    if show_radius:
        spots = np.asarray(adata_sp.obsm["spatial"])
        if spots.shape[0] > 1:
            try:
                from sklearn.neighbors import NearestNeighbors

                nn = NearestNeighbors(n_neighbors=2).fit(spots)
                radius = float(np.median(nn.kneighbors(spots, return_distance=True)[0][:, 1]) / 2.0)
                circle = plt.Circle(center, radius, edgecolor="black", facecolor="none", linestyle="--", linewidth=1.0, alpha=0.65)
                ax.add_patch(circle)
            except Exception:
                pass
    ax.scatter([center[0]], [center[1]], s=160, c="black", marker="x", linewidth=2, label="spot center")
    if coords.shape[0] > 0:
        sc = ax.scatter(coords[:, 0], coords[:, 1], s=cell_size, c=colors, cmap="tab20", alpha=0.9)
        if labels is not None:
            handles, _ = sc.legend_elements(num=min(len(labels), 10))
            ax.legend(handles, labels[: len(handles)], title=celltype_key, frameon=False, fontsize=8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("spatial x")
    ax.set_ylabel("spatial y")
    ax.set_title(f"SpaLMC local manifold: {spot_name}")
    return _save_or_show(fig, save_path, dpi=dpi)


def plot_celltype_spatial_panels(
    adata_sc,
    adata_sp=None,
    celltype_key: str = "celltype",
    save_path: str | Path | None = None,
    dpi: int = 150,
    ncols: int = 4,
    cell_size: float = 5,
    spot_size: float = 12,
    max_celltypes: int | None = None,
    show_image: bool | str = "auto",
    library_id: str | None = None,
    img_key: str = "hires",
    scale_factor: float | None = None,
    alpha_img: float = 0.9,
    bw: bool = False,
    show_all_cells: bool = False,
    background_cell_size: float = 2,
    background_alpha: float = 0.12,
    invert_y: bool = True,
):
    """Plot one reconstructed spatial panel per cell type.

    This visualization is intended for checking whether each cell population
    occupies plausible tissue regions after SpaLMC mapping.
    """

    plt = _require_matplotlib()
    if "spalmc_spatial" not in adata_sc.obsm:
        raise KeyError('`adata_sc.obsm["spalmc_spatial"]` is required.')
    if celltype_key not in adata_sc.obs:
        raise KeyError(f'`adata_sc.obs["{celltype_key}"]` is required.')

    coords = np.asarray(adata_sc.obsm["spalmc_spatial"])
    labels = adata_sc.obs[celltype_key].astype("category")
    categories = labels.cat.categories.astype(str).tolist()
    if max_celltypes is not None:
        categories = categories[:max_celltypes]

    n = len(categories)
    ncols = max(1, min(ncols, n))
    nrows = int(np.ceil(n / ncols))
    image, sf = (None, 1.0)
    if adata_sp is not None and (show_image is True or show_image == "auto"):
        image, sf = _get_spatial_image(adata_sp, library_id=library_id, img_key=img_key, scale_factor=scale_factor)
        if show_image is True and image is None:
            raise ValueError("No spatial image found in `adata_sp.uns['spatial']`.")
    plot_coords = coords * sf

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.4 * ncols, 3.2 * nrows),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    spots = None
    if adata_sp is not None and "spatial" in adata_sp.obsm:
        spots = np.asarray(adata_sp.obsm["spatial"]) * sf

    palette = plt.get_cmap("tab20")
    for idx, ct in enumerate(categories):
        ax = axes[idx // ncols][idx % ncols]
        _draw_spatial_background(ax, image, alpha_img=alpha_img, bw=bw)
        if image is None and spots is not None:
            ax.scatter(spots[:, 0], spots[:, 1], s=spot_size, c="lightgray", alpha=0.55, linewidth=0)
        if show_all_cells:
            ax.scatter(
                plot_coords[:, 0],
                plot_coords[:, 1],
                s=background_cell_size,
                c="lightgray",
                alpha=background_alpha,
                linewidth=0,
            )
        mask = labels.astype(str).to_numpy() == ct
        ax.scatter(
            plot_coords[mask, 0],
            plot_coords[mask, 1],
            s=cell_size,
            color=palette(idx % 20),
            alpha=0.9,
            linewidth=0,
        )
        ax.set_title(f"{ct} (n={int(mask.sum())})", fontsize=9)
        _finish_spatial_axis(ax, image=image, invert_y=invert_y)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.suptitle("SpaLMC reconstructed spatial distribution by cell type", y=1.01)
    fig.tight_layout()
    return _save_or_show(fig, save_path, dpi=dpi)


def plot_spot_celltype_abundance(
    adata_sc,
    adata_sp,
    celltype_key: str = "celltype",
    save_path: str | Path | None = None,
    dpi: int = 150,
    ncols: int = 4,
    spot_size: float = 24,
    max_celltypes: int | None = None,
    show_image: bool | str = "auto",
    library_id: str | None = None,
    img_key: str = "hires",
    scale_factor: float | None = None,
    alpha_img: float = 0.9,
    bw: bool = False,
    cmap: str = "magma",
    invert_y: bool = True,
):
    """Plot hard-assigned cell-type counts on spot centers, one panel per type."""

    plt = _require_matplotlib()
    if "spalmc_spot_id" not in adata_sc.obs:
        raise KeyError('`adata_sc.obs["spalmc_spot_id"]` is required.')
    if celltype_key not in adata_sc.obs:
        raise KeyError(f'`adata_sc.obs["{celltype_key}"]` is required.')
    if "spatial" not in adata_sp.obsm:
        raise KeyError('`adata_sp.obsm["spatial"]` is required.')

    labels = adata_sc.obs[celltype_key].astype("category")
    categories = labels.cat.categories.astype(str).tolist()
    if max_celltypes is not None:
        categories = categories[:max_celltypes]

    count_table = (
        adata_sc.obs.assign(
            _spot=adata_sc.obs["spalmc_spot_id"].astype(str),
            _celltype=labels.astype(str),
        )
        .groupby(["_spot", "_celltype"], observed=False)
        .size()
        .unstack(fill_value=0)
    )
    image, sf = (None, 1.0)
    if show_image is True or show_image == "auto":
        image, sf = _get_spatial_image(adata_sp, library_id=library_id, img_key=img_key, scale_factor=scale_factor)
        if show_image is True and image is None:
            raise ValueError("No spatial image found in `adata_sp.uns['spatial']`.")

    coords = np.asarray(adata_sp.obsm["spatial"]) * sf
    spot_names = adata_sp.obs_names.astype(str).tolist()

    n = len(categories)
    ncols = max(1, min(ncols, n))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.4 * ncols, 3.2 * nrows),
        squeeze=False,
        sharex=True,
        sharey=True,
    )

    for idx, ct in enumerate(categories):
        ax = axes[idx // ncols][idx % ncols]
        _draw_spatial_background(ax, image, alpha_img=alpha_img, bw=bw)
        vals = np.asarray([count_table.loc[s, ct] if s in count_table.index and ct in count_table.columns else 0 for s in spot_names])
        vmax = max(float(np.nanmax(vals)), 1.0)
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=vals, s=spot_size, cmap=cmap, vmin=0, vmax=vmax, linewidth=0)
        ax.set_title(ct, fontsize=9)
        _finish_spatial_axis(ax, image=image, invert_y=invert_y)
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.suptitle("SpaLMC hard-assigned cell-type abundance by spot", y=1.01)
    fig.tight_layout()
    return _save_or_show(fig, save_path, dpi=dpi)


def plot_spot_celltype_probabilities(
    adata_sp,
    celltype_names: list[str] | None = None,
    save_path: str | Path | None = None,
    dpi: int = 150,
    ncols: int = 4,
    spot_size: float = 24,
    max_celltypes: int | None = None,
    use_scanpy_spatial: bool | str = False,
    show_image: bool | str = "auto",
    spatial_key: str = "spatial",
    library_id: str | None = None,
    img_key: str = "hires",
    x: str = "x",
    y: str = "y",
    scale_factor: float | None = None,
    alpha_img: float = 1.0,
    bw: bool = False,
    frameon: bool = False,
    font_size: int = 12,
    cmap: str = "viridis",
):
    """Plot per-spot probability of each cell type estimated by SpaLMC.

    By default this uses a manual image-backed scatter overlay. Set
    ``use_scanpy_spatial=True`` only when you specifically want Scanpy's native
    spatial layout.
    """

    plt = _require_matplotlib()
    if spatial_key not in adata_sp.obsm:
        if x in adata_sp.obs and y in adata_sp.obs:
            adata_sp.obsm[spatial_key] = adata_sp.obs[[x, y]].to_numpy(dtype=float)
        else:
            raise KeyError(f'`adata_sp.obsm["{spatial_key}"]` is required, or obs columns `{x}`/`{y}`.')
    if "spalmc_celltype_prob" not in adata_sp.obsm:
        raise KeyError('`adata_sp.obsm["spalmc_celltype_prob"]` is required. Run `model.add_to_adata()` first.')
    probs = np.asarray(adata_sp.obsm["spalmc_celltype_prob"])
    if celltype_names is None:
        celltype_names = adata_sp.uns.get("spalmc_celltype_categories", [f"type_{i}" for i in range(probs.shape[1])])
    celltype_names = [str(x) for x in celltype_names]
    if max_celltypes is not None:
        celltype_names = celltype_names[:max_celltypes]
        probs = probs[:, :max_celltypes]

    plot_keys = [f"spalmc_prob_{name}" for name in celltype_names]
    for idx, key in enumerate(plot_keys):
        adata_sp.obs[key] = probs[:, idx]

    has_spatial_image = "spatial" in adata_sp.uns and bool(adata_sp.uns.get("spatial"))
    if use_scanpy_spatial == "auto":
        do_scanpy = has_spatial_image
    else:
        do_scanpy = bool(use_scanpy_spatial)

    if do_scanpy:
        try:
            import matplotlib as mpl
            import scanpy as sc
        except ImportError as exc:
            if use_scanpy_spatial is True:
                raise ImportError("scanpy is required for image-backed spatial plotting.") from exc
            do_scanpy = False

    if do_scanpy:
        if not has_spatial_image and (spot_size is None or scale_factor is None):
            raise ValueError(
                "When `adata_sp.uns['spatial']` is absent, provide both `spot_size` "
                "and `scale_factor` or set `use_scanpy_spatial=False`."
            )
        with mpl.rc_context(
            {
                "font.size": font_size,
                "axes.titlesize": font_size,
                "axes.labelsize": font_size,
                "legend.fontsize": max(font_size - 2, 1),
            }
        ):
            sc.pl.spatial(
                adata_sp,
                color=plot_keys,
                cmap=cmap,
                show=False,
                frameon=frameon,
                spot_size=spot_size,
                scale_factor=scale_factor,
                alpha_img=alpha_img,
                bw=bw,
                ncols=ncols,
                vmin=0,
                vmax=1,
            )
        fig = plt.gcf()
        for ax, name in zip(fig.axes, celltype_names):
            if hasattr(ax, "set_title"):
                ax.set_title(name)
        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    image, sf = (None, 1.0)
    if show_image is True or show_image == "auto":
        image, sf = _get_spatial_image(adata_sp, library_id=library_id, img_key=img_key, scale_factor=scale_factor)
        if show_image is True and image is None:
            raise ValueError("No spatial image found in `adata_sp.uns['spatial']`.")

    coords = np.asarray(adata_sp.obsm[spatial_key]) * sf
    n = len(celltype_names)
    ncols = max(1, min(ncols, n))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.4 * ncols, 3.2 * nrows),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    for idx, name in enumerate(celltype_names):
        ax = axes[idx // ncols][idx % ncols]
        _draw_spatial_background(ax, image, alpha_img=alpha_img, bw=bw)
        sc = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=probs[:, idx],
            s=spot_size,
            cmap=cmap,
            vmin=0,
            vmax=1,
            linewidth=0,
        )
        ax.set_title(name, fontsize=9)
        _finish_spatial_axis(ax, image=image, invert_y=True)
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")
    fig.suptitle("SpaLMC per-spot cell-type probabilities", y=1.01)
    fig.tight_layout()
    return _save_or_show(fig, save_path, dpi=dpi)
