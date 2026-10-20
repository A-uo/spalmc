"""Review figures for SpaLMC runs; truth is used only by these display helpers.

Coordinates are rendered as returned, without jitter, permutation matching, or
geometric registration. All plots are saved as 300 dpi PNG and vector SVG.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle
from matplotlib.ticker import MaxNLocator


SPOT_RADIUS = 0.5
CELL_RADIUS = 0.1
RULE_NAMES = {"capacity": "Capacity assignment", "native_output": "Capacity assignment",
              "native_capacity": "Capacity assignment", "native_argmax": "Native argmax",
              "score_argmax": "Score argmax"}
RULE_COLORS = {"capacity": "#2477ad", "native_output": "#2477ad",
               "native_capacity": "#2477ad", "native_argmax": "#2477ad",
               "score_argmax": "#e28b2d"}


def _style():
    return plt.rc_context({
        "font.family": "DejaVu Sans", "font.size": 22,
        "axes.titlesize": 25, "axes.labelsize": 26,
        "xtick.labelsize": 22, "ytick.labelsize": 22,
        "figure.titlesize": 29, "legend.fontsize": 19,
        "svg.fonttype": "none", "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _save(fig, output: Path) -> Path:
    path = Path(output)
    if path.suffix.lower() in {".png", ".svg"}:
        path = path.with_suffix("")
    path.parent.mkdir(parents=True, exist_ok=True)
    png = path.with_suffix(".png")
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png


def _figure_output(output, name):
    path = Path(output)
    return path if path.suffix.lower() in {".png", ".svg"} else path / name


def _radii(observed):
    radii = (float(observed.get("spot_radius", SPOT_RADIUS)),
             float(observed.get("cell_radius", CELL_RADIUS)))
    if not all(np.isfinite(radius) and radius > 0 for radius in radii):
        raise ValueError("Displayed spot and cell radii must be positive and finite")
    return radii


def _arrays(observed, prediction, truth):
    ids = np.asarray(observed["cell_ids"]).astype(str)
    predicted_ids = np.asarray(prediction["cell_ids"]).astype(str)
    if not np.array_equal(ids, predicted_ids):
        raise ValueError("Figures require the exact original cell-ID order; no matching is performed")
    centers = np.asarray(observed["centers"], dtype=float)
    coordinates = np.asarray(prediction["coordinates"], dtype=float)
    members = np.asarray(prediction["membership"], dtype=int)
    if coordinates.shape != (len(ids), 2) or members.shape != (len(ids),):
        raise ValueError("Prediction needs one coordinate and member per original cell ID")
    if not np.isfinite(coordinates).all() or not np.isfinite(centers).all():
        raise ValueError("Coordinates and spot centers must be finite")
    real = true_member = annotation = None
    if truth is not None:
        if "cell_ids" in truth and not np.array_equal(ids, np.asarray(truth["cell_ids"]).astype(str)):
            raise ValueError("Held-out truth must retain the exact original cell-ID order")
        real = np.asarray(truth["truth"], dtype=float)
        true_member = np.asarray(truth["true_member"], dtype=int)
        if real.shape != coordinates.shape or true_member.shape != members.shape:
            raise ValueError("Held-out truth needs one coordinate and member per original cell ID")
        if not np.isfinite(real).all():
            raise ValueError("Held-out coordinates must be finite")
        if "source_annotation" in truth:
            annotation = np.asarray(truth["source_annotation"]).astype(str)
            if annotation.shape != ids.shape:
                raise ValueError("Annotations must have one entry per original cell ID")
    return ids, centers, coordinates, members, real, true_member, annotation


def _colors(ids, annotation=None):
    labels = np.asarray(annotation if annotation is not None else ids).astype(str)
    categories = sorted(set(labels))
    cmap = plt.get_cmap("tab20")
    palette = {value: cmap(i % 20) for i, value in enumerate(categories)}
    return np.asarray([palette[value] for value in labels])


def _spots(ax, centers, radius=SPOT_RADIUS):
    patches = [Circle(tuple(c), radius) for c in centers]
    ax.add_collection(PatchCollection(patches, facecolor="none", edgecolor="#555555",
                                     linewidth=0.65, alpha=0.8, zorder=1))


def _cells(ax, points, colors, incorrect=None, radius=CELL_RADIUS):
    if len(points) == 0:
        return
    edges = np.full(len(points), "#ffffff", dtype=object)
    widths = np.full(len(points), 0.35)
    if incorrect is not None:
        edges[np.asarray(incorrect)] = "#c52525"
        widths[np.asarray(incorrect)] = 1.4
    patches = [Circle(tuple(p), radius) for p in points]
    ax.add_collection(PatchCollection(patches, facecolor=colors, edgecolor=edges,
                                     linewidth=widths, alpha=0.85, zorder=3))


def _coordinate_axis(ax, limits):
    ax.set(xlim=limits[0], ylim=limits[1], xlabel="x", ylabel="y")
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.grid(False)


def plot_overview(observed: dict, prediction: dict, truth: dict | None,
                  output: Path, title: str) -> Path:
    """Show native predicted coordinates alongside optional held-out coordinates.

    ``output`` is a figure directory, or an explicit PNG/SVG filename.
    """
    ids, centers, coordinates, members, real, true_member, annotation = _arrays(observed, prediction, truth)
    spot_radius, cell_radius = _radii(observed)
    colors = _colors(ids, annotation)
    shown = [centers, coordinates] + ([] if real is None else [real])
    all_points = np.concatenate(shown)
    margin = max(spot_radius, cell_radius) * 1.1
    lo = all_points.min(0) - margin
    hi = all_points.max(0) + margin
    limits = ((lo[0], hi[0]), (lo[1], hi[1]))
    with _style():
        columns = 1 if real is None else 2
        fig, axes = plt.subplots(1, columns, figsize=(11 * columns, 8), squeeze=False)
        panels = [(coordinates, "Predicted coordinates")]
        if real is not None:
            panels.insert(0, (real, "True coordinates (held-out)"))
        for ax, (points, subtitle) in zip(axes[0], panels):
            _spots(ax, centers, radius=spot_radius)
            _cells(ax, points, colors, radius=cell_radius)
            _coordinate_axis(ax, limits)
            ax.set_title(subtitle, pad=18)
        fig.suptitle(title, y=1.02)
        color_note = "Annotation colors are held-out display information." if annotation is not None else "Colors identify original cell IDs."
        fig.text(0.5, -0.015,
                 f"Spot radius = {spot_radius:g}; cell radius = {cell_radius:g}. {color_note}\n"
                 "Native coordinates; no registration, jitter, or identity permutation.",
                 ha="center", fontsize=18)
        fig.tight_layout(w_pad=3)
        return _save(fig, _figure_output(output, "coordinate_overview"))


def plot_spot(observed: dict, prediction: dict, truth: dict | None,
              output: Path, title: str, spot: int) -> Path:
    """Compare one spot's actual/predicted members, keeping original IDs visible.

    Red outlines mark an incorrect predicted member in the prediction panel or
    a true member assigned elsewhere in the truth panel. At most 30 labels are
    displayed per panel; data exports are not truncated by this plotting helper.
    """
    ids, centers, coordinates, members, real, true_member, annotation = _arrays(observed, prediction, truth)
    spot_radius, cell_radius = _radii(observed)
    if not 0 <= spot < len(centers):
        raise ValueError("spot must be an observed spot index")
    selected = members == spot
    union = selected if true_member is None else (selected | (true_member == spot))
    colors = _colors(ids, annotation)
    if annotation is None:
        # The compact local palette remains stable across the two panels.
        colors[union] = _colors(ids[union])
    center = centers[spot]
    relevant = [center.reshape(1, 2), coordinates[selected]]
    if real is not None:
        relevant.append(real[true_member == spot])
    points = np.concatenate(relevant)
    extent = max(spot_radius * 1.25, float(np.max(np.abs(points - center))) + cell_radius * 1.5)
    limits = ((center[0]-extent, center[0]+extent), (center[1]-extent, center[1]+extent))
    with _style():
        columns = 1 if real is None else 2
        fig, axes = plt.subplots(1, columns, figsize=(10 * columns, 9), squeeze=False)
        panels = [(coordinates, selected, "Predicted members")]
        if real is not None:
            panels.insert(0, (real, true_member == spot, "True members (held-out)"))
        for ax, (xy, mask, subtitle) in zip(axes[0], panels):
            _spots(ax, centers[[spot]], radius=spot_radius)
            rows = np.flatnonzero(mask)
            incorrect = None if true_member is None else members[rows] != true_member[rows]
            _cells(ax, xy[rows], colors[rows], incorrect, radius=cell_radius)
            for row in rows[:30]:
                ax.annotate(ids[row], xy=xy[row], xytext=(5, 6), textcoords="offset points",
                            fontsize=15, zorder=5, bbox=dict(facecolor="white", alpha=.65,
                                                           edgecolor="none", pad=.25))
            _coordinate_axis(ax, limits)
            ax.set_title(f"{subtitle}\n{len(rows)} original cell IDs", pad=18)
        fig.suptitle(f"{title} | Spot {spot}", y=1.03)
        label_note = " Only the first 30 IDs per panel are labeled." if int(union.sum()) > 30 else ""
        fig.text(.5, -.025,
                 "Same original ID keeps its color. Red outline: incorrect membership.\n"
                 f"Spot radius = {spot_radius:g}; cell radius = {cell_radius:g}. No identity matching or coordinate alignment.{label_note}",
                 ha="center", fontsize=17)
        fig.tight_layout(w_pad=3)
        return _save(fig, _figure_output(output, f"spot_{spot}_comparison"))


def plot_training(history: pd.DataFrame, output: Path, title: str) -> Path:
    """Render recorded training objectives without implying coordinate accuracy."""
    if history.empty:
        raise ValueError("Training history is empty")
    frame = history.copy()
    epoch = frame["epoch"] if "epoch" in frame else np.arange(1, len(frame)+1)
    losses = [key for key in frame if key.startswith("loss") and pd.api.types.is_numeric_dtype(frame[key])]
    if not losses:
        controls = {"epoch", "temperature", "learning_rate", "lr", "stage", "step", "seed"}
        losses = [key for key in frame.select_dtypes(include="number") if key not in controls]
    if not losses:
        raise ValueError("Training history has no numeric objective columns")
    total = next((key for key in ("loss_total", "total_loss", "loss", "total", "objective") if key in losses), losses[0])
    components = [key for key in losses if key != total]
    with _style():
        fig, axes = plt.subplots(1, 2 if components else 1,
                                 figsize=(20 if components else 12, 8), squeeze=False)
        axes[0, 0].plot(epoch, frame[total], lw=2.5, color="#2477ad")
        axes[0, 0].set(xlabel="Epoch", ylabel="Training objective", title="Total objective")
        if components:
            ax = axes[0, 1]
            for key in components:
                ax.plot(epoch, frame[key], lw=1.8, label=key.removeprefix("loss_").replace("_", " ").title())
            ax.set(xlabel="Epoch", ylabel="Recorded component value", title="Objective components")
            ax.legend(fontsize=16, loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
        fig.suptitle(title, y=1.03)
        fig.text(.5, -.025, "Training objectives are not held-out coordinate accuracy measures.",
                 ha="center", fontsize=18)
        fig.tight_layout(w_pad=3)
        return _save(fig, output)


def _rule_name(rule):
    return RULE_NAMES.get(str(rule), str(rule).replace("_", " ").title())


def plot_evaluation(metrics: pd.DataFrame, curves: dict, counts: dict,
                    output: Path, title: str) -> list[Path]:
    """Keep capacity and score-argmax evaluations separate; never select maxima.

    ``output`` is a directory. Required count columns are ``true_cells`` and
    ``predicted_cells``; ROC frames require ``fpr`` and ``mean_tpr`` (or ``tpr``).
    """
    required = {"decision_rule", "member_accuracy", "macro_f1", "macro_auc", "count_mae"}
    if metrics.empty or not required.issubset(metrics.columns):
        raise ValueError(f"Metrics require nonempty rows and columns: {sorted(required)}")
    if metrics.decision_rule.duplicated().any():
        raise ValueError("Evaluation plots require exactly one row per decision rule")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = metrics.reset_index(drop=True)
    rules = rows.decision_rule.astype(str).tolist()
    colors = [RULE_COLORS.get(rule, plt.get_cmap("tab10")(i)) for i, rule in enumerate(rules)]
    paths = []
    with _style():
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        for ax, key, ylabel in zip(axes, ("member_accuracy", "macro_f1"),
                                   ("Member accuracy", "Macro F1")):
            values = rows[key].to_numpy(dtype=float)
            ax.bar(np.arange(len(rules)), values, color=colors, width=.62)
            ax.set_xticks(np.arange(len(rules)), [_rule_name(rule) for rule in rules])
            ax.set(ylabel=ylabel, ylim=(0, 1.15))
            ax.grid(axis="y", alpha=.17)
            ax.set_axisbelow(True)
            for i, value in enumerate(values):
                label = f"{value:.2%}" if key == "member_accuracy" else f"{value:.3f}"
                ax.text(i, value+.025, label, ha="center", va="bottom", fontsize=23, fontweight="bold")
        fig.suptitle(f"{title} | Membership evaluation", y=1.02)
        fig.text(.5, -.015, "All original cell IDs; F1 averages over all observed spots. Both decision rules are shown.",
                 ha="center", fontsize=18)
        fig.tight_layout(w_pad=3)
        paths.append(_save(fig, output / "membership_accuracy_f1"))

        fig, ax = plt.subplots(figsize=(11, 9))
        for i, (rule, color) in enumerate(zip(rules, colors)):
            if rule not in curves:
                raise ValueError(f"Missing ROC curve for decision rule {rule}")
            curve = curves[rule]
            ykey = "mean_tpr" if "mean_tpr" in curve else "tpr"
            # Add the conventional ROC origin for display only; reported AUCs
            # retain the exact one-vs-rest calculation, not plotted-grid area.
            x = np.r_[0.0, curve.fpr.to_numpy(dtype=float)]
            y = np.r_[0.0, curve[ykey].to_numpy(dtype=float)]
            ax.plot(x, y, color=color, lw=2.5, linestyle="-" if i == 0 else "--")
            ax.text(.40, .25-i*.09,
                    f"{_rule_name(rule)}: AUC {rows.loc[i, 'macro_auc']:.3f}",
                    transform=ax.transAxes, color=color, fontsize=19)
        ax.plot([0, 1], [0, 1], color="#888888", lw=1, linestyle=":")
        ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="False positive rate", ylabel="True positive rate",
               title=f"{title} | Mean one-vs-rest ROC")
        ax.grid(alpha=.15)
        fig.text(.5, -.01,
                 "AUC = mean exact per-spot AUC. Shared assignment scores can produce identical curves.",
                 ha="center", fontsize=17)
        fig.tight_layout()
        paths.append(_save(fig, output / "membership_roc"))

        fig, axes = plt.subplots(1, len(rules), figsize=(9 * len(rules), 8), squeeze=False)
        maximum = 1
        for rule in rules:
            if rule not in counts:
                raise ValueError(f"Missing count table for decision rule {rule}")
            table = counts[rule]
            maximum = max(maximum, int(table[["true_cells", "predicted_cells"]].to_numpy().max()))
        for i, (rule, color, ax) in enumerate(zip(rules, colors, axes[0])):
            table = counts[rule]
            groups = table.groupby(["true_cells", "predicted_cells"], dropna=False).size().reset_index(name="spots")
            ax.scatter(groups.true_cells, groups.predicted_cells, color=color,
                       s=55+45*np.sqrt(groups.spots), alpha=.75, edgecolor="white", linewidth=.7)
            ax.plot([0, maximum], [0, maximum], color="#777777", linestyle="--", lw=1.5)
            ax.set(xlim=(-.5, maximum+.5), ylim=(-.5, maximum+.5),
                   xlabel="True cells per spot", ylabel="Predicted unique cells per spot",
                   title=_rule_name(rule))
            ax.xaxis.set_major_locator(MaxNLocator(5, integer=True))
            ax.yaxis.set_major_locator(MaxNLocator(5, integer=True))
            ax.set_aspect("equal", adjustable="box")
            ax.text(.04, .95, f"Count MAE = {rows.loc[i, 'count_mae']:.3f}",
                    transform=ax.transAxes, va="top", fontsize=23,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=.9))
        fig.suptitle(f"{title} | Cell counts within spots", y=1.02)
        fig.text(.5, -.015, "One count per unique original cell ID. Point size indicates the number of spots sharing a count pair.",
                 ha="center", fontsize=18)
        fig.tight_layout(w_pad=3)
        paths.append(_save(fig, output / "spot_cell_counts"))
    return paths
