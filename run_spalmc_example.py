"""Example command-line runner for SpaLMC.

This script loads paired scRNA-seq and spatial AnnData files, runs SpaLMC with
the paper-example hyperparameters, and exports the main mapping results.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from spalmc import SpaLMC, pp_adatas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SpaLMC on paired AnnData inputs.")
    parser.add_argument("--adata-sc", required=True, type=Path, help="Path to the scRNA-seq AnnData .h5ad file.")
    parser.add_argument("--adata-sp", required=True, type=Path, help="Path to the spatial AnnData .h5ad file.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for SpaLMC outputs.")
    parser.add_argument("--celltype-key", default=None, help="Column in adata_sc.obs with cell-type labels.")
    parser.add_argument("--layer-key", default=None, help="Optional AnnData layer to use instead of .X.")
    parser.add_argument(
        "--spot-prior-csv",
        default=None,
        type=Path,
        help="Optional spot x cell-type prior CSV. Rows should match spatial spot names.",
    )
    parser.add_argument("--genes", default=None, type=Path, help="Optional TXT/CSV marker gene list.")
    parser.add_argument("--max-epochs", default=500, type=int, help="Maximum training epochs.")
    parser.add_argument("--eval-every", default=20, type=int, help="Print training loss every N epochs.")
    parser.add_argument("--patience", default=80, type=int, help="Early-stopping patience.")
    parser.add_argument("--top-k-spots", default=None, type=int, help="Optional sparse candidate spots per cell.")
    parser.add_argument("--device", default=None, help="Torch device, for example 'cuda' or 'cpu'.")
    parser.add_argument("--random-state", default=0, type=int, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    adata_sc = ad.read_h5ad(args.adata_sc)
    adata_sp = ad.read_h5ad(args.adata_sp)
    spot_prior = pd.read_csv(args.spot_prior_csv, index_col=0) if args.spot_prior_csv else None

    adata_sc, adata_sp = pp_adatas(
        adata_sc,
        adata_sp,
        genes=args.genes,
        celltype_key=args.celltype_key,
        density_key="auto",
        copy=False,
    )

    model = SpaLMC(
        adata_sc=adata_sc,
        adata_sp=adata_sp,
        celltype_key=args.celltype_key,
        layer_key=args.layer_key,
        spot_celltype_prior=spot_prior,
        latent_dim=16,
        hidden_dim=128,
        dropout=0.3,
        temperature=0.05,
        top_k_spots=args.top_k_spots,
        lambda_entropy=0.001,
        lambda_comp=0.1,
        lambda_capacity=0.5,
        lambda_mani=0.01,
        lambda_spatial=0.01,
        lambda_fill=0.1,
        lambda_anchor=0.05,
        lambda_offset=0.05,
        device=args.device,
        random_state=args.random_state,
    )
    model.fit(
        max_epochs=args.max_epochs,
        lr=2e-4,
        weight_decay=1e-3,
        verbose=True,
        eval_every=args.eval_every,
        early_stopping=True,
        patience=args.patience,
    )
    model.add_to_adata()

    pred_cells = pd.DataFrame(
        {
            "cell": model.cell_names_,
            "x": adata_sc.obsm["spalmc_spatial"][:, 0],
            "y": adata_sc.obsm["spalmc_spatial"][:, 1],
            "spot": adata_sc.obs["spalmc_spot_id"].astype(str).to_numpy(),
            "assignment_prob": adata_sc.obs["spalmc_assignment_prob"].to_numpy(),
        }
    )
    if args.celltype_key and args.celltype_key in adata_sc.obs:
        pred_cells[args.celltype_key] = adata_sc.obs[args.celltype_key].astype(str).to_numpy()

    pred_cells.to_csv(args.output_dir / "SpaLMC_pred_single_cell_xy.csv", index=False)
    model.get_spot_fill_summary().to_csv(args.output_dir / "SpaLMC_spot_fill_summary.csv", index=False)
    model.loss_history.to_csv(args.output_dir / "SpaLMC_training_history.csv", index=False)

    spot_probs = model.get_spot_celltype_probabilities(as_dataframe=True)
    if spot_probs is not None:
        spot_probs.to_csv(args.output_dir / "SpaLMC_pred_cell_type_proportion.csv")

    adata_sc.write_h5ad(args.output_dir / "adata_sc_with_spalmc.h5ad")
    model.save_model(args.output_dir / "SpaLMC_model.pt")


if __name__ == "__main__":
    main()
