# SpaLMC Example Run Guide

This guide shows how to install a reproducible Python environment and run the
example SpaLMC script with the following hyperparameters:

| Parameter             | Value   |
| --------------------- | ------- |
| `lambda_entropy`      | `0.001` |
| `lambda_comp`         | `0.1`   |
| `lambda_capacity`     | `0.5`   |
| `lambda_mani`         | `0.01`  |
| `lambda_spatial`      | `0.01`  |
| `lambda_fill`         | `0.1`   |
| `lambda_anchor`       | `0.05`  |
| `lambda_offset`       | `0.05`  |
| `tau` / `temperature` | `0.05`  |
| `d` / `latent_dim`    | `16`    |
| `h` / `hidden_dim`    | `128`   |
| `dropout`             | `0.3`   |
| `lr`                  | `2e-4`  |
| `weight_decay`        | `1e-3`  |

## 1. Create the Environment

From the project root, create and activate the Conda environment:

```bash
conda env create -f environment_spalmc.yml
conda activate spalmc
```

The default environment installs the CPU version of PyTorch. For CUDA training,
install the CUDA-enabled PyTorch build that matches your driver after activating
the environment. For example:

```bash
conda remove cpuonly pytorch
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
```

## 2. Input Requirements

The example script expects two `.h5ad` files:

- `adata_sc`: single-cell reference data. Genes must be in `adata_sc.var_names`.
- `adata_sp`: spatial transcriptomics data. Genes must be in `adata_sp.var_names`.

Spatial coordinates must be available as `adata_sp.obsm["spatial"]`. If this key
is missing, SpaLMC will also look for `adata_sp.obs["x"]` and `adata_sp.obs["y"]`.

If you use cell-type composition supervision, provide:

- `--celltype-key`: a column in `adata_sc.obs`.
- `--spot-prior-csv`: an optional spot-by-cell-type prior table. Its row names
  should match `adata_sp.obs_names`, and its columns should match the cell-type
  categories in `adata_sc.obs[--celltype-key]`.

## 3. Run SpaLMC

Example command:

```bash
python scripts/run_spalmc_example.py \
  --adata-sc data/example_sc.h5ad \
  --adata-sp data/example_sp.h5ad \
  --output-dir results/spalmc_example \
  --celltype-key cell_type \
  --spot-prior-csv data/example_spot_celltype_prior.csv \
  --max-epochs 500 \
  --top-k-spots 64 \
  --device cuda
```

For CPU-only execution, replace `--device cuda` with `--device cpu`, or omit the
argument and let SpaLMC choose the available device automatically.

If you do not have a spot cell-type prior, omit `--spot-prior-csv`. If you do
not have cell-type labels, omit both `--celltype-key` and `--spot-prior-csv`.
The composition loss term will then be skipped internally.

## 4. Outputs

The script writes the following files to `--output-dir`:

- `SpaLMC_pred_single_cell_xy.csv`: reconstructed single-cell spatial coordinates.
- `SpaLMC_pred_cell_type_proportion.csv`: predicted per-spot cell-type proportions, when cell-type labels are available.
- `SpaLMC_spot_fill_summary.csv`: hard and soft cell mass assigned to each spot.
- `SpaLMC_training_history.csv`: training loss history.
- `adata_sc_with_spalmc.h5ad`: input single-cell AnnData with SpaLMC results in `.obsm` and `.obs`.
- `SpaLMC_model.pt`: trained SpaLMC model checkpoint.

SpaLMC stores reconstructed coordinates in `adata_sc.obsm["spalmc_spatial"]`,
assigned spot IDs in `adata_sc.obs["spalmc_spot_id"]`, assignment confidence in
`adata_sc.obs["spalmc_assignment_prob"]`, and relative within-spot offsets in
`adata_sc.obsm["spalmc_relative_offset"]`.# spalmc
