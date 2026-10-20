# SpaLMC

## Install

```bash
conda env create -f environment.yml
conda activate spalmc
python -m pip install -e ".[notebook]" --no-deps
python -m ipykernel install --user --name spalmc --display-name "Python (SpaLMC)"
```

## Run

From this folder, open the notebook:

```bash
jupyter lab notebooks/run_spalmc.ipynb
```

Select the **Python (SpaLMC)** kernel and run all cells in order.

- `MODE = 'pretrained'`: load the included checkpoints and visualize results.
- `MODE = 'train'`: train both stages from scratch.
- `DATASETS`: choose either or both included datasets.
- `SHOW_REFERENCE`: show reference coordinates alongside predictions.
- `SPOTS`: choose a spot for the detailed view.

The notebook contains seven blocks: setup, configuration, data loading, expression integration, cell localization, overall visualization, and within-spot visualization. It does not calculate benchmark metrics.

## Outputs

Results are saved under `outputs/notebook_<mode>_visualization/<dataset>/`:

- `cells.csv`: original cell IDs, assigned spots, and predicted coordinates.
- `prediction.npz`: prediction arrays.
- `spalmc_cells.h5ad`: cell data with predicted coordinates and assignments.
- `figures/`: overall and within-spot views in PNG and SVG formats.
- `checkpoints/`: saved models.
