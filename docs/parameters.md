# Parameters

Use `configs/default.yml` for the supplied examples and `configs/custom_h5ad.yml` for your own inputs. Paths are relative to the configuration file.

## Main controls

| Setting | Default | Purpose |
| --- | --- | --- |
| `mode` | `train` | Train models or load checkpoints with `pretrained`. |
| `decoder` | `occupied_anchor` | Decode within-spot coordinates. |
| `model.device` | `cpu` | Training and inference device. |
| `model.random_state` | `0` | Random seed. |
| `model.latent_dim` | `32` | Shared expression embedding size. |
| `model.hidden_dim` | `256` | Expression encoder hidden size. |
| `model.n_layers` | `2` | Hidden layer count. |
| `model.dropout` | `0.1` | Expression encoder dropout. |
| `model.temperature` | `0.1` | Cell-to-spot assignment temperature. |
| `model.spot_radius` | `0.5` | Spot radius in the input coordinate units. |
| `training.max_epochs` | `200` | Expression integration epochs. |
| `training.lr` | `0.001` | Expression integration learning rate. |
| `occupied.epochs` | `200` | Localization epochs. |
| `occupied.lr` | `0.001` | Localization learning rate. |
| `occupied.residual_fraction` | `0.35` | Residual bound relative to anchor spacing. |
| `occupied.global_weight` | `0.1` | Global relation loss weight. |
| `occupied.kappa` | `0.3` | Neighborhood expression guidance weight. |
| `occupied.residual_weight` | `0.01` | Relative residual penalty weight. |

The original first-stage objective and its weights are retained in this release. All weights are listed in `configs/default.yml`. `lambda_comp` is inactive unless both cell-type annotations and a spot composition prior are supplied.

Checkpoint loading requires matching model settings, input IDs, genes and preprocessing. Retrain after changing these settings. The notebook defaults to `pretrained` for quick visualization; set `MODE = 'train'` to train from scratch.
