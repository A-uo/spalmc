"""Portable SpaLMC workflow: observations -> model -> named outputs -> evaluation.

Training functions never open evaluation truth. Paths are relative to the YAML
file, and the release has no imports from the surrounding research workspace.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import time

import anndata as ad
import numpy as np
import pandas as pd
import torch
import yaml

from .model import SpaLMC
from .data import get_expression_matrix
from .occupied_decoder import OccupiedAnchorDecoder, build_groups, relation_objective
from .evaluation import membership_and_counts, multiclass_roc


def arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def effective_model_config(settings=None):
    signature = inspect.signature(SpaLMC.__init__)
    result = {key: value.default for key, value in signature.parameters.items()
              if value.default is not inspect.Parameter.empty and key != 'spot_celltype_prior'}
    result.update(device='cpu', spot_radius=.5, spotwise_coordinate_allocation=False,
                  density_key=None)
    unknown = set(settings or {}) - set(result)
    if unknown:
        raise ValueError(f'Unknown model options: {sorted(unknown)}')
    result.update(settings or {})
    if result['device'] is None:
        result['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    result['device'] = str(result['device'])
    if result['learnable_radius']:
        raise ValueError('This fixed-radius workflow requires learnable_radius=False.')
    if result['spot_radius'] is None or result['spot_radius'] <= 0:
        raise ValueError('Provide a positive spot_radius in the same units as spot coordinates.')
    return result


def effective_training_config(settings=None):
    result = dict(max_epochs=200, lr=.001, weight_decay=.0001,
                  early_stopping=False, verbose=True, eval_every=25)
    unknown = set(settings or {}) - set(result)
    if unknown:
        raise ValueError(f'Unknown training options: {sorted(unknown)}')
    result.update(settings or {})
    if result['max_epochs'] < 1:
        raise ValueError('max_epochs must be positive.')
    return result


def effective_occupied_config(settings=None):
    result = dict(epochs=200, lr=.001, residual_fraction=.35, kappa=.3,
                  temperature_start=.5, temperature_end=.2,
                  global_weight=.1, residual_weight=.01, seed=0)
    unknown = set(settings or {}) - set(result)
    if unknown:
        raise ValueError(f'Unknown occupied-anchor options: {sorted(unknown)}')
    result.update(settings or {})
    if result['epochs'] < 1 or not 0 <= result['residual_fraction'] < .5:
        raise ValueError('epochs must be positive; residual_fraction must be in [0, 0.5).')
    if min(result['temperature_start'], result['temperature_end'], result['lr']) <= 0:
        raise ValueError('Temperatures and learning rate must be positive.')
    return result


def load_inputs(path=None, sc_path=None, sp_path=None):
    """Read observations as NPZ or a pair of h5ad files; retain original IDs."""
    if path is not None:
        if sc_path is not None or sp_path is not None:
            raise ValueError('Use either observed NPZ or paired h5ad inputs.')
        data = arrays(path)
        required = {'counts', 'spot_counts', 'centers', 'cell_ids', 'genes'}
        if not required.issubset(data):
            raise ValueError(f'Observed NPZ is missing {sorted(required-set(data))}')
        x, y = data['counts'], data['spot_counts']
        centers = np.asarray(data['centers'], dtype=np.float64)
        ids = data['cell_ids'].astype(str)
        raw_genes = data['genes']
        genes = np.asarray(['gene_'+str(value) for value in raw_genes]) if np.issubdtype(raw_genes.dtype, np.number) else raw_genes.astype(str)
        spots = data.get('spot_ids', np.asarray([f'spot_{i}' for i in range(len(y))])).astype(str)
        if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
            raise ValueError('Counts must be cell x gene and spot x the same genes.')
        if len(genes) != x.shape[1] or len(ids) != len(x) or len(spots) != len(y):
            raise ValueError('IDs and gene names must match matrix dimensions.')
        if not np.isfinite(x).all() or not np.isfinite(y).all() or np.min(x) < 0 or np.min(y) < 0:
            raise ValueError('Observed NPZ counts must be finite and nonnegative.')
        sc = ad.AnnData(x.copy(), obs=pd.DataFrame(index=ids), var=pd.DataFrame(index=genes))
        sp = ad.AnnData(y.copy(), obs=pd.DataFrame(index=spots), var=pd.DataFrame(index=genes))
        sp.obsm['spatial'] = centers
        sc.uns['training_genes'] = genes.tolist()
    else:
        if sc_path is None or sp_path is None:
            raise ValueError('Supply observed NPZ, or both sc_path and sp_path.')
        sc, sp = ad.read_h5ad(sc_path), ad.read_h5ad(sp_path)
        for obj in (sc, sp):
            obj.obs_names = obj.obs_names.astype(str)
            obj.var_names = obj.var_names.astype(str)
    for obj, name in [(sc, 'reference'), (sp, 'spots')]:
        if not obj.obs_names.is_unique or not obj.var_names.is_unique:
            raise ValueError(f'{name} IDs and genes must be unique. Resolve duplicates explicitly.')
        if obj.n_obs < 2 or obj.n_vars < 1:
            raise ValueError(f'{name} needs at least two observations and one gene.')
    if 'spatial' not in sp.obsm:
        raise ValueError("Set spot coordinates in adata_sp.obsm['spatial'] (n_spots, 2).")
    centers = np.asarray(sp.obsm['spatial'])
    if centers.shape != (sp.n_obs, 2) or not np.isfinite(centers).all():
        raise ValueError('Spot centers must be a finite n_spots x 2 matrix.')
    if len(np.unique(centers, axis=0)) != len(centers):
        raise ValueError('Distinct spots require distinct observed centers.')
    return sc, sp


def _digest_arrays(items):
    digest = hashlib.sha256()
    for name, value in items:
        value = np.asarray(value)
        digest.update(name.encode()); digest.update(str(value.shape).encode())
        if value.dtype.kind in 'UOS':
            digest.update(json.dumps(value.astype(str).tolist(), ensure_ascii=False).encode())
        else:
            digest.update(str(value.dtype).encode()); digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def input_fingerprint(model):
    values = [('cell_ids', model.cell_names_), ('spot_ids', model.spot_names_),
              ('genes', model.gene_names_), ('sc', model.x_sc_np_),
              ('spots', model.x_sp_np_), ('centers', model.spot_coords_np_),
              ('target_mass', model.target_spot_mass_np_)]
    for name in ['cell_type_onehot', 'spot_prior']:
        value = model.tensors_.get(name)
        if value is not None:
            values.append((name, value.detach().cpu().numpy()))
    return _digest_arrays(values)


def weights_fingerprint(model):
    return _digest_arrays([(name, value.detach().cpu().numpy())
                           for name, value in sorted(model.model.state_dict().items())])


def make_model(sc, sp, model_config=None):
    config = effective_model_config(model_config)
    for obj in (sc, sp):
        matrix = get_expression_matrix(obj, config['layer_key'])
        if not np.isfinite(matrix).all():
            raise ValueError('The expression layer must contain finite values.')
        if config['normalize'] and np.any(matrix < 0):
            raise ValueError('Normalization expects nonnegative counts; select a raw-count layer or disable normalization for preprocessed expression.')
    model = SpaLMC(sc, sp, **config).prepare_data()
    model._release_config = config
    model._release_input_fingerprint = input_fingerprint(model)
    return model


def fit_stage1(sc, sp, model_config=None, training_config=None, checkpoint=None):
    """Fresh training, or exact-input/config restoration of a release checkpoint."""
    model = make_model(sc, sp, model_config)
    training = effective_training_config(training_config)
    if checkpoint is None:
        model.fit(**training)
    else:
        payload = torch.load(Path(checkpoint), map_location=model.device, weights_only=False)
        if payload.get('format') != 'spalmc-release-stage1-v1':
            raise ValueError('Use a release-format stage1 checkpoint with complete metadata.')
        expected = dict(payload['model_config']); requested = dict(model._release_config)
        expected.pop('device'); requested.pop('device')
        # This switch affects the final coordinate readout, not stage1 fitting.
        expected.pop('spotwise_coordinate_allocation'); requested.pop('spotwise_coordinate_allocation')
        if expected != requested or payload['input_fingerprint'] != model._release_input_fingerprint:
            raise ValueError('Checkpoint model configuration, data, genes or original IDs do not match.')
        if payload['training_config'] != training:
            raise ValueError('Checkpoint training settings do not match the requested settings.')
        model.model.load_state_dict(payload['state_dict'])
        model.loss_history = pd.DataFrame(payload['history'])
        model.results_ = {}; model._outputs_stale = True
    model._release_training = training
    return model


def save_stage1(model, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(format='spalmc-release-stage1-v1', state_dict=model.model.state_dict(),
        model_config=model._release_config, training_config=model._release_training,
        input_fingerprint=model._release_input_fingerprint,
        history=model.loss_history.to_dict('records')), path)


def predict_stage1(model):
    coordinates = model.transform()
    assignment = model.get_assignment()
    return dict(cell_ids=np.asarray(model.cell_names_, dtype=str),
                spot_ids=np.asarray(model.spot_names_, dtype=str),
                membership=model.results_['hard_spot'].astype(np.int64).copy(),
                assignment=assignment.copy(), coordinates=coordinates.copy())


def fit_occupied(model, settings=None, checkpoint=None):
    """Freeze stage1 embeddings/members and fit the optional occupied decoder."""
    settings = effective_occupied_config(settings)
    if model.device.type != 'cpu':
        raise ValueError('The occupied-anchor release path is CPU-only. Set model.device: cpu.')
    if model.coordinate_mode != 'hard':
        raise ValueError('Occupied anchors require coordinate_mode=hard.')
    baseline = predict_stage1(model)
    raw = model._forward_eval()
    member = torch.as_tensor(baseline['membership'], dtype=torch.long)
    inputs = torch.cat([raw['z_sc'], raw['z_sp'][member], model.tensors_['spatial_context'][member]], dim=1).detach()
    radius = float(model.spot_radius_)
    groups, protocol = build_groups(np.asarray(model.adata_sp_aligned.obsm['spatial'], dtype=float), baseline['membership'],
        raw['z_sc'].numpy(), raw['z_sp'].numpy(), model.model.offset_decoder.disk_anchors.detach().numpy(), radius=radius)
    # Match the existing revision's initialization and training seed convention.
    decoder = OccupiedAnchorDecoder(model.model.offset_decoder, settings['residual_fraction'])
    context = dict(inputs=inputs, centers=torch.as_tensor(model.spot_coords_np_),
        groups=groups, latent=raw['z_sc'].detach(), edge_sc=model.tensors_['edge_sc'],
        baseline=baseline, radius=radius, protocol=protocol)
    if checkpoint is not None:
        payload = torch.load(Path(checkpoint), map_location='cpu', weights_only=False)
        if payload.get('format') != 'spalmc-release-occupied-v1':
            raise ValueError('Use a release-format occupied-anchor checkpoint.')
        if payload['stage1_weights'] != weights_fingerprint(model) or payload['input_fingerprint'] != input_fingerprint(model):
            raise ValueError('Occupied checkpoint requires the exact stage1 weights, inputs and members.')
        membership_hash = _digest_arrays([('membership', baseline['membership'])])
        if payload.get('membership_fingerprint') != membership_hash or payload.get('model_config') != model._release_config:
            raise ValueError('Occupied checkpoint requires identical model configuration and decoded members.')
        if payload['settings'] != settings:
            raise ValueError('Occupied checkpoint settings differ from requested settings.')
        decoder.load_state_dict(payload['state_dict'])
        history = pd.DataFrame(payload['history'])
    else:
        torch.manual_seed(settings['seed'])
        optimizer = torch.optim.Adam(decoder.parameters(), lr=settings['lr'])
        rows = []
        for epoch in range(settings['epochs']):
            temperature = settings['temperature_start'] * (settings['temperature_end']/settings['temperature_start']) ** (epoch/max(1, settings['epochs']-1))
            decoder.train()
            coordinates, details = decoder(inputs, context['centers'], groups, radius=radius,
                kappa=settings['kappa'], temperature=temperature)
            loss = relation_objective(coordinates, details, context['latent'], context['edge_sc'], radius=radius,
                kappa=settings['kappa'], global_weight=settings['global_weight'], residual_weight=settings['residual_weight'])
            optimizer.zero_grad(); loss['total'].backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 5.)
            optimizer.step()
            rows.append(dict(epoch=epoch+1, temperature=temperature,
                **{name: float(value.detach()) for name, value in loss.items()}))
            if (epoch+1) % 25 == 0 or epoch == 0:
                print(f'[Occupied anchors] epoch {epoch+1:04d}: total={rows[-1]["total"]:.5f}', flush=True)
        history = pd.DataFrame(rows)
    return decoder, context, history


def save_occupied(model, decoder, settings, history, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(format='spalmc-release-occupied-v1', state_dict=decoder.state_dict(),
        settings=effective_occupied_config(settings), input_fingerprint=input_fingerprint(model),
        stage1_weights=weights_fingerprint(model), model_config=model._release_config,
        membership_fingerprint=_digest_arrays([('membership', predict_stage1(model)['membership'])]),
        history=history.to_dict('records')), path)


def predict_occupied(model, decoder, context, settings=None):
    settings = effective_occupied_config(settings)
    decoder.eval()
    with torch.no_grad():
        coordinates, details = decoder(context['inputs'], context['centers'], context['groups'],
            radius=context['radius'], kappa=settings['kappa'], temperature=settings['temperature_end'])
    prediction = {name: value.copy() for name, value in context['baseline'].items()}
    prediction['coordinates'] = coordinates.numpy()
    n = len(coordinates)
    for name in ['anchor_index', 'residual_bound']:
        prediction[name] = np.zeros(n, dtype=int if name == 'anchor_index' else float)
    prediction['anchor_offsets'] = np.zeros((n, 2))
    prediction['residual_offsets'] = np.zeros((n, 2))
    diagnostics = []
    for detail in details:
        group = detail['group']; idx = group['indices']; q = detail['relaxed'].numpy()
        prediction['anchor_index'][idx] = detail['hard'].numpy().argmax(1)
        prediction['anchor_offsets'][idx] = detail['base'].numpy()*context['radius']
        prediction['residual_offsets'][idx] = detail['residual'].numpy()*context['radius']
        prediction['residual_bound'][idx] = detail['bounds'].numpy()*context['radius']
        diagnostics.append(dict(spot=group['spot'], cells=len(idx),
            occupied_slots=len(np.unique(prediction['anchor_index'][idx])),
            relaxed_row_mass_error=float(np.max(abs(q.sum(1)-1))),
            relaxed_column_mass_error=float(np.max(abs(q.sum(0)-1)))))
    return prediction, pd.DataFrame(diagnostics)


def observed_from_model(model):
    return dict(centers=model.spot_coords_np_, cell_ids=np.asarray(model.cell_names_, dtype=str),
                genes=np.asarray(model.gene_names_, dtype=str), spot_radius=float(model.spot_radius_),
                cell_radius=.1)


def export_results(model, prediction, outdir, metadata=None):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    ids = np.asarray(model.cell_names_, dtype=str)
    np.testing.assert_array_equal(prediction['cell_ids'], ids)
    member = prediction['membership']; scores = prediction['assignment']
    np.savez_compressed(outdir/'prediction.npz', **prediction)
    maximum = scores.argmax(1)
    frame = pd.DataFrame(dict(cell_id=ids, predicted_spot_index=member,
        predicted_spot_id=prediction['spot_ids'][member],
        x=prediction['coordinates'][:, 0], y=prediction['coordinates'][:, 1],
        soft_argmax_spot_index=maximum, soft_max_probability=scores.max(1),
        assigned_member_probability=scores[np.arange(len(ids)), member]))
    if model.celltype_key is not None and model.celltype_key in model.adata_sc_aligned.obs:
        frame['reference_cell_type'] = model.adata_sc_aligned.obs[model.celltype_key].astype(str).to_numpy()
    frame.to_csv(outdir/'cells.csv', index=False)
    pd.DataFrame(dict(spot_id=prediction['spot_ids'],
        assigned_cells=np.bincount(member, minlength=len(prediction['spot_ids'])),
        soft_assignment_mass=scores.sum(0), target_cell_mass=model.target_spot_mass_np_)).to_csv(outdir/'spot_counts.csv', index=False)
    model.loss_history.to_csv(outdir/'stage1_loss_history.csv', index=False)
    output = model.adata_sc_aligned.copy()
    output.obsm['spalmc_spatial'] = prediction['coordinates'].copy()
    output.obsm['spalmc_assignment'] = scores.copy()
    for column in frame.columns[1:]:
        output.obs['spalmc_'+column] = frame[column].to_numpy()
    output.uns['spalmc_spot_ids'] = prediction['spot_ids']
    output.uns['spalmc_coordinate_note'] = 'Model coordinates; precise identity-level positions are not established by spot counts alone.'
    output.write_h5ad(outdir/'spalmc_cells.h5ad', compression='gzip')
    write_json(outdir/'run_metadata.json', dict(metadata or {},
        model_config=model._release_config, training_config=model._release_training,
        input_fingerprint=input_fingerprint(model), stage1_weights=weights_fingerprint(model),
        cell_count=len(ids), spot_count=len(prediction['spot_ids']), gene_count=len(model.gene_names_),
        cell_id_order='original reference IDs, no identity permutation',
        soft_probabilities='assignment outputs, not calibrated coordinate confidence',
        truth_used_in_training=False))
    return frame


def evaluate_saved(observed, truthpath, outdir):
    """Score saved predictions only; independently report both membership rules."""
    outdir = Path(outdir)
    prediction = arrays(outdir/'prediction.npz')
    truth = arrays(truthpath)
    ids = np.asarray(observed['cell_ids']).astype(str)
    np.testing.assert_array_equal(prediction['cell_ids'], ids)
    if 'cell_ids' not in truth:
        raise ValueError('Evaluation truth requires cell_ids for explicit identity alignment.')
    truth_ids = truth['cell_ids'].astype(str)
    if len(set(truth_ids)) != len(truth_ids) or set(truth_ids) != set(ids):
        raise ValueError('Truth must contain exactly the original reference IDs, without duplicates.')
    index = {value: i for i, value in enumerate(truth_ids)}
    order = np.asarray([index[value] for value in ids])
    for key in ['truth', 'true_member', 'source_annotation']:
        if key in truth:
            truth[key] = truth[key][order]
    truth['cell_ids'] = ids
    n_spots = len(prediction['spot_ids'])
    auc, curve, spots_auc, _ = multiclass_roc(truth['true_member'], prediction['assignment'], np.arange(len(ids)), n_spots)
    metrics = []; curves = {}; counts = {}
    for rule, member in [('native_capacity' if model_rule(outdir) == 'capacity' else 'native_argmax', prediction['membership']),
                         ('score_argmax', prediction['assignment'].argmax(1))]:
        summary, cells, spot_counts = membership_and_counts(ids, truth['true_member'], member, n_spots)
        summary.update(auc); summary['decision_rule'] = rule
        metrics.append(summary); curves[rule] = curve.copy(); counts[rule] = spot_counts
        cells.to_csv(outdir/f'evaluation_cells_{rule}.csv', index=False)
        spot_counts.to_csv(outdir/f'evaluation_counts_{rule}.csv', index=False)
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(outdir/'metrics.csv', index=False)
    curve.to_csv(outdir/'roc_curve.csv', index=False)
    spots_auc.to_csv(outdir/'per_spot_auc.csv', index=False)
    return metrics, curves, counts, truth


def model_rule(outdir):
    return json.loads((Path(outdir)/'run_metadata.json').read_text(encoding='utf-8'))['model_config']['hard_assignment_mode']


def read_config(path):
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text(encoding='utf-8'))
    for dataset in config['datasets']:
        for key in ['observed', 'sc_path', 'sp_path', 'truth', 'stage1_checkpoint', 'occupied_checkpoint']:
            if dataset.get(key) is not None:
                dataset[key] = str((path.parent/Path(dataset[key])).resolve())
    config['output_dir'] = str((path.parent/Path(config['output_dir'])).resolve())
    return config


def run_dataset(dataset, config):
    """Run a full dataset without opening its optional truth until export."""
    from . import visualization as viz
    started = time.perf_counter()
    folder = Path(config['output_dir'])/dataset['name']; folder.mkdir(parents=True, exist_ok=True)
    mode = config.get('mode', 'train'); decoder_name = config.get('decoder', 'occupied_anchor')
    if mode not in {'train', 'pretrained'} or decoder_name not in {'original_neural', 'occupied_anchor', 'energy'}:
        raise ValueError('Choose mode=train/pretrained and decoder=original_neural/occupied_anchor/energy.')
    if mode == 'pretrained' and (not dataset.get('stage1_checkpoint') or
            (decoder_name == 'occupied_anchor' and not dataset.get('occupied_checkpoint'))):
        raise ValueError('Pretrained mode requires explicit matching checkpoint paths.')
    torch.set_num_threads(int(config.get('threads', 2)))
    sc, sp = load_inputs(dataset.get('observed'), dataset.get('sc_path'), dataset.get('sp_path'))
    model_config = dict(config.get('model', {}))
    model_config['spotwise_coordinate_allocation'] = decoder_name == 'energy'
    write_json(folder/'status.json', dict(phase='stage1', mode=mode, dataset=dataset['name']))
    model = fit_stage1(sc, sp, model_config, config.get('training'),
        checkpoint=dataset.get('stage1_checkpoint') if mode == 'pretrained' else None)
    save_stage1(model, folder/'checkpoints/stage1.pt')
    prediction = predict_stage1(model)
    if decoder_name == 'occupied_anchor':
        write_json(folder/'status.json', dict(phase='occupied_anchor', mode=mode, dataset=dataset['name']))
        settings = effective_occupied_config(config.get('occupied'))
        decoder, context, history = fit_occupied(model, settings,
            checkpoint=dataset.get('occupied_checkpoint') if mode == 'pretrained' else None)
        save_occupied(model, decoder, settings, history, folder/'checkpoints/occupied.pt')
        prediction, diagnostics = predict_occupied(model, decoder, context, settings)
        history.to_csv(folder/'occupied_loss_history.csv', index=False)
        diagnostics.to_csv(folder/'occupancy_diagnostics.csv', index=False)
        write_json(folder/'occupied_protocol.json', dict(settings=settings, **context['protocol'],
            approximation='hard Hungarian forward; biased Sinkhorn straight-through gradient, 40 iterations',
            output='actual occupied anchor + bounded relative residual; relaxed weights are not calibrated probabilities'))
    export_results(model, prediction, folder, dict(dataset=dataset['name'], mode=mode, decoder=decoder_name))
    observed = observed_from_model(model)
    metrics = pd.DataFrame(); truth = None; figures = []
    if dataset.get('truth') is not None:
        write_json(folder/'status.json', dict(phase='evaluation_after_prediction_export', mode=mode, dataset=dataset['name']))
        metrics, curves, counts, truth = evaluate_saved(observed, dataset['truth'], folder)
        figures.extend(viz.plot_evaluation(metrics, curves, counts, folder/'figures', dataset['name']))
    figures.append(viz.plot_overview(observed, prediction, truth, folder/'figures', dataset['name']))
    member_count = np.bincount(prediction['membership'], minlength=sp.n_obs)
    selected_spot = int(np.argmax(np.where(member_count <= 20, member_count, -1)))
    figures.append(viz.plot_spot(observed, prediction, truth, folder/'figures', dataset['name'], spot=selected_spot))
    figures.append(viz.plot_training(model.loss_history, folder/'figures/stage1_training', dataset['name']+' | Stage 1 training'))
    if decoder_name == 'occupied_anchor':
        figures.append(viz.plot_training(history, folder/'figures/occupied_training', dataset['name']+' | Occupied-anchor training'))
    write_json(folder/'figure_manifest.json', [str(Path(path).relative_to(folder)) for path in figures])
    write_json(folder/'status.json', dict(phase='complete', mode=mode, dataset=dataset['name'],
        elapsed_seconds=time.perf_counter()-started, original_cells=sc.n_obs, spots=sp.n_obs))
    print(dataset['name'], 'complete:', folder, flush=True)
    return dict(folder=folder, prediction=prediction, metrics=metrics,
                figures=figures, model=model)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--mode', choices=['train', 'pretrained'])
    parser.add_argument('--decoder', choices=['original_neural', 'occupied_anchor', 'energy'])
    parser.add_argument('--dataset')
    args = parser.parse_args(); config = read_config(args.config)
    if args.mode: config['mode'] = args.mode
    if args.decoder: config['decoder'] = args.decoder
    if args.dataset:
        config['datasets'] = [d for d in config['datasets'] if d['name'] == args.dataset]
        if not config['datasets']:
            parser.error('Requested dataset is not in the configuration.')
    if config.get('mode', 'train') == 'pretrained':
        for dataset in config['datasets']:
            if not dataset.get('stage1_checkpoint') or (config.get('decoder', 'occupied_anchor') == 'occupied_anchor' and not dataset.get('occupied_checkpoint')):
                parser.error('Pretrained mode requires explicit checkpoint paths in YAML.')
    for dataset in config['datasets']:
        run_dataset(dataset, config)


if __name__ == '__main__':
    main()
