"""Validate the English visualization notebook and its exported predictions."""
from pathlib import Path
import hashlib
import json
import re
import nbformat
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def main():
    manifest = json.loads((ROOT / 'source_manifest.json').read_text(encoding='utf-8'))
    for name, expected in manifest['original_core'].items():
        assert hashlib.sha256((ROOT / 'src/spalmc' / name).read_bytes()).hexdigest() == expected
    books = [nbformat.read(ROOT / 'notebooks' / name, as_version=4)
             for name in ['run_spalmc.ipynb', 'run_spalmc.executed.ipynb']]
    for book in books:
        nbformat.validate(book)
        for cell in book.cells:
            assert not re.search(r'[\u3400-\u9fff]', cell.source), 'Notebook text must be English.'
            if cell.cell_type == 'code':
                compile(cell.source, '<notebook>', 'exec')
                assert 'evaluate_saved' not in cell.source and 'plot_evaluation' not in cell.source
    assert not re.search(r'[\u3400-\u9fff]', (ROOT / 'README.md').read_text(encoding='utf-8'))
    code = [cell for cell in books[1].cells if cell.cell_type == 'code']
    assert len(code) == 7 and all(c.execution_count is not None for c in code)
    assert not any(o.output_type == 'error' for c in code for o in c.outputs)
    images = sum('image/png' in o.get('data', {}) for c in code for o in c.outputs)
    assert images == 4
    rows = []
    for ds, expected in manifest['examples'].items():
        folder = ROOT / 'outputs/notebook_pretrained_visualization' / ds
        with np.load(ROOT / 'examples' / ds / 'observed.npz', allow_pickle=False) as obs:
            ids = obs['cell_ids'].astype(str); centers = obs['centers']
        with np.load(folder / 'prediction.npz', allow_pickle=False) as pred:
            np.testing.assert_array_equal(pred['cell_ids'], ids)
            xy = pred['coordinates']; members = pred['membership']; scores = pred['assignment']
            assert np.isfinite(xy).all() and np.isfinite(scores).all()
            assert (np.linalg.norm(xy - centers[members], axis=1) <= .5 + 1e-6).all()
        frame = pd.read_csv(folder / 'cells.csv')
        np.testing.assert_array_equal(frame.cell_id.to_numpy(str), ids)
        assert not (folder / 'metrics.csv').exists()
        assert len(list((folder / 'figures').glob('*.png'))) == 2
        assert len(list((folder / 'figures').glob('*.svg'))) == 2
        assert (folder / 'spalmc_cells.h5ad').is_file()
        rows.append(dict(dataset=ds, cells=len(ids), spots=len(centers), identity_preserved=True,
                         finite_coordinates=True, bounded_coordinates=True, figures=2))
    result = dict(passed=True, executed_code_cells=7, embedded_images=images,
                  language='English', mode='pretrained', benchmark_metrics_computed=False,
                  original_core_bytes_unchanged=True, datasets=rows)
    (ROOT / 'validation.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
