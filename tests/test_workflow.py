"""Small end-to-end release checks, using only synthetic observed counts.

Run after installing the release: python -m unittest discover -s tests -v.
No benchmark data, historical experiment directory or evaluation truth is read.
"""
from pathlib import Path
import tempfile
import unittest

import anndata as ad
import numpy as np
import pandas as pd
import torch

from spalmc import workflow as workflow


class WorkflowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory(prefix='spalmc_release_test_',
            dir=Path(__file__).resolve().parent)
        cls.folder = Path(cls.temp.name)
        rng = np.random.default_rng(37)
        groups = np.repeat(np.arange(4), 5)
        means = np.array([
            [12, 9, 2, 1, 3, 2, 4, 1],
            [2, 1, 12, 8, 3, 1, 2, 3],
            [1, 4, 2, 2, 11, 9, 2, 3],
            [3, 2, 1, 4, 2, 3, 13, 9],
        ], dtype=float)
        counts = rng.poisson(means[groups]).astype(np.float32)
        spot_counts = np.stack([counts[groups == j].sum(0) for j in range(4)])
        centers = np.array([[0., 0.], [.7, 0.], [0., .7], [.7, .7]])
        cls.ids = np.array([f'synthetic_cell_{i:02d}' for i in range(len(counts))])
        cls.observed = cls.folder/'observed.npz'
        np.savez_compressed(cls.observed, counts=counts, spot_counts=spot_counts,
            centers=centers, cell_ids=cls.ids,
            genes=np.array([f'gene_{i}' for i in range(counts.shape[1])]))
        cls.model_config = dict(celltype_key=None, lambda_comp=0., hidden_dim=16,
            latent_dim=4, n_layers=1, dropout=0., use_batch_norm=False,
            offset_hidden_dim=12, offset_n_layers=1, k_sc=3, k_spatial=2,
            top_k_spots=None, spot_radius=.5, random_state=13)
        cls.training = dict(max_epochs=2, verbose=False, eval_every=1)
        cls.settings = dict(epochs=2, seed=17)
        sc, sp = workflow.load_inputs(cls.observed)
        cls.model = workflow.fit_stage1(sc, sp, cls.model_config, cls.training)
        cls.baseline = workflow.predict_stage1(cls.model)
        cls.checkpoint = cls.folder/'stage1.pt'
        workflow.save_stage1(cls.model, cls.checkpoint)
        cls.decoder, cls.context, cls.history = workflow.fit_occupied(cls.model, cls.settings)
        cls.prediction, cls.diagnostics = workflow.predict_occupied(
            cls.model, cls.decoder, cls.context, cls.settings)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_stage1_checkpoint_roundtrip_preserves_named_predictions(self):
        sc, sp = workflow.load_inputs(self.observed)
        restored = workflow.fit_stage1(sc, sp, self.model_config, self.training,
            checkpoint=self.checkpoint)
        prediction = workflow.predict_stage1(restored)
        np.testing.assert_array_equal(prediction['cell_ids'], self.ids)
        np.testing.assert_array_equal(prediction['membership'], self.baseline['membership'])
        np.testing.assert_allclose(prediction['assignment'], self.baseline['assignment'], atol=1e-7, rtol=0)
        np.testing.assert_allclose(prediction['coordinates'], self.baseline['coordinates'], atol=1e-7, rtol=0)
        pd.testing.assert_frame_equal(restored.loss_history, self.model.loss_history)
        self.assertEqual(len(restored.loss_history), 2)

    def test_checkpoint_rejects_changed_identity_and_model_configuration(self):
        sc, sp = workflow.load_inputs(self.observed)
        renamed = sc.copy()
        renamed.obs_names = np.array(['different_id', *self.ids[1:]])
        with self.assertRaisesRegex(ValueError, 'configuration|IDs|match'):
            workflow.fit_stage1(renamed, sp, self.model_config, self.training,
                checkpoint=self.checkpoint)
        changed = dict(self.model_config, temperature=.23)
        with self.assertRaisesRegex(ValueError, 'configuration|match'):
            workflow.fit_stage1(sc, sp, changed, self.training, checkpoint=self.checkpoint)

    def test_occupied_training_preserves_members_and_geometric_support(self):
        prediction = self.prediction
        self.assertEqual(len(self.history), 2)
        self.assertTrue(np.isfinite(self.history.select_dtypes(include='number')).all().all())
        np.testing.assert_array_equal(prediction['cell_ids'], self.ids)
        np.testing.assert_array_equal(prediction['membership'], self.baseline['membership'])
        np.testing.assert_array_equal(prediction['assignment'], self.baseline['assignment'])
        centers = self.model.spot_coords_np_
        coordinates = prediction['coordinates']
        member = prediction['membership']
        distance = np.linalg.norm(coordinates[:, None] - centers[None], axis=2)
        self.assertLessEqual(np.max(distance[np.arange(len(member)), member]), .5 + 2e-6)
        np.testing.assert_array_equal(distance.argmin(1), member)
        self.assertTrue((self.diagnostics.occupied_slots == self.diagnostics.cells).all())
        for spot in np.unique(member):
            rows = np.flatnonzero(member == spot)
            self.assertEqual(len(np.unique(prediction['anchor_index'][rows])), len(rows))
            if len(rows) < 2:
                continue
            actual = np.linalg.norm(coordinates[rows, None] - coordinates[None, rows], axis=2)
            anchors = prediction['anchor_offsets'][rows]
            anchor_distance = np.linalg.norm(anchors[:, None] - anchors[None], axis=2)
            off_diagonal = ~np.eye(len(rows), dtype=bool)
            self.assertTrue(np.all(actual[off_diagonal] >= .3*anchor_distance[off_diagonal] - 3e-6))
        path = self.folder/'occupied.pt'
        workflow.save_occupied(self.model, self.decoder, self.settings, self.history, path)
        decoder, context, history = workflow.fit_occupied(self.model, self.settings, checkpoint=path)
        restored, _ = workflow.predict_occupied(self.model, decoder, context, self.settings)
        np.testing.assert_allclose(restored['coordinates'], coordinates, atol=1e-7, rtol=0)
        pd.testing.assert_frame_equal(history, self.history)
        # Identical neural weights do not authorize changing the membership rule:
        # the occupied model was trained conditional on the original members.
        sc, sp = workflow.load_inputs(self.observed)
        changed = workflow.make_model(sc, sp, dict(self.model_config, hard_assignment_mode='argmax'))
        changed.model.load_state_dict(self.model.model.state_dict())
        with self.assertRaisesRegex(ValueError, 'configuration|members'):
            workflow.fit_occupied(changed, self.settings, checkpoint=path)

    def test_export_retains_cell_identity_and_member_probability(self):
        folder = self.folder/'exported'
        frame = workflow.export_results(self.model, self.prediction, folder,
            dict(dataset='synthetic', mode='train', decoder='occupied_anchor'))
        output = ad.read_h5ad(folder/'spalmc_cells.h5ad')
        np.testing.assert_array_equal(output.obs_names.to_numpy(), self.ids)
        np.testing.assert_array_equal(frame.cell_id.to_numpy(), self.ids)
        np.testing.assert_allclose(output.obsm['spalmc_spatial'], self.prediction['coordinates'])
        member = self.prediction['membership']
        np.testing.assert_allclose(frame.assigned_member_probability.to_numpy(),
            self.prediction['assignment'][np.arange(len(member)), member])
        count = pd.read_csv(folder/'spot_counts.csv')
        self.assertEqual(int(count.assigned_cells.sum()), len(self.ids))
        np.testing.assert_array_equal(count.assigned_cells.to_numpy(), np.bincount(member, minlength=4))
        saved = workflow.arrays(folder/'prediction.npz')
        np.testing.assert_array_equal(saved['cell_ids'], self.ids)


if __name__ == '__main__':
    unittest.main()
