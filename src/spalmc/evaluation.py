"""Only member accuracy, macro F1, ROC/AUC, and per-spot cell counts.

One reference ID has at most one predicted member. Missing IDs count as wrong in
accuracy and as false negatives in F1. No truth-based matching or repair occurs.
ROC curves use the exact same score-covered ID intersection across methods.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_curve, roc_auc_score


def align_prediction(cell_ids, predicted_ids, predicted_members, scores, n_spots):
    ids = np.asarray(cell_ids).astype(str)
    returned = np.asarray(predicted_ids).astype(str)
    member = np.asarray(predicted_members)
    if ids.ndim != 1 or returned.ndim != 1:
        raise ValueError('Cell IDs must be one-dimensional')
    if len(set(ids)) != len(ids) or len(set(returned)) != len(returned):
        raise ValueError('Duplicate IDs must be resolved by a predeclared adapter rule')
    if member.shape != (len(returned),) or not np.issubdtype(member.dtype, np.integer):
        raise ValueError('Members must be an integer per returned ID')
    if np.any((member < -1) | (member >= n_spots)):
        raise ValueError('Member outside observed spot indices')
    index = {value: row for row, value in enumerate(ids)}
    unknown = sorted(set(returned)-set(ids))
    if unknown:
        raise ValueError(f'Unknown original IDs: {unknown[:5]}')
    rows = np.array([index[value] for value in returned], dtype=int)
    aligned_member = np.full(len(ids), -1, dtype=int)
    aligned_member[rows] = member
    aligned_scores = np.full((len(ids), n_spots), np.nan)
    if scores is not None:
        matrix = np.asarray(scores, dtype=float)
        if matrix.shape != (len(returned), n_spots) or not np.isfinite(matrix).all():
            raise ValueError('Scores must be finite returned-cell × observed-spot values')
        aligned_scores[rows] = matrix
    score_covered = np.isfinite(aligned_scores).all(1)
    return aligned_member, aligned_scores, score_covered


def membership_and_counts(cell_ids, true_members, predicted_members, n_spots):
    ids = np.asarray(cell_ids).astype(str)
    truth = np.asarray(true_members)
    prediction = np.asarray(predicted_members)
    if truth.shape != ids.shape or prediction.shape != ids.shape or len(ids) == 0:
        raise ValueError('Truth and prediction must have one row per original ID')
    if not np.issubdtype(truth.dtype, np.integer) or not np.issubdtype(prediction.dtype, np.integer):
        raise ValueError('Spot indices must be integers')
    if np.any((truth < 0) | (truth >= n_spots)) or np.any((prediction < -1) | (prediction >= n_spots)):
        raise ValueError('Spot indices are outside the observed range')
    labels = np.arange(n_spots)
    true_count = np.bincount(truth, minlength=n_spots)
    pred_count = np.bincount(prediction[prediction >= 0], minlength=n_spots)
    correct_count = np.bincount(truth[truth == prediction], minlength=n_spots)
    tp = correct_count
    fp = pred_count - tp
    fn = true_count - tp
    denominator = 2*tp + fp + fn
    f1 = np.divide(2*tp, denominator, out=np.zeros(n_spots, dtype=float), where=denominator>0)
    macro_f1 = float(f1.mean())
    np.testing.assert_allclose(macro_f1, f1_score(truth, prediction, labels=labels, average='macro', zero_division=0))
    table = pd.DataFrame(dict(spot=labels, true_cells=true_count, predicted_cells=pred_count,
        correct_members=tp, false_positive_members=fp, false_negative_members=fn,
        f1=f1, count_error=pred_count-true_count, absolute_count_error=abs(pred_count-true_count)))
    cells = pd.DataFrame(dict(cell_id=ids, true_spot=truth, predicted_spot=prediction,
                              member_correct=truth == prediction, returned=prediction>=0))
    summary = dict(cell_count=len(ids), spot_count=n_spots, returned_cell_count=int(np.sum(prediction>=0)),
        correct_members=int(np.sum(truth == prediction)), member_accuracy=float(np.mean(truth == prediction)),
        macro_f1=macro_f1, count_mae=float(np.mean(abs(pred_count-true_count))),
        true_cell_total=int(true_count.sum()), predicted_cell_total=int(pred_count.sum()))
    return summary, cells, table


def multiclass_roc(true_members, scores, shared_rows, n_spots):
    truth = np.asarray(true_members)[shared_rows]
    matrix = np.asarray(scores, dtype=float)[shared_rows]
    if len(truth) < 2 or not np.isfinite(matrix).all():
        raise ValueError('ROC needs finite scores on at least two shared original IDs')
    grid = np.linspace(0, 1, 1001)
    tpr_curves = []
    area = []
    rows = []
    exact = []
    for spot in range(n_spots):
        binary = truth == spot
        if not binary.any() or binary.all():
            continue
        fpr, tpr, thresholds = roc_curve(binary, matrix[:, spot], drop_intermediate=False)
        auc = float(roc_auc_score(binary, matrix[:, spot]))
        area.append(auc)
        tpr_curves.append(np.interp(grid, fpr, tpr))
        rows.append(dict(spot=spot, positives=int(binary.sum()), negatives=int((~binary).sum()), auc=auc))
        exact.extend(dict(spot=spot, fpr=float(f), tpr=float(t), threshold=float(threshold))
                     for f,t,threshold in zip(fpr,tpr,thresholds))
    if not area:
        raise ValueError('No spot has both positives and negatives on shared ROC IDs')
    mean_tpr = np.mean(tpr_curves, axis=0)
    summary = dict(macro_auc=float(np.mean(area)), roc_cell_count=len(truth), roc_spot_count=len(area),
                   auc_definition='mean exact one-vs-rest AUC over eligible observed spots')
    curve = pd.DataFrame(dict(fpr=grid, mean_tpr=mean_tpr))
    return summary, curve, pd.DataFrame(rows), pd.DataFrame(exact)
