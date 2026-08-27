# =========================================================================
# Copyright (C) 2024. The FuxiCTR Library. All rights reserved.
# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================


from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
import numpy as np
import pandas as pd
import multiprocessing as mp
from collections import OrderedDict


def evaluate_metrics(
    y_true, y_pred, metrics, group_id=None, y_logits=None, topk_list=None
):
    """Evaluate a list of metrics on predictions.

    Supports ``logloss``, ``AUC``, ``gAUC``, ``avgAUC``, ``MRR``, and ``NDCG@k``.
    Group-level metrics (``gAUC``, ``avgAUC``, ``MRR``, ``NDCG``) require
    ``group_id`` to be provided.

    Args:
        y_true (array-like): Ground-truth binary labels.
        y_pred (array-like): Predicted probabilities or scores.
        metrics (list): List of metric names to compute.
        group_id (array-like, optional): Group identifiers for group-level metrics.

    Returns:
        OrderedDict: Mapping from metric name to computed value.

    Raises:
        ValueError: If an unsupported metric is requested.
    """
    return_dict = OrderedDict()
    is_multiclass = isinstance(y_pred, np.ndarray) and y_pred.ndim > 1
    multiclass_labels = None
    observed_labels = None
    if is_multiclass:
        multiclass_labels = np.arange(y_pred.shape[1])
        observed_labels = np.unique(y_true.astype(np.int64))
    group_metrics = []
    for metric in metrics:
        if metric in ["logloss", "binary_crossentropy"]:
            if is_multiclass:
                return_dict[metric] = log_loss(y_true, y_pred, labels=multiclass_labels)
            else:
                return_dict[metric] = log_loss(y_true, y_pred)
        elif metric == "AUC":
            if is_multiclass:
                if len(observed_labels) < 2:
                    raise ValueError(
                        "AUC requires at least 2 classes in y_true, but got {}.".format(
                            observed_labels.tolist()
                        )
                    )
                auc_scores = y_pred[:, observed_labels]
                auc_scores = auc_scores / np.clip(
                    auc_scores.sum(axis=1, keepdims=True), 1e-12, None
                )
                if len(observed_labels) == 2:
                    positive_label = observed_labels[-1]
                    return_dict[metric] = roc_auc_score(
                        (y_true == positive_label).astype(np.int64), auc_scores[:, -1]
                    )
                else:
                    return_dict[metric] = roc_auc_score(
                        y_true,
                        auc_scores,
                        multi_class="ovr",
                        labels=observed_labels,
                    )
            else:
                return_dict[metric] = roc_auc_score(y_true, y_pred)
        elif metric == "accuracy":
            if is_multiclass:
                return_dict[metric] = accuracy_score(y_true, np.argmax(y_pred, axis=-1))
            else:
                return_dict[metric] = accuracy_score(
                    y_true, (y_pred >= 0.5).astype(int)
                )
        elif metric in ["gAUC", "avgAUC", "MRR"] or metric.startswith("NDCG"):
            if is_multiclass:
                raise ValueError(
                    "metric={} is not supported for multiclass predictions.".format(
                        metric
                    )
                )
            return_dict[metric] = 0
            group_metrics.append(metric)
        else:
            raise ValueError("metric={} not supported.".format(metric))
    if is_multiclass and y_logits is not None and topk_list:
        return_dict.update(compute_topk_label_distribution(y_true, y_logits, topk_list))
    if len(group_metrics) > 0:
        assert group_id is not None, "group_index is required."
        metric_funcs = []
        for metric in group_metrics:
            try:
                metric_funcs.append(eval(metric))
            except:
                raise NotImplementedError("metrics={} not implemented.".format(metric))
        score_df = pd.DataFrame(
            {"group_index": group_id, "y_true": y_true, "y_pred": y_pred}
        )
        results = []
        pool = mp.Pool(processes=mp.cpu_count() // 2)
        for idx, df in score_df.groupby("group_index"):
            results.append(pool.apply_async(evaluate_block, args=(df, metric_funcs)))
        pool.close()
        pool.join()
        results = [res.get() for res in results]
        sum_results = np.array(results).sum(0)
        average_result = list(sum_results[:, 0] / sum_results[:, 1])
        return_dict.update(dict(zip(group_metrics, average_result)))
    return return_dict


def compute_topk_label_distribution(y_true, y_logits, topk_list):
    """Compute true-label ratios within top-k samples ranked by each class logit.

    Args:
        y_true (np.ndarray): Integer class labels of shape ``(n_samples,)``.
        y_logits (np.ndarray): Logit matrix of shape ``(n_samples, num_classes)``.
        topk_list (list): List of positive integer top-k cutoffs.

    Returns:
        OrderedDict: Flattened metrics keyed as
            ``top{K}_by_logit_class{c}_label{label}_ratio``.
    """
    metrics = OrderedDict()
    if y_logits.ndim != 2:
        raise ValueError("y_logits must be a 2D array for multiclass top-k analysis.")

    num_samples, num_classes = y_logits.shape
    class_labels = np.unique(y_true.astype(np.int64))
    valid_topk = []
    for topk in topk_list:
        topk = int(topk)
        if topk <= 0:
            raise ValueError(
                "topk values must be positive integers, but got {}.".format(topk)
            )
        valid_topk.append(min(topk, num_samples))

    for class_idx in range(num_classes):
        ranked_index = np.argsort(y_logits[:, class_idx])[::-1]
        for topk in valid_topk:
            top_labels = y_true[ranked_index[:topk]]
            for label in class_labels:
                ratio = float(np.mean(top_labels == label))
                metrics[
                    "top{}_by_logit_class{}_label{}_ratio".format(
                        topk, class_idx, int(label)
                    )
                ] = ratio
    return metrics


def build_topk_distribution_tables(y_true, y_logits, topk_list):
    """Build per-topk tables for multiclass logit ranking analysis.

    Each returned DataFrame represents one top-k cutoff. Rows are the ranking
    class (which logit column is used for sorting), and columns contain label
    ratios and counts within that top-k slice.
    """
    y_true = normalize_distribution_values(y_true)
    if y_logits.ndim != 2:
        raise ValueError("y_logits must be a 2D array for multiclass top-k analysis.")

    num_samples, num_classes = y_logits.shape
    class_values = list(pd.unique(y_true))
    valid_topk = []
    for topk in topk_list:
        topk = int(topk)
        if topk <= 0:
            raise ValueError(
                "topk values must be positive integers, but got {}.".format(topk)
            )
        valid_topk.append(min(topk, num_samples))

    table_dict = OrderedDict()
    for topk in valid_topk:
        rows = []
        for class_idx in range(num_classes):
            ranked_index = np.argsort(y_logits[:, class_idx])[::-1]
            top_labels = y_true[ranked_index[:topk]]
            row = {
                "rank_by_logit_class": class_idx,
                "topk": topk,
                "sample_count": int(len(top_labels)),
            }
            for label_value in class_values:
                label_name = format_distribution_value(label_value)
                count = int(np.sum(top_labels == label_value))
                row["label_{}_count".format(label_name)] = count
                row["label_{}_ratio".format(label_name)] = float(
                    count / max(len(top_labels), 1)
                )
            rows.append(row)
        table_dict[topk] = pd.DataFrame(rows)
    return table_dict


def build_binary_topk_distribution_tables(y_true, y_score, topk_list):
    """Build per-topk tables for binary score ranking analysis.

    Rows are generated for a single ranking score column, which keeps the CSV
    shape aligned with multiclass outputs.
    """
    y_true = normalize_distribution_values(y_true)
    y_score = y_score.reshape(-1)
    num_samples = len(y_true)
    class_values = list(pd.unique(y_true))
    valid_topk = []
    for topk in topk_list:
        topk = int(topk)
        if topk <= 0:
            raise ValueError(
                "topk values must be positive integers, but got {}.".format(topk)
            )
        valid_topk.append(min(topk, num_samples))

    ranked_index = np.argsort(y_score)[::-1]
    table_dict = OrderedDict()
    for topk in valid_topk:
        top_labels = y_true[ranked_index[:topk]]
        row = {
            "rank_by_logit_class": 1,
            "topk": topk,
            "sample_count": int(len(top_labels)),
        }
        for label_value in class_values:
            label_name = format_distribution_value(label_value)
            count = int(np.sum(top_labels == label_value))
            row["label_{}_count".format(label_name)] = count
            row["label_{}_ratio".format(label_name)] = float(
                count / max(len(top_labels), 1)
            )
        table_dict[topk] = pd.DataFrame([row])
    return table_dict


def normalize_distribution_values(values):
    values = np.asarray(values).reshape(-1)
    if np.issubdtype(values.dtype, np.floating):
        rounded_values = np.rint(values)
        if np.allclose(values, rounded_values, equal_nan=True):
            values = rounded_values.astype(np.int64)
    return values


def format_distribution_value(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value)


def evaluate_block(df, metric_funcs):
    """Evaluate a list of metric functions on a single group DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with ``y_true`` and ``y_pred`` columns.
        metric_funcs (list): List of callable metric functions.

    Returns:
        list: List of ``(value, weight)`` tuples.
    """
    res_list = []
    for fn in metric_funcs:
        v = fn(df.y_true.values, df.y_pred.values)
        if type(v) == tuple:
            res_list.append(v)
        else:  # add group weight
            res_list.append((v, 1))
    return res_list


def avgAUC(y_true, y_pred):
    """Compute average AUC used in MIND news recommendation.

    Args:
        y_true (array-like): Ground-truth labels.
        y_pred (array-like): Predicted scores.

    Returns:
        tuple: ``(auc_value, weight)`` or ``(0, 0)`` for all-same-label groups.
    """
    if np.sum(y_true) > 0 and np.sum(y_true) < len(y_true):
        auc = roc_auc_score(y_true, y_pred)
        return (auc, 1)
    else:  # in case all negatives or all positives for a group
        return (0, 0)


def gAUC(y_true, y_pred):
    """Compute group AUC defined in the DIN paper.

    Args:
        y_true (array-like): Ground-truth labels.
        y_pred (array-like): Predicted scores.

    Returns:
        tuple: ``(weighted_auc, n_samples)`` or ``(0, 0)`` for all-same-label groups.
    """
    if np.sum(y_true) > 0 and np.sum(y_true) < len(y_true):
        auc = roc_auc_score(y_true, y_pred)
        n_samples = len(y_true)
        return (auc * n_samples, n_samples)
    else:  # in case all negatives or all positives for a group
        return (0, 0)


def MRR(y_true, y_pred):
    """Compute Mean Reciprocal Rank.

    Args:
        y_true (array-like): Ground-truth binary relevance labels.
        y_pred (array-like): Predicted scores for ranking.

    Returns:
        float: MRR score.
    """
    order = np.argsort(y_pred)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    mrr = np.sum(rr_score) / (np.sum(y_true) + 1e-12)
    return mrr


class NDCG(object):
    """Normalized discounted cumulative gain metric.

    Computes DCG at a given cutoff ``k`` and normalizes by the ideal DCG.

    Args:
        k (int): Rank cutoff for DCG computation. Default: ``1``.
    """

    def __init__(self, k=1):
        self.topk = k

    def dcg_score(self, y_true, y_pred):
        """Compute discounted cumulative gain at ``self.topk``.

        Args:
            y_true (array-like): Ground-truth relevance labels.
            y_pred (array-like): Predicted scores for ranking.

        Returns:
            float: DCG score.
        """
        order = np.argsort(y_pred)[::-1]
        y_true = np.take(y_true, order[: self.topk])
        gains = 2**y_true - 1
        discounts = np.log2(np.arange(len(y_true)) + 2)
        return np.sum(gains / discounts)

    def __call__(self, y_true, y_pred):
        """Compute NDCG at ``self.topk``.

        Args:
            y_true (array-like): Ground-truth relevance labels.
            y_pred (array-like): Predicted scores for ranking.

        Returns:
            float: NDCG score in ``[0, 1]``.
        """
        idcg = self.dcg_score(y_true, y_true)
        dcg = self.dcg_score(y_true, y_pred)
        return dcg / (idcg + 1e-12)
