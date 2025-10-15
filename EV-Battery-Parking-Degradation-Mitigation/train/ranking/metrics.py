"""ランキング評価指標をまとめて扱うユーティリティ集。"""

from __future__ import annotations

from typing import Dict, Iterable, Iterator, Tuple

import numpy as np
import pandas as pd


def _groupwise_sorted(
    df: pd.DataFrame, score_col: str, group_col: str
) -> Iterator[Tuple[str, pd.DataFrame]]:
    """イベント（group）ごとにスコア降順へ並べ替えたデータを返す内部ジェネレーター。"""

    for gid, group in df.groupby(group_col):
        yield gid, group.sort_values(score_col, ascending=False)


def top_k_accuracy_at_k(
    df: pd.DataFrame,
    score_col: str,
    group_col: str,
    label_col: str,
    k: int = 1,
) -> float:
    """イベントごとに上位k件へ正解が含まれる割合（HitRate@k）を計算する。"""

    hits = []
    for _, group in _groupwise_sorted(df, score_col, group_col):
        topk = group.head(k)
        hits.append(int(topk[label_col].max() > 0))
    return float(np.mean(hits)) if hits else float("nan")


def mean_reciprocal_rank(
    df: pd.DataFrame, score_col: str, group_col: str, label_col: str
) -> float:
    """イベントごとの最初の正解順位に基づいて平均逆順位（MRR）を求める。"""

    reciprocal_ranks = []
    for _, group in _groupwise_sorted(df, score_col, group_col):
        labels = group[label_col].to_numpy(dtype=int)
        hit_indices = np.where(labels == 1)[0]
        if len(hit_indices):
            reciprocal_ranks.append(1.0 / (hit_indices[0] + 1))
        else:
            reciprocal_ranks.append(0.0)
    return float(np.mean(reciprocal_ranks)) if reciprocal_ranks else float("nan")


def ndcg_at_k(
    df: pd.DataFrame,
    score_col: str,
    group_col: str,
    label_col: str,
    k: int = 3,
) -> float:
    """イベントごとの正規化DCG (NDCG@k) を計算し平均する。"""

    def dcg(y_true: np.ndarray) -> float:
        gains = (2 ** y_true - 1) / np.log2(np.arange(1, len(y_true) + 1) + 1)
        return float(gains.sum())

    ndcgs = []
    for _, group in _groupwise_sorted(df, score_col, group_col):
        y_true = group[label_col].to_numpy(dtype=int)
        top_true = y_true[:k]
        ideal = np.sort(y_true)[::-1][:k]
        denom = dcg(ideal)
        ndcgs.append(dcg(top_true) / denom if denom > 0 else 0.0)
    return float(np.mean(ndcgs)) if ndcgs else float("nan")


def calculate_ranking_metrics(
    df_scored: pd.DataFrame,
    score_col: str,
    group_col: str,
    label_col: str,
    topk: Iterable[int],
) -> Dict[str, float]:
    """Hit@k, NDCG@k, MRR をまとめて計算し辞書で返却する。"""

    metrics: Dict[str, float] = {}
    for k in topk:
        metrics[f"Hit@{k}"] = top_k_accuracy_at_k(
            df_scored, score_col, group_col, label_col, k
        )
        metrics[f"NDCG@{k}"] = ndcg_at_k(
            df_scored, score_col, group_col, label_col, k
        )
    metrics["MRR"] = mean_reciprocal_rank(df_scored, score_col, group_col, label_col)
    return metrics
