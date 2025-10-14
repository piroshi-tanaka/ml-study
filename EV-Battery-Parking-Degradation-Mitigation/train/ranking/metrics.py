"""
ランキング評価指標（充電イベント＝group 単位で計算し、全体平均を返す）。

実装済み:
- top_k_accuracy_at_k: Top-k 正解が含まれるか（含まれれば1、なければ0）の平均
- mean_reciprocal_rank (MRR): 最初に正解が出現する順位の逆数の平均
- mean_average_precision (MAP): AP を group ごとに計算し平均
- ndcg_at_k: 正規化 DCG（上位 k まで）
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


def _groupwise_sorted(df: pd.DataFrame, score_col: str, group_col: str) -> Iterable[Tuple[str, pd.DataFrame]]:
    """
    group_id ごとにスコア降順へ並べ替えた DataFrame を返すジェネレータ。
    """
    for gid, g in df.groupby(group_col):
        yield gid, g.sort_values(score_col, ascending=False)


def top_k_accuracy_at_k(df: pd.DataFrame, score_col: str, group_col: str, label_col: str, k: int = 1) -> float:
    """
    各 group で上位 k 件の中に正例（label=1）が1つでも含まれるかを 0/1 で評価し、平均を返す。
    """
    hits = []
    for _, g in _groupwise_sorted(df, score_col, group_col):
        topk = g.head(k)
        hits.append(int(topk[label_col].max() > 0))
    return float(np.mean(hits)) if len(hits) else float("nan")


def mean_reciprocal_rank(df: pd.DataFrame, score_col: str, group_col: str, label_col: str) -> float:
    """
    各 group で最初に正例が現れる順位の逆数（1/rank）を計算し、その平均を返す。
    正例が存在しない場合は 0 とする。
    """
    rrs = []
    for _, g in _groupwise_sorted(df, score_col, group_col):
        # find rank of first relevant
        ranks = np.where(g[label_col].to_numpy(dtype=int) == 1)[0]
        if len(ranks):
            rrs.append(1.0 / (ranks[0] + 1))
        else:
            rrs.append(0.0)
    return float(np.mean(rrs)) if len(rrs) else float("nan")


def mean_average_precision(df: pd.DataFrame, score_col: str, group_col: str, label_col: str) -> float:
    """
    各 group で AP（正例位置 i における precision@i を平均）を計算し、全 group 平均を返す。
    正例がない group は 0 としてカウント。
    """
    aps = []
    for _, g in _groupwise_sorted(df, score_col, group_col):
        y = g[label_col].to_numpy(dtype=int)
        if y.sum() == 0:
            aps.append(0.0)
            continue
        cum_hits = np.cumsum(y)
        precision_at_i = cum_hits / (np.arange(len(y)) + 1)
        ap = float(np.sum(precision_at_i * y) / max(int(y.sum()), 1))
        aps.append(ap)
    return float(np.mean(aps)) if len(aps) else float("nan")


def ndcg_at_k(df: pd.DataFrame, score_col: str, group_col: str, label_col: str, k: int = 3) -> float:
    """
    NDCG@k を group ごとに計算して平均。
    gain は 2^rel-1（rel は {0,1}）とし、順位 i の割引は log2(i+1) で行う一般的な定義。
    """
    def dcg(y: np.ndarray) -> float:
        # gains: 2^rel - 1 with rel in {0,1}
        gains = (2 ** y - 1) / np.log2(np.arange(1, len(y) + 1) + 1)
        return float(gains.sum())

    ndcgs = []
    for _, g in _groupwise_sorted(df, score_col, group_col):
        y_true = g[label_col].to_numpy(dtype=int)
        y_sorted = y_true[:k]
        ideal = np.sort(y_true)[::-1][:k]
        denom = dcg(ideal)
        ndcgs.append(dcg(y_sorted) / denom if denom > 0 else 0.0)
    return float(np.mean(ndcgs)) if len(ndcgs) else float("nan")
