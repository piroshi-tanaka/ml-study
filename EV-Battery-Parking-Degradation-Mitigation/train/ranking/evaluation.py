"""評価指標の計算と検証結果のサマリ生成を担当するモジュール。"""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple, TYPE_CHECKING, Union

import numpy as np
import pandas as pd

from .metrics import calculate_ranking_metrics

if TYPE_CHECKING:  # 循環参照を避けるため型チェック時のみ import
    from .pipeline import TrainedUserModel


def evaluate_user_model(
    model: "TrainedUserModel",
    topk: Iterable[int],
    return_scored: bool = False,
) -> Union[Dict[str, float], Tuple[Dict[str, float], pd.DataFrame]]:
    """検証データにスコアを付与し、ランキング指標を計算する。"""

    val_df = model.validation_data.copy()
    # AutoGluon の predict_proba が DataFrame の場合と ndarray の場合に対応
    proba = model.predictor.predict_proba(val_df[model.features])
    if isinstance(proba, pd.DataFrame):
        if 1 in proba.columns:
            val_df["score"] = proba[1].to_numpy()
        else:
            val_df["score"] = proba.iloc[:, -1].to_numpy()
    else:
        val_df["score"] = np.asarray(proba)

    metrics = calculate_ranking_metrics(
        val_df,
        score_col="score",
        group_col="event_id",
        label_col="label",
        topk=topk,
    )
    return (metrics, val_df) if return_scored else metrics


def summarize_validation_scores(
    val_scored: pd.DataFrame,
    topk: Iterable[int],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """検証結果をイベント単位・クラスタ単位で集計し、分析用テーブルを返す。"""

    topk_list = sorted({int(k) for k in topk}) if topk else []
    event_records: List[Dict[str, object]] = []
    cluster_stats: Dict[str, Dict[str, float]] = {}

    if val_scored.empty:
        event_columns = [
            "event_id",
            "actual_cluster",
            "actual_score",
            "actual_rank",
            "top1_cluster",
            "top1_score",
            "candidate_count",
            "top_candidates",
            "top_scores",
            "is_top1_correct",
        ] + [f"hit_at_{k}" for k in topk_list]
        cluster_columns = (
            [
                "cluster",
                "event_count",
                "share",
                "candidate_rows",
                "correct_top1",
                "incorrect_top1",
                "top1_accuracy",
                "actual_score_mean",
                "mean_rank",
            ]
            + [f"hit_at_{k}_count" for k in topk_list]
            + [f"hit_at_{k}_rate" for k in topk_list]
        )
        return pd.DataFrame(columns=event_columns), pd.DataFrame(columns=cluster_columns)

    for event_id, group in val_scored.groupby("event_id"):
        group_sorted = group.sort_values("score", ascending=False).reset_index(drop=True)
        candidate_count = len(group_sorted)
        actual_row = group_sorted[group_sorted["label"] == 1].head(1)
        actual_cluster = (
            str(actual_row.iloc[0]["candidate_cluster"]) if not actual_row.empty else None
        )
        actual_score = (
            float(actual_row.iloc[0]["score"]) if not actual_row.empty else float("nan")
        )
        actual_rank = None
        if actual_cluster is not None:
            match_idx = group_sorted[
                group_sorted["candidate_cluster"].astype(str) == actual_cluster
            ].index
            if len(match_idx):
                actual_rank = int(match_idx[0]) + 1

        top1_cluster = (
            str(group_sorted.iloc[0]["candidate_cluster"]) if candidate_count else None
        )
        top1_score = float(group_sorted.iloc[0]["score"]) if candidate_count else float("nan")

        max_k = topk_list[-1] if topk_list else min(3, candidate_count)
        top_candidates = ""
        top_scores = ""
        if candidate_count:
            top_subset = group_sorted.head(max_k)
            top_candidates = ",".join(top_subset["candidate_cluster"].astype(str).tolist())
            top_scores = ",".join(f"{s:.6f}" for s in top_subset["score"].tolist())

        hit_flags = {}
        for k in topk_list:
            if actual_cluster is None:
                hit_flags[k] = False
            else:
                hit_flags[k] = bool(
                    actual_cluster
                    in group_sorted.head(k)["candidate_cluster"].astype(str).tolist()
                )

        event_record: Dict[str, object] = {
            "event_id": event_id,
            "actual_cluster": actual_cluster,
            "actual_score": actual_score,
            "actual_rank": actual_rank,
            "top1_cluster": top1_cluster,
            "top1_score": top1_score,
            "candidate_count": candidate_count,
            "top_candidates": top_candidates,
            "top_scores": top_scores,
            "is_top1_correct": bool(actual_cluster is not None and top1_cluster == actual_cluster),
        }
        for k in topk_list:
            event_record[f"hit_at_{k}"] = hit_flags[k]
        event_records.append(event_record)

        if actual_cluster is None:
            continue

        stats = cluster_stats.setdefault(
            actual_cluster,
            {
                "event_count": 0,
                "candidate_rows": 0,
                "top1_correct": 0,
                "actual_score_total": 0.0,
                "actual_score_count": 0,
                "rank_total": 0.0,
                "rank_count": 0,
                **{f"hit_at_{k}_count": 0 for k in topk_list},
            },
        )

        stats["event_count"] += 1
        stats["candidate_rows"] += candidate_count
        stats["top1_correct"] += int(top1_cluster == actual_cluster)
        if not pd.isna(actual_score):
            stats["actual_score_total"] += actual_score
            stats["actual_score_count"] += 1
        if actual_rank is not None:
            stats["rank_total"] += actual_rank
            stats["rank_count"] += 1
        for k in topk_list:
            stats[f"hit_at_{k}_count"] += int(hit_flags[k])

    event_df = pd.DataFrame(event_records)

    total_events = sum(stats["event_count"] for stats in cluster_stats.values())
    cluster_rows: List[Dict[str, object]] = []
    for cluster, stats in cluster_stats.items():
        row: Dict[str, object] = {
            "cluster": cluster,
            "event_count": stats["event_count"],
            "share": stats["event_count"] / total_events if total_events else float("nan"),
            "candidate_rows": stats["candidate_rows"],
            "correct_top1": stats["top1_correct"],
            "incorrect_top1": stats["event_count"] - stats["top1_correct"],
            "top1_accuracy": (
                stats["top1_correct"] / stats["event_count"]
                if stats["event_count"]
                else float("nan")
            ),
            "actual_score_mean": (
                stats["actual_score_total"] / stats["actual_score_count"]
                if stats["actual_score_count"]
                else float("nan")
            ),
            "mean_rank": (
                stats["rank_total"] / stats["rank_count"]
                if stats["rank_count"]
                else float("nan")
            ),
        }
        for k in topk_list:
            row[f"hit_at_{k}_count"] = stats[f"hit_at_{k}_count"]
            row[f"hit_at_{k}_rate"] = (
                stats[f"hit_at_{k}_count"] / stats["event_count"]
                if stats["event_count"]
                else float("nan")
            )
        cluster_rows.append(row)

    cluster_df = pd.DataFrame(cluster_rows)
    if not cluster_df.empty:
        cluster_df = cluster_df.sort_values("event_count", ascending=False).reset_index(
            drop=True
        )

    return event_df, cluster_df
