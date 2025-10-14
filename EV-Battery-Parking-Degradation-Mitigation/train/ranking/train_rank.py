"""
AutoGluon を用いて「充電直後に最初に発生する長時間放置クラスタ」をランキング予測するための学習スクリプト。

方針: ランキング問題を（充電, 候補クラスタ）ペアの二値分類に落とし込み、
正例（実際の放置クラスタ）に対する陽性確率をランキングスコアとして用います。

使い方（例）:
  python -m train.ranking.train_rank \
      --sessions_csv EV-Battery-Parking-Degradation-Mitigation/eda/ev_sessions_test.csv \
      --outdir EV-Battery-Parking-Degradation-Mitigation/train/outputs/ranking_model \
      --time_limit 600
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor

from .dataset import (
    build_candidate_pool_per_vehicle,
    build_charge_to_next_long_table,
    build_ranking_training_data,
    compute_cluster_centroids_by_vehicle,
    get_feature_columns,
    load_sessions,
    prepare_sessions,
)
from .metrics import mean_average_precision, mean_reciprocal_rank, ndcg_at_k, top_k_accuracy_at_k


def _groupwise_split(df_rank: pd.DataFrame, group_col: str, val_ratio: float = 0.2, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    group_id 単位（＝充電イベント単位）で学習/検証を分割します。
    ランダムに group をシャッフルして、先頭の一定割合を検証用へ。
    """
    gids = df_rank[group_col].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(gids)
    n_val = max(1, int(len(gids) * float(val_ratio)))
    val_gids = set(gids[:n_val])
    train = df_rank[~df_rank[group_col].isin(val_gids)].copy()
    val = df_rank[df_rank[group_col].isin(val_gids)].copy()
    return train, val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions_csv", type=str, required=True, help="Path to sessions CSV")
    ap.add_argument("--outdir", type=str, required=True, help="Output dir for model and artifacts")
    ap.add_argument("--time_limit", type=int, default=600, help="AutoGluon fit time limit (seconds)")
    ap.add_argument("--neg_sample_k", type=int, default=20, help="Max negative candidates per group (None=all)")
    ap.add_argument("--presets", type=str, default="medium_quality_faster_train", help="AutoGluon presets")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) データ読込と前処理（長時間放置リンクまで）
    sessions = load_sessions(Path(args.sessions_csv))
    sessions = prepare_sessions(sessions, long_park_threshold_minutes=360)

    # 2) 充電→最初の長時間放置 の対応表（ラベル情報）を作成
    c2p = build_charge_to_next_long_table(sessions)
    # ラベルが存在する charge のみを学習に使用
    c2p_labeled = c2p.dropna(subset=["park_cluster"]).copy()
    if c2p_labeled.empty:
        raise SystemExit("No labeled charge→long-park pairs found. Check input data.")

    # 3) 候補集合と座標重心の作成
    centroids = compute_cluster_centroids_by_vehicle(sessions)
    cand_pool = build_candidate_pool_per_vehicle(sessions, top_n_per_vehicle=10, global_top_n=20)

    # 4) ランキング学習用データを構築（候補展開＋特徴量）
    df_rank = build_ranking_training_data(
        df_sessions=sessions,
        charge_to_long=c2p,
        candidate_pool=cand_pool,
        centroids_by_vehicle=centroids,
        negative_sample_k=(None if args.neg_sample_k <= 0 else args.neg_sample_k),
    )
    # 5) group_id 単位で学習/検証分割
    train_df, val_df = _groupwise_split(df_rank, group_col="group_id", val_ratio=0.2, seed=42)

    # 6) 特徴量列を抽出し、AutoGluon で学習
    features, cat_cols = get_feature_columns(train_df)
    label = "label"
    predictor = TabularPredictor(
        label=label,
        path=str(outdir / "autogluon"),
        problem_type="binary",
        eval_metric="roc_auc",
    )

    predictor.fit(
        train_data=train_df[[*features, label]],
        time_limit=args.time_limit,
        presets=args.presets,
        hyperparameters=None,
        ag_args_fit={"num_cpus": 0},
        # AutoGluon は型からカテゴリ判定を自動推定します（ここでは明示指定は行いません）。
    )

    # 7) 検証データに対して確率スコアを付与し、ランキング指標を算出
    val_df = val_df.copy()
    proba = predictor.predict_proba(val_df[features])
    # 2値分類の predict_proba は陽性クラスの確率を返す（モデルにより DataFrame/ndarray の差がある）
    if isinstance(proba, pd.DataFrame):
        # 列名が [0,1] の場合は 1 列目が陽性クラス
        if 1 in proba.columns:
            val_df["score"] = proba[1].to_numpy()
        else:
            val_df["score"] = proba.iloc[:, -1].to_numpy()
    else:
        val_df["score"] = np.asarray(proba)

    metrics = {
        "top1": top_k_accuracy_at_k(val_df, "score", "group_id", label, k=1),
        "top3": top_k_accuracy_at_k(val_df, "score", "group_id", label, k=3),
        "MRR": mean_reciprocal_rank(val_df, "score", "group_id", label),
        "MAP": mean_average_precision(val_df, "score", "group_id", label),
        "NDCG@3": ndcg_at_k(val_df, "score", "group_id", label, k=3),
    }
    pd.Series(metrics).to_csv(outdir / "val_metrics.csv")
    val_df.to_csv(outdir / "val_scored_rows.csv", index=False)

    print("Validation metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
