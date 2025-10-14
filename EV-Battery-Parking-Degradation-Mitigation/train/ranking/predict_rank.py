"""
推論スクリプト: 学習済みモデルを用いて、各充電イベントに対する「次の長時間放置クラスタ」を
スコア順にランキングし、上位 Top-k を出力します。

使い方:
  python -m train.ranking.predict_rank \
      --sessions_csv EV-Battery-Parking-Degradation-Mitigation/eda/ev_sessions_test.csv \
      --model_dir EV-Battery-Parking-Degradation-Mitigation/train/outputs/ranking_model/autogluon \
      --top_k 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions_csv", type=str, required=True)
    ap.add_argument("--model_dir", type=str, required=True)
    ap.add_argument("--top_k", type=int, default=3)
    args = ap.parse_args()

    # 1) データ読込と前処理（学習時と同じ規約で整形）
    sessions = load_sessions(Path(args.sessions_csv))
    sessions = prepare_sessions(sessions, long_park_threshold_minutes=360)
    c2p = build_charge_to_next_long_table(sessions)

    # 2) 全ての充電イベントに対して候補展開・特徴量作成（ラベルの有無に関わらず）
    centroids = compute_cluster_centroids_by_vehicle(sessions)
    cand_pool = build_candidate_pool_per_vehicle(sessions, top_n_per_vehicle=10, global_top_n=20)
    df_rank = build_ranking_training_data(
        df_sessions=sessions,
        charge_to_long=c2p,
        candidate_pool=cand_pool,
        centroids_by_vehicle=centroids,
        negative_sample_k=None,  # 推論ではサンプリングせず全候補を使用
    )
    features, _ = get_feature_columns(df_rank)

    # 3) モデル読込と確率スコア推定
    predictor = TabularPredictor.load(args.model_dir)
    proba = predictor.predict_proba(df_rank[features])
    if isinstance(proba, pd.DataFrame):
        scores = proba[1].to_numpy() if 1 in proba.columns else proba.iloc[:, -1].to_numpy()
    else:
        scores = np.asarray(proba)
    df_rank = df_rank.assign(score=scores)

    # 4) 充電イベント（group_id）ごとにスコア降順に並べ、Top-k を抽出
    topk_rows = []
    for gid, g in df_rank.groupby("group_id"):
        gg = g.sort_values("score", ascending=False).head(args.top_k)
        topk_rows.append(
            {
                "group_id": gid,
                "hashvin": gg["hashvin"].iloc[0],
                "charge_cluster": gg["charge_cluster"].iloc[0],
                "weekday": gg["weekday"].iloc[0],
                "charge_start_hour": gg["charge_start_hour"].iloc[0],
                "ranked_candidates": ",".join(map(str, gg["candidate_cluster"].tolist())),
                "scores": ",".join(f"{s:.6f}" for s in gg["score"].tolist()),
            }
        )

    out = pd.DataFrame(topk_rows)
    out_path = Path(args.model_dir).parent / "predictions_topk.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved top-{args.top_k} predictions to: {out_path}")


if __name__ == "__main__":
    main()
