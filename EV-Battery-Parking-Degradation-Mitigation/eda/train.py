# filename: train_parking_predictor.py
# 使い方:
#   python train_parking_predictor.py path/to/sessions.csv
#
# 前提: pip install autogluon.tabular==1.*  (環境に合わせて)

import sys
import os
import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.model_selection import train_test_split
import numpy as np

def top_k_accuracy(y_true, proba_df, k=3):
    """predict_probaのDataFrameを受け取り、Top-k Accuracyを返す"""
    classes = proba_df.columns.tolist()
    topk_preds = proba_df.values.argsort(axis=1)[:, -k:][:, ::-1]  # 各行で確率上位kのインデックス
    class_idx = {c:i for i,c in enumerate(classes)}
    y_idx = np.array([class_idx[y] for y in y_true])
    hit = (topk_preds == y_idx.reshape(-1,1)).any(axis=1)
    return float(hit.mean())

def main(csv_path):
    if not os.path.exists(csv_path):
        print(f"CSVが見つかりません: {csv_path}")
        sys.exit(1)

    # ---- データ読み込み ----
    df = TabularDataset(csv_path)

    # 必須列チェック（最低限）
    required_cols = ["hashvin", "charge_cluster_id", "inactive_cluster_id", "time_vin", "month"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"必要カラムが不足しています: {missing}")
        sys.exit(1)

    # ---- 型の簡易整形（カテゴリはそのまま渡してOK）----
    # 数値として入っていてもカテゴリ扱いのほうが効くケースが多いので明示的にstrへ
    for col in ["hashvin", "charge_cluster_id", "inactive_cluster_id", "time_vin", "month"]:
        df[col] = df[col].astype(str)

    label = "inactive_cluster_id"
    features = ["hashvin", "charge_cluster_id", "time_vin", "month"]
    # もし week も使いたければ:
    # if "week" in df.columns:
    #     df["week"] = df["week"].astype(str)
    #     features.append("week")

    # ---- 学習/評価データ分割 ----
    # 既知hashvinのみが対象という前提なので、そのままランダム分割
    # クラス不均衡が強い場合は stratify を試す（失敗時はフォールバック）
    try:
        train_df, test_df = train_test_split(
            df[features + [label]],
            test_size=0.2,
            random_state=42,
            stratify=df[label]
        )
    except Exception:
        train_df, test_df = train_test_split(
            df[features + [label]],
            test_size=0.2,
            random_state=42
        )

    # ---- 学習 ----
    predictor = TabularPredictor(
        label=label,
        problem_type="multiclass",
        eval_metric="accuracy",
        path="AutogluonModels/parking_predictor"
    ).fit(
        train_data=train_df,
        presets="best_quality",   # 小～中規模データで強い
        time_limit=None           # 時間制限が必要なら秒数を設定
    )

    # ---- 評価 ----
    print("\n== Evaluate on Test ==")
    metrics = predictor.evaluate(test_df, silent=True)
    print(metrics)

    # Top-3 accuracy を手計算
    proba = predictor.predict_proba(test_df[features])
    k3 = top_k_accuracy(test_df[label].tolist(), proba, k=3)
    print(f"top-3 accuracy: {k3:.4f}")

    # ---- 推論例（testの先頭1件）----
    sample = test_df[features].head(1)
    pred = predictor.predict(sample)
    pred_proba = predictor.predict_proba(sample)
    print("\n== Sample Prediction ==")
    print(sample)
    print("pred:", pred.values[0])
    print("proba top-3:")
    print(pred_proba.T.sort_values(by=pred_proba.index[0], ascending=False).head(3))

    print("\nモデル保存先:", predictor.path)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使い方: python train_parking_predictor.py path/to/sessions.csv")
        sys.exit(1)
    main(sys.argv[1])