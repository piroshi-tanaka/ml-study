"""
AutoGluon Tabular を用いた学習・評価ユーティリティ。

- §6: AutoGluonによる学習設定とハイパーパラメータ指定
- §7: 指定された評価指標（Top-1, クラス別Recall, 混同行列, Top-1@τ）を算出
- §8: 結果一式（特徴量CSV・予測CSV・metrics.json・HEAD情報）を保存"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.metrics import accuracy_score, confusion_matrix

from .pipeline import HashvinResult, save_head_details


def _select_feature_columns(df: pd.DataFrame, include_station: Optional[bool]) -> List[str]:
    """学習に使用する安全な列のみを抽出する（§4, §6）。"""
    core_cols = [col for col in ["dow", "hour_sin", "hour_cos", "soc_start"] if col in df.columns]
    behavior_cols = [
        col for col in ["is_return_band", "is_commute_band", "weekend_flag"] if col in df.columns
    ]
    station_cols = [col for col in df.columns if col.startswith("station_")]

    if include_station is None:
        include_station = bool(station_cols)
    if not include_station:
        station_cols = []

    cluster_cols = [
        col
        for col in df.columns
        if col.startswith("dist_to_")
        or col.startswith("freq_hashvin_")
        or col.startswith("recency_")
        or col.startswith("time_compat_")
    ]
    return core_cols + behavior_cols + station_cols + cluster_cols


def _prepare_training_frame(df: pd.DataFrame, feature_cols: List[str], label: str) -> pd.DataFrame:
    """特徴列と目的変数だけを抽出したDataFrameを返す。"""
    keep_cols = feature_cols + [label]
    return df[keep_cols].copy()


def _save_split_tables(result: HashvinResult, output_dir: Path) -> None:
    """train/valid/testの特徴テーブルをCSVで保存（§5の出力要件）。"""
    for split, df in result.split_datasets.items():
        path = output_dir / f"features_{split}.csv"
        df.to_csv(path, index=False)


def _compute_head_confusion(y_true: pd.Series, y_pred: pd.Series, head_clusters: List[str]) -> Dict[str, object]:
    """HEAD間の混同行列を算出（§7-4）。"""
    if y_true.empty or not head_clusters:
        return {"labels": [], "matrix": []}
    mask = y_true.isin(head_clusters)
    if mask.sum() == 0:
        return {"labels": head_clusters, "matrix": []}
    cm = confusion_matrix(y_true[mask], y_pred[mask], labels=head_clusters)
    return {"labels": head_clusters, "matrix": cm.tolist()}


def _compute_class_recalls(y_true: pd.Series, y_pred: pd.Series, head_clusters: List[str]) -> Dict[str, Optional[float]]:
    """クラス別Recall（HEADクラスのみ）を計算（§7-3）。"""
    recalls: Dict[str, Optional[float]] = {}
    for cid in head_clusters:
        mask = y_true == cid
        if mask.sum() == 0:
            recalls[cid] = None
        else:
            recalls[cid] = float(np.mean(y_pred[mask] == cid))
    return recalls


def _compute_threshold_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    proba_df: pd.DataFrame,
    thresholds: Iterable[float],
) -> Dict[str, Dict[str, Optional[float]]]:
    """Top-1@τ の Coverage/Accuracy を算出しレポート用にまとめる（§7 任意指標）。"""
    if proba_df.empty:
        return {str(t): {"coverage": None, "accuracy": None} for t in thresholds}
    max_proba = proba_df.max(axis=1)
    metrics: Dict[str, Dict[str, Optional[float]]] = {}
    for tau in thresholds:
        mask = max_proba >= tau
        coverage = float(mask.mean())
        if mask.sum() == 0:
            accuracy = None
        else:
            accuracy = float(np.mean(y_pred[mask] == y_true[mask]))
        metrics[str(tau)] = {"coverage": coverage, "accuracy": accuracy}
    return metrics


def _save_predictions(
    df: pd.DataFrame,
    y_pred: pd.Series,
    proba_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """テストデータの予測結果と確率をCSV出力（§7, §8の可観測性向上）。"""
    export_df = df[["session_uid", "charge_end_time", "y_class"]].copy()
    export_df.rename(columns={"y_class": "y_true"}, inplace=True)
    export_df["y_pred"] = y_pred
    if not proba_df.empty:
        export_df["max_proba"] = proba_df.max(axis=1)
        joined = pd.concat([export_df, proba_df], axis=1)
    else:
        joined = export_df
    joined.to_csv(output_path, index=False)


def train_and_evaluate(
    result: HashvinResult,
    output_dir: Path,
    enable_station_type: Optional[bool] = None,
    autogluon_presets: str = "medium_quality_faster_train",
    autogluon_time_limit: Optional[int] = None,
    eval_thresholds: Iterable[float] = (0.5, 0.7, 0.9),
) -> Dict[str, object]:
    """
    要件§6（学習設定）・§7（評価指標）・§8（出力物）を一括で実行する。

    - AutoGluon Tabular で学習（train/valid）
    - testでTop-1 Accuracy（Strict/Head-only）、クラス別Recall、HEAD混同行列、Top-1@τを算出
    - 各種CSV/JSONを `result/<hashvin>/` 配下に保存
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    # §8: 結果閲覧用に特徴テーブルとHEAD情報をディレクトリに保存
    _save_split_tables(result, output_dir)
    save_head_details(result.head_details, output_dir / "head_clusters.json")

    label_col = result.label_col
    train_df = result.split_datasets.get("train", pd.DataFrame())
    valid_df = result.split_datasets.get("valid", pd.DataFrame())
    test_df = result.split_datasets.get("test", pd.DataFrame())

    # §4・§6: 生成済み特徴から学習で使う列を抽出（ステーション種別はフラグで切替）
    feature_cols = _select_feature_columns(result.features, enable_station_type)
    if not feature_cols:
        raise ValueError("No feature columns selected. Check feature generation.")

    # train/valid/test をAutoGluonに渡せる形式に整形
    train_model_df = _prepare_training_frame(train_df, feature_cols, label_col)
    valid_model_df = _prepare_training_frame(valid_df, feature_cols, label_col) if not valid_df.empty else None
    test_model_df = _prepare_training_frame(test_df, feature_cols, label_col) if not test_df.empty else None

    metrics: Dict[str, object] = {
        "hashvin": result.hashvin,
        "head_clusters": result.head_clusters,
        "feature_columns": feature_cols,
    }

    # hashvinによっては教師データが不足するため、その際はスキップ情報を返す
    if train_model_df.empty or train_model_df[label_col].nunique() < 2:
        metrics["status"] = "skipped_training"
        metrics["reason"] = "Not enough labeled data to train a model."
        (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        return metrics

    # §6: AutoGluon Tabular で学習（評価指標は accuracy を指定）
    predictor_path = output_dir / "autogluon"
    predictor = TabularPredictor(
        label=label_col,
        eval_metric="accuracy",
        path=str(predictor_path),
    )
    fit_kwargs = {
        "train_data": train_model_df,
        "presets": autogluon_presets,
        "time_limit": autogluon_time_limit,
    }
    if valid_model_df is not None and not valid_model_df.empty:
        fit_kwargs["tuning_data"] = valid_model_df
    predictor.fit(**fit_kwargs)

    metrics["status"] = "trained"
    train_score = predictor.evaluate(train_model_df, silent=True)
    metrics["train_score"] = train_score

    if test_model_df is None or test_model_df.empty:
        metrics["note"] = "No test data available for evaluation."
        (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        return metrics

    y_pred = predictor.predict(test_model_df, as_pandas=True)
    proba_df = predictor.predict_proba(test_model_df, as_pandas=True)

    y_true = test_model_df[label_col]
    strict_acc = float(accuracy_score(y_true, y_pred))
    head_only_mask = y_true.isin(result.head_clusters)
    head_only_acc = float(accuracy_score(y_true[head_only_mask], y_pred[head_only_mask])) if head_only_mask.sum() > 0 else None

    class_recalls = _compute_class_recalls(y_true, y_pred, result.head_clusters)
    confusion = _compute_head_confusion(y_true, y_pred, result.head_clusters)
    threshold_metrics = _compute_threshold_metrics(y_true, y_pred, proba_df, eval_thresholds)

    # §7: 必須指標（Top-1, クラス別Recall, 混同行列）と任意指標（Top-1@τ）を取りまとめる
    metrics.update(
        {
            "strict_top1_accuracy": strict_acc,
            "head_only_top1_accuracy": head_only_acc,
            "class_recalls": class_recalls,
            "confusion_matrix": confusion,
            "threshold_metrics": threshold_metrics,
        }
    )

    # §7・§8: テストセットの予測詳細をCSVで保存し、分析できる状態にする
    _save_predictions(test_df, y_pred, proba_df, output_dir / "predictions_test.csv")

    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return metrics
