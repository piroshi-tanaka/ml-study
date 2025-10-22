from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from .pipeline import HashvinResult


def _select_feature_columns(df: pd.DataFrame, include_station: Optional[bool]) -> List[str]:
    """AutoGluonに渡す学習用列を抽出する。

    include_stationがFalseならstation_列を除外する。
    """
    exclude = {
        "label",
        "split",
        "hashvin",
        "session_uid",
        "true_cluster",
        "next_long_inactive_cluster",
    }
    station_cols = [col for col in df.columns if col.startswith("station_")]
    if include_station is None:
        include_station = bool(station_cols)

    feature_cols: List[str] = []
    for col in df.columns:
        if col in exclude:
            continue
        if not include_station and col.startswith("station_"):
            continue
        feature_cols.append(col)
    return list(dict.fromkeys(feature_cols))


def _prepare_training_frame(df: pd.DataFrame, feature_cols: List[str], label: str) -> pd.DataFrame:
    """特徴列とラベル列だけを抽出したDataFrameを返す。"""
    keep_cols = list(dict.fromkeys(feature_cols + [label]))
    available_cols = [col for col in keep_cols if col in df.columns]
    data = df[available_cols].copy()
    if data.columns.duplicated().any():
        data = data.loc[:, ~data.columns.duplicated()]
    return data


def _save_split_tables(result: HashvinResult, output_dir: Path) -> None:
    """train/valid/testの特徴テーブルをCSVで保存する。"""
    for split, df in result.split_datasets.items():
        path = output_dir / f"features_{split}.csv"
        df.to_csv(path, index=False)


def _extract_positive_proba(proba_df: pd.DataFrame) -> pd.Series:
    """AutoGluonの確率出力から陽性クラス(1)の列を安全に取得する。"""
    if proba_df.empty:
        return pd.Series(dtype=float)
    for key in [1, "1", True, "True"]:
        if key in proba_df.columns:
            return proba_df[key]
    return proba_df.iloc[:, -1]


def _aggregate_top_predictions(test_df: pd.DataFrame, proba_positive: pd.Series) -> pd.DataFrame:
    """セッションごとに最も確率が高い候補を抽出する。"""
    enriched = test_df.copy()
    enriched["proba_positive"] = proba_positive.reindex(test_df.index)
    idx = enriched.groupby("session_uid")["proba_positive"].idxmax()
    top_predictions = enriched.loc[idx].copy()
    top_predictions["predicted_cluster"] = top_predictions["candidate_cluster"]
    top_predictions["is_correct"] = top_predictions["predicted_cluster"] == top_predictions["true_cluster"]
    return top_predictions


def _compute_threshold_metrics(top_predictions: pd.DataFrame, thresholds: Iterable[float]) -> Dict[str, Dict[str, Optional[float]]]:
    """確率しきい値ごとのカバレッジと精度を算出する。"""
    metrics: Dict[str, Dict[str, Optional[float]]] = {}
    if top_predictions.empty:
        return metrics
    total = float(len(top_predictions))
    for tau in thresholds:
        subset = top_predictions[top_predictions["proba_positive"] >= tau]
        coverage = float(len(subset)) / total if total > 0 else 0.0
        accuracy = float(subset["is_correct"].mean()) if not subset.empty else None
        metrics[str(tau)] = {
            "coverage": coverage,
            "accuracy": accuracy,
            "count": int(len(subset)),
        }
    return metrics


def _build_hashvin_summary(
    result: HashvinResult,
    top_predictions: pd.DataFrame,
    session_info: pd.DataFrame,
    coverage_series: pd.Series,
    candidate_counts: pd.Series,
    top1_accuracy: float,
    balanced_acc: Optional[float],
    auc: Optional[float],
) -> pd.DataFrame:
    """hashvin単位のサマリーを整形する。"""
    total_sessions = int(session_info.shape[0])
    train_sessions = int((session_info["split"] == "train").sum())
    valid_sessions = int((session_info["split"] == "valid").sum())
    test_sessions = int((session_info["split"] == "test").sum())
    correct = int(top_predictions["is_correct"].sum())
    incorrect = int(len(top_predictions) - correct)
    coverage = float(coverage_series.mean()) if not coverage_series.empty else 0.0
    avg_candidates = float(candidate_counts.mean()) if not candidate_counts.empty else 0.0

    summary = pd.DataFrame(
        [
            {
                "hashvin": result.hashvin,
                "total_sessions": total_sessions,
                "train_sessions": train_sessions,
                "valid_sessions": valid_sessions,
                "test_sessions": test_sessions,
                "top1_accuracy": top1_accuracy,
                "balanced_accuracy": balanced_acc,
                "candidate_auc": auc,
                "coverage": coverage,
                "avg_candidate_count": avg_candidates,
                "correct_sessions": correct,
                "incorrect_sessions": incorrect,
            }
        ]
    )
    return summary


def _build_cluster_summary(
    result: HashvinResult,
    top_predictions: pd.DataFrame,
    session_info: pd.DataFrame,
) -> pd.DataFrame:
    """hashvin×放置クラスタ別の評価サマリーを作成する。"""
    if session_info.empty:
        return pd.DataFrame(
            columns=[
                "hashvin",
                "true_cluster",
                "total_sessions",
                "share_of_hashvin",
                "train_sessions",
                "valid_sessions",
                "test_sessions",
                "correct_sessions",
                "incorrect_sessions",
                "accuracy",
            ]
        )

    total_sessions = session_info.shape[0]
    total_counts = session_info.groupby("true_cluster")["session_uid"].count()
    train_counts = session_info[session_info["split"] == "train"].groupby("true_cluster")["session_uid"].count()
    valid_counts = session_info[session_info["split"] == "valid"].groupby("true_cluster")["session_uid"].count()
    test_counts = session_info[session_info["split"] == "test"].groupby("true_cluster")["session_uid"].count()

    per_cluster = top_predictions.groupby("true_cluster").agg(
        test_sessions=("session_uid", "count"),
        correct_sessions=("is_correct", "sum"),
    )
    per_cluster["incorrect_sessions"] = per_cluster["test_sessions"] - per_cluster["correct_sessions"]
    per_cluster["accuracy"] = per_cluster.apply(
        lambda row: float(row["correct_sessions"] / row["test_sessions"]) if row["test_sessions"] else None,
        axis=1,
    )
    per_cluster_dict = per_cluster.to_dict(orient="index")

    rows: List[Dict[str, object]] = []
    for cluster_id, total in total_counts.items():
        cluster_stats = per_cluster_dict.get(cluster_id, {})
        rows.append(
            {
                "hashvin": result.hashvin,
                "true_cluster": cluster_id,
                "total_sessions": int(total),
                "share_of_hashvin": float(total / total_sessions) if total_sessions else 0.0,
                "train_sessions": int(train_counts.get(cluster_id, 0)),
                "valid_sessions": int(valid_counts.get(cluster_id, 0)),
                "test_sessions": int(test_counts.get(cluster_id, 0)),
                "correct_sessions": int(cluster_stats.get("correct_sessions", 0)),
                "incorrect_sessions": int(cluster_stats.get("incorrect_sessions", 0)),
                "accuracy": cluster_stats.get("accuracy"),
            }
        )
    return pd.DataFrame(rows)


def _save_predictions(candidate_df: pd.DataFrame, top_predictions: pd.DataFrame, output_dir: Path) -> None:
    """候補レベルとセッションレベルの予測をCSVで保存する。"""
    candidate_df.to_csv(output_dir / "predictions_candidates.csv", index=False)
    top_predictions.to_csv(output_dir / "predictions_session_top1.csv", index=False)


def train_and_evaluate(
    result: HashvinResult,
    output_dir: Path,
    enable_station_type: Optional[bool] = None,
    autogluon_presets: str = "medium_quality_faster_train",
    autogluon_time_limit: Optional[int] = None,
    eval_thresholds: Iterable[float] = (0.5, 0.7, 0.9),
) -> Dict[str, object]:
    """AutoGluonで候補スコアリングモデルを学習し、評価指標を保存する。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_split_tables(result, output_dir)

    label_col = result.label_col
    train_df = result.split_datasets.get("train", pd.DataFrame())
    valid_df = result.split_datasets.get("valid", pd.DataFrame())
    test_df = result.split_datasets.get("test", pd.DataFrame())

    feature_cols = _select_feature_columns(result.features, enable_station_type)
    if not feature_cols:
        raise ValueError("特徴量が選択できませんでした。パイプラインの出力内容を確認してください。")

    train_model_df = _prepare_training_frame(train_df, feature_cols, label_col)
    valid_model_df = _prepare_training_frame(valid_df, feature_cols, label_col) if not valid_df.empty else None
    test_model_df = _prepare_training_frame(test_df, feature_cols, label_col) if not test_df.empty else None

    for frame in [train_model_df, valid_model_df, test_model_df]:
        if frame is not None and frame.columns.duplicated().any():
            frame.drop(columns=list(frame.columns[frame.columns.duplicated()]), inplace=True)

    feature_cols_clean = [col for col in feature_cols if col in train_model_df.columns]
    metrics: Dict[str, object] = {
        "hashvin": result.hashvin,
        "feature_columns": feature_cols_clean,
    }

    if train_model_df.empty or train_model_df[label_col].nunique() < 2:
        metrics["status"] = "skipped_training"
        metrics["reason"] = "教師データが不足しているため学習を中断しました。"
        (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        return metrics

    predictor_path = output_dir / "autogluon"
    predictor = TabularPredictor(
        label=label_col,
        eval_metric="accuracy",
        path=str(predictor_path),
        problem_type="binary",
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
    metrics["train_score"] = predictor.evaluate(train_model_df, silent=True)

    feature_importance_df: Optional[pd.DataFrame] = None
    try:
        feature_importance_df = predictor.feature_importance(train_model_df, silent=True)
    except Exception as exc:  # pylint: disable=broad-except
        metrics["feature_importance_error"] = str(exc)
    else:
        fi_path = output_dir / "feature_importance.csv"
        feature_importance_df.to_csv(fi_path, index=True)
        metrics["feature_importance_path"] = str(fi_path)
        fi_with_hashvin = feature_importance_df.reset_index().rename(columns={"index": "feature"})
        fi_with_hashvin.insert(0, "hashvin", result.hashvin)
        summary_path = output_dir.parent / "feature_importance_summary.csv"
        if summary_path.exists():
            existing = pd.read_csv(summary_path)
            existing = existing[existing["hashvin"] != result.hashvin]
            combined = pd.concat([existing, fi_with_hashvin], ignore_index=True)
        else:
            combined = fi_with_hashvin
        combined.to_csv(summary_path, index=False)

    if test_model_df is None or test_model_df.empty:
        metrics["note"] = "テストデータが存在しないため評価をスキップしました。"
        (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        return metrics

    y_pred_candidate = predictor.predict(test_model_df, as_pandas=True)
    proba_df = predictor.predict_proba(test_model_df, as_pandas=True)
    positive_proba = _extract_positive_proba(proba_df)

    test_candidates = test_df.copy()
    test_candidates["pred_label"] = y_pred_candidate
    test_candidates["proba_positive"] = positive_proba.reindex(test_candidates.index)

    top_predictions = _aggregate_top_predictions(test_candidates, positive_proba)
    if {"session_uid", "true_cluster", "split"}.issubset(result.features.columns):
        session_info = result.features[["session_uid", "true_cluster", "split"]].drop_duplicates(subset=["session_uid"])
    else:
        session_rows = []
        for split_name, split_df in result.split_datasets.items():
            if {"session_uid", "true_cluster"}.issubset(split_df.columns):
                info = split_df[["session_uid", "true_cluster"]].drop_duplicates(subset=["session_uid"]).copy()
                info["split"] = split_name
                session_rows.append(info)
        session_info = pd.concat(session_rows, ignore_index=True) if session_rows else pd.DataFrame(
            columns=["session_uid", "true_cluster", "split"]
        )

    coverage_series = test_candidates.groupby("session_uid")["label"].max()
    candidate_counts = test_candidates.groupby("session_uid")["candidate_cluster"].size()

    top1_accuracy = float(top_predictions["is_correct"].mean()) if not top_predictions.empty else 0.0
    try:
        balanced_acc = float(
            balanced_accuracy_score(top_predictions["true_cluster"], top_predictions["predicted_cluster"])
        )
    except ValueError:
        balanced_acc = None
    try:
        auc = float(roc_auc_score(test_model_df[label_col], positive_proba))
    except ValueError:
        auc = None

    threshold_metrics = _compute_threshold_metrics(top_predictions, eval_thresholds)

    summary_hashvin = _build_hashvin_summary(
        result,
        top_predictions,
        session_info,
        coverage_series,
        candidate_counts,
        top1_accuracy,
        balanced_acc,
        auc,
    )
    hashvin_summary_path = output_dir / "summary_hashvin.csv"
    summary_hashvin.to_csv(hashvin_summary_path, index=False)

    summary_cluster = _build_cluster_summary(result, top_predictions, session_info)
    cluster_summary_path = output_dir / "summary_cluster.csv"
    summary_cluster.to_csv(cluster_summary_path, index=False)

    _save_predictions(test_candidates, top_predictions, output_dir)

    metrics.update(
        {
            "top1_accuracy": top1_accuracy,
            "balanced_accuracy": balanced_acc,
            "candidate_auc": auc,
            "threshold_metrics": threshold_metrics,
            "hashvin_summary_path": str(hashvin_summary_path),
            "cluster_summary_path": str(cluster_summary_path),
            "coverage_mean": float(coverage_series.mean()) if not coverage_series.empty else 0.0,
            "candidate_count_mean": float(candidate_counts.mean()) if not candidate_counts.empty else 0.0,
        }
    )

    try:
        leaderboard_df = predictor.leaderboard(test_model_df, extra_info=True, silent=True)
    except Exception as exc:  # pylint: disable=broad-except
        metrics["leaderboard_error"] = str(exc)
    else:
        leaderboard_path = output_dir / "model_leaderboard.csv"
        leaderboard_df.to_csv(leaderboard_path, index=False)
        metrics["model_leaderboard_path"] = str(leaderboard_path)
        metrics["model_leaderboard_top"] = leaderboard_df.head(5).to_dict(orient="records")

    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return metrics
