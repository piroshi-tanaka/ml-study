"""
AutoGluon Tabular を用いた学習・評価ユーティリティ。

- §6: AutoGluonによる学習設定とハイパーパラメータ指定
- §7: 指定された評価指標（Top-1, クラス別Recall, 混同行列, Top-1@τ）を算出
- §8: 結果一式（特徴量CSV・予測CSV・metrics.json・HEAD情報）を保存"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

from .pipeline import HashvinResult, save_head_details


def _select_feature_columns(df: pd.DataFrame, include_station: Optional[bool]) -> List[str]:
    """学習に使用する安全な列のみを抽出する（§4, §6）。"""
    # 若手メモ: ここで列名を白リスト化しておくと、誤ってリーク列（例: charge_end_soc）が混入したときに即座に気づける。
    # include_stationをNoneにしているのは「データにstation_列が存在したら自動で採用」するための簡便なフラグ。
    time_candidates = [
        "dow",
        "hour_sin",
        "hour_cos",
        "time_raw_charge_start",
        "time_raw_timestamp",
        "time_band_2h",
        "time_dow_category",
    ]
    time_cols: List[str] = [col for col in time_candidates if col in df.columns]
    # time_ プレフィックスの列は可変（将来的に追加する可能性がある）ため動的に拾う。
    time_cols.extend(col for col in df.columns if col.startswith("time_") and col not in time_cols)
    daily_cluster_cols = [col for col in df.columns if col.startswith("daily_bin_") and col.endswith("_cluster")]
    daily_minutes_cols = [col for col in df.columns if col.startswith("daily_bin_") and col.endswith("_minutes")]
    time_cols.extend(daily_cluster_cols + daily_minutes_cols)

    vector_cols = [
        col for col in ["prev_to_charge_delta_lat_m", "prev_to_charge_delta_lon_m", "prev_to_charge_distance_m"] if col in df.columns
    ]

    core_cols = time_cols + [col for col in ["soc_start"] if col in df.columns] + vector_cols
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
        or col.startswith("transition_from_prev_to_")
    ]
    ordered_cols = core_cols + behavior_cols + station_cols + cluster_cols
    # 重複列名があれば最初の出現を優先し、後続は削除する。
    return list(dict.fromkeys(ordered_cols))


def _prepare_training_frame(df: pd.DataFrame, feature_cols: List[str], label: str) -> pd.DataFrame:
    """特徴列と目的変数だけを抽出したDataFrameを返す。"""
    keep_cols = list(dict.fromkeys(feature_cols + [label]))
    available_cols = [col for col in keep_cols if col in df.columns]
    data = df[available_cols].copy()
    if data.columns.duplicated().any():
        data = data.loc[:, ~data.columns.duplicated()]
    return data


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


def _to_native(value: object) -> object:
    """NumPyスカラなどをPython標準スカラへ変換するヘルパー。"""
    if isinstance(value, np.generic):
        return value.item()
    return value


def _compute_prediction_impacts(
    predictor: TabularPredictor,
    feature_cols: Sequence[str],
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    y_pred: pd.Series,
    proba_df: pd.DataFrame,
    top_features: Sequence[str],
) -> pd.DataFrame:
    """主要特徴を基準値に差し替えた際の確率差分（impact）を求める。"""
    if not top_features or test_features.empty or proba_df.empty:
        return pd.DataFrame(index=test_features.index)

    usable_features = [feat for feat in top_features if feat in feature_cols and feat in test_features.columns]
    if not usable_features:
        return pd.DataFrame(index=test_features.index)

    if train_features.empty:
        baseline_values = pd.Series(0.0, index=usable_features)
    else:
        baseline_values = train_features[usable_features].median()
    baseline_values = baseline_values.fillna(0.0)
    # 若手メモ: baselineは「その特徴を平均的な値に置き換えたらどうなるか」を評価する基準。
    # 学習データが空でも動くようにゼロ初期化しておくと、ハンドリング漏れで落ちる心配が少ない。

    class_columns = list(proba_df.columns)
    class_index_map = {label: idx for idx, label in enumerate(class_columns)}
    mapped = y_pred.map(class_index_map)
    if mapped.isnull().any():
        raise ValueError("Predicted labels are missing from probability columns.")
    # 若手メモ: AutoGluonのpredict/probaはモデルによって列順が異なるので、予測ラベルをインデックスに変換しておく。
    # mapがNaNを返したらラベル名の不一致＝致命的な整合性エラーなので即例外。
    target_indices = mapped.astype(int).to_numpy()
    row_indices = np.arange(len(test_features))
    base_probs = proba_df.to_numpy()
    base_scores = base_probs[row_indices, target_indices]

    impacts = pd.DataFrame(index=test_features.index)
    for feat in usable_features:
        baseline_val = baseline_values.get(feat, 0.0)
        modified = test_features.copy()
        modified[feat] = baseline_val
        # 若手メモ: 1列だけ基準値に差し替えたバージョンで再推論し、元の確率との差をimpactとして記録する。
        # ここではTop-1ラベルの確率だけを比較しており、確率が下がるほど「その特徴が予測を押し上げた」と解釈できる。
        proba_modified = predictor.predict_proba(modified, as_pandas=True)
        mod_scores = proba_modified.to_numpy()[row_indices, target_indices]
        impacts[f"impact_{feat}"] = base_scores - mod_scores
    return impacts


def _compute_class_stats(
    result: HashvinResult,
    y_true: pd.Series,
    y_pred: pd.Series,
) -> Dict[str, Dict[str, Optional[float]]]:
    """放置クラスタ別の件数・シェア・正答率などを集計して返す。"""
    features = result.features
    total_count = len(features)
    overall_counts = features["y_class"].value_counts()
    split_counts = {
        split: features.loc[features["split"] == split, "y_class"].value_counts() for split in ["train", "valid", "test"]
    }
    correct_mask = y_pred == y_true
    correct_counts = y_true[correct_mask].value_counts()
    incorrect_counts = y_true[~correct_mask].value_counts()

    class_labels = sorted(
        set(overall_counts.index)
        .union(correct_counts.index)
        .union(incorrect_counts.index)
        .union(result.head_clusters)
    )
    stats: Dict[str, Dict[str, Optional[float]]] = {}
    for cid in class_labels:
        overall = int(overall_counts.get(cid, 0))
        share = float(overall / total_count) if total_count > 0 else None
        train_count = int(split_counts["train"].get(cid, 0))
        valid_count = int(split_counts["valid"].get(cid, 0))
        test_count = int(split_counts["test"].get(cid, 0))
        correct = int(correct_counts.get(cid, 0))
        incorrect = int(incorrect_counts.get(cid, 0))
        accuracy = float(correct / test_count) if test_count > 0 else None
        stats[cid] = {
            "is_head": cid in result.head_clusters,
            "overall_count": overall,
            "overall_share": share,
            "train_count": train_count,
            "valid_count": valid_count,
            "test_count": test_count,
            "test_correct": correct,
            "test_incorrect": incorrect,
            "test_accuracy": accuracy,
        }
    return stats


def _save_predictions(
    df: pd.DataFrame,
    y_pred: pd.Series,
    proba_df: pd.DataFrame,
    explanation_df: Optional[pd.DataFrame],
    output_path: Path,
) -> None:
    """テストデータの予測結果と確率をCSV出力（§7, §8の可観測性向上）。"""
    # 若手メモ: session_uidと充電時刻をセットで残すことで、運用時の「どのセッション？」追跡が容易になる。
    # 予測確率にimpact列を結合すると説明資料が作りやすくなるため、explanation_dfは後から横結合している。
    export_df = pd.DataFrame(index=df.index)
    if "session_uid" in df.columns:
        export_df["session_uid"] = df["session_uid"]
    else:
        export_df["session_uid"] = df.index.astype(str)

    if "charge_start_time" in df.columns:
        export_df["charge_start_time"] = df["charge_start_time"]
    else:
        export_df["charge_start_time"] = pd.NaT

    if "charge_end_time" in df.columns:
        export_df["charge_end_time"] = df["charge_end_time"]

    export_df["y_true"] = df.get("y_class", pd.Series(index=df.index, dtype=object))
    export_df["y_pred"] = y_pred
    if not proba_df.empty:
        export_df["max_proba"] = proba_df.max(axis=1)
        joined = pd.concat([export_df, proba_df], axis=1)
    else:
        joined = export_df
    if explanation_df is not None and not explanation_df.empty:
        joined = pd.concat([joined, explanation_df], axis=1)
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
    # DataFrameに重複列が残っていれば除去し、列名の整合性を保つ。
    for frame in [train_model_df, valid_model_df, test_model_df]:
        if frame is not None:
            dup_cols = frame.columns[frame.columns.duplicated()]
            if not dup_cols.empty:
                frame.drop(columns=list(dup_cols), inplace=True)
    train_model_df = train_model_df.loc[:, ~train_model_df.columns.duplicated()]
    if valid_model_df is not None:
        valid_model_df = valid_model_df.loc[:, ~valid_model_df.columns.duplicated()]
    if test_model_df is not None:
        test_model_df = test_model_df.loc[:, ~test_model_df.columns.duplicated()]

    feature_cols_clean = [col for col in dict.fromkeys(feature_cols) if col in train_model_df.columns]
    train_features_only = train_model_df[feature_cols_clean]
    # 若手メモ: AutoGluon用のDataFrameは学習と推論で同じ列順を保証することが重要。
    # train_features_onlyはのちほど影響度計算に再利用するので破壊しないこと。

    metrics: Dict[str, object] = {
        "hashvin": result.hashvin,
        "head_clusters": result.head_clusters,
        "feature_columns": feature_cols_clean,
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

    feature_importance_df: Optional[pd.DataFrame] = None
    # 若手メモ: feature_importanceは学習完了後すぐに取得し、モデルの振る舞いをチームで共有できる形（CSV）に残す。
    # AutoGluonは稀にモデル構成次第でfeature_importanceを出せないため、try/exceptで握りつぶさずmetricsに理由を記録する。
    try:
        feature_importance_df = predictor.feature_importance(train_model_df, silent=True)
    except Exception as exc:
        metrics["feature_importance_error"] = str(exc)
    else:
        fi_path = output_dir / "feature_importance.csv"
        feature_importance_df.to_csv(fi_path, index=True)
        metrics["feature_importance_path"] = str(fi_path)
        top_importance = feature_importance_df["importance"].head(10)
        metrics["feature_importance_top"] = {str(idx): float(_to_native(val)) for idx, val in top_importance.items()}

    explanation_top_k = 5
    if feature_importance_df is not None and not feature_importance_df.empty:
        ranked_features = [feat for feat in feature_importance_df.index.tolist() if feat in feature_cols]
    else:
        ranked_features = []
    top_explain_features: List[str] = ranked_features[:explanation_top_k] if ranked_features else feature_cols[:explanation_top_k]

    if test_model_df is None or test_model_df.empty:
        metrics["note"] = "No test data available for evaluation."
        (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        return metrics

    y_pred = predictor.predict(test_model_df, as_pandas=True)
    proba_df = predictor.predict_proba(test_model_df, as_pandas=True)
    test_features_only = test_model_df[feature_cols]

    y_true = test_model_df[label_col]
    strict_acc = float(accuracy_score(y_true, y_pred))
    strict_balanced_acc = float(balanced_accuracy_score(y_true, y_pred))
    head_only_mask = y_true.isin(result.head_clusters)
    if head_only_mask.sum() > 0:
        head_only_acc = float(accuracy_score(y_true[head_only_mask], y_pred[head_only_mask]))
        head_only_balanced_acc = float(balanced_accuracy_score(y_true[head_only_mask], y_pred[head_only_mask]))
    else:
        head_only_acc = None
        head_only_balanced_acc = None

    class_recalls = _compute_class_recalls(y_true, y_pred, result.head_clusters)
    confusion = _compute_head_confusion(y_true, y_pred, result.head_clusters)
    threshold_metrics = _compute_threshold_metrics(y_true, y_pred, proba_df, eval_thresholds)
    class_stats = _compute_class_stats(result, y_true, y_pred)

    # §7: 必須指標（Top-1, クラス別Recall, 混同行列）と任意指標（Top-1@τ）を取りまとめる
    metrics.update(
        {
            "strict_top1_accuracy": strict_acc,
            "head_only_top1_accuracy": head_only_acc,
            "strict_balanced_accuracy": strict_balanced_acc,
            "head_only_balanced_accuracy": head_only_balanced_acc,
            "class_recalls": class_recalls,
            "confusion_matrix": confusion,
            "threshold_metrics": threshold_metrics,
            "class_stats": class_stats,
        }
    )

    leaderboard_df: Optional[pd.DataFrame] = None
    # 若手メモ: leaderboardは複数モデルを束ねるAutoGluonの「どのサブモデルが効いたか」の一覧。
    # extra_info=Trueにすると学習時間なども記録でき、後続のチューニング判断に役立つ。
    try:
        leaderboard_df = predictor.leaderboard(test_model_df, extra_info=True, silent=True)
    except Exception as exc:
        metrics["leaderboard_error"] = str(exc)
    else:
        leaderboard_path = output_dir / "model_leaderboard.csv"
        leaderboard_df.to_csv(leaderboard_path, index=False)
        metrics["model_leaderboard_path"] = str(leaderboard_path)
        metrics["model_leaderboard_top"] = [
            {k: _to_native(v) for k, v in row.items()}
            for row in leaderboard_df.head(5).to_dict(orient="records")
        ]

    explanation_df: Optional[pd.DataFrame] = None
    # 若手メモ: 予測理由を定量化するのがOJTの肝。impact計算は推論時間が伸びるので、障害が出ても握りつぶさずmetricsに残す。
    try:
        explanation_df = _compute_prediction_impacts(
            predictor,
            feature_cols_clean,
            train_features_only,
            test_features_only,
            y_pred,
            proba_df,
            top_explain_features,
        )
    except Exception as exc:
        metrics["explanation_error"] = str(exc)
        explanation_df = None
    else:
        if explanation_df is not None and not explanation_df.empty:
            metrics["explanation_features"] = top_explain_features


    # §7・§8: テストセットの予測詳細をCSVで保存し、分析できる状態にする
    _save_predictions(test_df, y_pred, proba_df, explanation_df, output_dir / "predictions_test.csv")

    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return metrics
