"""特徴量生成のロジックをまとめたモジュール。"""

from __future__ import annotations

from typing import Dict, List, TYPE_CHECKING

import numpy as np
import pandas as pd

from .candidates import generate_candidates
from .config import RankingConfig
from .utils import bearing_degree, gaussian_kernel, haversine_km

if TYPE_CHECKING:  # 循環インポートを避けるため型チェック時のみ読み込む
    from .pipeline import UserData


def _lookup(
    df: pd.DataFrame,
    keys: Dict[str, object],
    value_col: str,
    default: float = 0.0,
) -> float:
    """複数キーで DataFrame を検索し、該当する値を返す小さなヘルパー。"""

    if df.empty:
        return default
    query = pd.Series(True, index=df.index)
    for col, val in keys.items():
        query &= df[col] == val
    result = df.loc[query, value_col]
    if result.empty:
        return default
    return float(result.iloc[0])


def build_feature_row(
    candidate: str,
    event: pd.Series,
    user_data: "UserData",
    config: RankingConfig,
) -> Dict[str, object]:
    """1 つの候補クラスタについて学習に使う特徴量を組み立てる。"""

    weekday = int(event["weekday"])
    hour = int(event["charge_start_hour"])
    charge_cluster = str(event["charge_cluster"])

    # 滞在統計（presence）: 曜日×時間帯でどれだけ停めているか
    presence_prob = _lookup(
        user_data.presence,
        {"weekday": weekday, "hour": hour, "cluster": candidate},
        "presence_prob",
    )
    presence_weight = _lookup(
        user_data.presence,
        {"weekday": weekday, "hour": hour, "cluster": candidate},
        "presence_weight",
    )
    long_ratio = _lookup(
        user_data.presence,
        {"cluster": candidate},
        "long_park_ratio",
    )

    # 長時間放置の開始確率
    start_prob = _lookup(
        user_data.start_prob,
        {"weekday": weekday, "hour": hour, "cluster": candidate},
        "start_prob",
    )

    # 充電クラスタから放置クラスタへの遷移確率（なければ時間帯ベースで補完）
    charge_prob = _lookup(
        user_data.charge_prior,
        {
            "charge_cluster": charge_cluster,
            "weekday": weekday,
            "hour": hour,
            "park_cluster": candidate,
        },
        "prob",
    )
    hour_prob = _lookup(
        user_data.hour_prior,
        {"weekday": weekday, "hour": hour, "park_cluster": candidate},
        "prob",
    )
    if charge_prob == 0.0:
        charge_prob = hour_prob

    profile = user_data.cluster_profile[
        user_data.cluster_profile["cluster"] == candidate
    ]
    mean_lat = profile["mean_lat"].iloc[0] if not profile.empty else np.nan
    mean_lon = profile["mean_lon"].iloc[0] if not profile.empty else np.nan
    peak_hour = profile["peak_hour"].iloc[0] if not profile.empty else np.nan
    peak_std = profile["peak_hour_std"].iloc[0] if not profile.empty else np.nan

    charge_profile = user_data.cluster_profile[
        user_data.cluster_profile["cluster"] == charge_cluster
    ]
    charge_lat = (
        charge_profile["mean_lat"].iloc[0]
        if not charge_profile.empty
        else event.get("charge_end_lat", np.nan)
    )
    charge_lon = (
        charge_profile["mean_lon"].iloc[0]
        if not charge_profile.empty
        else event.get("charge_end_lon", np.nan)
    )

    dist = haversine_km(charge_lat, charge_lon, mean_lat, mean_lon)
    bearing = bearing_degree(charge_lat, charge_lon, mean_lat, mean_lon)

    # クラスタごとの集中時間帯と現在時刻の相性をガウスカーネルで評価
    delta_to_peak = np.nan
    compat_time = 0.0
    if pd.notnull(peak_hour):
        diff = abs(hour - peak_hour)
        delta_to_peak = min(diff, 24 - diff)
        compat_time = (
            gaussian_kernel(delta_to_peak, config.kernel_sigma_hour) * presence_prob
        )

    # ルールベースのスコア（滞在・遷移・開始確率を重み付けし距離で減点）
    score = (
        config.w_routine * presence_prob
        + config.w_charge * charge_prob
        + config.lambda_start * start_prob
        - config.gamma_distance * (dist if not np.isnan(dist) else 0.0)
    )

    prev_cluster = event.get("prev_charge_cluster")
    prev_same = 1 if pd.notnull(prev_cluster) and str(prev_cluster) == candidate else 0

    # 時刻・曜日の周期性はサイン・コサインに変換
    sin_hour = np.sin(2 * np.pi * hour / 24)
    cos_hour = np.cos(2 * np.pi * hour / 24)
    sin_week = np.sin(2 * np.pi * weekday / 7)
    cos_week = np.cos(2 * np.pi * weekday / 7)

    return {
        "hashvin": event["hashvin"],
        "event_id": event["event_id"],
        "candidate_cluster": candidate,
        "label": 1
        if pd.notnull(event.get("park_cluster"))
        and str(event["park_cluster"]) == candidate
        else 0,
        "weekday": weekday,
        "charge_cluster": charge_cluster,
        "charge_start_hour": hour,
        "weight_time": event["weight_time"],
        "age_days": event["age_days"],
        "presence_prob": presence_prob,
        "presence_weight": presence_weight,
        "long_park_ratio": long_ratio,
        "start_prob": start_prob,
        "charge_prior_prob": charge_prob,
        "hour_prior_prob": hour_prob,
        "compat_time": compat_time,
        "delta_to_peak": delta_to_peak,
        "peak_hour": peak_hour,
        "peak_hour_std": peak_std,
        "dist_km": dist,
        "bearing_deg": bearing,
        "candidate_score": score,
        "start_soc": event.get("start_soc"),
        "time_since_last_charge_min": event.get("time_since_last_charge_min"),
        "soc_drop_since_prev": event.get("soc_drop_since_prev"),
        "prev_charge_cluster": prev_cluster if pd.isna(prev_cluster) else str(prev_cluster),
        "prev_same_candidate": prev_same,
        "sin_hour": sin_hour,
        "cos_hour": cos_hour,
        "sin_week": sin_week,
        "cos_week": cos_week,
    }


def build_training_table(user_data: "UserData", config: RankingConfig) -> pd.DataFrame:
    """充電イベントを候補クラスタごとの行に展開し、学習テーブルを作成する。"""

    rows: List[Dict[str, object]] = []
    for _, event in user_data.links.iterrows():
        candidates = generate_candidates(event, user_data, config)
        for candidate in candidates:
            rows.append(build_feature_row(candidate, event, user_data, config))
    return pd.DataFrame(rows)
