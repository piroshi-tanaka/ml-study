"""候補クラスタを抽出するロジック。"""

from __future__ import annotations

from typing import List, TYPE_CHECKING

import pandas as pd

from .config import RankingConfig
from .utils import haversine_km

if TYPE_CHECKING:
    from .pipeline import UserData


def generate_candidates(
    event: pd.Series,
    user_data: "UserData",
    config: RankingConfig,
) -> List[str]:
    """充電イベントごとに候補クラスタを文脈に応じて抽出する。"""

    weekday = int(event["weekday"])
    hour = int(event["charge_start_hour"])
    charge_cluster = str(event["charge_cluster"])

    # 滞在統計: 曜日×時間帯でよく停めているクラスタ
    routine = user_data.presence[
        (user_data.presence["weekday"] == weekday)
        & (user_data.presence["hour"] == hour)
    ]
    routine_candidates = [
        str(c)
        for c in routine.sort_values("presence_prob", ascending=False)["cluster"]
        .head(config.m_routine)
        .tolist()
    ]

    # 長時間放置が始まりやすいクラスタ（start_prob）
    start_df = user_data.start_prob[
        (user_data.start_prob["weekday"] == weekday)
        & (user_data.start_prob["hour"] == hour)
    ]
    start_candidates = [
        str(c)
        for c in start_df.sort_values("start_prob", ascending=False)["cluster"]
        .head(config.m_routine)
        .tolist()
    ]

    # 同じ充電クラスタから過去に実際放置したクラスタ
    charge_df = user_data.charge_prior[
        (user_data.charge_prior["weekday"] == weekday)
        & (user_data.charge_prior["hour"] == hour)
        & (user_data.charge_prior["charge_cluster"] == charge_cluster)
    ]
    charge_candidates = [
        str(c)
        for c in charge_df.sort_values("prob", ascending=False)["park_cluster"]
        .head(config.n_charge_prior)
        .tolist()
    ]

    # 地理的に近いクラスタも補完として追加
    nearby_candidates: List[str] = []
    profile = user_data.cluster_profile
    if not profile.empty:
        charge_profile = profile[profile["cluster"] == charge_cluster]
        if not charge_profile.empty:
            lat = charge_profile["mean_lat"].iloc[0]
            lon = charge_profile["mean_lon"].iloc[0]
            distances = profile.apply(
                lambda r: haversine_km(lat, lon, r["mean_lat"], r["mean_lon"]), axis=1
            )
            nearby_candidates = (
                profile.assign(dist=distances)
                .query("dist <= @config.nearby_radius_km")
                .sort_values("dist")["cluster"]
                .head(config.l_nearby)
                .astype(str)
                .tolist()
            )

    union = routine_candidates + start_candidates + charge_candidates + nearby_candidates
    if pd.notnull(event.get("park_cluster")):
        # 学習データでは正解クラスタを必ず含める
        union.append(str(event["park_cluster"]))

    seen: List[str] = []
    for candidate in union:
        if pd.isna(candidate):
            continue
        candidate_str = str(candidate)
        if candidate_str not in seen:
            seen.append(candidate_str)
        if len(seen) >= config.k_candidates:
            break

    return seen
