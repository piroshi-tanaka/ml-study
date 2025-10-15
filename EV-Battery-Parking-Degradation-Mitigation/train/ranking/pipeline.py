"""hashvin ごとのランキング学習・推論ユーティリティ。"""

from __future__ import annotations


import json


from dataclasses import dataclass


from pathlib import Path


from typing import Dict, Iterable, List, Optional, Tuple, Union


import numpy as np


import pandas as pd


from autogluon.tabular import TabularPredictor


from .config import RankingConfig


from .metrics import calculate_ranking_metrics


from .utils import bearing_degree, gaussian_kernel, haversine_km


# ---------------------------------------------------------------------------


# 共通前処理


# ---------------------------------------------------------------------------


def load_sessions(csv_path: Path) -> pd.DataFrame:
    """セッションCSVから学習に必要な共通派生列を作成して返す。"""

    df = pd.read_csv(csv_path)
    # 入力直後に開始・終了時刻を日本時間へ統一し、タイムゾーン差異を吸収する

    for col in ["start_time", "end_time"]:
        ts = pd.to_datetime(df[col], errors="coerce")

        if pd.api.types.is_datetime64tz_dtype(ts.dtype):
            df[col] = ts.dt.tz_convert("Asia/Tokyo").dt.tz_localize(None)

        else:
            df[col] = ts.dt.tz_localize(
                "Asia/Tokyo", nonexistent="shift_forward", ambiguous="NaT"
            ).dt.tz_localize(None)

    df = df.sort_values(["hashvin", "start_time"]).reset_index(drop=True)
    # 滞在時間や曜日・開始時刻などモデルで使う派生列をまとめて追加する

    df["duration_minutes"] = pd.to_numeric(df["duration_minutes"], errors="coerce")

    df["weekday"] = df["start_time"].dt.dayofweek

    df["start_hour"] = df["start_time"].dt.hour

    df["date"] = df["start_time"].dt.date

    df["is_long_park"] = (df["session_type"] == "inactive") & (
        df["duration_minutes"] >= 360
    )

    return df


# ---------------------------------------------------------------------------


# ユーザー（hashvin）単位の統計データ


# ---------------------------------------------------------------------------


@dataclass
class UserData:
    """ユーザー単位で学習に必要なテーブルを保持するデータクラス。"""

    hashvin: str

    sessions: pd.DataFrame

    links: pd.DataFrame

    presence: pd.DataFrame

    start_prob: pd.DataFrame

    charge_prior: pd.DataFrame

    hour_prior: pd.DataFrame

    cluster_profile: pd.DataFrame


class UserDataBuilder:
    """hashvin ごとの統計テーブルを構築するヘルパー。"""

    def __init__(self, sessions: pd.DataFrame, config: RankingConfig) -> None:
        self.sessions = sessions.copy()

        self.config = config

        self.ref_time = self.sessions["start_time"].max()

        self._annotate_time_weight()

    # ---- 内部ユーティリティ --------------------------------------------------

    def _annotate_time_weight(self) -> None:
        """イベントの経過日数と時間減衰重みを計算する。"""

        # 参照時刻からの経過日数を日単位で計算し、減衰重みの基礎とする
        age_days = (
            self.ref_time - self.sessions["start_time"]
        ).dt.total_seconds() / 86400

        self.sessions["age_days"] = age_days

        if self.config.use_decay_weight and self.config.halflife_days > 0:
            # 半減期パラメータに従って、古いイベントほど指数的に重みを下げる
            weight = np.exp(-np.log(2) * age_days / self.config.halflife_days)

        else:
            weight = np.ones(len(self.sessions))

        if self.config.window_days > 0:
            # 指定期間より前のイベントは学習対象から除外するため重み0にする
            weight = np.where(age_days <= self.config.window_days, weight, 0.0)

        self.sessions["time_weight"] = weight

    # ---- テーブル構築 --------------------------------------------------

    def build_links(self) -> pd.DataFrame:
        """充電イベントに続く最初の長時間放置をリンクしたテーブルを作成する。"""

        charges = self.sessions[self.sessions["session_type"] == "charging"].copy()

        long_parks = self.sessions[self.sessions["is_long_park"]].copy()

        rows: List[Dict[str, object]] = []

        prev_charge_cluster = None

        prev_charge_end = None

        prev_end_soc = None

        event_id = 0

        for _, charge in charges.iterrows():
            event_id += 1

            start = charge["start_time"]

            end = charge["end_time"]

            subsequent_charges = charges[charges["start_time"] > end]

            next_charge_start = (
                subsequent_charges["start_time"].min()
                if not subsequent_charges.empty
                else None
            )

            if next_charge_start is not None:
                candidate_long = long_parks[
                    (long_parks["start_time"] >= end)
                    & (long_parks["start_time"] < next_charge_start)
                ]

            else:
                candidate_long = long_parks[long_parks["start_time"] >= end]

            candidate_long = candidate_long.sort_values("start_time")

            first_long = candidate_long.head(1)

            park_cluster = (
                first_long["session_cluster"].iloc[0]
                if not first_long.empty
                else np.nan
            )

            park_start = (
                first_long["start_time"].iloc[0] if not first_long.empty else pd.NaT
            )

            park_end = (
                first_long["end_time"].iloc[0] if not first_long.empty else pd.NaT
            )

            park_duration = (
                first_long["duration_minutes"].iloc[0]
                if not first_long.empty
                else np.nan
            )

            park_lat = (
                first_long["start_lat"].iloc[0] if not first_long.empty else np.nan
            )

            park_lon = (
                first_long["start_lon"].iloc[0] if not first_long.empty else np.nan
            )

            gap_minutes = (
                (park_start - end).total_seconds() / 60
                if pd.notnull(park_start)
                else np.nan
            )

            dist = haversine_km(
                charge.get("end_lat", np.nan),
                charge.get("end_lon", np.nan),
                park_lat,
                park_lon,
            )

            age_days = (self.ref_time - start).total_seconds() / 86400

            weight = (
                np.exp(-np.log(2) * age_days / self.config.halflife_days)
                if (self.config.use_decay_weight and self.config.halflife_days > 0)
                else 1.0
            )

            if self.config.window_days > 0 and age_days > self.config.window_days:
                weight = 0.0

            time_since_last = (
                (start - prev_charge_end).total_seconds() / 60
                if prev_charge_end is not None
                else np.nan
            )

            soc_drop = (
                (prev_end_soc - charge.get("start_soc", np.nan))
                if prev_end_soc is not None
                else np.nan
            )

            rows.append(
                {
                    "hashvin": charge["hashvin"],
                    "event_id": event_id,
                    "weekday": charge["weekday"],
                    "charge_cluster": str(charge["session_cluster"]),
                    "charge_start_time": start,
                    "charge_start_hour": charge["start_hour"],
                    "charge_end_time": end,
                    "charge_end_lat": charge.get("end_lat", np.nan),
                    "charge_end_lon": charge.get("end_lon", np.nan),
                    "park_cluster": park_cluster
                    if pd.isna(park_cluster)
                    else str(park_cluster),
                    "park_start_time": park_start,
                    "park_start_hour": park_start.hour
                    if pd.notnull(park_start)
                    else np.nan,
                    "park_duration_minutes": park_duration,
                    "park_start_lat": park_lat,
                    "park_start_lon": park_lon,
                    "gap_minutes": gap_minutes,
                    "dist_charge_to_park_km": dist,
                    "age_days": age_days,
                    "weight_time": weight,
                    "start_soc": charge.get("start_soc", np.nan),
                    "end_soc": charge.get("end_soc", np.nan),
                    "time_since_last_charge_min": time_since_last,
                    "soc_drop_since_prev": soc_drop,
                    "prev_charge_cluster": prev_charge_cluster
                    if prev_charge_cluster is None
                    else str(prev_charge_cluster),
                }
            )

            prev_charge_cluster = charge["session_cluster"]

            prev_charge_end = end

            prev_end_soc = charge.get("end_soc", np.nan)

        return pd.DataFrame(rows)

    def build_presence(self) -> pd.DataFrame:
        """曜日×時間帯ごとの存在確率テーブルを作成する。"""

        records: List[Dict[str, object]] = []
        # 6時間以上の放置で減衰重みが正の行だけを取り出す
        long_df = self.sessions[
            self.sessions["is_long_park"] & (self.sessions["time_weight"] > 0)
        ]

        for _, row in long_df.iterrows():
            start = row["start_time"]

            end = row["end_time"]

            weight = row["time_weight"]

            current = start.floor("H")

            if current > start:
                current -= pd.Timedelta(hours=1)

            while current < end:
                nxt = current + pd.Timedelta(hours=1)
                # 1時間ごとの重なりを分計算で取得し、滞在時間として積算する
                overlap = min(end, nxt) - max(start, current)

                minutes = overlap.total_seconds() / 60

                if minutes > 0:
                    weekday = (max(start, current)).dayofweek

                    records.append(
                        {
                            "weekday": weekday,
                            "hour": current.hour,
                            "cluster": str(row["session_cluster"]),
                            "weight": weight * (minutes / 60),
                        }
                    )

                current = nxt

        if not records:
            return pd.DataFrame(
                columns=[
                    "weekday",
                    "hour",
                    "cluster",
                    "presence_weight",
                    "presence_prob",
                    "long_park_ratio",
                ]
            )

        df = pd.DataFrame(records)

        grouped = (
            df.groupby(["weekday", "hour", "cluster"], as_index=False)["weight"]
            .sum()
            .rename(columns={"weight": "presence_weight"})
        )

        totals = (
            grouped.groupby(["weekday", "hour"], as_index=False)["presence_weight"]
            .sum()
            .rename(columns={"presence_weight": "total_weight"})
        )

        merged = grouped.merge(totals, on=["weekday", "hour"], how="left")

        alpha = self.config.alpha_smooth

        clusters = grouped["cluster"].nunique()

        # 平滑化しながら「曜日×時間帯にそのクラスタへ滞在する確率」をpresence_probとして持つ
        merged["presence_prob"] = (merged["presence_weight"] + alpha) / (
            merged["total_weight"] + alpha * clusters
        )

        cluster_total = (
            merged.groupby("cluster")["presence_weight"]
            .sum()
            .rename("cluster_total")
            .reset_index()
        )

        merged = merged.merge(cluster_total, on="cluster", how="left")

        total_sum = cluster_total["cluster_total"].sum()

        # クラスタ全体における滞在比率（どの放置先がホーム的か）も保持する
        merged["long_park_ratio"] = (
            merged["cluster_total"] / total_sum if total_sum > 0 else 0.0
        )

        return merged.drop(columns=["total_weight", "cluster_total"], errors="ignore")

    def build_start_prob(self) -> pd.DataFrame:
        """曜日×開始時刻ごとの長時間放置開始確率を計算する。"""

        df = self.sessions[
            self.sessions["is_long_park"] & (self.sessions["time_weight"] > 0)
        ]

        if df.empty:
            return pd.DataFrame(
                columns=["weekday", "hour", "cluster", "start_prob", "start_weight"]
            )

        grouped = df.groupby(
            ["weekday", "start_hour", "session_cluster"], as_index=False
        )["time_weight"].sum()

        grouped = grouped.rename(
            columns={
                "start_hour": "hour",
                "session_cluster": "cluster",
                "time_weight": "start_weight",
            }
        )

        grouped["cluster"] = grouped["cluster"].astype(str)

        totals = (
            grouped.groupby(["weekday", "hour"], as_index=False)["start_weight"]
            .sum()
            .rename(columns={"start_weight": "total_weight"})
        )

        merged = grouped.merge(totals, on=["weekday", "hour"], how="left")

        alpha = self.config.alpha_smooth

        clusters = grouped["cluster"].nunique()

        # 曜日×開始時刻ごとに「そのクラスタから放置を始めた重み付き件数」を確率化する
        merged["start_prob"] = (merged["start_weight"] + alpha) / (
            merged["total_weight"] + alpha * clusters
        )

        return merged.drop(columns=["total_weight"], errors="ignore")

    def build_charge_prior(self, links: pd.DataFrame) -> pd.DataFrame:
        """充電クラスタ×時間帯で条件付けした放置先確率を作成する。"""

        df = links.dropna(subset=["park_cluster"]).copy()

        if df.empty:
            return pd.DataFrame(
                columns=[
                    "charge_cluster",
                    "weekday",
                    "hour",
                    "park_cluster",
                    "prob",
                    "weight",
                ]
            )

        df["charge_cluster"] = df["charge_cluster"].astype(str)

        df["park_cluster"] = df["park_cluster"].astype(str)

        grouped = df.groupby(
            ["charge_cluster", "weekday", "charge_start_hour", "park_cluster"],
            as_index=False,
        )["weight_time"].sum()

        grouped = grouped.rename(
            columns={"charge_start_hour": "hour", "weight_time": "weight"}
        )

        totals = (
            grouped.groupby(["charge_cluster", "weekday", "hour"], as_index=False)[
                "weight"
            ]
            .sum()
            .rename(columns={"weight": "total_weight"})
        )

        merged = grouped.merge(
            totals, on=["charge_cluster", "weekday", "hour"], how="left"
        )

        alpha = self.config.alpha_smooth

        count = (
            grouped.groupby(["charge_cluster", "weekday", "hour"])["park_cluster"]
            .nunique()
            .rename("cluster_count")
            .reset_index()
        )

        merged = merged.merge(
            count, on=["charge_cluster", "weekday", "hour"], how="left"
        )

        # 特定の充電クラスタからどの放置クラスタへ流れやすいかを条件付き確率で保持する
        merged["prob"] = (merged["weight"] + alpha) / (
            merged["total_weight"] + alpha * merged["cluster_count"]
        )

        return merged.drop(columns=["total_weight"], errors="ignore")

    def build_hour_prior(self, links: pd.DataFrame) -> pd.DataFrame:
        """時間帯のみで条件付けした放置先確率を作成する。"""

        df = links.dropna(subset=["park_cluster"]).copy()

        if df.empty:
            return pd.DataFrame(
                columns=["weekday", "hour", "park_cluster", "prob", "weight"]
            )

        df["park_cluster"] = df["park_cluster"].astype(str)

        grouped = df.groupby(
            ["weekday", "charge_start_hour", "park_cluster"], as_index=False
        )["weight_time"].sum()

        grouped = grouped.rename(
            columns={"charge_start_hour": "hour", "weight_time": "weight"}
        )

        totals = (
            grouped.groupby(["weekday", "hour"], as_index=False)["weight"]
            .sum()
            .rename(columns={"weight": "total_weight"})
        )

        merged = grouped.merge(totals, on=["weekday", "hour"], how="left")

        alpha = self.config.alpha_smooth

        clusters = grouped["park_cluster"].nunique()

        merged["prob"] = (merged["weight"] + alpha) / (
            merged["total_weight"] + alpha * clusters
        )

        return merged.drop(columns=["total_weight"], errors="ignore")

    def build_cluster_profile(self) -> pd.DataFrame:
        """クラスタごとの代表座標・代表時刻を算出する。"""

        df = self.sessions[self.sessions["is_long_park"]].copy()

        if df.empty:
            return pd.DataFrame(
                columns=[
                    "cluster",
                    "mean_lat",
                    "mean_lon",
                    "peak_hour",
                    "peak_hour_std",
                ]
            )

        profile = (
            df.groupby("session_cluster")
            .agg(
                mean_lat=("start_lat", "mean"),
                mean_lon=("start_lon", "mean"),
                peak_hour=("start_hour", lambda s: s.value_counts().idxmax()),
                peak_hour_std=("start_hour", "std"),
            )
            .reset_index()
            .rename(columns={"session_cluster": "cluster"})
        )

        profile["cluster"] = profile["cluster"].astype(str)

        return profile

    def build(self) -> UserData:
        """各種テーブルをまとめた UserData を返す。"""

        links = self.build_links()

        presence = self.build_presence()

        start_prob = self.build_start_prob()

        charge_prior = self.build_charge_prior(links)

        hour_prior = self.build_hour_prior(links)

        cluster_profile = self.build_cluster_profile()

        return UserData(
            hashvin=self.sessions["hashvin"].iloc[0],
            sessions=self.sessions,
            links=links,
            presence=presence,
            start_prob=start_prob,
            charge_prior=charge_prior,
            hour_prior=hour_prior,
            cluster_profile=cluster_profile,
        )


# ---------------------------------------------------------------------------


# 候補生成と特徴量化


# ---------------------------------------------------------------------------


def _lookup(
    df: pd.DataFrame, keys: Dict[str, object], value_col: str, default: float = 0.0
) -> float:
    """指定した条件で DataFrame を検索し、値を取得するヘルパー。"""

    if df.empty:
        return default

    query = df

    for key, val in keys.items():
        query = query[query[key] == val]

        if query.empty:
            return default

    return float(query.iloc[0][value_col])


def generate_candidates(
    event: pd.Series, user_data: UserData, config: RankingConfig
) -> List[str]:
    """滞在頻度・開始確率・距離の3観点から候補クラスタを集める。"""

    weekday = int(event["weekday"])

    hour = int(event["charge_start_hour"])

    charge_cluster = str(event["charge_cluster"])

    # 滞在時間（presence_prob）が高い候補を曜日×時間帯から取得
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

    start_df = user_data.start_prob[
        (user_data.start_prob["weekday"] == weekday)
        & (user_data.start_prob["hour"] == hour)
    ]

    # 長時間放置の開始回数（start_prob）が多い候補を追加
    start_candidates = [
        str(c)
        for c in start_df.sort_values("start_prob", ascending=False)["cluster"]
        .head(config.m_routine)
        .tolist()
    ]

    charge_df = user_data.charge_prior[
        (user_data.charge_prior["weekday"] == weekday)
        & (user_data.charge_prior["hour"] == hour)
        & (user_data.charge_prior["charge_cluster"] == charge_cluster)
    ]

    # 同じ充電クラスタから過去に実際放置したクラスタを優先
    charge_candidates = [
        str(c)
        for c in charge_df.sort_values("prob", ascending=False)["park_cluster"]
        .head(config.n_charge_prior)
        .tolist()
    ]

    nearby_candidates: List[str] = []

    profile = user_data.cluster_profile

    if not profile.empty:
        charge_profile = profile[profile["cluster"] == charge_cluster]

        if not charge_profile.empty:
            lat = charge_profile["mean_lat"].iloc[0]

            lon = charge_profile["mean_lon"].iloc[0]

            # 地理的に近い放置先も補完として追加する
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

    union = (
        routine_candidates + start_candidates + charge_candidates + nearby_candidates
    )

    if pd.notnull(event.get("park_cluster")):
        union.append(str(event["park_cluster"]))

    seen: List[str] = []

    for c in union:
        if pd.isna(c):
            continue

        c_str = str(c)

        if c_str not in seen:
            seen.append(c_str)

        if len(seen) >= config.k_candidates:
            break

    return seen


def build_feature_row(
    candidate: str, event: pd.Series, user_data: UserData, config: RankingConfig
) -> Dict[str, object]:
    """候補クラスタに対する単一行の特徴量を作成する。"""

    weekday = int(event["weekday"])

    hour = int(event["charge_start_hour"])

    charge_cluster = str(event["charge_cluster"])

    # presence 系特徴量: 曜日×時間帯でどれだけ滞在しているか
    presence_prob = _lookup(
        user_data.presence,
        {"weekday": weekday, "hour": hour, "cluster": candidate},
        "presence_prob",
        0.0,
    )

    # presence_weight は滞在時間そのものを保持
    presence_weight = _lookup(
        user_data.presence,
        {"weekday": weekday, "hour": hour, "cluster": candidate},
        "presence_weight",
        0.0,
    )

    # long_park_ratio でホーム・職場など長時間滞在しやすさを表す
    long_ratio = _lookup(
        user_data.presence, {"cluster": candidate}, "long_park_ratio", 0.0
    )

    # start_prob は長時間放置の開始頻度を表す時系列特徴
    start_prob = _lookup(
        user_data.start_prob,
        {"weekday": weekday, "hour": hour, "cluster": candidate},
        "start_prob",
        0.0,
    )

    # charge_prior から同じ充電クラスタ→放置クラスタの過去実績確率を取得
    charge_prob = _lookup(
        user_data.charge_prior,
        {
            "charge_cluster": charge_cluster,
            "weekday": weekday,
            "hour": hour,
            "park_cluster": candidate,
        },
        "prob",
        0.0,
    )

    # hour_prior は時間帯のみの履歴で疎な場合のバックアップ
    hour_prob = _lookup(
        user_data.hour_prior,
        {"weekday": weekday, "hour": hour, "park_cluster": candidate},
        "prob",
        0.0,
    )

    if charge_prob == 0.0:
        # 充電クラスタに実績が無い場合は時間帯ベースの確率で補完する
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

    delta_to_peak = np.nan

    compat_time = 0.0

    if pd.notnull(peak_hour):
        diff = abs(hour - peak_hour)

        delta_to_peak = min(diff, 24 - diff)

        # peak_hour と現在時刻の近さをガウスカーネルで評価し、滞在確率と掛け合わせる
        compat_time = (
            gaussian_kernel(delta_to_peak, config.kernel_sigma_hour) * presence_prob
        )

    # ルールベーススコア: 滞在頻度・充電実績・開始頻度を重み付けし距離で減点
    score = (
        config.w_routine * presence_prob
        + config.w_charge * charge_prob
        + config.lambda_start * start_prob
        - config.gamma_distance * (dist if not np.isnan(dist) else 0.0)
    )

    prev_cluster = event.get("prev_charge_cluster")

    # 直前の充電クラスタと同じ候補かどうかで履歴継続フラグを作る
    prev_same = 1 if pd.notnull(prev_cluster) and str(prev_cluster) == candidate else 0

    # 時刻と曜日の周期性をサイン・コサインに変換して回帰しやすくする
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
        "prev_charge_cluster": prev_cluster
        if pd.isna(prev_cluster)
        else str(prev_cluster),
        "prev_same_candidate": prev_same,
        "sin_hour": sin_hour,
        "cos_hour": cos_hour,
        "sin_week": sin_week,
        "cos_week": cos_week,
    }


def build_training_table(user_data: UserData, config: RankingConfig) -> pd.DataFrame:
    """ユーザーの充電イベントを候補行に展開し、特徴量テーブルを返す。"""

    rows: List[Dict[str, object]] = []

    for _, event in user_data.links.iterrows():
        candidates = generate_candidates(event, user_data, config)

        for cand in candidates:
            rows.append(build_feature_row(cand, event, user_data, config))

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------


# 学習・評価・推論


# ---------------------------------------------------------------------------


@dataclass
class TrainedUserModel:
    """学習済みモデルと関連データをまとめたコンテナ。"""

    hashvin: str

    predictor: TabularPredictor

    features: List[str]

    train_data: pd.DataFrame

    validation_data: pd.DataFrame

    training_table: pd.DataFrame

    user_data: UserData


def train_user_model(
    training_table: pd.DataFrame,
    user_data: UserData,
    config: RankingConfig,
    model_root: Optional[Path] = None,
) -> Optional[TrainedUserModel]:
    """1ユーザー分の学習データから AutoGluon モデルを学習する。"""

    if training_table.empty or training_table["label"].sum() == 0:
        return None

    if training_table["label"].nunique() < 2:
        # 正例のみ（または負例のみ）では分類器が学習できないためスキップ

        return None

    dataset = training_table.sort_values("age_days").reset_index(drop=True)

    split_idx = max(1, int(len(dataset) * 0.8))

    train_df = dataset.iloc[:split_idx]

    val_df = dataset.iloc[split_idx:]

    if val_df.empty:
        val_df = train_df.copy()

    features = [c for c in dataset.columns if c not in {"label", "event_id", "hashvin"}]

    model_path: Optional[Path] = None

    if model_root is not None:
        model_path = model_root / f"autogluon_{user_data.hashvin}"

        model_path.mkdir(parents=True, exist_ok=True)

    predictor = TabularPredictor(
        label="label",
        problem_type="binary",
        path=str(model_path) if model_path else None,
        eval_metric="roc_auc",
    )

    try:
        predictor.fit(
            train_data=train_df[[*features, "label"]],
            tuning_data=val_df[[*features, "label"]] if not val_df.empty else None,
            time_limit=config.time_limit,
            presets=config.ag_presets,
        )

    except Exception:
        # モデルが学習できない場合は None を返して呼び出し元でスキップ

        return None

    return TrainedUserModel(
        hashvin=user_data.hashvin,
        predictor=predictor,
        features=features,
        train_data=train_df[[*features, "label"]].copy(),
        validation_data=val_df[["event_id", *features, "label"]].copy(),
        training_table=dataset,
        user_data=user_data,
    )


def evaluate_user_model(
    model: TrainedUserModel,
    topk: Iterable[int],
    return_scored: bool = False,
) -> Union[Dict[str, float], Tuple[Dict[str, float], pd.DataFrame]]:
    """検証データへスコアを付与し、ランキング評価指標を算出する。"""

    val_df = model.validation_data.copy()
    # AutoGluonの predict_proba は DataFrame または ndarray で返るため分岐させる
    proba = model.predictor.predict_proba(val_df[model.features])
    if isinstance(proba, pd.DataFrame):
        if 1 in proba.columns:
            val_df["score"] = proba[1].to_numpy()
        else:
            val_df["score"] = proba.iloc[:, -1].to_numpy()
    else:
        val_df["score"] = np.asarray(proba)

    metrics = calculate_ranking_metrics(
        val_df, score_col="score", group_col="event_id", label_col="label", topk=topk
    )
    return (metrics, val_df) if return_scored else metrics


def summarize_validation_scores(
    val_scored: pd.DataFrame,
    topk: Iterable[int],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """検証スコアをイベント単位・クラスタ単位に集計し、分析用テーブルを返す。"""

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

        return pd.DataFrame(columns=event_columns), pd.DataFrame(
            columns=cluster_columns
        )

    # イベント（充電×候補セット）単位でランキング結果を整理する
    for event_id, group in val_scored.groupby("event_id"):
        group_sorted = group.sort_values("score", ascending=False).reset_index(
            drop=True
        )

        candidate_count = len(group_sorted)

        actual_row = group_sorted[group_sorted["label"] == 1].head(1)

        actual_cluster = (
            str(actual_row.iloc[0]["candidate_cluster"])
            if not actual_row.empty
            else None
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

        top1_score = (
            float(group_sorted.iloc[0]["score"]) if candidate_count else float("nan")
        )

        max_k = topk_list[-1] if topk_list else min(3, candidate_count)

        top_candidates = ""

        top_scores = ""

        if candidate_count:
            top_subset = group_sorted.head(max_k)

            top_candidates = ",".join(
                top_subset["candidate_cluster"].astype(str).tolist()
            )

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
            "is_top1_correct": bool(
                actual_cluster is not None and top1_cluster == actual_cluster
            ),
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

    # 実際に放置されたクラスタごとの指標をまとめる
    for cluster, stats in cluster_stats.items():
        row: Dict[str, object] = {
            "cluster": cluster,
            "event_count": stats["event_count"],
            "share": stats["event_count"] / total_events
            if total_events
            else float("nan"),
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


def predict_topk_for_user(model: TrainedUserModel, top_k: int) -> pd.DataFrame:
    """学習済みモデルを用いて Top-k 候補を算出する。"""

    scoring_df = model.training_table.copy()

    proba = model.predictor.predict_proba(scoring_df[model.features])

    if isinstance(proba, pd.DataFrame):
        if 1 in proba.columns:
            scoring_df["score"] = proba[1].to_numpy()

        else:
            scoring_df["score"] = proba.iloc[:, -1].to_numpy()

    else:
        scoring_df["score"] = np.asarray(proba)

    records: List[Dict[str, object]] = []

    for event_id, group in scoring_df.groupby("event_id"):
        top_rows = group.sort_values("score", ascending=False).head(top_k)

        records.append(
            {
                "hashvin": model.hashvin,
                "event_id": event_id,
                "candidate_clusters": ",".join(
                    top_rows["candidate_cluster"].astype(str).tolist()
                ),
                "scores": ",".join(f"{s:.6f}" for s in top_rows["score"].tolist()),
            }
        )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------


# パイプライン統合（Notebook からの利用想定）


# ---------------------------------------------------------------------------


class RankingPipeline:
    """ノートブックから学習・推論をまとめて呼び出すためのラッパークラス。"""

    def __init__(
        self,
        sessions: pd.DataFrame,
        config: Optional[RankingConfig] = None,
        model_root: Optional[Path] = None,
    ) -> None:
        self.sessions = sessions

        self.config = config or RankingConfig()

        self.model_root = model_root

        self.result_root = (
            Path(self.config.result_root)
            if self.config.result_root is not None
            else None
        )

        self.user_models: Dict[str, TrainedUserModel] = {}

        self.metrics: Dict[str, Dict[str, float]] = {}

    def fit_all(self) -> Dict[str, Dict[str, float]]:
        """hashvinごとにモデルを学習・評価し、指標を集計して返す。"""

        self.user_models.clear()

        self.metrics.clear()

        for hashvin, df_user in self.sessions.groupby("hashvin"):
            builder = UserDataBuilder(df_user, self.config)

            user_data = builder.build()

            training_table = build_training_table(user_data, self.config)

            if training_table.empty:
                continue

            trained = train_user_model(
                training_table, user_data, self.config, self.model_root
            )

            if trained is None:
                continue

            metrics, val_scored = evaluate_user_model(
                trained, self.config.topk_eval, return_scored=True
            )

            self.user_models[hashvin] = trained

            self.metrics[hashvin] = metrics

            # 評価指標と検証スコアを保存し、後から分析できるようにする

            self._save_user_results(trained, metrics, val_scored)

        return self.metrics

    def predict_all(self, top_k: int = 3) -> pd.DataFrame:
        """学習済みモデルがあるユーザーについて Top-k 候補をまとめて返す。"""

        if not self.user_models:
            raise RuntimeError("先に fit_all() を実行してください。")

        frames = [
            predict_topk_for_user(model, top_k) for model in self.user_models.values()
        ]

        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def get_user_model(self, hashvin: str) -> Optional[TrainedUserModel]:
        """特定ユーザーの学習済みモデルを取得する。"""

        return self.user_models.get(hashvin)

    def _save_user_results(
        self,
        model: TrainedUserModel,
        metrics: Dict[str, float],
        scored_validation: pd.DataFrame,
    ) -> None:
        """評価指標・検証スコア・集計結果をユーザー別に保存する。"""

        if self.result_root is None:
            return

        user_dir = self.result_root / model.hashvin

        user_dir.mkdir(parents=True, exist_ok=True)

        metrics_path = user_dir / "metrics.json"

        with metrics_path.open("w", encoding="utf-8") as fh:
            json.dump(metrics, fh, ensure_ascii=False, indent=2)

        scored_path = user_dir / "validation_scores.csv"

        scored_validation.to_csv(scored_path, index=False)

        event_summary, cluster_summary = summarize_validation_scores(
            scored_validation, self.config.topk_eval
        )

        event_path = user_dir / "event_summary.csv"

        event_summary.to_csv(event_path, index=False)

        cluster_path = user_dir / "cluster_summary.csv"

        cluster_summary.to_csv(cluster_path, index=False)

        try:
            feature_importance = model.predictor.feature_importance(model.train_data)

        except Exception:
            feature_importance = None

        if feature_importance is not None:
            fi_df = (
                feature_importance
                if isinstance(feature_importance, pd.DataFrame)
                else pd.DataFrame(feature_importance)
            )

            if not fi_df.empty:
                fi_path = user_dir / "feature_importance.csv"

                fi_df.to_csv(fi_path, index=True)
