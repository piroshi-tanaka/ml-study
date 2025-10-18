
"""EV長時間放置クラスタ予測の前処理パイプライン。

要件.md のMVP＋α要件（§0〜§8）に対応する補助クラスと関数を提供する。
- §0〜§2: hashvin単位の独立処理・時系列Split
- §3: HEADクラスタ抽出（K≤10制限＋OTHER統合）
- §4: 特徴量生成（MVP4本＋αを切り替え可能）
- §5〜§7: 学習テーブル整形と辞書出力
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from pandas import Timestamp

OTHER_LABEL = "OTHER"  # §3 HEAD外統合クラス


def ensure_datetime(series: pd.Series) -> pd.Series:
    """Seriesをタイムゾーン付きdatetime64に変換し、リーク防止の前提を整える。"""
    if pd.api.types.is_datetime64_any_dtype(series):
        if getattr(series.dt, "tz", None) is None:
            return series.dt.tz_localize("UTC")
        return series
    return pd.to_datetime(series, utc=True, errors="coerce")


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """2地点間のハバーサイン距離[km]を返す（§4.2 距離特徴用）。"""
    if any(pd.isna([lat1, lon1, lat2, lon2])):
        return float("nan")
    r = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c


def log1p_safe(value: float) -> float:
    """距離などゼロ付近の値をlog1pに通す際の丸め誤差を補正する。"""
    if value < 0:
        value = 0.0
    return math.log1p(value)


def assign_splits(n_rows: int, train_ratio: float = 0.8, valid_ratio: float = 0.1) -> List[str]:
    """時系列順にtrain/valid/testを割り当てる（§2）。サンプルが少ない場合も安全に分割する。"""
    if n_rows == 0:
        return []
    if n_rows <= 2:
        return ["train"] + ["test"] * (n_rows - 1)

    train_end = max(int(n_rows * train_ratio), 1)
    valid_end = max(train_end + int(n_rows * valid_ratio), train_end + 1)
    valid_end = min(valid_end, n_rows - 1)

    splits = ["train"] * train_end
    splits.extend(["valid"] * (valid_end - train_end))
    splits.extend(["test"] * (n_rows - valid_end))
    return splits


@dataclass
class FeatureToggleConfig:
    """特徴量のON/OFFをまとめる設定。MVP4本と＋αを柔軟に切り替えられる。"""

    use_distance: bool = True  # MVP: 距離
    use_frequency: bool = True  # MVP: 頻度
    use_recency: bool = True  # MVP: Recency
    use_time_compat: bool = True  # MVP: time_compat
    use_behavior_flags: bool = True  # ＋α: 行動帯フラグ
    use_station_type: bool = False  # ＋α: ステーション種別


@dataclass
class PipelineConfig:
    """hashvin単位の特徴生成に関する設定。"""

    head_k: int = 10
    min_long_inactive_minutes: int = 360
    laplace_alpha: float = 1.0
    recency_default_hours: float = 1e6
    time_bin_hours: int = 4
    station_type_map: Optional[Dict[str, str]] = None
    min_delay_samples: int = 3
    max_time_shift_hours: float = 24.0
    feature_toggles: FeatureToggleConfig = field(default_factory=FeatureToggleConfig)

    @property
    def enable_station_type(self) -> bool:
        """後方互換のための補助プロパティ。"""
        return self.feature_toggles.use_station_type


@dataclass
class HeadClusterInfo:
    """HEADクラスタのメタ情報（§3 指標A〜D + §4辞書）を保持する。"""

    cluster_id: str
    source: str  # "after_charge" / "anytime"
    count_after_charge: int = 0
    hours_after_charge: float = 0.0
    count_anytime: int = 0
    hours_anytime: float = 0.0
    centroid_lat: Optional[float] = None
    centroid_lon: Optional[float] = None
    delay_hours: float = 0.0
    p_start_matrix: List[List[float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "cluster_id": self.cluster_id,
            "source": self.source,
            "count_after_charge": self.count_after_charge,
            "hours_after_charge": self.hours_after_charge,
            "count_anytime": self.count_anytime,
            "hours_anytime": self.hours_anytime,
            "centroid_lat": self.centroid_lat,
            "centroid_lon": self.centroid_lon,
            "delay_hours": self.delay_hours,
            "p_start_matrix": self.p_start_matrix,
        }


@dataclass
class HashvinResult:
    """hashvin単位の特徴テーブルとメタ情報。"""

    hashvin: str
    head_clusters: List[str]
    head_details: List[HeadClusterInfo]
    features: pd.DataFrame
    split_datasets: Dict[str, pd.DataFrame]
    label_col: str


class HeadSelector:
    """§3 HEADクラスタ抽出の責務を担うクラス。"""

    def __init__(self, config: PipelineConfig, sessions_df: pd.DataFrame) -> None:
        self.config = config
        self.sessions_df = sessions_df

    def select(self, train_df: pd.DataFrame) -> Tuple[List[str], List[HeadClusterInfo]]:
        if train_df.empty:
            return [], []

        after_charge = self._from_after_charge(train_df)
        train_cutoff = train_df["charge_end_time"].max()
        anytime = self._from_anytime(train_cutoff)

        head_infos: Dict[str, HeadClusterInfo] = {}

        after_charge = after_charge.sort_values(
            ["count_after_charge", "hours_after_charge"], ascending=[False, False]
        )
        for _, row in after_charge.iterrows():
            cid = str(row["cluster_id"])
            if cid not in head_infos and len(head_infos) < self.config.head_k:
                head_infos[cid] = HeadClusterInfo(
                    cluster_id=cid,
                    source="after_charge",
                    count_after_charge=int(row["count_after_charge"]),
                    hours_after_charge=float(row["hours_after_charge"]),
                )

        anytime = anytime.sort_values(["count_anytime", "hours_anytime"], ascending=[False, False])
        for _, row in anytime.iterrows():
            if len(head_infos) >= self.config.head_k:
                break
            cid = str(row["cluster_id"])
            if cid in head_infos:
                info = head_infos[cid]
                info.count_anytime = int(row["count_anytime"])
                info.hours_anytime = float(row["hours_anytime"])
            else:
                head_infos[cid] = HeadClusterInfo(
                    cluster_id=cid,
                    source="anytime",
                    count_anytime=int(row["count_anytime"]),
                    hours_anytime=float(row["hours_anytime"]),
                )

        return list(head_infos.keys()), list(head_infos.values())

    def _from_after_charge(self, train_df: pd.DataFrame) -> pd.DataFrame:
        agg = (
            train_df.groupby("next_long_inactive_cluster")
            .agg(
                count_after_charge=("next_long_inactive_cluster", "size"),
                hours_after_charge=("inactive_duration_minutes", lambda x: x.sum() / 60.0),
            )
            .reset_index()
        )
        agg.rename(columns={"next_long_inactive_cluster": "cluster_id"}, inplace=True)
        return agg

    def _from_anytime(self, train_cutoff: Timestamp) -> pd.DataFrame:
        sessions = self.sessions_df[
            (self.sessions_df["session_type"] == "inactive")
            & (self.sessions_df["duration_minutes"] >= self.config.min_long_inactive_minutes)
            & (self.sessions_df["start_time"] <= train_cutoff)
        ].copy()
        sessions = sessions[sessions["session_cluster"] != "-1"]
        agg = (
            sessions.groupby("session_cluster")
            .agg(
                count_anytime=("session_cluster", "size"),
                hours_anytime=("duration_minutes", lambda x: x.sum() / 60.0),
            )
            .reset_index()
        )
        agg.rename(columns={"session_cluster": "cluster_id"}, inplace=True)
        return agg


class VisitStatisticsBuilder:
    """§4で必要となる重心・遅延・時間分布を組み立てる。"""

    def __init__(self, config: PipelineConfig, sessions_df: pd.DataFrame) -> None:
        self.config = config
        self.sessions_df = sessions_df

    def build(self, train_df: pd.DataFrame, head_clusters: Sequence[str]) -> Tuple[
        Dict[str, Tuple[Optional[float], Optional[float]]],
        Dict[str, float],
        Dict[str, np.ndarray],
    ]:
        if not head_clusters:
            return {}, {}, {}
        train_cutoff = train_df["charge_end_time"].max()
        centroids = self._centroids(train_df, head_clusters, train_cutoff)
        delays = self._delays(train_df, head_clusters)
        p_start = self._p_start(head_clusters, train_cutoff)
        return centroids, delays, p_start

    def _centroids(
        self,
        train_df: pd.DataFrame,
        head_clusters: Sequence[str],
        train_cutoff: Timestamp,
    ) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
        centroids: Dict[str, Tuple[Optional[float], Optional[float]]] = {}

        if {"inactive_lat", "inactive_lon"}.issubset(train_df.columns):
            grouped = train_df.groupby("next_long_inactive_cluster")[
                ["inactive_lat", "inactive_lon"]
            ].median()
            for cid in head_clusters:
                if cid in grouped.index:
                    lat, lon = grouped.loc[cid].tolist()
                    centroids[cid] = (float(lat), float(lon))

        candidates = self.sessions_df[
            (self.sessions_df["session_type"] == "inactive")
            & (self.sessions_df["session_cluster"].isin(head_clusters))
            & (self.sessions_df["start_time"] <= train_cutoff)
        ]
        if not candidates.empty:
            med = candidates.groupby("session_cluster")[
                ["start_lat", "start_lon"]
            ].median()
            for cid in head_clusters:
                if cid in med.index and cid not in centroids:
                    lat, lon = med.loc[cid].tolist()
                    centroids[cid] = (float(lat), float(lon))

        for cid in head_clusters:
            centroids.setdefault(cid, (None, None))
        return centroids

    def _delays(self, train_df: pd.DataFrame, head_clusters: Sequence[str]) -> Dict[str, float]:
        if train_df.empty:
            return {cid: 0.0 for cid in head_clusters}

        delays = train_df["inactive_start_time"] - train_df["charge_start_time"]
        delays_hours = delays.dt.total_seconds() / 3600.0
        overall = float(np.median(delays_hours.dropna())) if not delays_hours.dropna().empty else 0.0

        delay_map: Dict[str, float] = {}
        grouped = train_df.groupby("next_long_inactive_cluster")
        for cid in head_clusters:
            if cid in grouped.groups:
                values = delays_hours.loc[grouped.groups[cid]].dropna()
                if len(values) >= self.config.min_delay_samples:
                    delay_map[cid] = float(np.median(values))
        for cid in head_clusters:
            delay_map.setdefault(cid, overall)
            delay_map[cid] = float(np.clip(delay_map[cid], 0.0, self.config.max_time_shift_hours))
        return delay_map

    def _p_start(self, head_clusters: Sequence[str], train_cutoff: Timestamp) -> Dict[str, np.ndarray]:
        alpha = self.config.laplace_alpha
        bins_per_day = int(24 / self.config.time_bin_hours)
        matrices: Dict[str, np.ndarray] = {}

        sessions = self.sessions_df[
            (self.sessions_df["session_type"] == "inactive")
            & (self.sessions_df["duration_minutes"] >= self.config.min_long_inactive_minutes)
            & (self.sessions_df["session_cluster"].isin(head_clusters))
            & (self.sessions_df["start_time"] <= train_cutoff)
        ]

        for cid in head_clusters:
            matrix = np.full((7, bins_per_day), alpha, dtype=float)
            cluster_sessions = sessions[sessions["session_cluster"] == cid]
            if not cluster_sessions.empty:
                tz = cluster_sessions["start_time"].iloc[0].tz
                for ts in cluster_sessions["start_time"]:
                    localized = ts.tz_convert(tz) if tz else ts
                    dow = localized.weekday()
                    hour = localized.hour + localized.minute / 60.0
                    bin_idx = int(hour // self.config.time_bin_hours) % bins_per_day
                    matrix[dow, bin_idx] += 1.0
            matrix /= matrix.sum()
            matrices[cid] = matrix

        return matrices


class ClassFeatureAssembler:
    """§4.2 クラスタ依存特徴を生成する。"""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.toggles = config.feature_toggles

    def transform(
        self,
        model_df: pd.DataFrame,
        head_clusters: Sequence[str],
        centroids: Dict[str, Tuple[Optional[float], Optional[float]]],
        delays: Dict[str, float],
        p_start: Dict[str, np.ndarray],
    ) -> pd.DataFrame:
        records: Dict[str, List[float]] = {}
        toggles = self.toggles

        for cid in head_clusters:
            if toggles.use_distance:
                records[f"dist_to_{cid}"] = []
            if toggles.use_frequency:
                records[f"freq_hashvin_{cid}"] = []
            if toggles.use_recency:
                records[f"recency_{cid}_h"] = []
            if toggles.use_time_compat:
                records[f"time_compat_{cid}"] = []

        freq_state = {cid: self.config.laplace_alpha for cid in head_clusters}
        last_visit_state = {cid: None for cid in head_clusters}

        for _, row in model_df.iterrows():
            charge_lat = row.get("charge_lat")
            charge_lon = row.get("charge_lon")
            charge_start: Timestamp = row["charge_start_time"]

            for cid in head_clusters:
                centroid_lat, centroid_lon = centroids.get(cid, (None, None))

                if toggles.use_distance:
                    if centroid_lat is None or centroid_lon is None or pd.isna(charge_lat) or pd.isna(charge_lon):
                        dist_value = float("nan")
                    else:
                        dist_value = haversine_km(charge_lat, charge_lon, centroid_lat, centroid_lon)
                    records[f"dist_to_{cid}"].append(log1p_safe(dist_value) if not math.isnan(dist_value) else float("nan"))

                if toggles.use_frequency:
                    records[f"freq_hashvin_{cid}"].append(freq_state.get(cid, self.config.laplace_alpha))

                if toggles.use_recency:
                    last_visit = last_visit_state.get(cid)
                    if last_visit is None:
                        recency = self.config.recency_default_hours
                    else:
                        delta = (charge_start - last_visit).total_seconds() / 3600.0
                        recency = float(max(delta, 0.0))
                    records[f"recency_{cid}_h"].append(recency)

                if toggles.use_time_compat:
                    matrix = p_start.get(cid)
                    delay = delays.get(cid, 0.0)
                    score = self._time_compat_score(charge_start, matrix, delay)
                    records[f"time_compat_{cid}"].append(score)

            if row["split"] == "train":
                target = row["next_long_inactive_cluster"]
                if target in head_clusters:
                    freq_state[target] = freq_state.get(target, self.config.laplace_alpha) + 1.0
                    last_visit_state[target] = row["inactive_start_time"]

        return pd.DataFrame(records, index=model_df.index)

    def _time_compat_score(
        self,
        charge_start: Timestamp,
        matrix: Optional[np.ndarray],
        delay_hours: float,
    ) -> float:
        if matrix is None or charge_start is pd.NaT:
            return 1.0 / (7 * max(1, int(24 / self.config.time_bin_hours)))
        target_time = charge_start + pd.Timedelta(hours=delay_hours)
        bins_per_day = int(24 / self.config.time_bin_hours)
        dow = target_time.weekday()
        hour = target_time.hour + target_time.minute / 60.0
        bin_idx = int(hour // self.config.time_bin_hours) % bins_per_day
        neighbor_idx = (bin_idx - 1) % bins_per_day
        primary = matrix[dow, bin_idx]
        neighbor = matrix[dow, neighbor_idx]
        return float(0.75 * primary + 0.25 * neighbor)


class CommonFeatureAssembler:
    """§4.1 共通特徴（時間情報・SOC等）を生成する。"""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.toggles = config.feature_toggles

    def transform(self, model_df: pd.DataFrame) -> pd.DataFrame:
        charge_start = model_df["charge_start_time"]
        hour = charge_start.dt.hour + charge_start.dt.minute / 60.0
        hour_rad = 2 * math.pi * hour / 24.0

        columns: Dict[str, pd.Series] = {
            "dow": charge_start.dt.weekday,
            "hour_sin": np.sin(hour_rad),
            "hour_cos": np.cos(hour_rad),
            "soc_start": model_df.get("charge_start_soc"),
        }

        if self.toggles.use_behavior_flags:
            columns.update(
                {
                    "is_return_band": ((charge_start.dt.hour >= 18) | (charge_start.dt.hour < 6)).astype(int),
                    "is_commute_band": (
                        ((charge_start.dt.hour >= 7) & (charge_start.dt.hour < 10))
                        | ((charge_start.dt.hour >= 9) & (charge_start.dt.hour < 18))
                    ).astype(int),
                    "weekend_flag": charge_start.dt.weekday.isin([5, 6]).astype(int),
                }
            )

        common_df = pd.DataFrame(columns, index=model_df.index)

        if self.toggles.use_station_type:
            station_df = self._station_type_feature(model_df)
            if not station_df.empty:
                common_df = pd.concat([common_df, station_df], axis=1)

        return common_df

    def _station_type_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        station_col = None
        for candidate in ["station_type", "charge_station_type"]:
            if candidate in df.columns:
                station_col = candidate
                break
        values = None
        if station_col is not None:
            values = df[station_col]
        elif self.config.station_type_map is not None and "charge_cluster" in df.columns:
            values = df["charge_cluster"].map(self.config.station_type_map)
        if values is None:
            return pd.DataFrame(index=df.index)
        return pd.get_dummies(values.fillna("unknown"), prefix="station", dtype=np.uint8)


class HashvinProcessor:
    """hashvin単位で要件§0〜§8の流れを実行する。"""

    def __init__(self, hashvin: str, charge_df: pd.DataFrame, sessions_df: pd.DataFrame, config: PipelineConfig) -> None:
        self.hashvin = hashvin
        self.charge_df = charge_df.copy()
        self.sessions_df = sessions_df.copy()
        self.config = config
        self._prepare_dataframe_types()

    def _prepare_dataframe_types(self) -> None:
        """入力データの型を整え、後続処理の前提を満たす。"""
        for col in ["charge_start_time", "charge_end_time", "inactive_start_time", "inactive_end_time"]:
            if col in self.charge_df.columns:
                self.charge_df[col] = ensure_datetime(self.charge_df[col])
        for col in ["start_time", "end_time"]:
            if col in self.sessions_df.columns:
                self.sessions_df[col] = ensure_datetime(self.sessions_df[col])

        for col in [
            "charge_durations_minutes",
            "inactive_duration_minutes",
            "start_soc",
            "end_soc",
            "charge_start_soc",
            "charge_end_soc",
        ]:
            if col in self.charge_df.columns:
                self.charge_df[col] = pd.to_numeric(self.charge_df[col], errors="coerce")

        if "next_long_inactive_cluster" in self.charge_df.columns:
            self.charge_df["next_long_inactive_cluster"] = self.charge_df["next_long_inactive_cluster"].astype(str)
        if "session_cluster" in self.sessions_df.columns:
            self.sessions_df["session_cluster"] = (
                self.sessions_df["session_cluster"].astype(str).str.replace("^I_", "", regex=True)
            )

    def _select_model_rows(self) -> pd.DataFrame:
        """教師データ対象（next_long_inactive_clusterあり）を抽出し、時間順に整列する。"""
        df = self.charge_df.copy()
        df = df[df["next_long_inactive_cluster"].notna()].copy()
        df.sort_values("charge_end_time", inplace=True)
        df.reset_index(drop=True, inplace=True)
        df["session_order"] = np.arange(len(df))
        df["session_uid"] = df["hashvin"].astype(str) + "_" + df["session_order"].astype(str)
        return df

    def build_features(self) -> HashvinResult:
        """要件§2〜§7に沿って学習用テーブルを構築し、HashvinResultを返す。"""
        model_df = self._select_model_rows()
        if model_df.empty:
            empty_split = {split: model_df for split in ["train", "valid", "test"]}
            return HashvinResult(self.hashvin, [], [], model_df, empty_split, "y_class")

        model_df["split"] = assign_splits(len(model_df))  # §2: 時系列Split

        head_selector = HeadSelector(self.config, self.sessions_df)
        head_clusters, head_details = head_selector.select(model_df[model_df["split"] == "train"])

        if not head_clusters:
            model_df["y_class"] = OTHER_LABEL
            split_datasets = {
                split: model_df[model_df["split"] == split].copy() for split in ["train", "valid", "test"]
            }
            return HashvinResult(self.hashvin, [], head_details, model_df, split_datasets, "y_class")

        stats_builder = VisitStatisticsBuilder(self.config, self.sessions_df)
        centroids, delays, p_start = stats_builder.build(model_df[model_df["split"] == "train"], head_clusters)

        for info in head_details:
            lat, lon = centroids.get(info.cluster_id, (None, None))
            info.centroid_lat = lat
            info.centroid_lon = lon
            info.delay_hours = delays.get(info.cluster_id, 0.0)
            matrix = p_start.get(info.cluster_id)
            info.p_start_matrix = matrix.tolist() if matrix is not None else []

        class_assembler = ClassFeatureAssembler(self.config)
        class_features = class_assembler.transform(model_df, head_clusters, centroids, delays, p_start)

        common_assembler = CommonFeatureAssembler(self.config)
        common_features = common_assembler.transform(model_df)

        feature_df = pd.concat([model_df, common_features, class_features], axis=1)

        # 予測時に未確定の値は特徴量として保持しない
        leak_cols = [
            "charge_durations_minutes",
            "charge_duration_minutes",
            "charge_end_soc",
            "soc_delta",
            "end_soc",
        ]
        existing_leak_cols = [col for col in leak_cols if col in feature_df.columns]
        if existing_leak_cols:
            feature_df = feature_df.drop(columns=existing_leak_cols)

        # §3: HEAD外はOTHERにマージして教師を整形
        feature_df["y_class"] = feature_df["next_long_inactive_cluster"].where(
            feature_df["next_long_inactive_cluster"].isin(head_clusters), OTHER_LABEL
        )

        # §6: 学習/検証/テスト用に分割テーブルを用意
        split_datasets = {
            split: feature_df[feature_df["split"] == split].copy() for split in ["train", "valid", "test"]
        }

        return HashvinResult(
            hashvin=self.hashvin,
            head_clusters=list(head_clusters),
            head_details=head_details,
            features=feature_df,
            split_datasets=split_datasets,
            label_col="y_class",
        )


def save_head_details(head_details: Sequence[HeadClusterInfo], output_path: Path) -> None:
    """HEAD抽出結果をJSONで保存（§3・§8）。"""
    data = [info.to_dict() for info in head_details]
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
