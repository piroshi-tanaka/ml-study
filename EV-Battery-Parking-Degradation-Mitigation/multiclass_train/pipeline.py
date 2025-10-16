"""
EV長時間放置クラスタ予測の前処理パイプライン。

要件.md のMVP＋α要件（§0〜§8）に対応する補助関数と `HashvinProcessor` クラスを提供する。
具体的な対応は以下の通り。

- §0〜§2: hashvin完全独立・リーク防止・時系列Split → `ensure_datetime`, `assign_splits`, `_prepare_dataframe_types`
- §3: HEADクラスタ抽出（K≤10＋OTHER） → `_select_head_clusters` 系列
- §4: 共通特徴＋クラス依存特徴（距離/頻度/Recency/time_compat） → `_generate_common_features`, `_generate_class_feature_frame`
- §5〜§7: 学習テーブル生成・教師整形・辞書出力 → `build_features`, `save_head_details`
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

# Constants --------------------------------------------------------------------

OTHER_LABEL = "OTHER"  # §3: HEAD外クラスをまとめるラベル
EPS = 1e-9


# Utilities --------------------------------------------------------------------

def ensure_datetime(series: pd.Series) -> pd.Series:
    """Seriesをタイムゾーン付きdatetime64に変換し、リークを防ぐ下準備とする。"""
    if pd.api.types.is_datetime64_any_dtype(series):
        if getattr(series.dt, "tz", None) is None:
            return series.dt.tz_localize("UTC")
        return series
    return pd.to_datetime(series, utc=True, errors="coerce")


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """2地点のハバーサイン距離[km]を返す（§4.2 距離特徴用）。"""
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
    """距離等のゼロ付近をlog1pに通す際の丸め誤差ガード。"""
    if value < 0:
        value = 0.0
    return math.log1p(value)


def assign_splits(n_rows: int, train_ratio: float = 0.8, valid_ratio: float = 0.1) -> List[str]:
    """
    時系列順のtrain/valid/test割当に加え、サンプルが少ない場合のフォールバックも実装する。

    要件§2の「過去のみで学習・辞書作成」を徹底するため、充電終了時刻の昇順に基づく。
    """
    if n_rows == 0:
        return []
    if n_rows <= 2:
        # Minimal fallback: 1 train, rest test, no valid
        return ["train"] + ["test"] * (n_rows - 1)

    train_end = max(int(n_rows * train_ratio), 1)
    valid_end = max(train_end + int(n_rows * valid_ratio), train_end + 1)
    valid_end = min(valid_end, n_rows - 1)  # ensure at least 1 test

    splits = ["train"] * train_end
    splits.extend(["valid"] * (valid_end - train_end))
    splits.extend(["test"] * (n_rows - valid_end))
    return splits


# Configuration ----------------------------------------------------------------

@dataclass
class PipelineConfig:
    """hashvin単位の特徴生成設定（§0〜§5で指定されたパラメータ群）。"""

    head_k: int = 10
    min_long_inactive_minutes: int = 360
    laplace_alpha: float = 1.0
    recency_default_hours: float = 1e6
    time_bin_hours: int = 4
    enable_station_type: bool = False
    station_type_map: Optional[Dict[str, str]] = None
    min_delay_samples: int = 3
    max_time_shift_hours: float = 24.0


@dataclass
class HeadClusterInfo:
    """HEADクラスタに関する指標（§3 指標A〜D）と辞書情報を保持する。"""
    cluster_id: str
    source: str  # "after_charge" or "anytime"
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
    """ハッシュVIN単位の特徴テーブルとメタ情報（§5の出力）をパッケージ化。"""
    hashvin: str
    head_clusters: List[str]
    head_details: List[HeadClusterInfo]
    features: pd.DataFrame
    split_datasets: Dict[str, pd.DataFrame]
    label_col: str


# Core processing --------------------------------------------------------------

class HashvinProcessor:
    """
    要件§0〜§8をhashvin単位で実装する前処理クラス。

    主な役割:
      - §0〜§2: 型変換と時系列Splitによりリークを防止
      - §3: HEADクラスタ抽出（K≤10制約＋OTHERマージ）
      - §4〜§5: 特徴量生成（共通＋クラス依存）と教師ラベル作成
      - §7: 評価時に利用する辞書（HEAD情報）出力
    """
    def __init__(
        self,
        hashvin: str,
        charge_df: pd.DataFrame,
        sessions_df: pd.DataFrame,
        config: PipelineConfig,
    ) -> None:
        self.hashvin = hashvin
        self.charge_df = charge_df.copy()
        self.sessions_df = sessions_df.copy()
        self.config = config
        self._prepare_dataframe_types()

    def _prepare_dataframe_types(self) -> None:
        """入力データの型を整える（§0 リーク防止の前処理）。"""
        dt_cols_charge = ["charge_start_time", "charge_end_time", "inactive_start_time", "inactive_end_time"]
        for col in dt_cols_charge:
            if col in self.charge_df.columns:
                self.charge_df[col] = ensure_datetime(self.charge_df[col])

        dt_cols_sessions = ["start_time", "end_time"]
        for col in dt_cols_sessions:
            if col in self.sessions_df.columns:
                self.sessions_df[col] = ensure_datetime(self.sessions_df[col])

        numeric_cols = [
            "charge_durations_minutes",
            "inactive_duration_minutes",
            "start_soc",
            "end_soc",
            "charge_start_soc",
            "charge_end_soc",
        ]
        for col in numeric_cols:
            if col in self.charge_df.columns:
                self.charge_df[col] = pd.to_numeric(self.charge_df[col], errors="coerce")

        # Ensure cluster ids are strings
        if "next_long_inactive_cluster" in self.charge_df.columns:
            self.charge_df["next_long_inactive_cluster"] = self.charge_df["next_long_inactive_cluster"].astype(str)
        if "session_cluster" in self.sessions_df.columns:
            self.sessions_df["session_cluster"] = (
                self.sessions_df["session_cluster"].astype(str).str.replace("^I_", "", regex=True)
            )

    def _select_model_rows(self) -> pd.DataFrame:
        """教師対象行のみを抽出し、充電終了時刻順に並べた学習基礎テーブルを返す（§1.2）。"""
        df = self.charge_df.copy()
        df = df[df["next_long_inactive_cluster"].notna()].copy()
        df.sort_values("charge_end_time", inplace=True)
        df.reset_index(drop=True, inplace=True)
        df["session_order"] = np.arange(len(df))
        df["session_uid"] = df["hashvin"].astype(str) + "_" + df["session_order"].astype(str)
        return df

    def _head_from_after_charge(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """§3-1 充電直後6h放置実績（指標A/B）でHEAD候補を評価するための集計を返す。"""
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

    def _head_from_anytime(self, train_cutoff: Timestamp) -> pd.DataFrame:
        """§3-2 日常長居地（指標C/D）でHEAD不足を補うための集計を返す。"""
        sessions = self.sessions_df.copy()
        sessions = sessions[
            (sessions["session_type"] == "inactive")
            & (sessions["duration_minutes"] >= self.config.min_long_inactive_minutes)
            & (sessions["start_time"] <= train_cutoff)
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

    def _select_head_clusters(self, train_df: pd.DataFrame) -> Tuple[List[str], List[HeadClusterInfo]]:
        """§3 HEAD抽出：指標A/B優先、残りを指標C/Dで補完し、K≤10に制限した結果を返す。"""
        if train_df.empty:
            return [], []

        after_charge = self._head_from_after_charge(train_df)
        train_cutoff = train_df["charge_end_time"].max()
        anytime = self._head_from_anytime(train_cutoff)

        head_infos: Dict[str, HeadClusterInfo] = {}

        after_charge_sorted = after_charge.sort_values(
            ["count_after_charge", "hours_after_charge"], ascending=[False, False]
        )
        for _, row in after_charge_sorted.iterrows():
            cid = str(row["cluster_id"])
            if cid not in head_infos and len(head_infos) < self.config.head_k:
                head_infos[cid] = HeadClusterInfo(
                    cluster_id=cid,
                    source="after_charge",
                    count_after_charge=int(row["count_after_charge"]),
                    hours_after_charge=float(row["hours_after_charge"]),
                )

        anytime_sorted = anytime.sort_values(["count_anytime", "hours_anytime"], ascending=[False, False])
        for _, row in anytime_sorted.iterrows():
            if len(head_infos) >= self.config.head_k:
                break
            cid = str(row["cluster_id"])
            if cid in head_infos:
                info = head_infos[cid]
                info.count_anytime = int(row["count_anytime"])
                info.hours_anytime = float(row["hours_anytime"])
                continue
            head_infos[cid] = HeadClusterInfo(
                cluster_id=cid,
                source="anytime",
                count_anytime=int(row["count_anytime"]),
                hours_anytime=float(row["hours_anytime"]),
            )

        return list(head_infos.keys()), list(head_infos.values())

    def _compute_cluster_centroids(
        self,
        head_clusters: Sequence[str],
        train_df: pd.DataFrame,
        train_cutoff: Timestamp,
    ) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
        """§4.2 距離特徴用のクラスタ代表座標（中央値）をtrainデータから算出する。"""
        centroids: Dict[str, Tuple[Optional[float], Optional[float]]] = {}

        # Primary source: train_df inactive coordinates
        inactive_cols_available = {"inactive_lat", "inactive_lon"}.issubset(train_df.columns)
        if inactive_cols_available:
            grouped = train_df.groupby("next_long_inactive_cluster")[["inactive_lat", "inactive_lon"]].median()
            for cid in head_clusters:
                if cid in grouped.index:
                    lat, lon = grouped.loc[cid].tolist()
                    centroids[cid] = (float(lat), float(lon))

        # Fallback: sessions dataset within train cutoff
        sessions = self.sessions_df[
            (self.sessions_df["session_type"] == "inactive")
            & (self.sessions_df["session_cluster"].isin(head_clusters))
            & (self.sessions_df["start_time"] <= train_cutoff)
        ]
        if not sessions.empty:
            med = sessions.groupby("session_cluster")[["start_lat", "start_lon"]].median()
            for cid in head_clusters:
                if cid in med.index and cid not in centroids:
                    lat, lon = med.loc[cid].tolist()
                    centroids[cid] = (float(lat), float(lon))

        # Ensure keys exist
        for cid in head_clusters:
            centroids.setdefault(cid, (None, None))
        return centroids

    def _compute_delay_statistics(self, train_df: pd.DataFrame, head_clusters: Sequence[str]) -> Dict[str, float]:
        """§4.2 time_compat用の遅延中央値（L̃）をクラスタ別に計算し、データ不足時は全体値で代替。"""
        if train_df.empty:
            return {cid: 0.0 for cid in head_clusters}

        delays = train_df["inactive_start_time"] - train_df["charge_start_time"]
        delays_hours = delays.dt.total_seconds() / 3600.0
        overall_delay = float(np.median(delays_hours.dropna())) if not delays_hours.dropna().empty else 0.0

        delay_map: Dict[str, float] = {}
        grouped = train_df.groupby("next_long_inactive_cluster")
        for cid in head_clusters:
            if cid in grouped.groups:
                cluster_delays = delays_hours.loc[grouped.groups[cid]]
                valid_delays = cluster_delays.dropna()
                if len(valid_delays) >= self.config.min_delay_samples:
                    delay_map[cid] = float(np.median(valid_delays))

        for cid in head_clusters:
            delay_map.setdefault(cid, overall_delay)
            delay_map[cid] = float(np.clip(delay_map[cid], 0.0, self.config.max_time_shift_hours))
        return delay_map

    def _compute_p_start(
        self,
        head_clusters: Sequence[str],
        train_cutoff: Timestamp,
    ) -> Dict[str, np.ndarray]:
        """§4.2 time_compat の曜日×4hビン分布 p_start_Cj を作成（ラプラス平滑込み）。"""
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
            subsets = sessions[sessions["session_cluster"] == cid]
            if not subsets.empty:
                tz = subsets["start_time"].iloc[0].tz
                for ts in subsets["start_time"]:
                    localized = ts.tz_convert(tz) if tz else ts
                    dow = localized.weekday()
                    hour = localized.hour + localized.minute / 60.0
                    bin_idx = int(hour // self.config.time_bin_hours) % bins_per_day
                    matrix[dow, bin_idx] += 1.0
            matrix /= matrix.sum()
            matrices[cid] = matrix

        for cid in head_clusters:
            matrices.setdefault(cid, np.full((7, bins_per_day), 1.0 / (7 * bins_per_day), dtype=float))
        return matrices

    def _station_type_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        """§4.1 備考: 充電ステーション種別を後から有効化できるようOne-Hotを作成。"""
        if not self.config.enable_station_type:
            return pd.DataFrame(index=df.index)
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

    def _time_compat_score(
        self,
        row: pd.Series,
        matrix: np.ndarray,
        delay_hours: float,
    ) -> float:
        """充電開始時刻を遅延中央値だけシフトし、対応する曜日×時間帯の確率を返す（§4.2）。"""
        charge_start: Timestamp = row["charge_start_time"]
        if pd.isna(charge_start):
            return float(matrix.mean())
        target_time = charge_start + pd.Timedelta(hours=delay_hours)
        time_bin_hours = self.config.time_bin_hours
        bins_per_day = int(24 / time_bin_hours)

        dow = target_time.weekday()
        hour = target_time.hour + target_time.minute / 60.0
        bin_idx = int(hour // time_bin_hours) % bins_per_day

        primary = matrix[dow, bin_idx]
        # Optional smoothing with neighboring bins
        neighbor_bin = (bin_idx - 1) % bins_per_day
        neighbor_score = matrix[dow, neighbor_bin]
        return float(0.75 * primary + 0.25 * neighbor_score)

    def _generate_class_feature_frame(
        self,
        model_df: pd.DataFrame,
        head_clusters: Sequence[str],
        centroids: Dict[str, Tuple[Optional[float], Optional[float]]],
        delays: Dict[str, float],
        p_start: Dict[str, np.ndarray],
    ) -> pd.DataFrame:
        """
        §4.2 クラス依存特徴（距離/頻度/Recency/time_compat）を横展開して生成する。

        頻度・Recencyはtrain期間でのみ状態更新し、未来情報の混入を防ぐ（§0のリーク防止）。
        """
        feature_records: Dict[str, List[float]] = {}
        for cid in head_clusters:
            feature_records[f"dist_to_{cid}"] = []
            feature_records[f"freq_hashvin_{cid}"] = []
            feature_records[f"recency_{cid}_h"] = []
            feature_records[f"time_compat_{cid}"] = []
        freq_state = {cid: self.config.laplace_alpha for cid in head_clusters}
        last_visit_state = {cid: None for cid in head_clusters}

        for _, row in model_df.iterrows():
            charge_lat = row.get("charge_lat")
            charge_lon = row.get("charge_lon")
            charge_end_time: Timestamp = row["charge_end_time"]

            for cid in head_clusters:
                centroid_lat, centroid_lon = centroids.get(cid, (None, None))
                if centroid_lat is None or centroid_lon is None or pd.isna(charge_lat) or pd.isna(charge_lon):
                    distance = float("nan")
                else:
                    distance = haversine_km(charge_lat, charge_lon, centroid_lat, centroid_lon)
                feature_records[f"dist_to_{cid}"].append(log1p_safe(distance) if not math.isnan(distance) else float("nan"))
                feature_records[f"freq_hashvin_{cid}"].append(freq_state.get(cid, self.config.laplace_alpha))

                last_visit = last_visit_state.get(cid)
                if last_visit is None:
                    recency = self.config.recency_default_hours
                else:
                    delta = (charge_end_time - last_visit).total_seconds() / 3600.0
                    recency = float(max(delta, 0.0))
                feature_records[f"recency_{cid}_h"].append(recency)

                matrix = p_start.get(cid)
                delay = delays.get(cid, 0.0)
                if matrix is None:
                    feature_records[f"time_compat_{cid}"].append(1.0 / (7 * (24 // self.config.time_bin_hours)))
                else:
                    feature_records[f"time_compat_{cid}"].append(self._time_compat_score(row, matrix, delay))

            if row["split"] == "train":
                target = row["next_long_inactive_cluster"]
                if target in head_clusters:
                    freq_state[target] = freq_state.get(target, self.config.laplace_alpha) + 1.0
                    last_visit_state[target] = row["inactive_start_time"]

        return pd.DataFrame(feature_records, index=model_df.index)

    def _generate_common_features(
        self,
        model_df: pd.DataFrame,
        station_features: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        §4.1 共通特徴（曜日/時間/充電メタ/行動フェーズ）を生成し、必要に応じてステーション種別を連結。
        """
        dow = model_df["charge_end_time"].dt.weekday
        hour = model_df["charge_end_time"].dt.hour + model_df["charge_end_time"].dt.minute / 60.0
        hour_rad = 2 * math.pi * hour / 24.0

        common_features = {
            "dow": dow,
            "hour_sin": np.sin(hour_rad),
            "hour_cos": np.cos(hour_rad),
            "charge_durations_minutes": model_df.get("charge_durations_minutes"),
            "soc_start": model_df.get("charge_start_soc"),
            "soc_end": model_df.get("charge_end_soc"),
            "soc_delta": model_df.get("charge_end_soc") - model_df.get("charge_start_soc"),
            # 行動フェーズの粗いフラグ（§4.1-3）
            "is_return_band": ((model_df["charge_end_time"].dt.hour >= 18) | (model_df["charge_end_time"].dt.hour < 6)).astype(int),
            "is_commute_band": (
                ((model_df["charge_end_time"].dt.hour >= 7) & (model_df["charge_end_time"].dt.hour < 10))
                | ((model_df["charge_end_time"].dt.hour >= 9) & (model_df["charge_end_time"].dt.hour < 18))
            ).astype(int),
            "weekend_flag": model_df["charge_end_time"].dt.weekday.isin([5, 6]).astype(int),
        }
        common_df = pd.DataFrame(common_features, index=model_df.index)
        if not station_features.empty:
            common_df = pd.concat([common_df, station_features], axis=1)
        return common_df

    def build_features(self) -> HashvinResult:
        """
        要件§2〜§7の流れに沿って学習用特徴テーブルを構築し、HashvinResultを返す。

        フロー:
          1. 教師対象抽出と時系列Split（§1, §2）
          2. HEADクラスタ決定と辞書作成（§3, §4）
          3. 特徴量生成（共通＋クラス依存）とラベル整形（§4, §5）
          4. train/valid/testセットをまとめて返却（§6, §7準備）
        """
        model_df = self._select_model_rows()
        if model_df.empty:
            return HashvinResult(
                hashvin=self.hashvin,
                head_clusters=[],
                head_details=[],
                features=model_df,
                split_datasets={"train": model_df, "valid": model_df, "test": model_df},
                label_col="y_class",
            )

        model_df["split"] = assign_splits(len(model_df))  # §2: 時系列Split（train/valid/test）
        head_clusters, head_details = self._select_head_clusters(model_df[model_df["split"] == "train"])

        if not head_clusters:
            # No head clusters found -> everything becomes OTHER
            model_df["y_class"] = OTHER_LABEL
            return HashvinResult(
                hashvin=self.hashvin,
                head_clusters=[],
                head_details=head_details,
                features=model_df,
                split_datasets={
                    "train": model_df[model_df["split"] == "train"],
                    "valid": model_df[model_df["split"] == "valid"],
                    "test": model_df[model_df["split"] == "test"],
                },
                label_col="y_class",
            )

        train_df = model_df[model_df["split"] == "train"]
        train_cutoff = train_df["charge_end_time"].max()

        centroids = self._compute_cluster_centroids(head_clusters, train_df, train_cutoff)
        delays = self._compute_delay_statistics(train_df, head_clusters)
        p_start = self._compute_p_start(head_clusters, train_cutoff)

        # Attach centroid/delay/matrix info for export
        for info in head_details:
            lat, lon = centroids.get(info.cluster_id, (None, None))
            info.centroid_lat = lat
            info.centroid_lon = lon
            info.delay_hours = delays.get(info.cluster_id, 0.0)
            matrix = p_start.get(info.cluster_id)
            info.p_start_matrix = matrix.tolist() if matrix is not None else []

        # クラス依存特徴（距離/頻度/Recency/time_compat）を生成（§4.2）
        station_features = self._station_type_feature(model_df)
        class_feature_df = self._generate_class_feature_frame(model_df, head_clusters, centroids, delays, p_start)
        # 共通特徴を結合（§4.1）
        common_df = self._generate_common_features(model_df, station_features)

        feature_df = pd.concat([model_df, common_df, class_feature_df], axis=1)

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
    """HEAD抽出結果をJSONに保存（§3, §8の検証・運用で再利用）。"""
    data = [info.to_dict() for info in head_details]
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
