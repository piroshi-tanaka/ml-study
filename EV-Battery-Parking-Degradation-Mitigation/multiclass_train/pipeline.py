
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
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from pandas import Timestamp

OTHER_LABEL = "OTHER"  # §3 HEAD外統合クラス
GLOBAL_TRANS_KEY = "__GLOBAL__"
DEFAULT_BIN_KEY = "__DEFAULT__"


def ensure_datetime(series: pd.Series) -> pd.Series:
    """JSTタイムゾーン付きのdatetime64に正規化するユーティリティ関数。

    Parameters
    ----------
    series : pd.Series
        日時情報を含む列。文字列・object型でも受け取り、すべてをJSTにそろえる。

    Returns
    -------
    pd.Series
        タイムゾーン付きdatetime64（Asia/Tokyo）。変換できない値はNaTになる。

    Notes
    -----
    下流の処理（時系列ソートやリーク防止チェック）はタイムゾーン付きdatetimeを前提にしているため、
    入力データが文字列・タイムゾーンなしのdatetimeであってもここで必ず統一しておく。
    """
    #  セッションCSVがtz無しのUTC相当で届くことがあるため、ここで必ずJSTに固定する。
    # 充電終了時刻を扱う後段ロジックが「タイムゾーン付き」を前提にしている点にも注意。
    if pd.api.types.is_datetime64_any_dtype(series):
        if getattr(series.dt, "tz", None) is None:
            return series.dt.tz_localize("Asia/Tokyo")
        return series
    parsed = pd.to_datetime(series, errors="coerce")
    if getattr(parsed.dt, "tz", None) is None:
        return parsed.dt.tz_localize("Asia/Tokyo")
    return parsed


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """地球上の2地点間距離をハバーサイン公式で計算する。

    距離特徴はモデルの中核なので、緯度経度が欠損している場合は NaN を返し、
    後段で適切に欠損処理できるようにしている。
    """
    #  単純なユークリッド距離だと地球曲率で誤差が出るので、EVの実移動距離感覚に合うハバーサインを採用。
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


def latlon_delta_to_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> Tuple[float, float, float]:
    """緯度経度差をメートル単位のベクトルと距離に変換する。"""
    if any(pd.isna([lat1, lon1, lat2, lon2])):
        return float("nan"), float("nan"), float("nan")
    avg_lat = (lat1 + lat2) / 2.0
    meters_lat = (lat2 - lat1) * 111_320.0
    meters_lon = (lon2 - lon1) * (40075000.0 * math.cos(math.radians(avg_lat)) / 360.0)
    distance_m = haversine_km(lat1, lon1, lat2, lon2) * 1000.0
    return meters_lat, meters_lon, distance_m


def log1p_safe(value: float) -> float:
    """log1pに通す前に0未満の値を補正し、安全に対数変換できるようにする。

    距離計算の丸め誤差などで負の極小値が生じても 0 にクリップして log1p を適用し、
    計算結果がNaNになることを防ぐ。
    """
    #  log1p(0)は0だが、負の極小値が入るとNaNになるので0へ押し上げてからlog1pをかける。
    if value < 0:
        value = 0.0
    return math.log1p(value)


def assign_splits(n_rows: int, train_ratio: float = 0.8, valid_ratio: float = 0.1) -> List[str]:
    """時系列に沿って train/valid/test ラベルを付与するユーティリティ。

    hashvinごとの件数が少なくても検証用データを確保できるよう最小件数を調整し、
    未来情報が学習に混入しないよう単純な割合分割ではなく時系列順を前提にする。
    """
    #  直近データで検証したいので「未来」ほど検証・テストに寄せる。件数が足りないときは最低1件ずつ確保。
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
    """特徴量の切替と時刻表現を管理する設定をまとめたクラス。

    各フラグは「この特徴量を使う／使わない」を明示し、実験時にON/OFFを切り替えやすくする。
    時刻の扱いは time_feature_mode で一括指定し、Cyclic（sin/cos）、Datetime（生データ）、
    Categorical（時間帯カテゴリ）、All（これら全て）から選択できる。
    また、直前遷移（use_prev_transition）、移動ベクトル（use_prev_vector）、当日行動ヒストグラム（use_daily_time_bins）
    の利用可否もここで制御する。
    """

    use_distance: bool = True
    use_frequency: bool = True
    use_recency: bool = True
    use_time_compat: bool = True
    use_behavior_flags: bool = True
    use_station_type: bool = False
    use_prev_transition: bool = True  # 直前クラスタ遷移確率を使う
    use_prev_vector: bool = True  # 直前クラスタ→充電地点の移動ベクトルを使う
    use_daily_time_bins: bool = True  # 当日（もしくは直近24h）の時間帯別クラスタ滞在を使う
    time_feature_mode: str = "cyclic"  # 'cyclic' / 'datetime' / 'categorical' / 'all'

@dataclass
class PipelineConfig:
    """パイプライン全体で共有するハイパーパラメータを保持する設定クラス。

    - head_k: HEADとして採用するクラスタの最大件数。
    - min_long_inactive_minutes: 「長時間放置」とみなす下限（分）。既定は6時間。
    - laplace_alpha: 頻度特徴のラプラス平滑に使う初期値。
    - frequency_window_days: 頻度カウントで参照する直近日数。
    - day_time_bin_hours: 当日行動ヒストグラムのビン幅（例：6時間単位）。
    - day_time_window_hours: rollingモードで何時間遡るか（例：24時間）。
    - day_time_window_mode: 'rolling'（直近ウィンドウ） or 'calendar'（当日0:00〜充電時刻）を指定。
    """

    head_k: int = 10
    min_long_inactive_minutes: int = 360
    laplace_alpha: float = 1.0
    recency_default_hours: float = 1e6
    time_bin_hours: int = 4
    station_type_map: Optional[Dict[str, str]] = None
    min_delay_samples: int = 3
    min_pair_delay_samples: int = 2
    max_time_shift_hours: float = 24.0
    frequency_window_days: float = 90.0
    day_time_bin_hours: int = 6  # 当日行動ヒストグラムのビン幅
    day_time_window_hours: int = 24  # rollingモードのウィンドウ長
    day_time_window_mode: str = "rolling"  # 'rolling' or 'calendar'
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
        """JSON出力や記録用に、dataclassを辞書へ変換するヘルパー。"""
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
    """HEADクラスタ（頻出放置候補）を抽出し、メタ情報を整理するヘルパークラス。

    - train_df（充電→放置ペア）から直後放置の実績が多いクラスタを優先的に採用する。
    - 枠が埋まらない場合のみ、inactiveセッション全体から補欠クラスタを追加する。
    - 抽出結果は HeadClusterInfo に保持し、距離・時間統計の付与に利用する。
    """

    def __init__(self, config: PipelineConfig, sessions_df: pd.DataFrame) -> None:
        """設定とセッション一覧を保持するだけの初期化処理。"""
        self.config = config
        self.sessions_df = sessions_df

    def select(self, train_df: pd.DataFrame) -> Tuple[List[str], List[HeadClusterInfo]]:
        """HEADクラスタのIDリストとメタ情報を返す。空の学習データなら即終了する。"""
        if train_df.empty:
            return [], []

        after_charge = self._from_after_charge(train_df)
        train_cutoff = train_df["charge_start_time"].max()

        head_infos: Dict[str, HeadClusterInfo] = {}

        after_charge = after_charge.sort_values(
            ["count_after_charge", "hours_after_charge"], ascending=[False, False]
        )
        for _, row in after_charge.iterrows():
            cid = str(row["cluster_id"])
            if cid not in head_infos and len(head_infos) < self.config.head_k:  # 未採用＆枠に余裕があればHEAD入り
                #  まずは「充電直後に長時間放置した実績」が多いクラスタをHEADに採用する。
                # 既に選ばれている（もしくは枠が埋まった）場合は何もしない。
                head_infos[cid] = HeadClusterInfo(
                    cluster_id=cid,
                    source="after_charge",
                    count_after_charge=int(row["count_after_charge"]),
                    hours_after_charge=float(row["hours_after_charge"]),
                )

        if len(head_infos) < self.config.head_k:  # 充電直後実績だけではK件に届かない場合のみ補欠を探す
            #  充電後実績だけでは枠が埋まらないときに限り、通常放置の頻度で補欠を追加する。
            anytime = self._from_anytime(train_cutoff)
            anytime = anytime.sort_values(["count_anytime", "hours_anytime"], ascending=[False, False])
            for _, row in anytime.iterrows():
                if len(head_infos) >= self.config.head_k:  # これ以上追加するとK制約を超えるので打ち切り
                    break  # HEAD枠が埋まったら補欠探索を終了する
                cid = str(row["cluster_id"])
                if cid in head_infos:  # 既存HEADクラスタは情報追記のみで新規採用はしない
                    # 既に採用済みのクラスタには、参考情報として日常放置の件数/時間を追記するだけで、新規追加はしない。
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

    def _from_after_charge(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """充電直後の長時間放置実績をクラスタごとに集計する補助メソッド。"""
        agg = (
            train_df.groupby("next_long_inactive_cluster")
            .agg(
                count_after_charge=("next_long_inactive_cluster", "size"),
                hours_after_charge=("inactive_time_minutes", lambda x: x.sum() / 60.0),
            )
            .reset_index()
        )
        agg.rename(columns={"next_long_inactive_cluster": "cluster_id"}, inplace=True)
        return agg

    def _from_anytime(self, train_cutoff: Timestamp) -> pd.DataFrame:
        """inactiveセッション全体から長時間放置クラスタの統計を取得する。"""
        sessions = self.sessions_df[
            (self.sessions_df["session_type"] == "inactive")  # 放置イベントのみ抽出（充電・走行は除外）
            & (
                self.sessions_df["duration_minutes"] >= self.config.min_long_inactive_minutes
            )  # 6時間以上の滞在に限定し、短時間放置をノイズ扱い
            & (self.sessions_df["start_time"] <= train_cutoff)  # 学習期間より未来の情報はリークなので除外
        ].copy()
        sessions = sessions[sessions["session_cluster"] != "-1"]  # -1はノイズクラスタ。HEAD候補には採用しない
        agg = (
            sessions.groupby("session_cluster")
            .agg(
                count_anytime=("session_cluster", "size"),
                hours_anytime=("duration_minutes", lambda x: x.sum() / 60.0),
            )
            .reset_index()
        )
        #  ここで得られるのは「充電条件を外した長時間放置の頻度」。HEADが埋まらないときの補欠要員として使う。
        agg.rename(columns={"session_cluster": "cluster_id"}, inplace=True)
        return agg


class VisitStatisticsBuilder:
    """§4で必要となる重心・遅延・時間分布を組み立てる。"""

    def __init__(self, config: PipelineConfig, sessions_df: pd.DataFrame) -> None:
        """設定とセッション一覧を保持するだけの初期化処理。"""
        self.config = config
        self.sessions_df = sessions_df

    def build(self, train_df: pd.DataFrame, head_clusters: Sequence[str]) -> Tuple[
        Dict[str, Tuple[Optional[float], Optional[float]]],
        Dict[str, float],
        Dict[Tuple[str, str], float],
        Dict[str, np.ndarray],
        Dict[str, Dict[object, Dict[str, float]]],
    ]:
        """重心・遅延・時間分布・遷移確率をまとめて計算し、後続の特徴量生成に渡す。"""
        if not head_clusters:
            return {}, {}, {}, {}, {}
        #  HEADクラスタに関する統計は「学習で何を信じるか」の土台。
        # 充電クラスタ×放置クラスタの遅延はデータが乏しいほど全体中央値へフォールバックするので、Noneではなく0.0で返すところに注目。
        train_cutoff = train_df["charge_start_time"].max()
        centroids = self._centroids(train_df, head_clusters, train_cutoff)
        cluster_delays, pair_delays = self._delays(train_df, head_clusters)
        p_start = self._p_start(head_clusters, train_cutoff)
        if self.config.feature_toggles.use_prev_transition:
            transitions = self._transitions(train_df, head_clusters)
        else:
            base_targets = set(head_clusters) | {OTHER_LABEL}
            if not base_targets:
                base_targets = {OTHER_LABEL}
            uniform_prob = 1.0 / len(base_targets)
            transitions = {
                GLOBAL_TRANS_KEY: {DEFAULT_BIN_KEY: {cid: uniform_prob for cid in base_targets}}
            }
        return centroids, cluster_delays, pair_delays, p_start, transitions

    def _centroids(
        self,
        train_df: pd.DataFrame,
        head_clusters: Sequence[str],
        train_cutoff: Timestamp,
    ) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
        """HEADクラスタごとの代表座標をtrainデータ＋通常セッションから推定する。"""
        centroids: Dict[str, Tuple[Optional[float], Optional[float]]] = {}

        if {"inactive_lat", "inactive_lon"}.issubset(train_df.columns):
            #  まずは学習用データ（充電→放置のペア）だけで重心を求め、実績から逆算する。
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
            # 学習期間外の通常放置セッションも統計に取り込み、欠損した重心を補う。
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

    def _delays(
        self, train_df: pd.DataFrame, head_clusters: Sequence[str]
    ) -> Tuple[Dict[str, float], Dict[Tuple[str, str], float]]:
        """クラスタ単位およびステーション×クラスタ単位の遅延統計を求める。"""
        if train_df.empty:
            base = {cid: 0.0 for cid in head_clusters}
            return base, {}

        delays = train_df["inactive_start_time"] - train_df["charge_start_time"]
        delays_hours = delays.dt.total_seconds() / 3600.0
        #  遅延の中央値は「充電→放置までの感覚」を表す。異常値が多いので平均ではなく中央値を採用している。
        overall = float(np.median(delays_hours.dropna())) if not delays_hours.dropna().empty else 0.0

        cluster_delay_map: Dict[str, float] = {}
        grouped = train_df.groupby("next_long_inactive_cluster")
        for cid in head_clusters:
            if cid in grouped.groups:
                values = delays_hours.loc[grouped.groups[cid]].dropna()
                if len(values) >= self.config.min_delay_samples:
                    cluster_delay_map[cid] = float(np.median(values))
        for cid in head_clusters:
            cluster_delay_map.setdefault(cid, overall)
            cluster_delay_map[cid] = float(np.clip(cluster_delay_map[cid], 0.0, self.config.max_time_shift_hours))

        pair_delay_map: Dict[Tuple[str, str], float] = {}
        if "charge_cluster" in train_df.columns:
            pair_group = (
                train_df.dropna(subset=["charge_cluster", "next_long_inactive_cluster"])
                .groupby(["charge_cluster", "next_long_inactive_cluster"])
            )
            #  充電ステーションごとに放置先の遅延傾向が異なるので、可能な限りペア単位で中央値を計算する。
            # サンプルが少ないときはmin_pair_delay_samplesで足切りし、過学習を避ける。
            for (charge_cluster, target_cluster), indexer in pair_group.groups.items():
                target_cluster_str = str(target_cluster)
                if target_cluster_str not in head_clusters:
                    continue
                values = delays_hours.loc[indexer].dropna()
                if len(values) >= self.config.min_pair_delay_samples:
                    charge_key = str(charge_cluster)
                    delay_val = float(np.median(values))
                    delay_val = float(np.clip(delay_val, 0.0, self.config.max_time_shift_hours))
                    pair_delay_map[(charge_key, target_cluster_str)] = delay_val

        return cluster_delay_map, pair_delay_map

    def _p_start(self, head_clusters: Sequence[str], train_cutoff: Timestamp) -> Dict[str, np.ndarray]:
        """曜日×時間帯の開始確率行列（time_compat辞書）を構築する。"""
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


    def _transitions(
        self, train_df: pd.DataFrame, head_clusters: Sequence[str]
    ) -> Dict[str, Dict[object, Dict[str, float]]]:
        """prevクラスタ×曜日×時間帯から次クラスタへの遷移確率を算出する。"""
        alpha = self.config.laplace_alpha
        bins_per_day = int(24 / self.config.time_bin_hours)
        counts_prev_time: Dict[str, Dict[Tuple[int, int], Dict[str, float]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        counts_prev_total: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        counts_global_time: Dict[Tuple[int, int], Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        counts_global_total: Dict[str, float] = defaultdict(float)

        if train_df.empty or "prev_long_inactive_cluster" not in train_df.columns:
            base_targets = set(head_clusters) | {OTHER_LABEL}
            if not base_targets:
                base_targets = {OTHER_LABEL}
            denominator = float(len(base_targets)) if base_targets else 1.0
            base_probs = {cid: 1.0 / denominator for cid in base_targets} if base_targets else {OTHER_LABEL: 1.0}
            return {GLOBAL_TRANS_KEY: {DEFAULT_BIN_KEY: base_probs}}

        valid_rows = train_df[
            train_df["prev_long_inactive_cluster"].notna()
            & train_df["next_long_inactive_cluster"].notna()
            & train_df["charge_start_time"].notna()
        ]

        for row in valid_rows.itertuples():
            prev_cluster = str(row.prev_long_inactive_cluster)
            target_cluster = str(row.next_long_inactive_cluster)
            charge_start: Timestamp = row.charge_start_time
            if charge_start is pd.NaT:
                continue
            local_time = (
                charge_start.tz_convert("Asia/Tokyo")
                if getattr(charge_start, "tz", None) is not None
                else charge_start.tz_localize("Asia/Tokyo")
            )
            hour = local_time.hour + local_time.minute / 60.0
            bin_idx = int(hour // self.config.time_bin_hours) % bins_per_day
            dow = local_time.weekday()
            key = (dow, bin_idx)
            counts_prev_time[prev_cluster][key][target_cluster] += 1.0
            counts_prev_total[prev_cluster][target_cluster] += 1.0
            counts_global_time[key][target_cluster] += 1.0
            counts_global_total[target_cluster] += 1.0

        all_targets = set(head_clusters) | set(counts_global_total.keys()) | {OTHER_LABEL}
        if not all_targets:
            all_targets = {OTHER_LABEL}

        def normalize(counts: Dict[str, float]) -> Dict[str, float]:
            targets = set(all_targets) | set(counts.keys())
            denom = sum(counts.get(cid, 0.0) + alpha for cid in targets)
            if denom == 0.0:
                uniform = 1.0 / len(targets) if targets else 1.0
                return {cid: uniform for cid in targets}
            return {cid: (counts.get(cid, 0.0) + alpha) / denom for cid in targets}

        transitions: Dict[str, Dict[object, Dict[str, float]]] = {}
        for prev_cluster, bucket in counts_prev_time.items():
            dest: Dict[object, Dict[str, float]] = {}
            for key, counts in bucket.items():
                dest[key] = normalize(counts)
            dest[DEFAULT_BIN_KEY] = normalize(counts_prev_total.get(prev_cluster, {}))
            transitions[prev_cluster] = dest

        global_bucket: Dict[object, Dict[str, float]] = {}
        for key, counts in counts_global_time.items():
            global_bucket[key] = normalize(counts)
        global_bucket[DEFAULT_BIN_KEY] = normalize(counts_global_total)
        transitions[GLOBAL_TRANS_KEY] = global_bucket
        return transitions
class ClassFeatureAssembler:
    """§4.2 クラスタ依存特徴を生成する。"""

    def __init__(self, config: PipelineConfig) -> None:
        """クラスタ依存特徴を作成する際の設定値とトグルを保持する。"""
        self.config = config
        self.toggles = config.feature_toggles

    def transform(
        self,
        model_df: pd.DataFrame,
        head_clusters: Sequence[str],
        centroids: Dict[str, Tuple[Optional[float], Optional[float]]],
        cluster_delays: Dict[str, float],
        pair_delays: Dict[Tuple[str, str], float],
        p_start: Dict[str, np.ndarray],
        transitions: Dict[str, Dict[object, Dict[str, float]]],
    ) -> pd.DataFrame:
        """距離・頻度・再訪間隔・時間適合度・遷移確率を横持ちで組み立てる。"""
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
            if toggles.use_prev_transition:
                records[f"transition_from_prev_to_{cid}"] = []

        head_count = max(len(head_clusters), 1)
        freq_events = {cid: deque() for cid in head_clusters}
        last_visit_state = {cid: None for cid in head_clusters}
        window_days = max(float(self.config.frequency_window_days), 0.0)
        freq_window = pd.Timedelta(days=window_days) if window_days > 0 else None
        #  dequeで「直近frequency_window_daysに起きた訪問時刻」だけを保持し、ウィンドウ外のものはその場で捨てる。
        # ここを素直なカウントで書くとO(N^2)になるので、双方向キューで効率良く削除している。

        for _, row in model_df.iterrows():
            charge_lat = row.get("charge_lat")
            charge_lon = row.get("charge_lon")
            charge_start: Timestamp = row["charge_start_time"]
            origin_cluster_raw = row.get("charge_cluster")
            origin_cluster = str(origin_cluster_raw) if pd.notna(origin_cluster_raw) else None
            prev_cluster_raw = row.get("prev_long_inactive_cluster")
            prev_cluster = str(prev_cluster_raw) if pd.notna(prev_cluster_raw) else None

            for cid in head_clusters:
                centroid_lat, centroid_lon = centroids.get(cid, (None, None))

                if toggles.use_distance:
                    if centroid_lat is None or centroid_lon is None or pd.isna(charge_lat) or pd.isna(charge_lon):
                        dist_value = float("nan")
                    else:
                        dist_value = haversine_km(charge_lat, charge_lon, centroid_lat, centroid_lon)
                        # メートル単位の整数に変換してモデルが扱いやすいスケールに調整
                        dist_value = round(dist_value * 1000.0, 0)
                    # NaNはそのまま残し、学習時に欠損扱いに任せる
                    records[f"dist_to_{cid}"].append(dist_value if not math.isnan(dist_value) else float("nan"))

                if toggles.use_frequency:
                    events = freq_events[cid]
                    if freq_window is not None and not pd.isna(charge_start):
                        cutoff = charge_start - freq_window
                        # ウィンドウ外（古すぎる訪問）は左端から順次捨てる。dequeを使っているのはこのポップ処理がO(1)になるため。
                        while events and events[0] < cutoff:
                            events.popleft()
                    records[f"freq_hashvin_{cid}"].append(len(events) + self.config.laplace_alpha)

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
                    delay = self._resolve_delay(origin_cluster, cid, cluster_delays, pair_delays)
                    #  delayはcharge_cluster→cidのペア遅延を優先し、なければクラスタ単体の中央値にフォールバックする。
                    # こうすることで「自宅→自宅」のような即時放置と「職場→自宅」のような長距離を区別できる。
                    score = self._time_compat_score(charge_start, matrix, delay)
                    records[f"time_compat_{cid}"].append(score)

                if toggles.use_prev_transition:
                    trans_prob = self._transition_score(prev_cluster, cid, charge_start, transitions, head_count)
                    records[f"transition_from_prev_to_{cid}"].append(trans_prob)

            if row["split"] == "train":
                target = row["next_long_inactive_cluster"]
                if target in head_clusters:
                    event_time = row.get("inactive_start_time")
                    if pd.notna(event_time):
                        freq_events[target].append(event_time)
                        last_visit_state[target] = event_time

        return pd.DataFrame(records, index=model_df.index)

    def _resolve_delay(
        self,
        origin_cluster: Optional[str],
        target_cluster: str,
        cluster_delays: Dict[str, float],
        pair_delays: Dict[Tuple[str, str], float],
    ) -> float:
        """ペア専用遅延があればそれを優先し、なければクラスタ遅延を返す。"""
        if origin_cluster is not None:
            key = (origin_cluster, target_cluster)
            if key in pair_delays:
                return pair_delays[key]
        return cluster_delays.get(target_cluster, 0.0)

    def _time_compat_score(
        self,
        charge_start: Timestamp,
        matrix: Optional[np.ndarray],
        delay_hours: float,
    ) -> float:
        """指定された遅延後の時間帯における開始確率を近傍補間付きで算出する。"""
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

    def _transition_score(
        self,
        prev_cluster: Optional[str],
        target_cluster: str,
        charge_start: Timestamp,
        transitions: Dict[str, Dict[object, Dict[str, float]]],
        head_cluster_count: int,
    ) -> float:
        """prevクラスタ・時間帯から次クラスタへの遷移確率を取得する。"""
        bucket: Optional[Dict[object, Dict[str, float]]] = None
        search_key = str(prev_cluster) if prev_cluster else None
        if search_key and search_key in transitions:
            bucket = transitions.get(search_key)
        if bucket is None:
            bucket = transitions.get(GLOBAL_TRANS_KEY, {})
        probs: Optional[Dict[str, float]] = None
        if charge_start is not pd.NaT and bucket:
            local_time = (
                charge_start.tz_convert("Asia/Tokyo")
                if getattr(charge_start, "tz", None) is not None
                else charge_start.tz_localize("Asia/Tokyo")
            )
            bins_per_day = int(24 / self.config.time_bin_hours)
            hour = local_time.hour + local_time.minute / 60.0
            bin_idx = int(hour // self.config.time_bin_hours) % bins_per_day
            key = (local_time.weekday(), bin_idx)
            probs = bucket.get(key)
        if probs is None and bucket is not None:
            probs = bucket.get(DEFAULT_BIN_KEY)
        if probs is None:
            global_bucket = transitions.get(GLOBAL_TRANS_KEY, {})
            if charge_start is not pd.NaT and global_bucket:
                local_time = (
                    charge_start.tz_convert("Asia/Tokyo")
                    if getattr(charge_start, "tz", None) is not None
                    else charge_start.tz_localize("Asia/Tokyo")
                )
                bins_per_day = int(24 / self.config.time_bin_hours)
                hour = local_time.hour + local_time.minute / 60.0
                bin_idx = int(hour // self.config.time_bin_hours) % bins_per_day
                key = (local_time.weekday(), bin_idx)
                probs = global_bucket.get(key)
            if probs is None:
                probs = global_bucket.get(DEFAULT_BIN_KEY) if global_bucket else None
        if probs is None or head_cluster_count == 0:
            return 1.0 / max(head_cluster_count, 1)
        return float(probs.get(target_cluster, probs.get(OTHER_LABEL, 1.0 / head_cluster_count)))

class CommonFeatureAssembler:
    """共通（クラスタ非依存）特徴量を生成する責務を持つクラス。

    充電開始時刻から派生する時間情報と、SOC・ステーション種別など「どの候補クラスタにも共通で使い回せる」
    情報をまとめて組み立てる。時刻の表現方法は FeatureToggleConfig.time_feature_mode で切り替える。
    """

    def __init__(self, config: PipelineConfig) -> None:
        """共通特徴の生成に必要な設定（時間表現のモードなど）を保持する。"""
        self.config = config
        self.toggles = config.feature_toggles

    def transform(self, model_df: pd.DataFrame) -> pd.DataFrame:
        """共通特徴量をDataFrameとして返す。

        Parameters
        ----------
        model_df : pd.DataFrame
            充電イベントを1行とした学習用テーブル。`charge_start_time` 列が必須。

        Returns
        -------
        pd.DataFrame
            共通特徴量のみを列に持つDataFrame。indexは `model_df` と同一。
        """
        charge_start = model_df["charge_start_time"]

        # タイムゾーン付きのままではモデルが扱いづらいので、一旦JSTへそろえた上でtz情報を外す。
        charge_start_local = charge_start.dt.tz_convert("Asia/Tokyo") if getattr(charge_start.dt, "tz", None) else charge_start
        charge_start_naive = charge_start_local.dt.tz_localize(None)

        columns: Dict[str, pd.Series] = {}

        # 時刻特徴の表現方法をまとめて判定（cyclic/datetime/categorical/all）。
        mode = (self.toggles.time_feature_mode or "cyclic").strip().lower()
        include_cyclic = mode in {"cyclic", "all"}
        include_raw = mode in {"datetime", "raw", "all"}
        include_categorical = mode in {"categorical", "all"}

        hour_fraction = charge_start_naive.dt.hour + charge_start_naive.dt.minute / 60.0
        hour_radian = 2 * math.pi * hour_fraction / 24.0

        if include_cyclic:
            # sin/cos で24時間周期を滑らかに表現。dowは曜日の整数インデックス（0=月〜6=日）。
            columns["dow"] = charge_start_naive.dt.weekday
            columns["hour_sin"] = np.sin(hour_radian)
            columns["hour_cos"] = np.cos(hour_radian)

        if include_raw:
            # AutoGluonはdatetime型も扱えるため、生の時刻とUNIX秒をあわせて渡せるようにする。
            columns["time_raw_charge_start"] = charge_start_naive
            # view で得られるUNIXナノ秒値を秒に換算し、NaT部分はmaskでNaNへ変換する。
            timestamp_series = charge_start_naive.view("int64").astype(float) / 1_000_000_000
            timestamp_series = timestamp_series.mask(charge_start_naive.isna(), np.nan)
            columns["time_raw_timestamp"] = timestamp_series

        if include_categorical:
            # 2時間幅のカテゴリ（00_02, 02_04, ...）と曜日カテゴリを作成し、決定木系モデルで扱いやすくする。
            hour_value = charge_start_naive.dt.hour.astype("float")
            band_start = (hour_value // 2) * 2
            band_labels = band_start.map(
                lambda h: "unknown" if np.isnan(h) else f"{int(h):02d}_{int((h + 2) % 24):02d}"
            )
            categories = [f"{i:02d}_{(i + 2) % 24:02d}" for i in range(0, 24, 2)]
            columns["time_band_2h"] = pd.Categorical(band_labels, categories=categories + ["unknown"], ordered=False)

            dow_labels = charge_start_naive.dt.weekday.map({0: "mon", 1: "tue", 2: "wed", 3: "thu", 4: "fri", 5: "sat", 6: "sun"})
            dow_labels = dow_labels.fillna("unknown")
            columns["time_dow_category"] = pd.Categorical(
                dow_labels, categories=["mon", "tue", "wed", "thu", "fri", "sat", "sun", "unknown"], ordered=False
            )

        if self.toggles.use_prev_vector and {"prev_inactive_lat", "prev_inactive_lon", "charge_lat", "charge_lon"}.issubset(model_df.columns):
            prev_lat = pd.to_numeric(model_df["prev_inactive_lat"], errors="coerce")
            prev_lon = pd.to_numeric(model_df["prev_inactive_lon"], errors="coerce")
            charge_lat_series = pd.to_numeric(model_df["charge_lat"], errors="coerce")
            charge_lon_series = pd.to_numeric(model_df["charge_lon"], errors="coerce")
            avg_lat = (prev_lat + charge_lat_series) / 2.0
            delta_lat = charge_lat_series - prev_lat
            delta_lon = charge_lon_series - prev_lon
            delta_lat_m = delta_lat * 111_320.0
            delta_lon_m = delta_lon * (40075000.0 * np.cos(np.radians(avg_lat)) / 360.0)
            distance_m = np.vectorize(
                lambda la1, lo1, la2, lo2: np.nan
                if np.any(np.isnan([la1, lo1, la2, lo2]))
                else haversine_km(la1, lo1, la2, lo2) * 1000.0
            )(prev_lat, prev_lon, charge_lat_series, charge_lon_series)
            columns["prev_to_charge_delta_lat_m"] = pd.Series(delta_lat_m, index=model_df.index)
            columns["prev_to_charge_delta_lon_m"] = pd.Series(delta_lon_m, index=model_df.index)
            columns["prev_to_charge_distance_m"] = pd.Series(distance_m, index=model_df.index)

        # SOCの開始値は常に保持しておく（欠損はそのままNaN）。
        columns["soc_start"] = model_df.get("charge_start_soc")

        if self.toggles.use_behavior_flags:
            # 帰宅帯（18-6時）や通勤帯（朝/日中）など、ドメイン知識ベースの粗いフラグ。
            hour_int = charge_start_naive.dt.hour
            columns.update(
                {
                    "is_return_band": ((hour_int >= 18) | (hour_int < 6)).astype("Int8", copy=False),
                    "is_commute_band": (
                        ((hour_int >= 7) & (hour_int < 10)) | ((hour_int >= 9) & (hour_int < 18))
                    ).astype("Int8", copy=False),
                    "weekend_flag": charge_start_naive.dt.weekday.isin([5, 6]).astype("Int8", copy=False),
                }
            )

        common_df = pd.DataFrame(columns, index=model_df.index)

        if self.toggles.use_station_type:
            # ステーション種別は別メソッドでone-hot化し、共通特徴と結合する。
            station_df = self._station_type_feature(model_df)
            if not station_df.empty:
                common_df = pd.concat([common_df, station_df], axis=1)

        return common_df

    def _station_type_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        """ステーション種別のone-hot表現を返す。情報が無ければ空DataFrame。"""
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
        """hashvinごとのデータセットと設定を保持し、以降の処理を実行できるようにする。"""
        self.hashvin = hashvin
        self.charge_df = charge_df.copy()
        self.sessions_df = sessions_df.copy()
        self.config = config
        self._prepare_dataframe_types()

    def _prepare_dataframe_types(self) -> None:
        """入力データの型を整え、後続処理の前提を満たす。"""
        #  生データは型が混在するため、最初にdatetime/数値へ矯正しないと統計計算でエラーになる。
        # tz付きdatetimeが揃っていれば「充電開始時刻で並べる」等の比較がズレなくなる点に注目。
        for col in ["charge_start_time", "charge_end_time", "inactive_start_time", "inactive_end_time"]:
            if col in self.charge_df.columns:
                self.charge_df[col] = ensure_datetime(self.charge_df[col])
        for col in ["start_time", "end_time"]:
            if col in self.sessions_df.columns:
                self.sessions_df[col] = ensure_datetime(self.sessions_df[col])
                
        for col in [
            "charge_durations_minutes",
            "inactive_time_minutes",
            "start_soc",
            "end_soc",
            "charge_start_soc",
            "charge_end_soc",
        ]:
            if col in self.charge_df.columns:
                # SOCは小数・文字列混在がありがち。to_numericで落ちる値はNaNにして後続処理に委ねる。
                self.charge_df[col] = pd.to_numeric(self.charge_df[col], errors="coerce")

        if "next_long_inactive_cluster" in self.charge_df.columns:
            self.charge_df["next_long_inactive_cluster"] = self.charge_df["next_long_inactive_cluster"].astype(str)
        if "session_cluster" in self.sessions_df.columns:
            # クラスタIDのプレフィックス（例: I_）は後段の結合条件で利用するため、絶対に削除しない。
            # 文字列化だけ行い、表示上の違和感があっても解析ロジックを優先する。
            self.sessions_df["session_cluster"] = self.sessions_df["session_cluster"].astype(str)

    def _select_model_rows(self) -> pd.DataFrame:
        """教師データ対象（next_long_inactive_clusterあり）を抽出し、時間順に整列する。"""
        #  予測対象は「充電完了時点で既に決まっている次の長時間放置」。ラベルが欠けている行は学習に使えないので除外する。
        # ソートキーはcharge_start_time。リーク防止でcharge_end_timeではない点を必ず押さえておく。
        df = self.charge_df.copy()
        df = df[df["next_long_inactive_cluster"].notna()].copy()
        df.sort_values("charge_start_time", inplace=True)
        df.reset_index(drop=True, inplace=True)
        df["session_order"] = np.arange(len(df))
        df["session_uid"] = df["hashvin"].astype(str) + "_" + df["session_order"].astype(str)
        # 直前の長時間放置クラスタ・座標を記録し、遷移特徴や移動ベクトルに利用する。
        df["prev_long_inactive_cluster"] = df["next_long_inactive_cluster"].shift(1)
        for coord in ["inactive_lat", "inactive_lon"]:
            if coord in df.columns:
                df[f"prev_{coord}"] = df[coord].shift(1)
        return df

    def _add_daily_time_bins(self, model_df: pd.DataFrame) -> pd.DataFrame:
        """当日（または直近24h）の時間帯別滞在クラスタを特徴量化する。"""
        if not self.config.feature_toggles.use_daily_time_bins:
            return model_df

        bin_hours = max(int(self.config.day_time_bin_hours), 1)
        window_mode = (self.config.day_time_window_mode or "rolling").lower()
        window_hours = max(int(self.config.day_time_window_hours), 1)

        sessions = self.sessions_df.copy()
        if "hashvin" in sessions.columns:
            sessions = sessions[sessions["hashvin"] == self.hashvin]
        sessions = sessions.dropna(subset=["start_time", "end_time"])

        def _normalize_ts(ts: Timestamp, tz: Optional[str] = "Asia/Tokyo") -> Timestamp:
            if ts is pd.NaT:
                return ts
            if getattr(ts, "tzinfo", None) is None:
                return ts.tz_localize(tz)
            return ts.tz_convert(tz)

        def _bin_labels(total_hours: int) -> List[str]:
            labels: List[str] = []
            for start_hour in range(0, total_hours, bin_hours):
                end_hour = start_hour + bin_hours
                labels.append(f"{start_hour:02d}_{end_hour:02d}")
            return labels

        column_buffers: Dict[str, List[object]] = {}

        for _, row in model_df.iterrows():
            charge_start: Timestamp = row.get("charge_start_time")
            if charge_start is pd.NaT:
                continue
            charge_start_local = _normalize_ts(charge_start)

            if window_mode == "calendar":
                window_start_local = charge_start_local.floor("D")
                window_end_local = charge_start_local
                total_hours = 24
            else:
                total_hours = max(window_hours, bin_hours)
                window_start_local = charge_start_local - pd.Timedelta(hours=total_hours)
                window_end_local = charge_start_local

            labels = _bin_labels(total_hours)
            for label in labels:
                column_buffers.setdefault(f"daily_bin_{label}_cluster", [])
                column_buffers.setdefault(f"daily_bin_{label}_minutes", [])

            mask = (sessions["end_time"] > window_start_local) & (sessions["start_time"] < window_end_local)
            window_sessions = sessions.loc[mask]

            for i, label in enumerate(labels):
                bin_start = window_start_local + pd.Timedelta(hours=i * bin_hours)
                bin_end = bin_start + pd.Timedelta(hours=bin_hours)
                if bin_start >= window_end_local:
                    column_buffers[f"daily_bin_{label}_cluster"].append("unknown")
                    column_buffers[f"daily_bin_{label}_minutes"].append(0.0)
                    continue
                if bin_end > window_end_local:
                    bin_end = window_end_local

                cluster_minutes: Dict[str, float] = defaultdict(float)
                for sess in window_sessions.itertuples():
                    sess_start = _normalize_ts(sess.start_time)
                    sess_end = _normalize_ts(sess.end_time)
                    if sess_end <= bin_start or sess_start >= bin_end:
                        continue
                    overlap_start = max(sess_start, bin_start)
                    overlap_end = min(sess_end, bin_end)
                    minutes = (overlap_end - overlap_start).total_seconds() / 60.0
                    if minutes <= 0:
                        continue
                    cluster_id = str(getattr(sess, "session_cluster", OTHER_LABEL) or OTHER_LABEL)
                    cluster_minutes[cluster_id] += minutes

                if cluster_minutes:
                    best_cluster, max_minutes = max(cluster_minutes.items(), key=lambda item: item[1])
                else:
                    best_cluster, max_minutes = "unknown", 0.0
                column_buffers[f"daily_bin_{label}_cluster"].append(best_cluster)
                column_buffers[f"daily_bin_{label}_minutes"].append(max_minutes)

        # 未処理行があればunknown/0で埋める
        for col, values in column_buffers.items():
            if len(values) < len(model_df):
                fill_value = "unknown" if col.endswith("_cluster") else 0.0
                values.extend([fill_value] * (len(model_df) - len(values)))
            model_df[col] = values
        return model_df

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

        # 当日行動ヒストグラムを付与（必要な場合のみ）
        model_df = self._add_daily_time_bins(model_df)

        stats_builder = VisitStatisticsBuilder(self.config, self.sessions_df)
        centroids, cluster_delays, pair_delays, p_start, transitions = stats_builder.build(
            model_df[model_df["split"] == "train"], head_clusters
        )

        for info in head_details:
            lat, lon = centroids.get(info.cluster_id, (None, None))
            info.centroid_lat = lat
            info.centroid_lon = lon
            info.delay_hours = cluster_delays.get(info.cluster_id, 0.0)
            matrix = p_start.get(info.cluster_id)
            info.p_start_matrix = matrix.tolist() if matrix is not None else []

        class_assembler = ClassFeatureAssembler(self.config)
        class_features = class_assembler.transform(
            model_df, head_clusters, centroids, cluster_delays, pair_delays, p_start, transitions
        )

        common_assembler = CommonFeatureAssembler(self.config)
        common_features = common_assembler.transform(model_df)

        feature_df = pd.concat([model_df, common_features, class_features], axis=1)
        if feature_df.columns.duplicated().any():
            # 若手メモ: 時刻特徴の表現モードによっては同名列が二重に生成されることがあるため、
            # 最初に出現した列を残して重複列を除去しておく（AutoGluonは重複列を許容しない）。
            feature_df = feature_df.loc[:, ~feature_df.columns.duplicated()]

        # 予測時に未確定の値は特徴量として保持しない
        leak_cols = [
            "charge_durations_minutes",
            "charge_duration_minutes",
            "charge_end_soc",
            "soc_delta",
            "end_soc",
            "charge_end_time",
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
