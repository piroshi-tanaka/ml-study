
"""EV長時間放置クラスタ予測の前処理パイプライン。

train_ranking要件.mdに基づき、候補生成＋二値スコアリング方式で放置場所を予測する。
- hashvin単位の独立処理・時系列Split
- 適応ウィンドウによる候補生成（充電間隔に応じた動的ウィンドウ）
- 候補依存特徴量と共通特徴量の生成
- 二値スコアリング用の学習テーブル構築
"""
from __future__ import annotations

import math
from bisect import bisect_left
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from pandas import Timestamp

OTHER_LABEL = "OTHER"
GLOBAL_TRANS_KEY = "__GLOBAL__"
DEFAULT_BIN_KEY = "__DEFAULT__"


def ensure_datetime(series: pd.Series) -> pd.Series:
    """JSTタイムゾーン付きのdatetime64に正規化するユーティリティ関数。

    Parameters
    ----------
    series : pd.Series
        日時情報を含む列。**入力データはJSTタイムゾーン付きが前提**。

    Returns
    -------
    pd.Series
        タイムゾーン付きdatetime64（Asia/Tokyo）。変換できない値はNaTになる。

    Notes
    -----
    入力データは既にJSTタイムゾーン付きであることを前提としているが、
    万が一タイムゾーンが欠けている場合はJSTとして解釈する。
    下流の処理（時系列ソートやリーク防止チェック）はタイムゾーン付きdatetimeを前提にしている。
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        if getattr(series.dt, "tz", None) is None:
            # タイムゾーンが欠けている場合はJSTとして解釈
            return series.dt.tz_localize("Asia/Tokyo")
        return series
    parsed = pd.to_datetime(series, errors="coerce")
    if getattr(parsed.dt, "tz", None) is None:
        # タイムゾーンが欠けている場合はJSTとして解釈
        return parsed.dt.tz_localize("Asia/Tokyo")
    return parsed


def haversine_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """地球上の2地点間距離をハバーサイン公式で計算する（メートル単位）。

    距離特徴はモデルの中核なので、緯度経度が欠損している場合は NaN を返し、
    後段で適切に欠損処理できるようにしている。

    Parameters
    ----------
    lat1, lon1 : float
        始点の緯度・経度
    lat2, lon2 : float
        終点の緯度・経度

    Returns
    -------
    float
        2地点間の距離（メートル）。欠損がある場合はNaN。
    """
    if any(pd.isna([lat1, lon1, lat2, lon2])):
        return float("nan")
    r = 6371000.0  # 地球の半径（メートル）
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
    distance_m = haversine_meters(lat1, lon1, lat2, lon2)
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


def compute_charge_interval_median(train_df: pd.DataFrame, default_days: float = 14.0) -> float:
    """充電開始時刻の間隔中央値（日数）を求め、候補生成用ウィンドウの基準値とする。
    
    この関数は、hashvinごとの充電習慣（充電間隔）を定量化し、候補生成ウィンドウの基準値を決定する。
    
    【活用箇所】
    1. 候補生成時の適応ウィンドウ幅の算出（CandidateGenerator._compute_window_days）
       - 充電間隔が短い人（例：3日ごと）→ウィンドウを短く（直近の最新傾向を重視）
       - 充電間隔が長い人（例：20日ごと）→ウィンドウを長く（十分なデータ量を確保）
    
    2. 放置回数・頻度特徴の集計期間
       - 短い間隔で充電する人は直近の短期間データから習慣を把握
       - 長い間隔の人は長期間のデータから習慣を抽出
    
    【ロジック詳細】
    - 充電開始時刻を時系列順に並べ、隣接する充電間の時間差を算出
    - 中央値を採用することで外れ値（極端に長い/短い間隔）の影響を抑制
    - データ不足時はdefault_days（既定14日）を返す
    
    Parameters
    ----------
    train_df : pd.DataFrame
        学習用の充電データ。charge_start_time列を含む。
    default_days : float
        充電間隔が算出できない場合の既定値（日数）。
    
    Returns
    -------
    float
        充電間隔の中央値（日数）。算出不可時はdefault_daysを返す。
    """
    if train_df.empty or "charge_start_time" not in train_df.columns:
        return default_days
    times = train_df["charge_start_time"].dropna().sort_values()
    if len(times) < 2:
        return default_days
    deltas = times.diff().dropna()
    if deltas.empty:
        return default_days
    median_days = float(deltas.median().total_seconds() / 86400.0)
    if not math.isfinite(median_days) or median_days <= 0:
        return default_days
    return median_days


def assign_splits(n_rows: int, train_ratio: float = 0.7, valid_ratio: float = 0.1) -> List[str]:
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
    """特徴量の切替と時刻表現を管理する設定クラス。

    各フラグは「この特徴量を使う／使わない」を明示し、実験時にON/OFFを切り替えやすくする。

    Attributes
    ----------
    use_distance : bool, default=True
        候補クラスタまでの距離特徴（cand_distance_m）を使用するか。
        充電地点から候補クラスタ重心までのハバーサイン距離（メートル）。
        近接優位の物理的制約を反映する最重要特徴。

    use_frequency : bool, default=True
        候補クラスタへの訪問頻度特徴（cand_freq_in_window, cand_freq_log）を使用するか。
        適応ウィンドウ内での訪問回数をカウント。個体の習慣の強さを表す。

    use_recency : bool, default=True
        候補クラスタへの最終訪問からの経過時間特徴（cand_recency_hours）を使用するか。
        最後に訪問してからの時間（時間）。再訪周期を反映。

    use_time_compat : bool, default=True
        時間相性特徴（cand_time_compat）を使用するか。
        曜日×時間帯の開始確率と遅延中央値から算出。HOME/WORKの時間習慣を反映。

    use_station_type : bool, default=False
        充電ステーション種別特徴（station_*）を使用するか。
        200V/急速などのone-hot表現。充電の性質が次行動に与える影響を捉える。

    use_prev_transition : bool, default=True
        直前クラスタからの遷移確率特徴（cand_transition_prev）を使用するか。
        前回の長時間放置クラスタから次クラスタへの遷移傾向を時間帯込みで捉える。

    use_daily_time_bins : bool, default=True
        当日（または直近時間）の時間帯別滞在クラスタ特徴（daily_bin_*）を使用するか。
        充電前の行動パターン（どのクラスタで何分滞在したか）を時間帯別に記録。

    time_feature_mode : str, default='cyclic'
        時刻特徴の表現方法。以下から選択：
        - 'cyclic': sin/cosで24時間周期を滑らかに表現（hour_sin, hour_cos, dow）
        - 'datetime': 生の時刻とUNIXタイムスタンプ（time_raw_charge_start, time_raw_timestamp）
        - 'categorical': 2時間幅のカテゴリと曜日カテゴリ（time_band_2h, time_dow_category）
        - 'all': 上記すべてを含める
    """

    use_distance: bool = True
    use_frequency: bool = True
    use_recency: bool = True
    use_time_compat: bool = True
    use_station_type: bool = False
    use_prev_transition: bool = True
    use_daily_time_bins: bool = True
    time_feature_mode: str = "cyclic"

@dataclass
class PipelineConfig:
    """パイプライン全体で共有するハイパーパラメータを保持する設定クラス。

    Attributes
    ----------
    min_long_inactive_minutes : int, default=360
        「長時間放置」とみなす下限時間（分）。既定は360分（6時間）。
        これ以上の滞在を「長時間放置」として予測対象とする。

    laplace_alpha : float, default=1.0
        頻度・遷移確率のラプラス平滑に使う初期カウント。
        ゼロ頻度の候補でも最小限の確率を持たせることで未知クラスタへの過適合を防ぐ。

    recency_default_hours : float, default=1e6
        未訪問クラスタのRecency初期値（時間）。
        一度も訪問したことがないクラスタは「非常に古い」として扱う。

    time_bin_hours : int, default=4
        時間相性（time_compat）と遷移確率で使う時間ビン幅（時間）。
        24時間を何時間ごとに区切るか。既定4時間なら6ビン（0-4, 4-8, ...）。

    station_type_map : Optional[Dict[str, str]], default=None
        充電クラスタIDからステーション種別へのマッピング辞書。
        例: {"C_0001": "200V", "C_0002": "急速"}

    min_delay_samples : int, default=3
        クラスタ単位の遅延中央値を採用する最小サンプル数。
        これ未満の場合は全体中央値を使用。

    min_pair_delay_samples : int, default=2
        充電クラスタ×放置クラスタのペア遅延を採用する最小サンプル数。
        これ未満の場合はクラスタ単体の遅延にフォールバック。

    max_time_shift_hours : float, default=24.0
        遅延時間の上限（時間）。異常値を除外し計算の安定性を確保。

    frequency_window_days : float, default=90.0
        頻度カウントで参照する固定ウィンドウ日数（日）。
        ※現在は使用していない可能性あり（適応ウィンドウが優先）。

    day_time_bin_hours : int, default=6
        当日行動ヒストグラム（daily_bin_*）のビン幅（時間）。
        例: 6時間なら4ビン（00-06, 06-12, 12-18, 18-24）。

    day_time_window_hours : int, default=24
        当日行動ヒストグラムをrollingモードで集計する際の遡り時間（時間）。

    day_time_window_mode : str, default='calendar'
        当日行動ヒストグラムの集計モード。
        - 'rolling': 充電時刻からday_time_window_hours時間遡る
        - 'calendar': 当日0:00から充電時刻まで

    feature_toggles : FeatureToggleConfig
        特徴量のON/OFF切り替え設定。

    candidate_top_k : int, default=6
        1セッションあたりの候補クラスタ数の上限。
        計算量と学習安定性のバランスから5〜8が推奨。

    candidate_min_unique : int, default=4
        適応ウィンドウ内で最低限確保したいユニーククラスタ数。
        これ未満の場合はウィンドウを拡張。

    candidate_min_events : int, default=8
        適応ウィンドウ内で最低限確保したい長時間放置イベント数。
        これ未満の場合はウィンドウを拡張。

    candidate_recent_charge_count : int, default=10
        直近候補として参照する充電回数。
        この回数分の直近充電で訪れたクラスタを候補に含める。

    candidate_window_alpha : float, default=6.0
        適応ウィンドウの基本倍率。
        基本ウィンドウ日数 = 充電間隔中央値 × candidate_window_alpha。
        例: 充電間隔3日 × 6.0 = 18日間。

    candidate_min_days : int, default=7
        適応ウィンドウの下限（日）。
        これより短くならないよう制限。

    candidate_max_days : int, default=56
        適応ウィンドウの上限（日）。
        これより長くならないよう制限。

    candidate_expand_factor : float, default=1.5
        ウィンドウ拡張時の倍率。
        データ不足時、現在のウィンドウ日数にこの値を掛けて拡張。

    default_charge_interval_days : float, default=14.0
        充電間隔中央値が算出できない場合の既定値（日）。
        データが少ない初期段階で使用。
    """

    min_long_inactive_minutes: int = 360
    laplace_alpha: float = 1.0
    recency_default_hours: float = 1e6
    time_bin_hours: int = 4
    station_type_map: Optional[Dict[str, str]] = None
    min_delay_samples: int = 3
    min_pair_delay_samples: int = 2
    max_time_shift_hours: float = 24.0
    frequency_window_days: float = 90.0
    day_time_bin_hours: int = 6
    day_time_window_hours: int = 24
    day_time_window_mode: str = "calendar"
    feature_toggles: FeatureToggleConfig = field(default_factory=FeatureToggleConfig)
    candidate_top_k: int = 6
    candidate_min_unique: int = 4
    candidate_min_events: int = 8
    candidate_recent_charge_count: int = 10
    candidate_window_alpha: float = 6.0
    candidate_min_days: int = 7
    candidate_max_days: int = 56
    candidate_expand_factor: float = 1.5
    default_charge_interval_days: float = 14.0

    @property
    def enable_station_type(self) -> bool:
        """後方互換のための補助プロパティ。"""
        return self.feature_toggles.use_station_type


@dataclass
class HashvinResult:
    """hashvin単位で特徴テーブルとメタ情報を束ねる結果オブジェクト。"""

    hashvin: str
    features: pd.DataFrame
    split_datasets: Dict[str, pd.DataFrame]
    label_col: str
    session_col: str = "session_uid"
    candidate_col: str = "candidate_cluster"
    meta: Dict[str, object] = field(default_factory=dict)


@dataclass
class CandidateInfo:
    """候補クラスタと候補生成時のメタ情報を束ねる簡易データクラス。"""

    cluster_id: str
    freq_in_window: int
    recency_hours: float
    source: str
    in_window: bool
    window_days: float
    events_in_window: int
    distance_order: Optional[int] = None


class CandidateHistory:
    """hashvin内の長時間放置履歴を逐次蓄積し、候補生成時の統計を提供するヘルパークラス。"""

    def __init__(self, default_recency_hours: float, recent_limit: int) -> None:
        self.default_recency_hours = float(default_recency_hours)
        self.recent_limit = max(int(recent_limit), 1)
        self.events_by_cluster: Dict[str, List[Timestamp]] = defaultdict(list)
        self.event_times: List[Timestamp] = []
        self.recent_clusters: deque[str] = deque(maxlen=self.recent_limit)

    def _insert_sorted(self, items: List[Timestamp], value: Timestamp) -> None:
        """時刻リストへ昇順を維持したまま値を挿入する。"""
        idx = bisect_left(items, value)
        items.insert(idx, value)

    def register_event(
        self,
        cluster_id: Optional[str],
        charge_end_time: Optional[Timestamp],
        inactive_start_time: Optional[Timestamp],
    ) -> None:
        """実績イベントを履歴に追加する。"""
        if cluster_id is None or pd.isna(cluster_id):
            return
        cid = str(cluster_id)
        event_time = inactive_start_time if pd.notna(inactive_start_time) else charge_end_time
        if event_time is None or event_time is pd.NaT:
            return
        self._insert_sorted(self.events_by_cluster[cid], event_time)
        self._insert_sorted(self.event_times, event_time)
        self.recent_clusters.append(cid)

    def clusters_in_window(self, window_start: Optional[Timestamp]) -> List[str]:
        """指定期間内に訪れたクラスタIDのリストを返す。"""
        if window_start is None:
            return list(self.events_by_cluster.keys())
        clusters: List[str] = []
        for cid, times in self.events_by_cluster.items():
            idx = bisect_left(times, window_start)
            if idx < len(times):
                clusters.append(cid)
        return clusters

    def count_in_window(self, cluster_id: str, window_start: Optional[Timestamp]) -> int:
        """指定期間内での訪問回数を返す。"""
        times = self.events_by_cluster.get(cluster_id, [])
        if not times:
            return 0
        if window_start is None:
            return len(times)
        idx = bisect_left(times, window_start)
        return max(len(times) - idx, 0)

    def total_events_in_window(self, window_start: Optional[Timestamp]) -> int:
        """全クラスタ合計の訪問回数を求める。"""
        if not self.event_times:
            return 0
        if window_start is None:
            return len(self.event_times)
        idx = bisect_left(self.event_times, window_start)
        return max(len(self.event_times) - idx, 0)

    def last_visit_hours(self, cluster_id: str, reference_time: Optional[Timestamp]) -> float:
        """指定クラスタの最終訪問からの経過時間（時間）を算出する。"""
        if reference_time is None or reference_time is pd.NaT:
            return self.default_recency_hours
        times = self.events_by_cluster.get(cluster_id)
        if not times:
            return self.default_recency_hours
        last_time = times[-1]
        delta = (reference_time - last_time).total_seconds() / 3600.0
        if not math.isfinite(delta) or delta < 0:
            return self.default_recency_hours
        return float(delta)

    def recent_candidate_ids(self) -> List[str]:
        """直近訪問したクラスタIDを重複なしの新しい順で返す。"""
        return list(dict.fromkeys(reversed(self.recent_clusters)))

    def known_clusters(self) -> List[str]:
        """履歴に一度でも登場したクラスタの一覧を返す。"""
        return list(self.events_by_cluster.keys())


class CandidateGenerator:
    """適応ウィンドウと履歴情報から候補クラスタ集合を構築する。
    
    放置実績があるクラスタに加え、日常的に長時間滞在しているクラスタも候補に含める。
    これにより、まだ充電後の放置実績はないが頻繁に訪れる場所も予測対象とする。
    """

    def __init__(
        self,
        config: PipelineConfig,
        centroids: Dict[str, Tuple[Optional[float], Optional[float]]],
        known_clusters: Sequence[str],
        base_window_days: float,
        sessions_df: Optional[pd.DataFrame] = None,
        train_cutoff: Optional[Timestamp] = None,
    ) -> None:
        """
        Parameters
        ----------
        config : PipelineConfig
            パイプライン設定
        centroids : Dict[str, Tuple[Optional[float], Optional[float]]]
            クラスタID→(緯度, 経度)の辞書
        known_clusters : Sequence[str]
            trainで観測された放置クラスタのリスト
        base_window_days : float
            充電間隔中央値（適応ウィンドウの基準値）
        sessions_df : Optional[pd.DataFrame]
            全セッションデータ。日常的な長時間滞在クラスタの抽出に使用。
        train_cutoff : Optional[Timestamp]
            学習データの最終時刻。これより未来の情報はリーク防止のため除外。
        """
        self.config = config
        self.centroids = centroids
        
        # 放置実績クラスタに加え、日常的に長時間滞在しているクラスタも候補プールに追加
        frequent_long_stay_clusters: List[str] = []
        if sessions_df is not None and not sessions_df.empty and train_cutoff is not None:
            # 学習期間内の長時間inactiveセッションから頻出クラスタを抽出
            long_stay_sessions = sessions_df[
                (sessions_df["session_type"] == "inactive")
                & (sessions_df["duration_minutes"] >= config.min_long_inactive_minutes)
                & (sessions_df["start_time"] <= train_cutoff)
            ]
            if not long_stay_sessions.empty:
                # ノイズクラスタ(-1)を除外し、頻度順で抽出
                cluster_counts = long_stay_sessions[
                    long_stay_sessions["session_cluster"] != "-1"
                ]["session_cluster"].value_counts()
                # 上位クラスタ（最大でcandidate_top_k × 3程度）を候補プールに追加
                top_n = min(len(cluster_counts), config.candidate_top_k * 3)
                frequent_long_stay_clusters = cluster_counts.head(top_n).index.astype(str).tolist()
        
        # 放置実績クラスタ + 日常的長時間滞在クラスタを統合（重複除去）
        self.known_clusters = list(dict.fromkeys(
            list(known_clusters) + frequent_long_stay_clusters
        ))
        
        self.base_window_days = float(base_window_days) if base_window_days and base_window_days > 0 else 14.0

    def _clip_days(self, days: float) -> float:
        return float(
            max(self.config.candidate_min_days, min(days, self.config.candidate_max_days))
        )

    def _compute_window_days(self, factor: float) -> float:
        return self._clip_days(self.base_window_days * max(factor, 1.0))

    def _reference_time(self, row: pd.Series) -> Optional[Timestamp]:
        charge_end = row.get("charge_end_time")
        if charge_end is not None and charge_end is not pd.NaT:
            return charge_end
        return row.get("charge_start_time")

    def _distance_order_candidates(self, row: pd.Series) -> List[str]:
        lat = row.get("charge_lat")
        lon = row.get("charge_lon")
        if pd.isna(lat) or pd.isna(lon):
            return []
        candidates: List[Tuple[float, str]] = []
        for cid, (clat, clon) in self.centroids.items():
            if clat is None or clon is None:
                continue
            dist_m = haversine_meters(lat, lon, clat, clon)
            candidates.append((dist_m, cid))
        candidates.sort(key=lambda item: item[0])
        return [cid for _, cid in candidates]

    def _ensure_true_candidate(
        self,
        top_candidates: List[CandidateInfo],
        candidate_map: Dict[str, CandidateInfo],
        true_cluster: Optional[str],
    ) -> List[CandidateInfo]:
        """真値クラスタが落選した場合でも必ず含めるように補正する。"""
        if true_cluster is None:
            return top_candidates
        true_cluster = str(true_cluster)
        present = any(info.cluster_id == true_cluster for info in top_candidates)
        if present:
            return top_candidates
        true_info = candidate_map.get(true_cluster)
        if true_info is None:
            return top_candidates
        if not top_candidates:
            return [true_info]
        replaced = [true_info] + top_candidates[:-1]
        return replaced

    def generate_candidates(
        self,
        row: pd.Series,
        history: CandidateHistory,
        true_cluster: Optional[str],
    ) -> Tuple[List[CandidateInfo], float, int]:
        """候補クラスタ一覧と採用したウィンドウ情報を返す。
        
        【候補生成の基本戦略】
        このメソッドは、充電セッションごとに「次の長時間放置先として妥当な候補クラスタ集合」を動的に生成する。
        候補生成は以下の3つの観点を組み合わせて行う：
        
        1. **履歴ベース（過去の放置実績）**
           - この充電クラスタから放置した履歴があるクラスタを優先
           - 利用回数が多いほど選ばれやすい（freq_in_window）
        
        2. **距離ベース（空間的近接性）**
           - 充電地点から近いクラスタほど選ばれやすい
           - 移動コストの観点から妥当な候補を補完
        
        3. **適応ウィンドウ（充電習慣への適応）**
           - 充電間隔が短い人 → ウィンドウを短く（直近の最新傾向を重視）
           - 充電間隔が長い人 → ウィンドウを長く（十分なデータ量を確保）
           - データ不足時は自動的にウィンドウを拡張（candidate_expand_factor倍ずつ）
        
        【候補集合の構成（優先度順）】
        1. ウィンドウ内候補（source="window"）：適応ウィンドウ内で実際に放置したクラスタ
        2. 直近訪問候補（source="recent"）：直近N回の充電で訪れたクラスタ
        3. 距離フォールバック候補（source="fallback"）：上記で不足する場合、距離順で補完
        4. 真値候補（source="label"）：学習時のみ、正解クラスタを必ず含める
        
        【TOPKに満たない場合の補完ロジック】
        - ウィンドウ内候補が少ない場合、自動的にウィンドウを拡張（最大candidate_max_daysまで）
        - それでも不足する場合、距離順で既知クラスタから補完
        - 最終的にcandidate_top_k件の候補を確保
        
        Parameters
        ----------
        row : pd.Series
            充電セッション情報（charge_lat, charge_lon, charge_start_timeなどを含む）
        history : CandidateHistory
            これまでの放置履歴を保持するオブジェクト
        true_cluster : Optional[str]
            学習時の正解クラスタID（推論時はNone）
        
        Returns
        -------
        Tuple[List[CandidateInfo], float, int]
            - 候補クラスタ情報のリスト（最大candidate_top_k件）
            - 採用したウィンドウ日数
            - ウィンドウ内の総イベント数
        """
        reference_time = self._reference_time(row)
        window_start: Optional[Timestamp]

        # ========================================
        # ステップ1: 適応ウィンドウの決定
        # ========================================
        # 充電間隔に応じてウィンドウ幅を調整する。
        # - 基本ウィンドウ = base_window_days（充電間隔中央値） × candidate_window_alpha（既定6倍）
        # - 最小限のデータ量を確保するため、不足時は自動拡張（candidate_expand_factor倍ずつ）
        # - 上限はcandidate_max_days（既定56日）、下限はcandidate_min_days（既定7日）
        
        if reference_time is None or reference_time is pd.NaT:
            # 時刻情報がない場合は最大ウィンドウを使用
            window_days = self.config.candidate_max_days
            window_start = None
        else:
            factor = 1.0
            while True:
                # 充電間隔の中央値に基づいてウィンドウ幅を計算
                # 例: 充電間隔が3日 → 基本ウィンドウ = 3日 × 6 = 18日
                #     充電間隔が20日 → 基本ウィンドウ = 20日 × 6 = 120日 → 上限56日に制限
                window_days = self._compute_window_days(factor)
                window_start = reference_time - pd.Timedelta(days=window_days)
                
                # ウィンドウ内の統計を取得
                clusters = history.clusters_in_window(window_start)  # ウィンドウ内で訪れたクラスタ
                events = history.total_events_in_window(window_start)  # ウィンドウ内の総放置回数
                
                # 終了条件:
                # 1. 十分なクラスタ数（candidate_min_unique以上、既定4個）
                # 2. 十分なイベント数（candidate_min_events以上、既定8回）
                # 3. または、ウィンドウが上限に達した
                if (
                    len(clusters) >= self.config.candidate_min_unique
                    and events >= self.config.candidate_min_events
                ) or window_days >= self.config.candidate_max_days:
                    break
                
                # データ不足の場合、ウィンドウを拡張して再試行
                # 既定: 1.5倍ずつ拡張（candidate_expand_factor=1.5）
                factor *= max(self.config.candidate_expand_factor, 1.0)

        # ========================================
        # ステップ2: ウィンドウ内候補の抽出（最優先）
        # ========================================
        # 適応ウィンドウ内で実際に放置したクラスタを候補として追加。
        # これらは「この充電クラスタから放置した履歴がある」クラスタであり、最も信頼性が高い。
        
        clusters = history.clusters_in_window(window_start)
        events_in_window = history.total_events_in_window(window_start)
        candidate_map: Dict[str, CandidateInfo] = {}

        for cid in clusters:
            # ウィンドウ内での訪問回数（利用回数）を取得
            freq = history.count_in_window(cid, window_start)
            # 最後に訪問してからの経過時間（時間）を取得
            recency = history.last_visit_hours(cid, reference_time)
            candidate_map[cid] = CandidateInfo(
                cluster_id=cid,
                freq_in_window=freq,  # 利用回数が多いほど後でランク上位になる
                recency_hours=recency,
                source="window",
                in_window=True,
                window_days=window_days,
                events_in_window=events_in_window,
            )

        # ========================================
        # ステップ3: 直近訪問候補の追加（補完）
        # ========================================
        # ウィンドウ外でも直近N回（candidate_recent_charge_count、既定10回）の充電で
        # 訪れたクラスタは候補に含める。最新の行動変化を捉えるため。
        
        for cid in history.recent_candidate_ids():
            if cid in candidate_map:
                continue  # 既にウィンドウ内候補として登録済みならスキップ
            freq = history.count_in_window(cid, window_start)
            recency = history.last_visit_hours(cid, reference_time)
            candidate_map[cid] = CandidateInfo(
                cluster_id=cid,
                freq_in_window=freq,
                recency_hours=recency,
                source="recent",
                in_window=False,
                window_days=window_days,
                events_in_window=events_in_window,
            )

        # ========================================
        # ステップ4: 距離ベースのフォールバック候補（不足時の補完）
        # ========================================
        # 上記でcandidate_top_k件に満たない場合、充電地点から距離が近いクラスタで埋める。
        # 既知クラスタ（trainで観測されたクラスタ or これまでに訪問したクラスタ）のみを対象とし、
        # 未知クラスタは除外（モデルが学習していないため予測精度が不安定）。
        
        known_pool = list(dict.fromkeys(self.known_clusters + history.known_clusters()))
        distance_ranking = self._distance_order_candidates(row)  # 距離昇順でソート
        order = 1
        for cid in distance_ranking:
            if cid in candidate_map:
                continue  # 既に候補に含まれていればスキップ
            if known_pool and cid not in known_pool:
                continue  # 既知クラスタに限定
            freq = history.count_in_window(cid, window_start)
            recency = history.last_visit_hours(cid, reference_time)
            candidate_map[cid] = CandidateInfo(
                cluster_id=cid,
                freq_in_window=freq,
                recency_hours=recency,
                source="fallback",
                in_window=False,
                window_days=window_days,
                events_in_window=events_in_window,
                distance_order=order,  # 距離順位を記録（近いほど小さい値）
            )
            order += 1
            # candidate_top_k件に達したら打ち切り
            if len(candidate_map) >= self.config.candidate_top_k:
                break

        # ========================================
        # ステップ5: 真値クラスタの強制追加（学習時のみ）
        # ========================================
        # 学習データ作成時、正解クラスタが候補に含まれていないと正例が作れないため、
        # 必ず候補に追加する。推論時はtrue_cluster=Noneなのでスキップされる。
        
        if true_cluster is not None:
            true_cluster = str(true_cluster)
            if true_cluster not in candidate_map:
                freq = history.count_in_window(true_cluster, window_start)
                recency = history.last_visit_hours(true_cluster, reference_time)
                candidate_map[true_cluster] = CandidateInfo(
                    cluster_id=true_cluster,
                    freq_in_window=freq,
                    recency_hours=recency,
                    source="label",
                    in_window=True,
                    window_days=window_days,
                    events_in_window=events_in_window,
                )

        # ========================================
        # ステップ6: 候補のランク付けとTOPK選択
        # ========================================
        # 優先度順にソート:
        # 1. 真値クラスタ（学習時のみ、最優先）
        # 2. source="window"（ウィンドウ内候補）> その他
        # 3. freq_in_window降順（利用回数が多いほど上位）
        # 4. recency_hours昇順（最近訪問したほど上位）
        # 5. distance_order昇順（距離が近いほど上位）
        # 6. cluster_id（同順位時の安定ソート用）
        
        ranked = sorted(
            candidate_map.values(),
            key=lambda info: (
                0 if true_cluster is not None and info.cluster_id == str(true_cluster) else 1,
                info.source != "window",  # ウィンドウ内候補を優先
                -info.freq_in_window,  # 利用回数降順
                info.recency_hours,  # 最近訪問したほど優先
                info.distance_order if info.distance_order is not None else 9999,  # 距離昇順
                info.cluster_id,
            ),
        )

        # TOPKに制限
        top_k = max(self.config.candidate_top_k, 1)
        limited = ranked[:top_k]
        
        # 最終チェック: 真値クラスタが落選していたら強制的に含める
        limited = self._ensure_true_candidate(limited, candidate_map, true_cluster)
        
        return limited, window_days, events_in_window


class CandidateFeatureAssembler:
    """候補クラスタに依存する特徴量を計算し、行レコードとして返す。"""

    def __init__(
        self,
        config: PipelineConfig,
        centroids: Dict[str, Tuple[Optional[float], Optional[float]]],
        cluster_delays: Dict[str, float],
        pair_delays: Dict[Tuple[str, str], float],
        p_start: Dict[str, np.ndarray],
        transitions: Dict[str, Dict[object, Dict[str, float]]],
    ) -> None:
        self.config = config
        self.centroids = centroids
        self.cluster_delays = cluster_delays
        self.pair_delays = pair_delays
        self.p_start = p_start
        self.transitions = transitions
        self.toggles = config.feature_toggles

    def _distance_to_candidate(self, row: pd.Series, cluster_id: str) -> float:
        if not self.toggles.use_distance:
            return 0.0
        charge_lat = row.get("charge_lat")
        charge_lon = row.get("charge_lon")
        if pd.isna(charge_lat) or pd.isna(charge_lon):
            return float("nan")
        centroid_lat, centroid_lon = self.centroids.get(cluster_id, (None, None))
        if centroid_lat is None or centroid_lon is None:
            return float("nan")
        return round(haversine_meters(charge_lat, charge_lon, centroid_lat, centroid_lon), 0)

    def _resolve_delay(self, origin_cluster: Optional[str], target_cluster: str) -> float:
        if origin_cluster is not None:
            key = (origin_cluster, target_cluster)
            if key in self.pair_delays:
                return self.pair_delays[key]
        return self.cluster_delays.get(target_cluster, 0.0)

    def _calc_time_kernel(
        self,
        charge_start: Timestamp,
        matrix: Optional[np.ndarray],
        delay_hours: float,
    ) -> float:
        if matrix is None or matrix.size == 0:
            return 0.0
        if getattr(charge_start, "tzinfo", None) is None:
            charge_start = charge_start.tz_localize("Asia/Tokyo")
        else:
            charge_start = charge_start.tz_convert("Asia/Tokyo")
        bins_per_day = matrix.shape[1]
        charge_time = charge_start + pd.Timedelta(hours=delay_hours)
        hour = charge_time.hour + charge_time.minute / 60.0
        bin_idx = int(hour // self.config.time_bin_hours) % bins_per_day
        dow = charge_time.weekday()
        return float(matrix[dow, bin_idx])

    def _time_compat_score(
        self,
        row: pd.Series,
        cluster_id: str,
    ) -> float:
        if not self.toggles.use_time_compat:
            return 0.0
        charge_start = row.get("charge_start_time")
        if charge_start is None or charge_start is pd.NaT:
            return 0.0
        origin_cluster_raw = row.get("charge_cluster")
        origin_cluster = str(origin_cluster_raw) if pd.notna(origin_cluster_raw) else None
        matrix = self.p_start.get(cluster_id)
        delay = self._resolve_delay(origin_cluster, cluster_id)
        return self._calc_time_kernel(charge_start, matrix, delay)

    def _transition_score(self, row: pd.Series, candidate_cluster: str) -> float:
        if not self.toggles.use_prev_transition:
            return 0.0
        prev_cluster_raw = row.get("prev_long_inactive_cluster")
        prev_cluster = str(prev_cluster_raw) if pd.notna(prev_cluster_raw) else None
        charge_start: Optional[Timestamp] = row.get("charge_start_time")
        if charge_start is not None and charge_start is not pd.NaT:
            if getattr(charge_start, "tzinfo", None) is None:
                charge_start = charge_start.tz_localize("Asia/Tokyo")
            else:
                charge_start = charge_start.tz_convert("Asia/Tokyo")
            hour = charge_start.hour + charge_start.minute / 60.0
            bin_idx = int(hour // self.config.time_bin_hours)
            dow = charge_start.weekday()
            key = (dow, bin_idx)
        else:
            key = DEFAULT_BIN_KEY

        def lookup(source: Optional[str]) -> Optional[float]:
            if source is None:
                return None
            bucket = self.transitions.get(source)
            if not bucket:
                return None
            if key in bucket:
                return bucket[key].get(candidate_cluster)
            return bucket.get(DEFAULT_BIN_KEY, {}).get(candidate_cluster)

        prob = lookup(prev_cluster)
        if prob is None:
            global_bucket = self.transitions.get(GLOBAL_TRANS_KEY, {})
            if isinstance(key, tuple) and key in global_bucket:
                prob = global_bucket[key].get(candidate_cluster)
            if prob is None:
                prob = global_bucket.get(DEFAULT_BIN_KEY, {}).get(candidate_cluster, 0.0)
        return float(prob if prob is not None else 0.0)

    def build_feature_dict(self, row: pd.Series, candidate: CandidateInfo) -> Dict[str, object]:
        """候補クラスタ1件分の特徴量を辞書形式で返す。"""
        features: Dict[str, object] = {}
        features["cand_distance_m"] = self._distance_to_candidate(row, candidate.cluster_id)
        features["cand_freq_in_window"] = candidate.freq_in_window + self.config.laplace_alpha
        features["cand_freq_log"] = log1p_safe(candidate.freq_in_window)
        features["cand_recency_hours"] = candidate.recency_hours
        features["cand_window_days"] = candidate.window_days
        features["cand_events_in_window"] = candidate.events_in_window
        features["cand_source"] = candidate.source
        features["cand_in_window"] = int(candidate.in_window)
        features["cand_distance_order"] = candidate.distance_order if candidate.distance_order is not None else -1
        features["cand_time_compat"] = self._time_compat_score(row, candidate.cluster_id)
        features["cand_transition_prev"] = self._transition_score(row, candidate.cluster_id)
        return features


class VisitStatisticsBuilder:
    """クラスタの重心・遅延・時間分布・遷移確率を算出する。"""

    def __init__(self, config: PipelineConfig, sessions_df: pd.DataFrame) -> None:
        """設定とセッション一覧を保持するだけの初期化処理。"""
        self.config = config
        self.sessions_df = sessions_df

    def build(self, train_df: pd.DataFrame, train_clusters: Sequence[str]) -> Tuple[
        Dict[str, Tuple[Optional[float], Optional[float]]],
        Dict[str, float],
        Dict[Tuple[str, str], float],
        Dict[str, np.ndarray],
        Dict[str, Dict[object, Dict[str, float]]],
    ]:
        """重心・遅延・時間分布・遷移確率をまとめて計算し、後続の特徴量生成に渡す。"""
        if not train_clusters:
            return {}, {}, {}, {}, {}
        train_cutoff = train_df["charge_start_time"].max()
        centroids = self._centroids(train_df, train_clusters, train_cutoff)
        cluster_delays, pair_delays = self._delays(train_df, train_clusters)
        p_start = self._p_start(train_clusters, train_cutoff)
        if self.config.feature_toggles.use_prev_transition:
            transitions = self._transitions(train_df, train_clusters)
        else:
            base_targets = set(train_clusters) | {OTHER_LABEL}
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
        train_clusters: Sequence[str],
        train_cutoff: Timestamp,
    ) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
        """クラスタごとの代表座標をtrainデータ＋通常セッションから推定する。"""
        centroids: Dict[str, Tuple[Optional[float], Optional[float]]] = {}

        if {"inactive_lat", "inactive_lon"}.issubset(train_df.columns):
            grouped = train_df.groupby("next_long_inactive_cluster")[
                ["inactive_lat", "inactive_lon"]
            ].median()
            for cid in train_clusters:
                if cid in grouped.index:
                    lat, lon = grouped.loc[cid].tolist()
                    centroids[cid] = (float(lat), float(lon))

        candidates = self.sessions_df[
            (self.sessions_df["session_type"] == "inactive")
            & (self.sessions_df["session_cluster"].isin(train_clusters))
            & (self.sessions_df["start_time"] <= train_cutoff)
        ]
        if not candidates.empty:
            med = candidates.groupby("session_cluster")[
                ["start_lat", "start_lon"]
            ].median()
            for cid in train_clusters:
                if cid in med.index and cid not in centroids:
                    lat, lon = med.loc[cid].tolist()
                    centroids[cid] = (float(lat), float(lon))

        for cid in train_clusters:
            centroids.setdefault(cid, (None, None))
        return centroids

    def _delays(
        self, train_df: pd.DataFrame, train_clusters: Sequence[str]
    ) -> Tuple[Dict[str, float], Dict[Tuple[str, str], float]]:
        """クラスタ単位およびステーション×クラスタ単位の遅延統計を求める。"""
        if train_df.empty:
            base = {cid: 0.0 for cid in train_clusters}
            return base, {}

        delays = train_df["inactive_start_time"] - train_df["charge_start_time"]
        delays_hours = delays.dt.total_seconds() / 3600.0
        overall = float(np.median(delays_hours.dropna())) if not delays_hours.dropna().empty else 0.0

        cluster_delay_map: Dict[str, float] = {}
        grouped = train_df.groupby("next_long_inactive_cluster")
        for cid in train_clusters:
            if cid in grouped.groups:
                values = delays_hours.loc[grouped.groups[cid]].dropna()
                if len(values) >= self.config.min_delay_samples:
                    cluster_delay_map[cid] = float(np.median(values))
        for cid in train_clusters:
            cluster_delay_map.setdefault(cid, overall)
            cluster_delay_map[cid] = float(np.clip(cluster_delay_map[cid], 0.0, self.config.max_time_shift_hours))

        pair_delay_map: Dict[Tuple[str, str], float] = {}
        if "charge_cluster" in train_df.columns:
            pair_group = (
                train_df.dropna(subset=["charge_cluster", "next_long_inactive_cluster"])
                .groupby(["charge_cluster", "next_long_inactive_cluster"])
            )
            for (charge_cluster, target_cluster), indexer in pair_group.groups.items():
                target_cluster_str = str(target_cluster)
                if target_cluster_str not in train_clusters:
                    continue
                values = delays_hours.loc[indexer].dropna()
                if len(values) >= self.config.min_pair_delay_samples:
                    charge_key = str(charge_cluster)
                    delay_val = float(np.median(values))
                    delay_val = float(np.clip(delay_val, 0.0, self.config.max_time_shift_hours))
                    pair_delay_map[(charge_key, target_cluster_str)] = delay_val

        return cluster_delay_map, pair_delay_map

    def _p_start(self, train_clusters: Sequence[str], train_cutoff: Timestamp) -> Dict[str, np.ndarray]:
        """曜日×時間帯の開始確率行列（time_compat辞書）を構築する。"""
        alpha = self.config.laplace_alpha
        bins_per_day = int(24 / self.config.time_bin_hours)
        matrices: Dict[str, np.ndarray] = {}

        sessions = self.sessions_df[
            (self.sessions_df["session_type"] == "inactive")
            & (self.sessions_df["duration_minutes"] >= self.config.min_long_inactive_minutes)
            & (self.sessions_df["session_cluster"].isin(train_clusters))
            & (self.sessions_df["start_time"] <= train_cutoff)
        ]

        for cid in train_clusters:
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
        self, train_df: pd.DataFrame, train_clusters: Sequence[str]
    ) -> Dict[str, Dict[object, Dict[str, float]]]:
        """prevクラスタ×曜日×時間帯から次クラスタへの遷移確率を算出する。"""
        alpha = self.config.laplace_alpha
        bins_per_day = int(24 / self.config.time_bin_hours)
        counts_prev_time: Dict[str, Dict[Tuple[int, int], Dict[str, float]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        counts_prev_total: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        counts_global_time: Dict[Tuple[int, int], Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        counts_global_total: Dict[str, float] = defaultdict(float)

        if train_df.empty or "prev_long_inactive_cluster" not in train_df.columns:
            base_targets = set(train_clusters) | {OTHER_LABEL}
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

        all_targets = set(train_clusters) | set(counts_global_total.keys()) | {OTHER_LABEL}
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

        # SOCの開始値は常に保持しておく（欠損はそのままNaN）。
        columns["soc_start"] = model_df.get("charge_start_soc")

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
        """候補生成＋二値スコアリング用テーブルを構築する。"""
        model_df = self._select_model_rows()
        if model_df.empty:
            empty = {split: model_df.copy() for split in ["train", "valid", "test"]}
            return HashvinResult(
                hashvin=self.hashvin,
                features=model_df,
                split_datasets=empty,
                label_col="label",
            )

        model_df["split"] = assign_splits(len(model_df))
        model_df = self._add_daily_time_bins(model_df)

        train_df = model_df[model_df["split"] == "train"].copy()
        train_clusters = sorted(train_df["next_long_inactive_cluster"].dropna().astype(str).unique().tolist())
        base_window_days = compute_charge_interval_median(train_df, self.config.default_charge_interval_days)

        stats_builder = VisitStatisticsBuilder(self.config, self.sessions_df)
        centroids, cluster_delays, pair_delays, p_start, transitions = stats_builder.build(
            train_df, train_clusters
        )

        candidate_history = CandidateHistory(
            default_recency_hours=self.config.recency_default_hours,
            recent_limit=self.config.candidate_recent_charge_count,
        )
        candidate_generator = CandidateGenerator(
            config=self.config,
            centroids=centroids,
            known_clusters=train_clusters,
            base_window_days=base_window_days,
            sessions_df=self.sessions_df,
            train_cutoff=train_df["charge_start_time"].max() if not train_df.empty else None,
        )
        candidate_feature = CandidateFeatureAssembler(
            config=self.config,
            centroids=centroids,
            cluster_delays=cluster_delays,
            pair_delays=pair_delays,
            p_start=p_start,
            transitions=transitions,
        )

        common_assembler = CommonFeatureAssembler(self.config)
        common_features = common_assembler.transform(model_df)
        base_df = pd.concat([model_df, common_features], axis=1)
        if base_df.columns.duplicated().any():
            base_df = base_df.loc[:, ~base_df.columns.duplicated()]

        records: List[Dict[str, object]] = []
        ordered_indices = base_df.sort_values("charge_start_time").index.tolist()
        for idx in ordered_indices:
            row = base_df.loc[idx]
            true_cluster = row.get("next_long_inactive_cluster")
            candidates, _, events_in_window = candidate_generator.generate_candidates(
                row, candidate_history, true_cluster
            )

            for candidate in candidates:
                feature_row = row.to_dict()
                feature_row.update(candidate_feature.build_feature_dict(row, candidate))
                feature_row["candidate_cluster"] = candidate.cluster_id
                feature_row["label"] = int(str(true_cluster) == candidate.cluster_id)
                feature_row["true_cluster"] = str(true_cluster)
                feature_row["cand_set_size"] = len(candidates)
                feature_row["cand_window_total_events"] = events_in_window
                records.append(feature_row)

            candidate_history.register_event(
                cluster_id=true_cluster,
                charge_end_time=row.get("charge_end_time"),
                inactive_start_time=row.get("inactive_start_time"),
            )

        feature_df = pd.DataFrame(records)
        if feature_df.empty:
            empty = {split: feature_df.copy() for split in ["train", "valid", "test"]}
            return HashvinResult(
                hashvin=self.hashvin,
                features=feature_df,
                split_datasets=empty,
                label_col="label",
            )

        # 生成した特徴量以外の元データ列を削除（充電クラスタは特徴量として保持）
        generated_feature_cols = {
            # 候補依存特徴
            "cand_distance_m", "cand_freq_in_window", "cand_freq_log", "cand_recency_hours",
            "cand_window_days", "cand_events_in_window", "cand_source", "cand_in_window",
            "cand_distance_order", "cand_time_compat", "cand_transition_prev",
            # 共通特徴
            "dow", "hour_sin", "hour_cos", "time_raw_charge_start", "time_raw_timestamp",
            "time_band_2h", "time_dow_category", "soc_start",
            # 当日行動ヒストグラム（daily_bin_で始まる列）
            # station種別（station_で始まる列）
            # メタ列
            "candidate_cluster", "label", "true_cluster", "cand_set_size",
            "cand_window_total_events", "session_uid", "split", "hashvin",
            # 充電クラスタ（特徴量として保持）
            "charge_cluster",
        }
        
        # daily_bin_とstation_で始まる列を追加
        for col in feature_df.columns:
            if col.startswith("daily_bin_") or col.startswith("station_"):
                generated_feature_cols.add(col)
        
        # 生成された特徴量以外の列を削除
        cols_to_drop = [col for col in feature_df.columns if col not in generated_feature_cols]
        if cols_to_drop:
            feature_df = feature_df.drop(columns=cols_to_drop)

        split_datasets = {
            split: feature_df[feature_df["split"] == split].copy() for split in ["train", "valid", "test"]
        }

        return HashvinResult(
            hashvin=self.hashvin,
            features=feature_df,
            split_datasets=split_datasets,
            label_col="label",
        )
