"""
行動分析の設定パラメータ
"""

from dataclasses import dataclass
from typing import Tuple, Literal


@dataclass
class BehaviorAnalysisConfig:
    """
    行動分析の設定パラメータ
    
    Attributes:
        hour_bin_size (int): 時間ビンのサイズ（時間単位）。デフォルト: 1
        topK_clusters (int): 上位K個のクラスタを採用。デフォルト: 20
        min_session_minutes (int): 最小セッション時間（分）。これより短いセッションは除外可能。デフォルト: 20
        window_months (int): 分析ウィンドウ期間（月）。デフォルト: 3
        day_start_hour (int): 日の開始時刻（06:00基準）。デフォルト: 6
        k_range (Tuple[int, int]): クラスタリングのk値探索範囲（KMeans用）。デフォルト: (3, 10)
        clustering_method (Literal['kmeans', 'hdbscan']): クラスタリング手法。デフォルト: 'kmeans'
        hdbscan_min_cluster_size (int): HDBSCANの最小クラスタサイズ。デフォルト: 5
        hdbscan_min_samples (int): HDBSCANの最小サンプル数。デフォルト: 3
        include_transition_features (bool): 遷移特徴量を含めるか。デフォルト: True
        topK_transitions (int): 上位K個の遷移パターンを採用。デフォルト: 15
        random_state (int): 再現性のための乱数シード。デフォルト: 42
        timezone (str): タイムゾーン。デフォルト: 'Asia/Tokyo'
    """
    hour_bin_size: int = 1
    topK_clusters: int = 20
    min_session_minutes: int = 20
    window_months: int = 3
    day_start_hour: int = 6
    k_range: Tuple[int, int] = (3, 10)
    clustering_method: Literal['kmeans', 'hdbscan'] = 'kmeans'
    hdbscan_min_cluster_size: int = 5
    hdbscan_min_samples: int = 3
    include_transition_features: bool = True
    topK_transitions: int = 15
    random_state: int = 42
    timezone: str = 'Asia/Tokyo'



