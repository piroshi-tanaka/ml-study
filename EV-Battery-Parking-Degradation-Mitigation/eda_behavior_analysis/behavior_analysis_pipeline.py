"""
EVユーザー行動パターン分析パイプライン

このモジュールは、EVユーザーの行動パターンを分析するための包括的なパイプラインを提供します。

主要コンポーネント:
- TimeRangeProcessor: 日界分割・時間ビン処理
- DailyBehaviorVectorizer: Step1実装（日次行動ベクトル生成）
- BehaviorPatternClusterer: Step2実装（クラスタリング）

Author: ML Engineer
Date: 2025-10-23
"""

from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional, Literal
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False


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


class TimeRangeProcessor:
    """
    時間範囲処理クラス
    
    06:00起点の日界処理、セッションの日跨ぎ分割、時間ビンへの按分を行います。
    """
    
    def __init__(self, config: BehaviorAnalysisConfig):
        """
        初期化
        
        Args:
            config (BehaviorAnalysisConfig): 設定パラメータ
        """
        self.config = config
        
    def get_date06(self, dt: pd.Timestamp) -> pd.Timestamp:
        """
        06:00起点の日付キーを取得
        
        Args:
            dt (pd.Timestamp): 対象の日時
            
        Returns:
            pd.Timestamp: 06:00起点の日付（その日の06:00）
            
        Examples:
            >>> # 2025-10-23 20:00 -> 2025-10-23 06:00
            >>> # 2025-10-23 03:00 -> 2025-10-22 06:00
        """
        if dt.hour < self.config.day_start_hour:
            # 06:00より前の場合は前日の06:00が起点
            base_date = dt.date() - timedelta(days=1)
        else:
            base_date = dt.date()
        
        return pd.Timestamp(year=base_date.year, month=base_date.month, 
                          day=base_date.day, hour=self.config.day_start_hour,
                          tz=self.config.timezone)
    
    def split_session_by_day06(self, start_time: pd.Timestamp, end_time: pd.Timestamp,
                               session_data: Dict) -> List[Dict]:
        """
        セッションを06:00起点の日界で分割
        
        Args:
            start_time (pd.Timestamp): セッション開始時刻
            end_time (pd.Timestamp): セッション終了時刻
            session_data (Dict): セッションのメタデータ
            
        Returns:
            List[Dict]: 日ごとに分割されたセッションのリスト
                各要素は {'date06': 日付キー, 'start': 開始時刻, 'end': 終了時刻, **session_data}
        """
        result = []
        current_date06 = self.get_date06(start_time)
        
        while True:
            next_date06 = current_date06 + timedelta(hours=24)
            
            # 現在の日バケットに含まれる範囲を計算
            segment_start = max(start_time, current_date06)
            segment_end = min(end_time, next_date06)
            
            if segment_start < segment_end:
                result.append({
                    'date06': current_date06,
                    'start': segment_start,
                    'end': segment_end,
                    **session_data
                })
            
            # 次の日へ
            if end_time <= next_date06:
                break
            current_date06 = next_date06
            
        return result
    
    def get_hour_bins(self) -> List[str]:
        """
        時間ビンのラベルリストを生成
        
        Returns:
            List[str]: 時間ビンのラベルリスト
                例: ['06-07', '07-08', ..., '05-06'] (hour_bin_size=1の場合)
        """
        bins = []
        hour_size = self.config.hour_bin_size
        start_hour = self.config.day_start_hour
        
        for i in range(0, 24, hour_size):
            h_start = (start_hour + i) % 24
            h_end = (start_hour + i + hour_size) % 24
            bins.append(f"{h_start:02d}-{h_end:02d}")
            
        return bins
    
    def assign_to_hour_bins(self, start_time: pd.Timestamp, end_time: pd.Timestamp,
                           date06: pd.Timestamp) -> Dict[str, float]:
        """
        セッションを時間ビンに按分
        
        Args:
            start_time (pd.Timestamp): セッション開始時刻
            end_time (pd.Timestamp): セッション終了時刻
            date06 (pd.Timestamp): 日付キー（06:00起点）
            
        Returns:
            Dict[str, float]: 時間ビンラベル -> 滞在時間（分）のマッピング
        """
        result = {}
        hour_size = self.config.hour_bin_size
        
        # 日の開始・終了時刻
        day_start = date06
        day_end = date06 + timedelta(hours=24)
        
        # セッションが日の範囲内にあることを確認
        start_time = max(start_time, day_start)
        end_time = min(end_time, day_end)
        
        if start_time >= end_time:
            return result
        
        # 時間ビンごとに重なり時間を計算
        for i in range(0, 24, hour_size):
            bin_start = day_start + timedelta(hours=i)
            bin_end = bin_start + timedelta(hours=hour_size)
            
            # 重なり区間を計算
            overlap_start = max(start_time, bin_start)
            overlap_end = min(end_time, bin_end)
            
            if overlap_start < overlap_end:
                minutes = (overlap_end - overlap_start).total_seconds() / 60
                h_start = (self.config.day_start_hour + i) % 24
                h_end = (self.config.day_start_hour + i + hour_size) % 24
                bin_label = f"{h_start:02d}-{h_end:02d}"
                result[bin_label] = minutes
                
        return result


class DailyBehaviorVectorizer:
    """
    日次行動ベクトル生成クラス（Step1）
    
    セッションデータから、hashvin × date06ごとの行動ベクトルを生成します。
    滞在時間ベースの特徴量と、場所間遷移ベースの特徴量を含みます。
    """
    
    def __init__(self, config: BehaviorAnalysisConfig):
        """
        初期化
        
        Args:
            config (BehaviorAnalysisConfig): 設定パラメータ
        """
        self.config = config
        self.time_processor = TimeRangeProcessor(config)
        self.top_clusters_ = None
        self.top_transitions_ = None
        
    def _ensure_timezone(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        start_time、end_timeにタイムゾーン情報を付与
        
        Args:
            df (pd.DataFrame): セッションデータ
            
        Returns:
            pd.DataFrame: タイムゾーン付きのデータフレーム
        """
        df = df.copy()
        
        for col in ['start_time', 'end_time']:
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                # ISO8601形式や混在形式に対応
                df[col] = pd.to_datetime(df[col], format='mixed')
            
            # タイムゾーン情報がない場合は付与
            if df[col].dt.tz is None:
                df[col] = df[col].dt.tz_localize(self.config.timezone)
            else:
                df[col] = df[col].dt.tz_convert(self.config.timezone)
                
        return df
    
    def _select_top_clusters(self, df: pd.DataFrame) -> List[str]:
        """
        滞在時間の多い上位Kクラスタを選定
        
        Args:
            df (pd.DataFrame): セッションデータ
            
        Returns:
            List[str]: 上位Kクラスタのリスト
        """
        cluster_minutes = df.groupby('session_cluster')['duration_minutes'].sum()
        top_clusters = cluster_minutes.nlargest(self.config.topK_clusters).index.tolist()
        
        print(f"上位{self.config.topK_clusters}クラスタを選定:")
        for i, cluster in enumerate(top_clusters[:10], 1):
            mins = cluster_minutes[cluster]
            print(f"  {i}. {cluster}: {mins:.1f}分 ({mins/60:.1f}時間)")
        if len(top_clusters) > 10:
            print(f"  ... (残り{len(top_clusters)-10}クラスタ)")
            
        return top_clusters
    
    def _select_top_transitions(self, df: pd.DataFrame) -> List[Tuple[str, str]]:
        """
        頻度の高い上位K遷移パターンを選定
        
        Args:
            df (pd.DataFrame): セッションデータ（時系列順にソート済み）
            
        Returns:
            List[Tuple[str, str]]: 上位K遷移パターンのリスト [(from_cluster, to_cluster), ...]
        """
        # hashvinごとに遷移を集計
        transitions = []
        
        for hashvin, group in df.groupby('hashvin'):
            group = group.sort_values('start_time')
            clusters = group['session_cluster'].values
            
            # 連続する遷移を抽出
            for i in range(len(clusters) - 1):
                from_cluster = clusters[i]
                to_cluster = clusters[i + 1]
                # 同じクラスタへの遷移は除外
                if from_cluster != to_cluster:
                    transitions.append((from_cluster, to_cluster))
        
        # 頻度集計
        transition_counts = pd.Series(transitions).value_counts()
        top_transitions = transition_counts.head(self.config.topK_transitions).index.tolist()
        
        print(f"\n上位{self.config.topK_transitions}遷移パターンを選定:")
        for i, (from_c, to_c) in enumerate(top_transitions[:10], 1):
            count = transition_counts[(from_c, to_c)]
            print(f"  {i}. {from_c} → {to_c}: {count}回")
        if len(top_transitions) > 10:
            print(f"  ... (残り{len(top_transitions)-10}遷移)")
        
        return top_transitions
    
    def _extract_daily_transitions(self, df_sessions: pd.DataFrame) -> pd.DataFrame:
        """
        日次の遷移パターンを抽出
        
        Args:
            df_sessions (pd.DataFrame): セッションデータ（タイムゾーン付与済み）
            
        Returns:
            pd.DataFrame: 日次遷移データ
                カラム: hashvin, date06, from_cluster, to_cluster, transition_time, time_period
        """
        transitions_list = []
        
        for hashvin, group in df_sessions.groupby('hashvin'):
            group = group.sort_values('start_time')
            
            for i in range(len(group) - 1):
                from_row = group.iloc[i]
                to_row = group.iloc[i + 1]
                
                from_cluster = from_row['session_cluster']
                to_cluster = to_row['session_cluster']
                
                # 同じクラスタへの遷移は除外
                if from_cluster == to_cluster:
                    continue
                
                # 遷移時刻（次のセッションの開始時刻）
                transition_time = to_row['start_time']
                date06 = self.time_processor.get_date06(transition_time)
                
                # 時間帯の判定（朝/昼/夕/夜）
                hour = transition_time.hour
                if 6 <= hour < 12:
                    time_period = 'morning'
                elif 12 <= hour < 17:
                    time_period = 'afternoon'
                elif 17 <= hour < 21:
                    time_period = 'evening'
                else:
                    time_period = 'night'
                
                transitions_list.append({
                    'hashvin': hashvin,
                    'date06': date06,
                    'from_cluster': from_cluster,
                    'to_cluster': to_cluster,
                    'transition_time': transition_time,
                    'time_period': time_period
                })
        
        return pd.DataFrame(transitions_list)
    
    def _add_transition_features(self, result_df: pd.DataFrame, df_sessions: pd.DataFrame) -> pd.DataFrame:
        """
        遷移特徴量を追加
        
        Args:
            result_df (pd.DataFrame): 日次行動ベクトル（滞在特徴量のみ）
            df_sessions (pd.DataFrame): セッションデータ
            
        Returns:
            pd.DataFrame: 遷移特徴量を追加した日次行動ベクトル
        """
        # 日次遷移データを抽出
        daily_transitions = self._extract_daily_transitions(df_sessions)
        
        # 各日×hashvinごとに遷移特徴量を集計
        for idx in result_df.index:
            hashvin, date06 = idx
            
            # その日の遷移データを取得
            day_trans = daily_transitions[
                (daily_transitions['hashvin'] == hashvin) &
                (daily_transitions['date06'] == date06)
            ]
            
            # 上位遷移パターンの出現回数
            for from_c, to_c in self.top_transitions_:
                trans_count = len(day_trans[
                    (day_trans['from_cluster'] == from_c) &
                    (day_trans['to_cluster'] == to_c)
                ])
                result_df.loc[idx, f'trans__{from_c}--{to_c}'] = trans_count
            
            # 時間帯別の遷移数
            for period in ['morning', 'afternoon', 'evening', 'night']:
                period_count = len(day_trans[day_trans['time_period'] == period])
                result_df.loc[idx, f'trans_count_{period}'] = period_count
            
            # 総遷移数
            result_df.loc[idx, 'trans_count_total'] = len(day_trans)
            
            # ユニーククラスタ訪問数
            if len(day_trans) > 0:
                unique_clusters = set(day_trans['from_cluster'].tolist() + day_trans['to_cluster'].tolist())
                result_df.loc[idx, 'unique_clusters_visited'] = len(unique_clusters)
            else:
                result_df.loc[idx, 'unique_clusters_visited'] = 0
        
        # 欠損値を0で埋める
        trans_cols = [c for c in result_df.columns if 'trans_' in c or 'unique_' in c]
        result_df[trans_cols] = result_df[trans_cols].fillna(0)
        
        return result_df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        セッションデータから日次行動ベクトルを生成
        
        Args:
            df (pd.DataFrame): セッションデータ
                必須カラム: hashvin, session_cluster, session_type, 
                           start_time, end_time, duration_minutes
                           
        Returns:
            pd.DataFrame: 日次行動ベクトル
                インデックス: (hashvin, date06)
                カラム: ratio__{hour_bin}__{cluster}, total_minutes, weekday, is_empty_day
        """
        print("=" * 60)
        print("Step1: 日次行動ベクトル生成を開始")
        print("=" * 60)
        
        # タイムゾーン処理
        df = self._ensure_timezone(df)
        
        # 上位クラスタ選定
        self.top_clusters_ = self._select_top_clusters(df)
        
        # 遷移パターン選定（オプション）
        if self.config.include_transition_features:
            self.top_transitions_ = self._select_top_transitions(df)
        
        # セッションを日ごとに分割
        print("\nセッションを06:00起点の日バケットに分割中...")
        daily_sessions = []
        
        for idx, row in df.iterrows():
            session_data = {
                'hashvin': row['hashvin'],
                'session_cluster': row['session_cluster'],
                'session_type': row['session_type'],
                'duration_minutes': row['duration_minutes']
            }
            
            splits = self.time_processor.split_session_by_day06(
                row['start_time'], row['end_time'], session_data
            )
            daily_sessions.extend(splits)
        
        daily_df = pd.DataFrame(daily_sessions)
        print(f"  元のセッション数: {len(df)}")
        print(f"  分割後のセグメント数: {len(daily_df)}")
        
        # 時間ビンへの按分
        print("\n時間ビンへの按分処理中...")
        hour_bins = self.time_processor.get_hour_bins()
        
        records = []
        for (hashvin, date06), group in daily_df.groupby(['hashvin', 'date06']):
            # 時間ビン×クラスタの滞在時間を集計
            bin_cluster_minutes = {}
            
            for _, row in group.iterrows():
                bin_minutes = self.time_processor.assign_to_hour_bins(
                    row['start'], row['end'], date06
                )
                
                cluster = row['session_cluster']
                for bin_label, minutes in bin_minutes.items():
                    key = (bin_label, cluster)
                    bin_cluster_minutes[key] = bin_cluster_minutes.get(key, 0) + minutes
            
            # ベクトル化
            record = {
                'hashvin': hashvin,
                'date06': date06
            }
            
            total_minutes = 0
            for hour_bin in hour_bins:
                for cluster in self.top_clusters_:
                    minutes = bin_cluster_minutes.get((hour_bin, cluster), 0)
                    record[f'ratio__{hour_bin}__{cluster}'] = minutes
                    total_minutes += minutes
                
                # OTHER集約
                other_minutes = sum(
                    minutes for (bin_label, clust), minutes in bin_cluster_minutes.items()
                    if bin_label == hour_bin and clust not in self.top_clusters_
                )
                record[f'ratio_OTHER__{hour_bin}'] = other_minutes
                total_minutes += other_minutes
            
            record['total_minutes'] = total_minutes
            record['weekday'] = date06.weekday()
            record['is_empty_day'] = 1 if total_minutes == 0 else 0
            
            records.append(record)
        
        result_df = pd.DataFrame(records)
        result_df = result_df.set_index(['hashvin', 'date06'])
        
        # 遷移特徴量の追加
        if self.config.include_transition_features:
            print("\n遷移特徴量の追加中...")
            result_df = self._add_transition_features(result_df, df)
        
        # 正規化（比率化）
        print("\n滞在比率への正規化中...")
        ratio_cols = [col for col in result_df.columns if col.startswith('ratio__')]
        
        for idx in result_df.index:
            total = result_df.loc[idx, 'total_minutes']
            if total > 0:
                result_df.loc[idx, ratio_cols] = result_df.loc[idx, ratio_cols] / total
        
        # 品質チェック
        print("\n品質チェック実行中...")
        self._quality_check(result_df)
        
        transition_features = len([c for c in result_df.columns if 'trans_' in c or 'unique_' in c])
        print(f"\n[OK] 日次行動ベクトル生成完了: {len(result_df)}行")
        print(f"  滞在特徴量: {len(ratio_cols)}次元")
        print(f"  遷移特徴量: {transition_features}次元")
        print(f"  空日数: {result_df['is_empty_day'].sum()}日")
        
        return result_df
    
    def _quality_check(self, df: pd.DataFrame):
        """
        品質チェック
        
        Args:
            df (pd.DataFrame): 日次行動ベクトル
        """
        ratio_cols = [col for col in df.columns if col.startswith('ratio__')]
        
        # 各行の合計が1±1e-6内かチェック
        row_sums = df[ratio_cols].sum(axis=1)
        non_empty_rows = df[df['is_empty_day'] == 0]
        
        if len(non_empty_rows) > 0:
            non_empty_sums = row_sums[non_empty_rows.index]
            max_deviation = (non_empty_sums - 1.0).abs().max()
            
            if max_deviation > 1e-3:  # 許容範囲を緩和（浮動小数点誤差考慮）
                warnings.warn(f"警告: 行合計の最大偏差が許容範囲を超えています: {max_deviation}")
            else:
                print(f"  [OK] 行合計チェック: OK (最大偏差={max_deviation:.2e})")
        
        # 各時間帯で合計が≤1を保証
        hour_bins = self.time_processor.get_hour_bins()
        max_hourly_violation = 0
        for hour_bin in hour_bins:
            bin_cols = [col for col in ratio_cols if f'__{hour_bin}__' in col]
            hourly_sums = df[bin_cols].sum(axis=1)
            max_hourly = hourly_sums.max()
            
            if max_hourly > 1.0 + 1e-3:
                max_hourly_violation = max(max_hourly_violation, max_hourly - 1.0)
        
        if max_hourly_violation > 0:
            warnings.warn(f"警告: 一部の時間帯で合計が1を超えています (最大超過: {max_hourly_violation:.3f})")


class BehaviorPatternClusterer:
    """
    行動パターンクラスタリングクラス（Step2）
    
    日次行動ベクトルから、hashvinごとに行動パターンを抽出します。
    """
    
    def __init__(self, config: BehaviorAnalysisConfig):
        """
        初期化
        
        Args:
            config (BehaviorAnalysisConfig): 設定パラメータ
        """
        self.config = config
        self.clustering_results_ = {}  # hashvin -> クラスタリング結果
        
    def fit_transform(self, daily_vector_df: pd.DataFrame, 
                     hashvin: str) -> pd.DataFrame:
        """
        特定のhashvinに対してクラスタリングを実行
        
        Args:
            daily_vector_df (pd.DataFrame): 日次行動ベクトル
            hashvin (str): 対象のhashvin
            
        Returns:
            pd.DataFrame: クラスタID付き日次ベクトル
                新カラム: day_pattern_cluster
        """
        print("=" * 60)
        print(f"Step2: 行動パターンクラスタリング (hashvin={hashvin})")
        print("=" * 60)
        
        # 対象hashvinのデータ抽出
        if isinstance(daily_vector_df.index, pd.MultiIndex):
            df = daily_vector_df.loc[hashvin].copy()
        else:
            df = daily_vector_df[daily_vector_df['hashvin'] == hashvin].copy()
        
        # 空日除外
        df_valid = df[df['is_empty_day'] == 0].copy()
        print(f"対象日数: {len(df_valid)}日（空日{len(df) - len(df_valid)}日を除外）")
        
        if len(df_valid) < self.config.k_range[0]:
            print(f"警告: データ数が少なすぎるためクラスタリングをスキップします")
            df['day_pattern_cluster'] = -1
            return df
        
        # 特徴量抽出
        feature_cols = [col for col in df_valid.columns 
                       if col.startswith('ratio__') or col.startswith('trans_') or col == 'unique_clusters_visited']
        X = df_valid[feature_cols].values
        
        # 標準化
        print("\n特徴量の標準化中...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # クラスタリング手法による分岐
        if self.config.clustering_method == 'kmeans':
            labels, clustering_info = self._cluster_kmeans(X_scaled)
        elif self.config.clustering_method == 'hdbscan':
            labels, clustering_info = self._cluster_hdbscan(X_scaled)
        else:
            raise ValueError(f"未対応のクラスタリング手法: {self.config.clustering_method}")
        
        # 結果を格納
        df_valid['day_pattern_cluster'] = labels
        df['day_pattern_cluster'] = -1  # 空日はデフォルト-1
        df.loc[df_valid.index, 'day_pattern_cluster'] = labels
        
        # クラスタリング結果を保存
        self.clustering_results_[hashvin] = {
            'scaler': scaler,
            'feature_cols': feature_cols,
            'labels': labels,
            'cluster_sizes': pd.Series(labels).value_counts().to_dict(),
            'method': self.config.clustering_method,
            **clustering_info
        }
        
        # クラスタサイズを表示
        print("\nクラスタサイズ:")
        for cluster_id, size in sorted(self.clustering_results_[hashvin]['cluster_sizes'].items()):
            cluster_label = f"Noise" if cluster_id == -1 else f"Cluster {cluster_id}"
            print(f"  {cluster_label}: {size}日")
        
        print(f"\n[OK] クラスタリング完了")
        
        return df
    
    def _cluster_kmeans(self, X: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        KMeansによるクラスタリング
        
        Args:
            X (np.ndarray): 標準化済み特徴量
            
        Returns:
            Tuple[np.ndarray, Dict]: (ラベル配列, クラスタリング情報)
        """
        print("\n最適クラスタ数を探索中（KMeans）...")
        best_k, metrics_history = self._find_optimal_k(X)
        print(f"  最適k値: {best_k}")
        
        print(f"\nKMeans (k={best_k}) でクラスタリング実行中...")
        kmeans = KMeans(n_clusters=best_k, random_state=self.config.random_state, n_init=10)
        labels = kmeans.fit_predict(X)
        
        return labels, {
            'kmeans': kmeans,
            'best_k': best_k,
            'metrics_history': metrics_history
        }
    
    def _cluster_hdbscan(self, X: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        HDBSCANによるクラスタリング
        
        Args:
            X (np.ndarray): 標準化済み特徴量
            
        Returns:
            Tuple[np.ndarray, Dict]: (ラベル配列, クラスタリング情報)
        """
        if not HDBSCAN_AVAILABLE:
            raise ImportError("HDBSCANが利用できません。'pip install hdbscan'でインストールしてください。")
        
        print("\nHDBSCANでクラスタリング実行中...")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.config.hdbscan_min_cluster_size,
            min_samples=self.config.hdbscan_min_samples,
            metric='euclidean',
            cluster_selection_method='eom'
        )
        labels = clusterer.fit_predict(X)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()
        
        print(f"  検出クラスタ数: {n_clusters}")
        print(f"  ノイズ点数: {n_noise}")
        
        return labels, {
            'hdbscan': clusterer,
            'n_clusters': n_clusters,
            'n_noise': n_noise
        }
    
    def _find_optimal_k(self, X: np.ndarray) -> Tuple[int, Dict]:
        """
        最適なk値を自動選択
        
        Args:
            X (np.ndarray): 標準化済み特徴量
            
        Returns:
            Tuple[int, Dict]: (最適k値, メトリクス履歴)
        """
        k_min, k_max = self.config.k_range
        k_range = range(k_min, min(k_max + 1, len(X)))
        
        metrics = {
            'k': [],
            'silhouette': [],
            'calinski_harabasz': [],
            'davies_bouldin': []
        }
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=self.config.random_state, n_init=10)
            labels = kmeans.fit_predict(X)
            
            # 各指標を計算
            sil = silhouette_score(X, labels)
            ch = calinski_harabasz_score(X, labels)
            db = davies_bouldin_score(X, labels)
            
            metrics['k'].append(k)
            metrics['silhouette'].append(sil)
            metrics['calinski_harabasz'].append(ch)
            metrics['davies_bouldin'].append(db)
            
            print(f"  k={k}: Silhouette={sil:.3f}, CH={ch:.1f}, DB={db:.3f}")
        
        # ランキングベースで最適k選択
        # Silhouette: 高いほど良い
        # Calinski-Harabasz: 高いほど良い
        # Davies-Bouldin: 低いほど良い
        
        df_metrics = pd.DataFrame(metrics)
        df_metrics['rank_sil'] = df_metrics['silhouette'].rank(ascending=False)
        df_metrics['rank_ch'] = df_metrics['calinski_harabasz'].rank(ascending=False)
        df_metrics['rank_db'] = df_metrics['davies_bouldin'].rank(ascending=True)  # 低いほど良い
        
        # 合成ランク（平均順位が最良のものを選択）
        df_metrics['avg_rank'] = (
            df_metrics['rank_sil'] + df_metrics['rank_ch'] + df_metrics['rank_db']
        ) / 3
        
        best_idx = df_metrics['avg_rank'].idxmin()
        best_k = df_metrics.loc[best_idx, 'k']
        
        return int(best_k), metrics
    
    def get_cluster_profiles(self, daily_vector_df: pd.DataFrame, 
                            hashvin: str) -> Dict[int, pd.DataFrame]:
        """
        各クラスタのプロファイル（平均ベクトル）を取得
        
        Args:
            daily_vector_df (pd.DataFrame): クラスタID付き日次ベクトル
            hashvin (str): 対象のhashvin
            
        Returns:
            Dict[int, pd.DataFrame]: クラスタID -> プロファイルデータフレーム
        """
        if isinstance(daily_vector_df.index, pd.MultiIndex):
            df = daily_vector_df.loc[hashvin].copy()
        else:
            df = daily_vector_df[daily_vector_df['hashvin'] == hashvin].copy()
        
        df_valid = df[df['is_empty_day'] == 0]
        ratio_cols = [col for col in df_valid.columns if col.startswith('ratio__')]
        
        profiles = {}
        for cluster_id in df_valid['day_pattern_cluster'].unique():
            if cluster_id == -1:
                continue
            
            cluster_data = df_valid[df_valid['day_pattern_cluster'] == cluster_id]
            
            # 平均ベクトル
            avg_vector = cluster_data[ratio_cols].mean()
            
            # 曜日分布
            weekday_dist = cluster_data['weekday'].value_counts(normalize=True).sort_index()
            
            # 代表日（平均ベクトルに最も近い日）
            distances = ((cluster_data[ratio_cols] - avg_vector) ** 2).sum(axis=1)
            representative_dates = distances.nsmallest(3).index.tolist()
            
            profiles[cluster_id] = {
                'avg_vector': avg_vector,
                'weekday_dist': weekday_dist,
                'representative_dates': representative_dates,
                'size': len(cluster_data)
            }
        
        return profiles


def load_session_data(filepath: str, min_duration_minutes: int = 0) -> pd.DataFrame:
    """
    セッションデータを読み込み
    
    Args:
        filepath (str): CSVファイルのパス
        min_duration_minutes (int): 最小セッション時間（分）。これより短いセッションは除外
        
    Returns:
        pd.DataFrame: セッションデータ
    """
    print(f"セッションデータを読み込み中: {filepath}")
    df = pd.read_csv(filepath)
    
    print(f"  読み込み行数: {len(df)}")
    
    if min_duration_minutes > 0:
        before_len = len(df)
        df = df[df['duration_minutes'] >= min_duration_minutes]
        print(f"  {min_duration_minutes}分未満のセッションを除外: {before_len - len(df)}件")
    
    print(f"  最終行数: {len(df)}")
    print(f"  hashvin数: {df['hashvin'].nunique()}")
    print(f"  期間: {df['start_time'].min()} ～ {df['end_time'].max()}")
    
    return df

