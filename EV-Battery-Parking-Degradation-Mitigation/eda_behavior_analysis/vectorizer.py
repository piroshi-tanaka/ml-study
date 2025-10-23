"""
日次行動ベクトル生成モジュール（Step1）

セッションデータから、hashvin × date06ごとの行動ベクトルを生成します。
滞在時間ベースの特徴量と、場所間遷移ベースの特徴量を含みます。
"""

from typing import List, Tuple
import pandas as pd
import warnings
from config import BehaviorAnalysisConfig
from time_processor import TimeRangeProcessor


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
        self.top_clusters_: List[str] = []
        self.top_transitions_: List[Tuple[str, str]] = []
        
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
            bin_cluster_minutes: Dict[Tuple[str, str], float] = {}
            
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
        
        # 各行の合計が1±1e-3内かチェック
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

