"""
時間範囲処理モジュール

06:00起点の日界処理、セッションの日跨ぎ分割、時間ビンへの按分を行います。
"""

from typing import List, Dict
import pandas as pd
from datetime import timedelta
from config import BehaviorAnalysisConfig


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
        result: List[Dict] = []
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

