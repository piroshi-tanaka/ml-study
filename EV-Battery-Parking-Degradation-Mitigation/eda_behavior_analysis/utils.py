"""
ユーティリティ関数
"""

import pandas as pd


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



