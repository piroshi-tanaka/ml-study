# -*- coding: utf-8 -*-
"""
session_detector使用例
====================
"""
import pandas as pd
from datetime import datetime, timedelta
from session_detector import (
    detect_sessions, SessionParams, validate_sessions, SessionType,
    extract_charge_to_inactive_sessions, get_session_data_by_boundary,
    extract_session_details, get_session_summary_by_type
)

# 使用例
def main():
    """基本的な使用例"""
    
    # 1. パラメータ設定
    params = SessionParams(
        DIST_TH_m=150.0,      # 移動判定距離閾値（m）
        PARK_TH_min=360.0,    # 長時間放置閾値（6時間）
        GAP_MAX_min=15.0,     # 充電ギャップ許容時間（分）
        WINDOW_min=60.0       # 充電後移動チェック時間窓（分）
    )
    
    # 2. サンプルデータ（実際のデータに置き換えてください）
    sample_data = {
        'hashvin': ['VEH001'] * 10,
        'tsu_current_time': [datetime(2024, 1, 1, 8, 0, 0) + timedelta(minutes=i*30) for i in range(10)],
        'tsu_igon_time': [None] * 5 + [datetime(2024, 1, 1, 10, 30, 0)] * 5,
        'tsu_latitude': [35.681236] * 10,
        'tsu_longitude': [139.767125] * 10,
        'soc': [20, 30, 40, 50, 60, 60, 59, 58, 57, 56],
        'charge_mode': ['Fast charging'] * 5 + ['Not charging'] * 5
    }
    df = pd.DataFrame(sample_data)
    
    # 3. セッション判定実行
    result = detect_sessions(df, params)
    
    # 4. 結果表示
    print("=== セッション判定結果 ===")
    print(f"処理行数: {len(result)}")
    
    # セッション統計
    if 'is_charging_stitched' in result.columns:
        print(f"充電セッション: {result['is_charging_stitched'].sum()}行")
    if 'is_moving' in result.columns:
        print(f"移動セッション: {result['is_moving'].sum()}行")
    if 'is_idling' in result.columns:
        print(f"アイドリングセッション: {result['is_idling'].sum()}行")
    if 'is_parking' in result.columns:
        print(f"パーキングセッション: {result['is_parking'].sum()}行")
    if 'is_inactive' in result.columns:
        print(f"放置セッション（inactive）: {result['is_inactive'].sum()}行")
    
    # session_type分布
    if 'session_type' in result.columns:
        print(f"\nセッションタイプ分布:")
        type_counts = result['session_type'].value_counts()
        for session_type, count in type_counts.items():
            print(f"  {session_type}: {count}行")
    
    # 5. 品質チェック
    quality = validate_sessions(result)
    print(f"\n品質チェック:")
    print(f"総カバレッジ: {quality.get('total_coverage', 0):.2%}")
    
    if 'session_type_distribution' in quality:
        print(f"セッションタイプ分布確認済み")
    
    # 6. 境界情報の表示
    print(f"\n=== 境界情報の例 ===")
    boundary_rows = result[result['has_boundary']]
    if len(boundary_rows) > 0:
        print(f"境界イベントがある行: {len(boundary_rows)}行")
        for idx, row in boundary_rows.head(3).iterrows():
            print(f"  行{idx}: session_type={row['session_type']}, boundary_events={row['boundary_events']}")
    
    # 7. 統合セッションの抽出例
    print(f"\n=== 統合セッション抽出例 ===")
    integrated_sessions = extract_charge_to_inactive_sessions(result, params)
    print(f"充電→放置（inactive）セッション数: {len(integrated_sessions)}")
    
    if len(integrated_sessions) > 0:
        print(f"利用可能なカラム: {list(integrated_sessions.columns)}")
        valid_sessions = integrated_sessions[integrated_sessions['is_valid_session']]
        print(f"有効セッション: {len(valid_sessions)}個")
        
        if len(valid_sessions) > 0 and 'is_excluded_no_movement' in valid_sessions.columns:
            excluded_sessions = valid_sessions[valid_sessions['is_excluded_no_movement']]
            print(f"充電後移動なし（除外対象）: {len(excluded_sessions)}個")
        else:
            print("充電後移動なし（除外対象）: 情報なし（有効セッションが0個または詳細情報なし）")
            excluded_sessions = pd.DataFrame()  # 空のDataFrame
        
        if len(valid_sessions) > 0 and 'total_session_duration_min' in valid_sessions.columns:
            avg_duration = valid_sessions['total_session_duration_min'].mean()
            print(f"平均セッション時間: {avg_duration:.1f}分")
    
    # 8. 境界データ取得例
    print(f"\n=== 境界データ取得例 ===")
    charge_end_data = get_session_data_by_boundary(result, 'charge_end')
    print(f"充電終了イベント: {len(charge_end_data)}件")
    
    movement_start_data = get_session_data_by_boundary(result, 'movement_start')
    print(f"移動開始イベント: {len(movement_start_data)}件")
    
    inactive_start_data = get_session_data_by_boundary(result, 'inactive_start')
    print(f"放置開始イベント: {len(inactive_start_data)}件")
    
    # 9. 新機能：セッション詳細データ抽出
    print(f"\n=== セッション詳細データ抽出例 ===")
    session_details = extract_session_details(result)
    print(f"セッション詳細データ数: {len(session_details)}件")
    
    if len(session_details) > 0:
        print(f"利用可能なカラム: {list(session_details.columns)}")
        print("\n各セッションタイプの件数:")
        for session_type in session_details['session_type'].unique():
            count = len(session_details[session_details['session_type'] == session_type])
            print(f"  {session_type}: {count}件")
        
        # 開始・終了イベントの例を表示
        print("\nセッション詳細データ例（最初の5件）:")
        for idx, row in session_details.head(5).iterrows():
            print(f"  {row['session_type']} {row['event_type']}: {row['timestamp']} SOC={row['soc']:.1f}%")
    
    # 10. 新機能：セッションサマリー取得
    print(f"\n=== セッションサマリー例 ===")
    
    # 充電セッションのサマリー
    charging_summary = get_session_summary_by_type(result, 'charging')
    print(f"充電セッションサマリー: {len(charging_summary)}件")
    if len(charging_summary) > 0:
        for idx, session in charging_summary.iterrows():
            print(f"  充電セッション{session['session_id']}: {session['duration_min']:.1f}分、SOC変化={session['soc_diff']:+.1f}%")
    
    # 放置セッションのサマリー
    inactive_summary = get_session_summary_by_type(result, 'inactive')
    print(f"放置セッションサマリー: {len(inactive_summary)}件")
    if len(inactive_summary) > 0:
        for idx, session in inactive_summary.iterrows():
            breakdown = session.get('inactive_breakdown', 'unknown')
            print(f"  放置セッション{session['session_id']}: {session['duration_min']:.1f}分、内訳={breakdown}")
    
    # 全セッションのサマリー
    all_summary = get_session_summary_by_type(result)
    print(f"\n全セッションサマリー: {len(all_summary)}件")
    if len(all_summary) > 0:
        print("セッション時間統計:")
        for session_type in all_summary['session_type'].unique():
            type_data = all_summary[all_summary['session_type'] == session_type]
            avg_duration = type_data['duration_min'].mean()
            print(f"  {session_type}: 平均{avg_duration:.1f}分")
    
    # 11. 要求されたデータ形式での出力例
    print(f"\n=== 要求されたデータ形式での出力 ===")
    
    # セッション詳細データをより分かりやすい形式で表示
    if len(session_details) > 0:
        print("セッション開始・終了の詳細データ:")
        print("-" * 80)
        
        # グループ化して表示
        for (hashvin, s_type, s_id), group in session_details.groupby(['hashvin', 'session_type', 'session_id']):
            start_row = group[group['event_type'] == 'start']
            end_row = group[group['event_type'] == 'end']
            
            print(f"車両: {hashvin}, セッション: {s_type} #{s_id}")
            
            if len(start_row) > 0:
                start = start_row.iloc[0]
                print(f"  開始: {start['timestamp']} SOC={start['soc']:.1f}% 距離={start['distance_from_prev']:.1f}m")
                
            if len(end_row) > 0:
                end = end_row.iloc[0]
                print(f"  終了: {end['timestamp']} SOC={end['soc']:.1f}% 距離={end['distance_from_prev']:.1f}m")
                print(f"  期間: {end['duration_min']:.1f}分, SOC差分: {end['soc_diff']:+.1f}%, 総移動距離: {end['total_distance']:.1f}m")
                
                if s_type == 'inactive' and 'inactive_breakdown' in end:
                    print(f"  放置内訳: {end['inactive_breakdown']}")
            
            print()
    
    return result, integrated_sessions

if __name__ == "__main__":
    result, integrated_sessions = main()
