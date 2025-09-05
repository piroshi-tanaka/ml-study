# -*- coding: utf-8 -*-
"""
セッション判定テスト実行スクリプト
===============================

天才テスト職人が作成したテストデータでsession_detector.pyをテストします。
"""

import pandas as pd
import sys
import os
sys.path.append('./chatgpt')

from session_detector import detect_sessions, SessionParams, validate_sessions, extract_session_details

def run_comprehensive_test():
    """包括的なセッション判定テスト"""
    print("🚀 セッション判定包括テスト開始")
    print("=" * 50)
    
    # テストデータ読み込み
    try:
        df = pd.read_csv('test_data_final.csv')
        print(f"✅ テストデータ読み込み完了: {len(df)}行")
    except Exception as e:
        print(f"❌ テストデータ読み込み失敗: {e}")
        return
    
    # パラメータ設定
    params = SessionParams(
        DIST_TH_m=150.0,      # 移動判定距離閾値（m）
        PARK_TH_min=360.0,    # 長時間放置閾値（分）= 6時間
        GAP_MAX_min=15.0,     # 充電ギャップ許容時間（分）
        WINDOW_min=60.0       # 充電後移動チェック時間窓（分）
    )
    
    print(f"📋 パラメータ設定:")
    print(f"  移動判定閾値: {params.DIST_TH_m}m")
    print(f"  長時間放置閾値: {params.PARK_TH_min}分")
    print(f"  充電ギャップ許容: {params.GAP_MAX_min}分")
    
    # セッション判定実行
    try:
        print("\n🔍 セッション判定実行中...")
        result = detect_sessions(df, params)
        print(f"✅ セッション判定完了: {len(result)}行")
    except Exception as e:
        print(f"❌ セッション判定失敗: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 品質チェック
    try:
        print("\n🔍 品質チェック実行中...")
        quality = validate_sessions(result)
        print("✅ 品質チェック完了")
        
        print(f"\n📊 品質チェック結果:")
        print(f"  総行数: {quality['total_rows']}")
        print(f"  セッションカバレッジ:")
        for session, count in quality['session_coverage'].items():
            print(f"    {session}: {count}行")
        
        if quality['errors']:
            print(f"❌ エラー: {len(quality['errors'])}件")
            for error in quality['errors']:
                print(f"    - {error}")
        else:
            print("✅ エラーなし")
            
    except Exception as e:
        print(f"❌ 品質チェック失敗: {e}")
    
    # テストケース別結果確認
    print(f"\n📋 テストケース別結果:")
    for tc_id in sorted(df['testcase_id'].unique()):
        tc_result = result[result['testcase_id'] == tc_id]
        
        # セッション統計
        session_stats = {}
        session_columns = ['is_charging_stitched', 'is_moving', 'is_idling', 'is_parking']
        for col in session_columns:
            if col in tc_result.columns:
                session_stats[col] = tc_result[col].sum()
        
        print(f"  {tc_id} ({len(tc_result)}行):")
        for session, count in session_stats.items():
            session_name = session.replace('is_', '').replace('_stitched', '')
            print(f"    {session_name}: {count}行")
    
    # セッション詳細データ抽出テスト
    try:
        print(f"\n🔍 セッション詳細抽出テスト...")
        session_details = extract_session_details(result)
        print(f"✅ セッション詳細抽出完了: {len(session_details)}イベント")
        
        # イベントタイプ別統計
        if len(session_details) > 0:
            event_stats = session_details.groupby(['session_type', 'event_type']).size()
            print(f"📊 イベント統計:")
            for (session_type, event_type), count in event_stats.items():
                print(f"  {session_type} {event_type}: {count}件")
                
    except Exception as e:
        print(f"❌ セッション詳細抽出失敗: {e}")
    
    # 特定テストケースの詳細確認
    print(f"\n🎯 重要テストケース詳細確認:")
    
    # TC001: 基本パターン
    tc001 = result[result['testcase_id'] == 'TC001']
    if len(tc001) > 0:
        charge_count = tc001['is_charging_stitched'].sum()
        move_count = tc001['is_moving'].sum()
        park_count = tc001['is_parking'].sum()
        print(f"  TC001 基本パターン: 充電{charge_count}回, 移動{move_count}回, 放置{park_count}回")
    
    # TC002: 充電→放置→充電
    tc002 = result[result['testcase_id'] == 'TC002']
    if len(tc002) > 0:
        no_move_after_charge = tc002['is_no_move_after_charge'].sum()
        charge_sessions = tc002['charge_session_id'].max() if 'charge_session_id' in tc002.columns else 0
        print(f"  TC002 放置→再充電: 充電後移動なし{no_move_after_charge}行, 充電セッション{charge_sessions}個")
    
    # TC004: 充電ギャップ
    tc004 = result[result['testcase_id'] == 'TC004']
    if len(tc004) > 0:
        charge_count = tc004['is_charging_stitched'].sum()
        charge_sessions = tc004['charge_session_id'].max() if 'charge_session_id' in tc004.columns else 0
        print(f"  TC004 充電ギャップ: 充電{charge_count}行, 充電セッション{charge_sessions}個（ギャップ補正確認）")
    
    # 結果保存
    try:
        result.to_csv('test_result.csv', index=False)
        print(f"\n💾 結果保存: test_result.csv ({len(result)}行)")
        
        if len(session_details) > 0:
            session_details.to_csv('test_session_details.csv', index=False)
            print(f"💾 セッション詳細保存: test_session_details.csv ({len(session_details)}行)")
            
    except Exception as e:
        print(f"⚠️  結果保存失敗: {e}")
    
    print(f"\n🎉 セッション判定包括テスト完了！")
    
    # 最終評価
    total_errors = len(quality.get('errors', []))
    if total_errors == 0:
        print("🏆 天才テスト職人のテストデータでセッション判定が完璧に動作しました！")
    else:
        print(f"🔧 {total_errors}件のエラーが検出されました。改善が必要です。")

if __name__ == "__main__":
    run_comprehensive_test()
