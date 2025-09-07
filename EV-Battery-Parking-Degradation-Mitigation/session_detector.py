#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vim: set fileencoding=utf-8 :
"""
EVバッテリー駐車劣化軽減のための5セッション判定モジュール
====================================================

【目的】
EVの時系列データから、充電・移動・放置の各セッションを正確に識別し、
バッテリー劣化に影響する放置パターンを分析可能にする。

【5つのセッションタイプ】
1. 充電セッション (charging): バッテリー充電中
2. 移動中セッション (moving): 実際に車両が移動している状態
3. アイドリングセッション (idling): IG-ONだが移動していない状態
4. パーキングセッション (parking): IG-OFFで駐車している状態
5. 非活動セッション (inactive): アイドリング + パーキングの総称（放置状態）

【特殊処理】
- 充電後移動なしの放置は分析対象外として除外判定
- セッション境界での情報を詳細に保持（複数イベントの同時発生に対応）
- 短時間のギャップは同一セッションとして連結処理

【実行順序（優先度順）】
1. 基礎データ準備（型変換・異常値除去・ソート）
2. 充電セッション判定（最優先・SOC条件も考慮）
3. IG-ON/OFF状態判定（車両状態の基本情報）
4. 移動距離計算（GPS座標からハバーサイン距離）
5. 各セッション判定（移動→アイドリング→パーキング順）
6. 放置状態統合（inactive = idling + parking）
7. 充電後移動なし判定（後処理での除外対象識別）
8. セッション境界情報の統合（データ取得しやすさの向上）
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Any, Dict, List
from enum import Enum
from geopy.distance import geodesic  # type: ignore


# ======================
# セッションタイプ定義
# ======================
class SessionType(Enum):
    """
    セッションタイプの定義
    
    EVの状態を5つのタイプに分類し、一貫性のある分析を可能にする。
    各タイプは互いに排他的で、優先順位がある（CHARGING > MOVING > IDLING > PARKING）
    """

    CHARGING = "charging"   # 充電セッション（最優先）
    MOVING = "moving"       # 移動中セッション（実際の車両移動）
    IDLING = "idling"       # アイドリングセッション（IG-ONだが移動なし）
    PARKING = "parking"     # パーキングセッション（IG-OFFで駐車）
    INACTIVE = "inactive"   # 非活動状態（IDLING + PARKINGの総称、放置分析用）


# ======================
# パラメータ定義
# ======================
@dataclass
class SessionParams:
    """
    セッション判定用パラメータ設定
    
    各種閾値を調整することで、分析対象や精度を制御可能。
    デフォルト値は一般的なEV使用パターンに基づいて設定。
    """

    # 移動判定関連
    DIST_TH_m: float = 150.0        # 移動判定距離閾値（メートル）
                                    # この距離以上移動した場合に「移動中」と判定      
    # 充電判定関連
    SOC_TH_pct: float = 5.0         # SOC変化閾値（パーセント）
                                    # 充電判定時のSOC変化の最小値
    # 放置時間分類
    PARK_TH_min: float = 360.0      # 長時間放置閾値（分）= 6時間
                                    # この時間以上の放置を「長時間放置」として分析対象
    SHORT_PARK_min: float = 180.0   # 短時間放置上限（分）= 3時間
                                    # 短時間放置の上限値
    LONG_PARK_min: float = 360.0    # 長時間放置下限（分）= 6時間
                                    # 長時間放置の下限値（PARK_TH_minと同値）
    # セッション連結処理
    GAP_MAX_min: float = 15.0       # 充電ギャップ許容時間（分）
                                    # この時間内の充電中断は同一充電セッションとして扱う


# ======================
# ユーティリティ関数
# ======================
def calculate_geodesic_distance(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """
    地球上の2点間の距離をgeodetic距離で計算
    
    【目的】
    GPS座標から車両の移動距離を正確に算出するため。
    地球の球面性を考慮した距離計算が必要。
    
    【使用ライブラリ】
    geopyのgeodesic: 高精度・効率的な地球上距離計算
    
    Args:
        lat1, lon1: 開始点の緯度・経度（度単位）
        lat2, lon2: 終了点の緯度・経度（度単位）

    Returns:
        float: 2点間の距離（メートル単位）
               エラー時や無効値の場合は0.0を返す
    """
    # データ品質チェック: NaN値は移動距離0として扱う
    # 理由: GPS信号が取得できない場合は移動していないと仮定
    if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2):
        return 0.0
        
    # 座標値の妥当性チェック: 地球上の有効な緯度経度範囲内か確認
    # 緯度: -90度～+90度、経度: -180度～+180度
    if not (
        -90 <= lat1 <= 90
        and -90 <= lat2 <= 90
        and -180 <= lon1 <= 180
        and -180 <= lon2 <= 180
    ):
        return 0.0

    try:
        # geopyで2点間の距離を計算
        point1 = (lat1, lon1)
        point2 = (lat2, lon2)
        
        # geodesicで距離計算（メートル単位で返る）
        distance = geodesic(point1, point2).meters
        return float(distance)

    except (ValueError, TypeError):
        # 計算エラー時は安全に0を返す
        # 例: 極端な座標値、数値変換エラーなど
        return 0.0


# ======================
# 前処理
# ======================
def preprocess_data(df: pd.DataFrame, params: SessionParams) -> pd.DataFrame:
    """
    Step1-3: セッション判定のための基礎データ準備
    
    【目的】
    生データをセッション判定可能な形に整備する。
    データ品質の確保と、後続処理の前提条件を満たすため。
    
    【処理内容】
    1. データ型の統一と品質チェック
    2. 時系列順のソート（重要: セッション判定の前提）
    3. 充電状態の基本判定
    4. IG-ON/OFF状態の判定
    5. 移動距離の計算
    
    Args:
        df: 生の時系列データ
        params: セッション判定パラメータ
        
    Returns:
        pd.DataFrame: 前処理済みのデータ（基本フラグ付与済み）
    """
    d = df.copy()

    # === データ型統一・品質チェック ===
    # 車両ID: 文字列型に統一（数値IDの場合の型不整合を防ぐ）
    d["hashvin"] = d["hashvin"].astype("string")
    
    # SOC（バッテリー残量）: 数値型に変換し、異常値を除去
    d["soc"] = pd.to_numeric(d["soc"], errors="coerce")
    d.loc[d["soc"] < 0, "soc"] = np.nan      # 負の値は物理的に不可能
    d.loc[d["soc"] > 100, "soc"] = np.nan    # 100%超過は異常値

    # === 時系列データ変換 ===
    # 時刻列をdatetime型に変換
    d["tsu_current_time"] = pd.to_datetime(d["tsu_current_time"])
    d["tsu_igon_time"] = pd.to_datetime(d["tsu_igon_time"], errors='coerce')
    
    # === 時系列ソート（重要）===
    # セッション判定は前の状態との比較が必要なため、時系列順が必須
    d = d.sort_values(["hashvin", "tsu_current_time"]).reset_index(drop=True)

    # === Step1: 充電セッション判定 ===
    # 基本充電モード判定: charge_modeの値から充電状態を判定
    d["is_charge_mode"] = d["charge_mode"].isin(
        ["100v charging", "200v charging", "Fast charging"]
    )
    
    # 初期充電判定: まずはcharge_modeベースで設定
    d["is_charging"] = d["is_charge_mode"].fillna(False).astype(bool)
    
    # === 重要: SOC条件による充電判定の補正（最適化版）===
    # 【課題】charge_modeがFalseでも実際は充電継続している場合がある
    # 【解決】前行がcharge_mode=Trueで現行がFalseの場合、SOC変化で判定
    # 【最適化】shift()を使ってベクトル化処理
    
    # hashvinごとに前行のcharge_modeとSOCを取得
    d["prev_is_charge_mode"] = d.groupby("hashvin")["is_charge_mode"].shift(1)
    d["prev_soc"] = d.groupby("hashvin")["soc"].shift(1)
    
    # 補正条件を定義
    # 1. 前行が充電中で現行が充電モードoff
    # 2. 前行と現行のSOCが両方とも有効
    # 3. SOCが維持または増加している
    correction_condition = (
        d["prev_is_charge_mode"].fillna(False) &  # 前行が充電中
        ~d["is_charge_mode"] &                     # 現行が充電モードoff
        d["prev_soc"].notna() &                    # 前行SOCが有効
        d["soc"].notna() &                         # 現行SOCが有効
        (d["soc"] >= d["prev_soc"])                # SOCが維持or増加
    )
    
    # 条件に合致する行を充電継続と判定
    d.loc[correction_condition, "is_charging"] = True
    
    # 一時的な列を削除（メモリ節約）
    d.drop(["prev_is_charge_mode", "prev_soc"], axis=1, inplace=True)

    # === Step2: IGONタイム変更イベントの検出 ===
    # 【目的】車両のIG-ON状態変化を捉える（ユーザーの行動変化を示す）
    # 【処理】hashvinごとにtsu_igon_timeの値が前行から変化した場合を検出
    d["igon_change"] = (
        d.groupby("hashvin")["tsu_igon_time"]
        .transform(lambda s: s.ne(s.shift()))  # 前行と値が異なる場合True
        .fillna(False)  # 最初の行はFalse（比較対象なし）
    )

    # === Step3: 移動距離計算（高速化版）===
    # 前観測点からの移動距離を計算（geodesic距離使用）
    # 【高速化ポイント】
    # 1. hashvinごとにグループ化してshift()で前行データを取得
    # 2. 有効データのみ効率的に計算（無駄な計算を排除）
    # 3. 車両変更時は自動的に距離0（shift結果がNaNになるため）
    
    # hashvinごとに前行の緯度・経度を取得
    # d["prev_lat"] = d.groupby("hashvin")["tsu_rawgnss_latitude"].shift(1)
    # d["prev_lon"] = d.groupby("hashvin")["tsu_rawgnss_longitude"].shift(1)
    
    # # 初期化：すべて0.0で開始
    # d["distance_prev_m"] = 0.0
    
    # # NaN（最初の行や車両変更時）以外で有効な前行データがある行を抽出
    # valid_prev = d["prev_lat"].notna() & d["prev_lon"].notna()
    
    # # 有効な前行データがある行のみ距離計算
    # if valid_prev.any():
    #     # apply()を使って効率的に距離計算
    #     def calc_distance_row(row):
    #         return calculate_geodesic_distance(
    #             row["prev_lat"],
    #             row["prev_lon"],
    #             row["tsu_rawgnss_latitude"],
    #             row["tsu_rawgnss_longitude"]
    #         )
        
    #     d.loc[valid_prev, "distance_prev_m"] = d[valid_prev].apply(calc_distance_row, axis=1)
    
    # # 一時的な列を削除（メモリ節約）
    # d.drop(["prev_lat", "prev_lon"], axis=1, inplace=True)
    
    # 移動判定: 設定閾値以上の距離移動があった場合
    # 【理由】GPS誤差やわずかな位置ずれを移動と誤認しないため
    d["is_moving_distance"] = d["distance_prev_m"] > params.DIST_TH_m
    return d


# ======================
# 充電セッション処理
# ======================
def stitch_charging_sessions(df: pd.DataFrame, params: SessionParams) -> pd.DataFrame:
    """充電セッションのギャップ補正と開始・終了フラグ作成"""
    d = df.copy()

    # 充電ギャップ補正
    rows = []
    for _, g in d.groupby("hashvin", sort=False):
        g = g.copy()

        # 短時間のFalseギャップを補正
        charge_blocks = []
        in_charge = False
        start_idx = None

        for i, (idx, row) in enumerate(g.iterrows()):
            if row["is_charging"] and not in_charge:
                # 充電開始
                in_charge = True
                start_idx = idx
            elif not row["is_charging"] and in_charge:
                # 充電が一時停止 - ギャップをチェック
                if i + 1 < len(g):
                    # 次の充電開始までの時間と距離をチェック
                    next_charge_idx = None
                    for j in range(i + 1, len(g)):
                        if g.iloc[j]["is_charging"]:
                            next_charge_idx = g.index[j]
                            break

                    if next_charge_idx is not None:
                        time_gap = (
                            g.loc[next_charge_idx, "tsu_current_time"]
                            - row["tsu_current_time"]
                        ).total_seconds() / 60.0
                        dist_gap = calculate_geodesic_distance(
                            row["tsu_rawgnss_latitude"],
                            row["tsu_rawgnss_longitude"],
                            g.loc[next_charge_idx, "tsu_rawgnss_latitude"],
                            g.loc[next_charge_idx, "tsu_rawgnss_longitude"],
                        )

                        # ギャップが許容範囲内なら継続
                        if (
                            time_gap <= params.GAP_MAX_min
                            and dist_gap <= params.DIST_TH_m
                        ):
                            continue

                # 充電終了
                charge_blocks.append((start_idx, idx))
                in_charge = False
                start_idx = None

        # 最後まで充電中の場合
        if in_charge and start_idx is not None:
            charge_blocks.append((start_idx, g.index[-1]))

        # 補正結果をマーキング
        g["is_charging_stitched"] = False
        g["charge_block_id"] = np.nan

        for bid, (start_idx, end_idx) in enumerate(charge_blocks, 1):
            # 範囲内のインデックスを取得して値を設定
            mask = (g.index >= start_idx) & (g.index <= end_idx)
            g.loc[mask, "is_charging_stitched"] = True
            g.loc[mask, "charge_block_id"] = bid

        rows.append(g)

    result = (
        pd.concat(rows)
        .sort_values(["hashvin", "tsu_current_time"])
        .reset_index(drop=True)
    )

    # 開始・終了フラグ作成
    result["charge_start_flag"] = (
        result.groupby("hashvin")["is_charging_stitched"]
        .transform(lambda s: s & ~s.shift(1, fill_value=False))
        .fillna(False)
    )

    result["charge_end_flag"] = (
        result.groupby("hashvin")["is_charging_stitched"]
        .transform(lambda s: ~s & s.shift(1, fill_value=False))
        .fillna(False)
    )

    result["charge_session_id"] = result.groupby("hashvin")[
        "charge_start_flag"
    ].cumsum()

    return result


# ======================
# セッション判定メイン処理
# ======================

# def determine_sessions(df: pd.DataFrame, params: SessionParams) -> pd.DataFrame:
#     """Step4-6: 各セッションの判定（優先度: parking→moving→idling）"""
#     # paramsは将来の拡張用に保持
#     _ = params  # 未使用警告を回避
#     d = df.copy()

#     # 初期化：すべてのセッションフラグをFalseに設定
#     d["is_parking"] = False
#     d["is_moving"] = False  
#     d["is_idling"] = False

#     # Step4: パーキングセッション判定（最優先）
#     # igon_changeがTRUEの行の一つ前の行で、igon_changeがFALSEかつis_chargingがFALSEの場合
#     # まず、igon_changeがTRUEの行を特定
#     igon_change_true_mask = d["igon_change"] == True
    
#     # 次の行でigon_changeがTRUEになる行（一つ前の行）を特定
#     next_igon_change_mask = d.groupby("hashvin")["igon_change"].shift(-1).fillna(False)
    
#     # パーキング条件：次の行でigon_changeがTRUE、かつ現在行でigon_changeがFALSE、かつis_chargingがFALSE
#     d.loc[
#         next_igon_change_mask & 
#         (d["igon_change"] == False) & 
#         (d["is_charging_stitched"] == False), 
#         "is_parking"
#     ] = True

#     d["parking_start_flag"] = (
#         d.groupby("hashvin")["is_parking"]
#         .transform(lambda s: s & ~s.shift(1, fill_value=False))
#         .fillna(False)
#     )

#     d["parking_end_flag"] = (
#         d.groupby("hashvin")["is_parking"]
#         .transform(lambda s: ~s & s.shift(1, fill_value=False))
#         .fillna(False)
#     )

#     d["parking_session_id"] = d.groupby("hashvin")["parking_start_flag"].cumsum()

#     # Step5: 移動中セッション判定（パーキング次点）
#     # IG-ON、充電中でない、パーキング中でない、かつ移動距離がある場合
#     d.loc[
#         ~d["is_charging_stitched"] & 
#         ~d["is_parking"] & 
#         d["is_moving_distance"], 
#         "is_moving"
#     ] = True

#     d["movement_start_flag"] = (
#         d.groupby("hashvin")["is_moving"]
#         .transform(lambda s: s & ~s.shift(1, fill_value=False))
#         .fillna(False)
#     )

#     d["movement_end_flag"] = (
#         d.groupby("hashvin")["is_moving"]
#         .transform(lambda s: ~s & s.shift(1, fill_value=False))
#         .fillna(False)
#     )

#     d["movement_session_id"] = d.groupby("hashvin")["movement_start_flag"].cumsum()

#     # Step6: アイドリングセッション判定（最低優先度）
#     # IG-ON、充電中でない、パーキング中でない、移動中でない場合
#     d.loc[
#         ~d["is_charging_stitched"] & 
#         ~d["is_parking"] & 
#         ~d["is_moving"], 
#         "is_idling"
#     ] = True

#     d["idling_start_flag"] = (
#         d.groupby("hashvin")["is_idling"]
#         .transform(lambda s: s & ~s.shift(1, fill_value=False))
#         .fillna(False)
#     )

#     d["idling_end_flag"] = (
#         d.groupby("hashvin")["is_idling"]
#         .transform(lambda s: ~s & s.shift(1, fill_value=False))
#         .fillna(False)
#     )

#     d["idling_session_id"] = d.groupby("hashvin")["idling_start_flag"].cumsum()

#     return d
# ======================
# セッション判定メイン処理（修正版）
# ======================
### is_pakingのstartとend 
# - is_parkingのstartはis_parkingがFALSEからTRUEに変わったTRUEの行がstart
#  →is_parkingがTRUEは、igonの前の行のこと。igonでエンジンが起動するので、その一つ前の行はエンジンOFF
#  - is_parkingのendはis_parkingがTRUEからFALSEに変わったFALSEの行がend
# 
#  ### is_movementのstartとend
#  - start条件
#  - is_movementがFALSEからTRUEに変わるFALSEの行
#  - end条件
#  - is_movementがTRUEからFALSEに変わるTRUE行
# 
#  ### is_idlingのstartとend条件
#  - start条件
#  - is_idlingがFALSEからTRUEに変わるFALSE行
#  - end条件
#  - is_idlingがTRUEからFALSEに変わるTRUE行
def determine_sessions(df: pd.DataFrame, params: SessionParams) -> pd.DataFrame:
    """Step4-6: 各セッションの判定
       優先度: parking → moving → idling
       ※開始・終了フラグの付与位置は要件どおり
         - parking: start=True行, end=False行
         - moving : start=False行, end=True行
         - idling : start=False行, end=True行
    """
    _ = params  # 予備
    d = df.copy()

    # 初期化
    d["is_parking"] = False
    d["is_moving"]  = False
    d["is_idling"]  = False

    # -----------------------------
    # Step4: パーキング判定（最優先）
    # 「次の行で igon_change が True」かつ 現在行が「igon_change False & 非充電」
    next_igon_change = (
        d.groupby("hashvin")["igon_change"]
        .shift(-1)
        .fillna(False)
        .astype(bool)
    )
    d.loc[
        next_igon_change &
        (~d["igon_change"].astype(bool)) &
        (~d["is_charging_stitched"].astype(bool)),
        "is_parking"
    ] = True

    # parking start/end（要件）
    # start: False→True になった True 行
    d["parking_start_flag"] = (
        d.groupby("hashvin")["is_parking"]
         .apply(lambda s: (s & ~s.shift(1, fill_value=False)))
         .reset_index(level=0, drop=True)
    )

    # end: True→False になった False 行
    d["parking_end_flag"] = (
        d.groupby("hashvin")["is_parking"]
         .apply(lambda s: (~s & s.shift(1, fill_value=False)))
         .reset_index(level=0, drop=True)
    )

    # parking_session_id は True 側の開始で採番し、True 行のみに付与
    parking_start_on_true = (
        d.groupby("hashvin")["is_parking"]
         .apply(lambda s: (s & ~s.shift(1, fill_value=False)))
         .reset_index(level=0, drop=True)
    )
    d["parking_session_id"] = (
        d.groupby("hashvin")[parking_start_on_true.name]
         .apply(lambda s: parking_start_on_true.loc[s.index].cumsum())
         .reset_index(level=0, drop=True)
    )
    # True 行のみIDを残し、その他は0（必要ならNaNでも可）
    d.loc[~d["is_parking"], "parking_session_id"] = 0
    d["parking_session_id"] = d["parking_session_id"].astype(int)

    # -----------------------------
    # Step5: 移動判定（パーキング以外で、移動距離がある等の条件）
    # 既存ロジックを尊重：非充電 & 非パーキング & is_moving_distance
    d.loc[
        (~d["is_charging_stitched"].astype(bool)) &
        (~d["is_parking"]) &
        (d["is_moving_distance"].astype(bool)),
        "is_moving"
    ] = True

    # moving start/end（要件）
    # start: False→True になる直前の False 行
    d["movement_start_flag"] = (
        d.groupby("hashvin")["is_moving"]
         .apply(lambda s: ((~s) & s.shift(-1, fill_value=False)))
         .reset_index(level=0, drop=True)
    )
    # end: True→False になる True 行
    d["movement_end_flag"] = (
        d.groupby("hashvin")["is_moving"]
         .apply(lambda s: (s & ~s.shift(-1, fill_value=False)))
         .reset_index(level=0, drop=True)
    )

    # movement_session_id は True 側の開始で採番し、True 行のみに付与
    movement_start_on_true = (
        d.groupby("hashvin")["is_moving"]
         .apply(lambda s: (s & ~s.shift(1, fill_value=False)))
         .reset_index(level=0, drop=True)
    )
    d["movement_session_id"] = (
        d.groupby("hashvin")[movement_start_on_true.name]
         .apply(lambda s: movement_start_on_true.loc[s.index].cumsum())
         .reset_index(level=0, drop=True)
    )
    d.loc[~d["is_moving"], "movement_session_id"] = 0
    d["movement_session_id"] = d["movement_session_id"].astype(int)

    # -----------------------------
    # Step6: アイドリング判定（残り：非充電 & 非パーキング & 非移動）
    d.loc[
        (~d["is_charging_stitched"].astype(bool)) &
        (~d["is_parking"]) &
        (~d["is_moving"]),
        "is_idling"
    ] = True

    # idling start/end（要件）
    # start: False→True になる直前の False 行
    d["idling_start_flag"] = (
        d.groupby("hashvin")["is_idling"]
         .apply(lambda s: ((~s) & s.shift(-1, fill_value=False)))
         .reset_index(level=0, drop=True)
    )
    # end: True→False になる True 行
    d["idling_end_flag"] = (
        d.groupby("hashvin")["is_idling"]
         .apply(lambda s: (s & ~s.shift(-1, fill_value=False)))
         .reset_index(level=0, drop=True)
    )

    # idling_session_id は True 側の開始で採番し、True 行のみに付与
    idling_start_on_true = (
        d.groupby("hashvin")["is_idling"]
         .apply(lambda s: (s & ~s.shift(1, fill_value=False)))
         .reset_index(level=0, drop=True)
    )
    d["idling_session_id"] = (
        d.groupby("hashvin")[idling_start_on_true.name]
         .apply(lambda s: idling_start_on_true.loc[s.index].cumsum())
         .reset_index(level=0, drop=True)
    )
    d.loc[~d["is_idling"], "idling_session_id"] = 0
    d["idling_session_id"] = d["idling_session_id"].astype(int)

    return d



# ======================
# 充電後移動なし判定
# ======================
def check_no_movement_after_charge(
    df: pd.DataFrame, params: SessionParams
) -> pd.DataFrame:
    """Step7: 充電後移動なし放置判定（改訂版）"""
    d = df.copy()
    d["is_no_move_after_charge"] = False

    for _, g in d.groupby("hashvin", sort=False):
        g = g.copy()
        charge_start_indices = g[g["charge_start_flag"]].index

        for cs_idx in charge_start_indices:
            # 充電開始後の次の移動開始を探す
            after_charge_start = g[g.index > cs_idx]
            movement_start_mask = after_charge_start["movement_start_flag"]
            
            if movement_start_mask.any():
                # 次の移動開始が見つかった場合
                next_movement_idx = after_charge_start[movement_start_mask].index[0]
                
                # 充電開始から次の移動開始までのデータを取得
                between_data = g[
                    (g.index >= cs_idx) & (g.index < next_movement_idx)
                ]
                
                # この期間の最大移動距離をチェック
                if len(between_data) > 0:
                    max_distance = between_data["distance_prev_m"].max()
                    
                    if max_distance <= params.DIST_TH_m:
                        # 移動なしと判定 - この期間の放置セッション（inactive = idle + parking）をマーク
                        inactive_data = between_data[between_data["is_idling"] | between_data["is_parking"]]
                        if len(inactive_data) > 0:
                            d.loc[inactive_data.index, "is_no_move_after_charge"] = True
            else:
                # 次の移動開始が見つからない場合、充電開始後の全データをチェック
                after_charge_data = g[g.index > cs_idx]
                
                if len(after_charge_data) > 0:
                    max_distance = after_charge_data["distance_prev_m"].max()
                    
                    if max_distance <= params.DIST_TH_m:
                        # 移動なしと判定 - 充電開始後の放置セッションをマーク
                        inactive_data = after_charge_data[after_charge_data["is_idling"] | after_charge_data["is_parking"]]
                        if len(inactive_data) > 0:
                            d.loc[inactive_data.index, "is_no_move_after_charge"] = True

    return d


# ======================
# 放置時間分類
# ======================
def classify_parking_duration(df: pd.DataFrame, params: SessionParams) -> pd.DataFrame:
    """パーキングセッションの時間分類"""
    d = df.copy()
    d["parking_category"] = "unknown"
    d["parking_duration_min"] = np.nan

    for _, g in d.groupby("hashvin", sort=False):
        g = g.copy()

        # パーキングセッションごとに継続時間を計算
        parking_sessions = g[g["parking_session_id"] > 0]["parking_session_id"].unique()

        for session_id in parking_sessions:
            session_data = g[g["parking_session_id"] == session_id]

            if len(session_data) == 0:
                continue

            start_time = session_data["tsu_current_time"].iloc[0]
            end_time = session_data["tsu_current_time"].iloc[-1]
            duration_min = (end_time - start_time).total_seconds() / 60.0

            # 分類
            if duration_min < params.SHORT_PARK_min:
                category = "短時間放置(<3h)"
            elif duration_min < params.LONG_PARK_min:
                category = "中時間放置(3-6h)"
            else:
                category = "長時間放置(≥6h)"

            d.loc[session_data.index, "parking_duration_min"] = duration_min
            d.loc[session_data.index, "parking_category"] = category

    return d


# ======================
# セッションタイプ統合処理
# ======================
# def determine_session_type(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     セッションフラグから統合session_type列を作成
    
#     【目的】
#     個別のセッションフラグ（is_charging, is_moving等）を統合し、
#     境界情報も保持しながら、分析しやすい単一の列を提供する。
    
#     【設計方針】
#     1. 主要セッション優先: 境界があっても主セッションを明確化
#     2. 境界情報分離: 複数イベント同時発生時の詳細情報を保持
#     3. 優先順位: CHARGING > MOVING > IDLING > PARKING
    
#     【出力】
#     - session_type: 主要セッションタイプ
#     - is_inactive: 放置状態統合フラグ
#     - inactive関連フラグ: 放置セッションの詳細情報
#     """
#     d = df.copy()


#     # === 主要セッションタイプの決定 ===
#     # 【方針】境界イベントがあっても、その行の主要な状態を明確にする
#     # 【優先順位】CHARGING > MOVING > IDLING > PARKING
#     #             （バッテリー劣化分析では充電状態が最重要）
#     def get_primary_session_type(row):
#         # 1. 充電優先（最重要）
#         # 充電中はバッテリー状態が変化するため、他の状態より重要
#         if row.get("is_charging_stitched", False):
#             return SessionType.CHARGING.value

#         # 2. 移動中
#         # 実際に車両が移動している状態（エネルギー消費）
#         if row.get("is_moving", False):
#             return SessionType.MOVING.value

#         # 3. アイドリング（放置の一種）
#         # IG-ONだが移動していない状態（エネルギー消費あり）
#         if row.get("is_idling", False):
#             return SessionType.IDLING.value

#         # 4. パーキング（放置の一種）
#         # IG-OFFで駐車している状態（エネルギー消費最小）
#         if row.get("is_parking", False):
#             return SessionType.PARKING.value

#         # デフォルト: 該当するセッションなし（データ不備等）
#         return None

#     # 主要セッションタイプを全行に適用
#     d["session_type"] = d.apply(get_primary_session_type, axis=1)

#     # === 放置状態（inactive）の統合処理 ===
#     # 【目的】idling + parking = inactive として、放置状態を統一的に扱う
#     # 【理由】バッテリー劣化分析では、IG-ON/OFFの違いより放置時間が重要
#     d["is_inactive"] = d["is_idling"] | d["is_parking"]
    
#     # inactive開始フラグ: 放置状態になった瞬間を検出
#     # 【ロジック】現在が放置中 かつ 前行が放置でない場合
#     d["inactive_start_flag"] = (
#         d.groupby("hashvin")["is_inactive"]
#         .transform(lambda s: s & ~s.shift(1, fill_value=False))
#         .fillna(False)
#     )
    
#     # inactive終了フラグ: 放置状態から抜けた瞬間を検出
#     # 【ロジック】現在が放置でない かつ 前行が放置中の場合
#     d["inactive_end_flag"] = (
#         d.groupby("hashvin")["is_inactive"]
#         .transform(lambda s: ~s & s.shift(1, fill_value=False))
#         .fillna(False)
#     )
    
#     # inactiveセッションID: 各放置セッションに一意のIDを付与
#     # 【用途】放置期間の集計や分析時のグループ化に使用
#     d["inactive_session_id"] = d.groupby("hashvin")["inactive_start_flag"].cumsum()

#     return d

import numpy as np
import pandas as pd

def determine_session_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    セッションフラグから統合 session_type と、inactive（= idling ∪ parking）の
    開始・終了フラグ/セッションIDを“ブロック駆動”で生成する。

    重要ポイント（ズレない理由）:
      1) まず is_inactive = is_idling | is_parking から「連続ブロック」を抽出
         - ブロック開始 = is_inactive が False→True になる最初の True 行（canonical start）
         - ブロック終了 = is_inactive が True→False になる最初の False 行（canonical end）
         - ブロック内の idling↔parking 切替は「モード切替」であり、inactive の境界ではない
      2) 各ブロックごとに、仕様どおりの“アンカー行”に start/end を1つずつ立てる
         - start アンカー:
             * ブロック開始行で parking==True なら、その True 行（parking の仕様）
             * そうでなく idling==True なら、その直前の False 行（idling の仕様）
             * 先頭行で前行が無い等の端ケースは、当該開始 True 行に立てる（安全フォールバック）
         - end アンカー:
             * ブロック終了直前（最後の True 行）が parking==True なら、終了は False 行（parking の仕様）
             * そうでなく idling==True なら、終了は最後の True 行（idling の仕様）
             * データ終端まで継続し終了が無い場合は end を立てない（未終了ブロック）

    これにより、
      - OR で起こる「区間内に start/end が複数発生」問題を根本から回避
      - 事後の“同時発生潰し”も不要（論理的に1ブロック=1 start + 1 end）

    前提:
      - hashvin 内で時系列順に並んでいること（本関数では並び替えは行わない）

    出力:
      - session_type: "charging"|"moving"|"idling"|"parking"|None（優先度 CHARGING > MOVING > IDLING > PARKING）
      - is_inactive: bool（idling or parking）
      - inactive_start_flag, inactive_end_flag: アンカー行にのみ True
      - inactive_session_id: 連続ブロックID（is_inactive==True 行にのみ >0）
      - inactive_start_source / inactive_end_source: どちらの仕様に基づくアンカーか
      - boundary_events / has_boundary: デバッグ用のイベント要約
    """

    d = df.copy()

    # --- 防御的初期化（欠けていても落ちない） ---
    if "hashvin" not in d.columns:
        d["hashvin"] = "__all__"

    for col in ["is_charging_stitched", "is_moving", "is_idling", "is_parking"]:
        if col not in d.columns:
            d[col] = False
        d[col] = d[col].fillna(False).astype(bool)

    # --- session_type（優先度: charging > moving > idling > parking） ---
    def _val(name: str) -> str:
        try:
            return getattr(SessionType, name.upper()).value  # noqa: F821
        except Exception:
            return name.lower()

    d["session_type"] = None
    # charging
    mask = d["is_charging_stitched"]
    d.loc[mask, "session_type"] = _val("CHARGING")
    # moving
    d.loc[d["session_type"].isna() & d["is_moving"], "session_type"] = _val("MOVING")
    # idling
    d.loc[d["session_type"].isna() & d["is_idling"], "session_type"] = _val("IDLING")
    # parking
    d.loc[d["session_type"].isna() & d["is_parking"], "session_type"] = _val("PARKING")

    # --- inactive 基本フラグ ---
    d["is_inactive"] = d["is_idling"] | d["is_parking"]

    # 出力列の器
    d["inactive_start_flag"] = False
    d["inactive_end_flag"]   = False
    d["inactive_start_source"] = None  # "idling(False_row)" | "parking(True_row)"
    d["inactive_end_source"]   = None  # "idling(True_row)"  | "parking(False_row)"
    d["inactive_session_id"] = 0

    # === ブロック駆動: hashvin ごとに処理 ===
    for vin, sub in d.groupby("hashvin", sort=False):
        idx = sub.index.to_numpy()
        idl = sub["is_idling"].to_numpy(dtype=bool)
        prk = sub["is_parking"].to_numpy(dtype=bool)
        ina = (idl | prk).astype(bool)

        if idx.size == 0:
            continue

        # 連続ブロックの canonical start/end（is_inactive の遷移のみで決定）
        prev_ina = np.r_[False, ina[:-1]]
        next_ina = np.r_[ina[1:], False]
        starts_true = np.where(ina & ~prev_ina)[0]   # ブロックの最初の True 行
        ends_false  = np.where(~ina & prev_ina)[0]   # ブロックの直後の False 行

        # セッションIDは「ブロック開始(True行)」の累積で付与し、True 行にのみ残す
        # ここはアンカー位置（前の False 行）とは切り離し、IDの安定性を担保
        block_start_true_flag = np.zeros_like(ina, dtype=bool)
        block_start_true_flag[starts_true] = True
        session_id_series = block_start_true_flag.cumsum()
        # is_inactive=False は 0 に
        session_id_series = np.where(ina, session_id_series, 0)

        # 結果を書き戻し（後でアンカーも立てる）
        d.loc[idx, "inactive_session_id"] = session_id_series

        # 各ブロックに 1 start + 1 end（未終了は end なし）
        for s_pos in starts_true:
            # --- start アンカー決定 ---
            if prk[s_pos]:
                # parking で始まっている → True 行が start アンカー
                start_anchor_local = s_pos
                start_source = "parking(True_row)"
            elif idl[s_pos]:
                # idling で始まっている → 直前の False 行が start アンカー
                if s_pos > 0:
                    start_anchor_local = s_pos - 1
                    start_source = "idling(False_row)"
                else:
                    # 先頭行で直前が無い場合のフォールバック（True 行に立てる）
                    start_anchor_local = s_pos
                    start_source = "idling(False_row|fallback_to_first_true)"
            else:
                # 理論上ここには来ないが、防御
                start_anchor_local = s_pos
                start_source = "unknown(start_on_true)"

            start_anchor_idx = idx[start_anchor_local]
            d.at[start_anchor_idx, "inactive_start_flag"] = True
            d.at[start_anchor_idx, "inactive_start_source"] = start_source

            # --- end アンカー決定 ---
            # この start 以降で最初に現れるブロック終了 False 行を取得
            e_candidates = ends_false[ends_false > s_pos]
            if e_candidates.size == 0:
                # 未終了（データ終端まで放置継続）の場合、end は立てない
                continue

            e_pos = int(e_candidates[0])            # ブロック直後の False 行（canonical end）
            last_true = e_pos - 1                   # 直前の True 行（ブロックの最終 True 行）
            if last_true < 0:
                # 防御（理論上あり得ない）
                last_true = 0

            if prk[last_true]:
                # parking で終わっている → False 行が end アンカー
                end_anchor_local = e_pos
                end_source = "parking(False_row)"
            elif idl[last_true]:
                # idling で終わっている → 直前の True 行が end アンカー
                end_anchor_local = last_true
                end_source = "idling(True_row)"
            else:
                # 防御（理論上あり得ない）
                end_anchor_local = e_pos
                end_source = "unknown(end_on_false)"

            end_anchor_idx = idx[end_anchor_local]
            d.at[end_anchor_idx, "inactive_end_flag"] = True
            d.at[end_anchor_idx, "inactive_end_source"] = end_source

    # --- 境界イベント（デバッグ用） ---
    def _events_row(row) -> str:
        ev = []
        if row.get("inactive_start_flag", False): ev.append("inactive_start")
        if row.get("inactive_end_flag", False):   ev.append("inactive_end")
        return "|".join(ev)

    d["boundary_events"] = d.apply(_events_row, axis=1)
    d["has_boundary"] = d["boundary_events"].astype(str).str.len().gt(0)

    # 型整備
    d["inactive_session_id"] = d["inactive_session_id"].astype(int)

    return d



# ======================
# 統合セッション抽出機能
# ======================
def extract_charge_to_inactive_sessions(
    df: pd.DataFrame, params: SessionParams
) -> pd.DataFrame:
    """
    充電開始→移動→放置（inactive）終了の統合セッションを抽出

    Returns:
        統合セッション情報のDataFrame
        - charge_to_inactive_session_id: 統合セッションID
        - session_start_time: 充電開始時刻
        - session_end_time: 放置終了時刻
        - charge_duration_min: 充電時間
        - movement_duration_min: 移動時間
        - inactive_duration_min: 放置時間（idle + parking）
        - total_session_duration_min: 全体時間
        - has_movement: 移動があったか
        - is_long_inactive: 長時間放置か
        - is_excluded_no_movement: 充電後移動なしで除外対象か
    """
    result_sessions: List[Dict[str, Any]] = []

    for hashvin, vehicle_data in df.groupby("hashvin"):
        vehicle_data = vehicle_data.sort_values("tsu_current_time").reset_index(
            drop=True
        )

        # 充電開始点を特定
        charge_starts = vehicle_data[vehicle_data["charge_start_flag"]].index.tolist()

        for i, charge_start_idx in enumerate(charge_starts):
            session_info = {
                "hashvin": hashvin,
                "charge_to_inactive_session_id": len(result_sessions) + 1,
                "session_start_time": vehicle_data.loc[
                    charge_start_idx, "tsu_current_time"
                ],
                "charge_start_idx": charge_start_idx,
                "charge_start_lat": vehicle_data.loc[charge_start_idx, "tsu_rawgnss_latitude"],
                "charge_start_lon": vehicle_data.loc[charge_start_idx, "tsu_rawgnss_longitude"],
            }

            # 充電終了を探す
            charge_end_mask = (vehicle_data.index > charge_start_idx) & vehicle_data[
                "charge_end_flag"
            ]
            if not charge_end_mask.any():
                continue

            charge_end_idx = vehicle_data[charge_end_mask].index[0]
            session_info["charge_end_idx"] = charge_end_idx
            session_info["charge_end_time"] = vehicle_data.loc[
                charge_end_idx, "tsu_current_time"
            ]
            session_info["charge_end_lat"] = vehicle_data.loc[charge_end_idx, "tsu_rawgnss_latitude"]
            session_info["charge_end_lon"] = vehicle_data.loc[charge_end_idx, "tsu_rawgnss_longitude"]

            # 充電後の移動・放置を探す
            after_charge = vehicle_data[vehicle_data.index > charge_end_idx]

            # 放置開始を探す（inactive = idle or parking）
            inactive_start_mask = after_charge["inactive_start_flag"]
            movement_occurred = after_charge["is_moving"].any()

            if inactive_start_mask.any():
                inactive_start_idx = after_charge[inactive_start_mask].index[0]
                session_info["inactive_start_idx"] = inactive_start_idx
                session_info["inactive_start_time"] = vehicle_data.loc[
                    inactive_start_idx, "tsu_current_time"
                ]
                session_info["inactive_start_lat"] = vehicle_data.loc[inactive_start_idx, "tsu_rawgnss_latitude"]
                session_info["inactive_start_lon"] = vehicle_data.loc[inactive_start_idx, "tsu_rawgnss_longitude"]

                # 放置終了を探す
                after_inactive_start = vehicle_data[
                    vehicle_data.index > inactive_start_idx
                ]
                inactive_end_mask = after_inactive_start["inactive_end_flag"]

                if inactive_end_mask.any():
                    inactive_end_idx = after_inactive_start[inactive_end_mask].index[0]
                    session_info["inactive_end_idx"] = inactive_end_idx
                    session_info["session_end_time"] = vehicle_data.loc[
                        inactive_end_idx, "tsu_current_time"
                    ]
                    session_info["inactive_end_lat"] = vehicle_data.loc[inactive_end_idx, "tsu_rawgnss_latitude"]
                    session_info["inactive_end_lon"] = vehicle_data.loc[inactive_end_idx, "tsu_rawgnss_longitude"]

                    # 各セッションの時間計算
                    charge_duration = (
                        session_info["charge_end_time"]
                        - session_info["session_start_time"]
                    ).total_seconds() / 60
                    total_duration = (
                        session_info["session_end_time"]
                        - session_info["session_start_time"]
                    ).total_seconds() / 60
                    inactive_duration = (
                        session_info["session_end_time"]
                        - session_info["inactive_start_time"]
                    ).total_seconds() / 60
                    movement_duration = (
                        total_duration - charge_duration - inactive_duration
                    )
                    
                    # 充電後移動なし判定（改善版）
                    # 【条件】1. 移動がない AND 2. 放置期間がある AND 3. 実際に充電終了から放置開始まで移動なし
                    charge_to_inactive_data = vehicle_data[
                        (vehicle_data.index >= charge_end_idx) & 
                        (vehicle_data.index <= inactive_start_idx)
                    ]
                    actual_movement_distance = charge_to_inactive_data["distance_prev_m"].sum()
                    
                    is_excluded = (
                        not movement_occurred and 
                        inactive_duration > 0 and
                        actual_movement_distance <= params.DIST_TH_m
                    )

                    session_info.update(
                        {
                            "charge_duration_min": charge_duration,
                            "movement_duration_min": max(0, movement_duration),
                            "inactive_duration_min": inactive_duration,
                            "total_session_duration_min": total_duration,
                            "has_movement": movement_occurred,
                            "is_long_inactive": inactive_duration >= params.LONG_PARK_min,
                            "is_excluded_no_movement": is_excluded,
                            "is_valid_session": True,
                        }
                    )

                    result_sessions.append(session_info)
                else:
                    # 放置終了が見つからない場合
                    session_info.update(
                        {
                            "session_end_time": None,
                            "inactive_end_idx": None,
                            "is_valid_session": False,
                            "reason": "inactive_end_not_found",
                        }
                    )
                    result_sessions.append(session_info)
            else:
                # 放置開始が見つからない場合（充電→移動→充電など）
                session_info.update(
                    {
                        "inactive_start_time": None,
                        "inactive_start_idx": None,
                        "session_end_time": None,
                        "has_movement": movement_occurred,
                        "is_valid_session": False,
                        "reason": "no_inactive_after_charge",
                    }
                )
                result_sessions.append(session_info)

    return pd.DataFrame(result_sessions)


# from typing import Any, Dict, List, Optional
# import pandas as pd

# def extract_charge_to_inactive_sessions(
#     df: pd.DataFrame, params
# ) -> pd.DataFrame:
#     """
#     充電開始→充電終了→（移動あり）→最初の長時間放置→放置終了 を 1 区間として抽出。

#     変更点:
#       - 移動の有無は is_no_move_after_charge で判定（True が1つでもあれば除外）
#       - 次の充電開始が存在しない場合は、その hashvin の最終行を探索終端に採用
#     使う主な列:
#       hashvin, tsu_current_time, charge_start_flag, charge_end_flag,
#       inactive_start_flag, inactive_end_flag, inactive_session_id,
#       is_inactive, is_no_move_after_charge
#     使う主なパラメータ:
#       params.LONG_PARK_min : 長時間放置しきい値 [分]
#     """
#     results: List[Dict[str, Any]] = []

#     for hashvin, vdf in df.groupby("hashvin"):
#         vdf = vdf.sort_values("tsu_current_time").reset_index(drop=True)

#         charge_starts: List[int] = vdf.index[vdf["charge_start_flag"]].tolist()

#         for idx_cs, cs_idx in enumerate(charge_starts):
#             # 充電終了
#             ce_idxs = vdf.index[(vdf.index > cs_idx) & (vdf["charge_end_flag"])]
#             if len(ce_idxs) == 0:
#                 results.append({
#                     "hashvin": hashvin,
#                     "charge_to_inactive_session_id": len(results) + 1,
#                     "session_start_time": vdf.loc[cs_idx, "tsu_current_time"],
#                     "is_valid_session": False,
#                     "reason": "charge_end_not_found",
#                 })
#                 continue
#             ce_idx = int(ce_idxs[0])

#             # 次の充電開始があればその直前、なければこの hashvin の最終行
#             if idx_cs + 1 < len(charge_starts):
#                 window_end_idx = charge_starts[idx_cs + 1] - 1
#             else:
#                 window_end_idx = int(vdf.index.max())

#             window_mask = (vdf.index > ce_idx) & (vdf.index <= window_end_idx)
#             win = vdf.loc[window_mask].copy()

#             rec: Dict[str, Any] = {
#                 "hashvin": hashvin,
#                 "charge_to_inactive_session_id": len(results) + 1,
#                 "charge_start_idx": cs_idx,
#                 "charge_start_time": vdf.loc[cs_idx, "tsu_current_time"],
#                 "charge_end_idx": ce_idx,
#                 "charge_end_time": vdf.loc[ce_idx, "tsu_current_time"],
#             }

#             # 放置候補（inactive_session_id 単位）
#             candidates: List[Dict[str, Any]] = []
#             if not win.empty and "inactive_session_id" in vdf.columns:
#                 start_rows = win.loc[win["inactive_start_flag"] == True]  # noqa: E712
#                 for sid in start_rows["inactive_session_id"].dropna().unique().tolist():
#                     sid_rows = win.loc[win["inactive_session_id"] == sid]
#                     if sid_rows.empty:
#                         continue

#                     inact_start_idx = int(sid_rows.index.min())

#                     # 終了フラグが撮れていない未完了放置は採用不可
#                     end_rows = sid_rows.loc[sid_rows["inactive_end_flag"] == True]  # noqa: E712
#                     if len(end_rows) == 0:
#                         continue
#                     inact_end_idx = int(end_rows.index.max())

#                     inact_start_time = vdf.loc[inact_start_idx, "tsu_current_time"]
#                     inact_end_time   = vdf.loc[inact_end_idx, "tsu_current_time"]
#                     inact_dur_min = (inact_end_time - inact_start_time).total_seconds() / 60.0

#                     # 充電終了→この放置開始までの区間に "充電後移動なし" があるか
#                     pre_inactive_slice = vdf.loc[(vdf.index >= ce_idx) & (vdf.index <= inact_start_idx)]
#                     no_move_flag = bool(pre_inactive_slice.get("is_no_move_after_charge", pd.Series([False])).any())

#                     candidates.append({
#                         "inactive_session_id": sid,
#                         "inactive_start_idx": inact_start_idx,
#                         "inactive_end_idx": inact_end_idx,
#                         "inactive_start_time": inact_start_time,
#                         "inactive_end_time": inact_end_time,
#                         "inactive_duration_min": inact_dur_min,
#                         "is_no_move_after_charge": no_move_flag,
#                     })

#             # 条件: 「移動あり」= is_no_move_after_charge が False かつ 「長時間放置」
#             chosen: Optional[Dict[str, Any]] = None
#             for c in sorted(candidates, key=lambda x: x["inactive_start_idx"]):
#                 if (not c["is_no_move_after_charge"]) and (c["inactive_duration_min"] >= params.LONG_PARK_min):
#                     chosen = c
#                     break

#             # 採用なし → 理由を付けて無効レコード
#             if chosen is None:
#                 if len(candidates) == 0:
#                     reason = "no_inactive_in_window"
#                 else:
#                     any_long = any(c["inactive_duration_min"] >= params.LONG_PARK_min for c in candidates)
#                     any_move = any(not c["is_no_move_after_charge"] for c in candidates)
#                     if (not any_move) and (not any_long):
#                         reason = "no_movement_and_no_long_inactive"
#                     elif not any_move:
#                         reason = "no_movement_after_charge"
#                     elif not any_long:
#                         reason = "only_short_inactive"
#                     else:
#                         reason = "unknown_filter_out"

#                 # 充電後移動なしの総合判定（ウィンドウ全体の参考値）
#                 overall_no_move = bool(win.get("is_no_move_after_charge", pd.Series([False])).any())

#                 rec.update({
#                     "inactive_session_id": None,
#                     "inactive_start_idx": None,
#                     "inactive_end_idx": None,
#                     "session_end_time": None,
#                     "charge_duration_min": (rec["charge_end_time"] - rec["charge_start_time"]).total_seconds() / 60.0,
#                     "movement_duration_min": None,
#                     "inactive_duration_min": None,
#                     "total_session_duration_min": None,
#                     "has_movement": (not overall_no_move),
#                     "is_long_inactive": False,
#                     "is_excluded_no_movement": overall_no_move,
#                     "is_no_move_after_charge": overall_no_move,  # 参考出力
#                     "is_valid_session": False,
#                     "reason": reason,
#                 })
#                 results.append(rec)
#                 continue

#             # 採用セッションの集計
#             session_end_time = chosen["inactive_end_time"]
#             charge_dur_min = (rec["charge_end_time"] - rec["charge_start_time"]).total_seconds() / 60.0
#             inactive_dur_min = chosen["inactive_duration_min"]
#             total_dur_min = (session_end_time - rec["charge_start_time"]).total_seconds() / 60.0
#             move_dur_min = max(0.0, total_dur_min - charge_dur_min - inactive_dur_min)

#             rec.update({
#                 "inactive_session_id": chosen["inactive_session_id"],
#                 "inactive_start_idx": chosen["inactive_start_idx"],
#                 "inactive_end_idx": chosen["inactive_end_idx"],
#                 "inactive_start_time": chosen["inactive_start_time"],
#                 "inactive_end_time": chosen["inactive_end_time"],
#                 "session_end_time": session_end_time,
#                 "charge_duration_min": charge_dur_min,
#                 "movement_duration_min": move_dur_min,
#                 "inactive_duration_min": inactive_dur_min,
#                 "total_session_duration_min": total_dur_min,
#                 "has_movement": True,  # is_no_move_after_charge を満たしていないことを条件に採用
#                 "is_long_inactive": True,
#                 "is_excluded_no_movement": False,
#                 "is_no_move_after_charge": False,  # 採用条件により False 固定
#                 "is_valid_session": True,
#             })
#             results.append(rec)

#     return pd.DataFrame(results)

