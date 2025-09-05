# -*- coding: utf-8 -*-
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
                                    
    # 充電後移動判定
    WINDOW_min: float = 60.0        # 充電後移動チェック時間窓（分）
                                    # 充電終了後この時間内での移動有無をチェック


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


def calculate_distance_prev(df: pd.DataFrame) -> pd.Series:
    """
    各観測点について、前の観測点からの移動距離を計算
    
    【目的】
    時系列データで車両がどれだけ移動したかを把握し、
    移動セッションと停止セッションを区別するため。
    
    【処理ロジック】
    1. 最初の行は距離0（比較対象なし）
    2. 同一車両（hashvin）内でのみ距離計算
    3. 車両が変わった場合は距離0（別車両との比較は無意味）
    4. ハバーサイン公式で正確な地球上距離を計算
    
    Args:
        df: 時系列順にソートされたDataFrame
            必須カラム: hashvin, tsu_rawgnss_latitude, tsu_rawgnss_longitude
            
    Returns:
        pd.Series: 各行の前観測点からの移動距離（メートル）
                   最初の行や車両変更時は0.0
    """
    distances = []

    for i in range(len(df)):
        # 最初の行は比較対象がないため距離0
        if i == 0:
            distances.append(0.0)
            continue

        # 現在行と前行のデータを取得
        row = df.iloc[i]
        prev_row = df.iloc[i - 1]

        # 同じ車両（hashvin）の場合のみ距離計算を実行
        # 異なる車両間の距離は意味がないため0とする
        if row["hashvin"] == prev_row["hashvin"]:
            # geodesic距離で前の位置からの移動距離を計算
            dist = calculate_geodesic_distance(
                prev_row["tsu_rawgnss_latitude"],
                prev_row["tsu_rawgnss_longitude"],
                row["tsu_rawgnss_latitude"],
                row["tsu_rawgnss_longitude"],
            )
            distances.append(dist)
        else:
            # 車両が変わった場合は距離0
            distances.append(0.0)

    # 元のDataFrameのインデックスを保持してSeriesとして返す
    return pd.Series(distances, index=df.index)


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
    
    # === 重要: SOC条件による充電判定の補正 ===
    # 【課題】charge_modeがFalseでも実際は充電継続している場合がある
    # 【解決】前行がcharge_mode=Trueで現行がFalseの場合、SOC変化で判定
    for _, g in d.groupby("hashvin", sort=False):
        g = g.copy()
        for i in range(1, len(g)):
            current_idx = g.index[i]
            prev_idx = g.index[i-1]
            
            # 条件: 前行が充電中、現行が充電モードoff
            if (d.loc[prev_idx, "is_charge_mode"] and 
                not d.loc[current_idx, "is_charge_mode"]):
                
                prev_soc = d.loc[prev_idx, "soc"]
                current_soc = d.loc[current_idx, "soc"]
                
                # SOCが維持or増加している場合は充電継続と判定
                # 理由: 充電中はSOCが減少することはないため
                if (pd.notna(prev_soc) and pd.notna(current_soc) and 
                    current_soc >= prev_soc):
                    d.loc[current_idx, "is_charging"] = True

    # === Step2: IGONタイム変更イベントの検出 ===
    # 【目的】車両のIG-ON状態変化を捉える（ユーザーの行動変化を示す）
    # 【処理】hashvinごとにtsu_igon_timeの値が前行から変化した場合を検出
    d["igon_change"] = (
        d.groupby("hashvin")["tsu_igon_time"]
        .transform(lambda s: s.ne(s.shift()))  # 前行と値が異なる場合True
        .fillna(False)  # 最初の行はFalse（比較対象なし）
    )

    # === Step3: 移動距離計算 ===
    # 前観測点からの移動距離を計算（geodesic距離使用）
    d["distance_prev_m"] = calculate_distance_prev(d)
    
    # 移動判定: 設定閾値以上の距離移動があった場合
    # 【理由】GPS誤差やわずかな位置ずれを移動と誤認しないため
    d["is_moving_distance"] = d["distance_prev_m"] > params.DIST_TH_m

    return d


# ======================
# セッションタイプ統合処理
# ======================
def determine_session_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    セッションフラグから統合session_type列を作成
    
    【目的】
    個別のセッションフラグ（is_charging, is_moving等）を統合し、
    境界情報も保持しながら、分析しやすい単一の列を提供する。
    
    【設計方針】
    1. 主要セッション優先: 境界があっても主セッションを明確化
    2. 境界情報分離: 複数イベント同時発生時の詳細情報を保持
    3. 優先順位: CHARGING > MOVING > IDLING > PARKING
    
    【出力】
    - session_type: 主要セッションタイプ
    - is_inactive: 放置状態統合フラグ
    - inactive関連フラグ: 放置セッションの詳細情報
    """
    d = df.copy()


    # === 主要セッションタイプの決定 ===
    # 【方針】境界イベントがあっても、その行の主要な状態を明確にする
    # 【優先順位】CHARGING > MOVING > IDLING > PARKING
    #             （バッテリー劣化分析では充電状態が最重要）
    def get_primary_session_type(row):
        # 1. 充電優先（最重要）
        # 充電中はバッテリー状態が変化するため、他の状態より重要
        if row.get("is_charging_stitched", False):
            return SessionType.CHARGING.value

        # 2. 移動中
        # 実際に車両が移動している状態（エネルギー消費）
        if row.get("is_moving", False):
            return SessionType.MOVING.value

        # 3. アイドリング（放置の一種）
        # IG-ONだが移動していない状態（エネルギー消費あり）
        if row.get("is_idling", False):
            return SessionType.IDLING.value

        # 4. パーキング（放置の一種）
        # IG-OFFで駐車している状態（エネルギー消費最小）
        if row.get("is_parking", False):
            return SessionType.PARKING.value

        # デフォルト: 該当するセッションなし（データ不備等）
        return None

    # 主要セッションタイプを全行に適用
    d["session_type"] = d.apply(get_primary_session_type, axis=1)

    # === 放置状態（inactive）の統合処理 ===
    # 【目的】idling + parking = inactive として、放置状態を統一的に扱う
    # 【理由】バッテリー劣化分析では、IG-ON/OFFの違いより放置時間が重要
    d["is_inactive"] = d["is_idling"] | d["is_parking"]
    
    # inactive開始フラグ: 放置状態になった瞬間を検出
    # 【ロジック】現在が放置中 かつ 前行が放置でない場合
    d["inactive_start_flag"] = (
        d.groupby("hashvin")["is_inactive"]
        .transform(lambda s: s & ~s.shift(1, fill_value=False))
        .fillna(False)
    )
    
    # inactive終了フラグ: 放置状態から抜けた瞬間を検出
    # 【ロジック】現在が放置でない かつ 前行が放置中の場合
    d["inactive_end_flag"] = (
        d.groupby("hashvin")["is_inactive"]
        .transform(lambda s: ~s & s.shift(1, fill_value=False))
        .fillna(False)
    )
    
    # inactiveセッションID: 各放置セッションに一意のIDを付与
    # 【用途】放置期間の集計や分析時のグループ化に使用
    d["inactive_session_id"] = d.groupby("hashvin")["inactive_start_flag"].cumsum()

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

        for charge_start_idx in charge_starts:
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
def determine_sessions(df: pd.DataFrame, params: SessionParams) -> pd.DataFrame:
    """Step4-6: 各セッションの判定（優先度: parking→moving→idling）"""
    # paramsは将来の拡張用に保持
    _ = params  # 未使用警告を回避
    d = df.copy()

    # 初期化：すべてのセッションフラグをFalseに設定
    d["is_parking"] = False
    d["is_moving"] = False  
    d["is_idling"] = False

    # Step4: パーキングセッション判定（最優先）
    # IG-OFF状態かつ充電中でない場合にパーキング
    d.loc[
        (d["tsu_igon_time"].isna() | (d["tsu_igon_time"] == "")) & 
        ~d["is_charging_stitched"], 
        "is_parking"
    ] = True

    d["parking_start_flag"] = (
        d.groupby("hashvin")["is_parking"]
        .transform(lambda s: s & ~s.shift(1, fill_value=False))
        .fillna(False)
    )

    d["parking_end_flag"] = (
        d.groupby("hashvin")["is_parking"]
        .transform(lambda s: ~s & s.shift(1, fill_value=False))
        .fillna(False)
    )

    d["parking_session_id"] = d.groupby("hashvin")["parking_start_flag"].cumsum()

    # Step5: 移動中セッション判定（パーキング次点）
    # IG-ON、充電中でない、パーキング中でない、かつ移動距離がある場合
    d.loc[
        (d["tsu_igon_time"].notna() & (d["tsu_igon_time"] != "")) &
        ~d["is_charging_stitched"] & 
        ~d["is_parking"] & 
        d["is_moving_distance"], 
        "is_moving"
    ] = True

    d["movement_start_flag"] = (
        d.groupby("hashvin")["is_moving"]
        .transform(lambda s: s & ~s.shift(1, fill_value=False))
        .fillna(False)
    )

    d["movement_end_flag"] = (
        d.groupby("hashvin")["is_moving"]
        .transform(lambda s: ~s & s.shift(1, fill_value=False))
        .fillna(False)
    )

    d["movement_session_id"] = d.groupby("hashvin")["movement_start_flag"].cumsum()

    # Step6: アイドリングセッション判定（最低優先度）
    # IG-ON、充電中でない、パーキング中でない、移動中でない場合
    d.loc[
        (d["tsu_igon_time"].notna() & (d["tsu_igon_time"] != "")) &
        ~d["is_charging_stitched"] & 
        ~d["is_parking"] & 
        ~d["is_moving"], 
        "is_idling"
    ] = True

    d["idling_start_flag"] = (
        d.groupby("hashvin")["is_idling"]
        .transform(lambda s: s & ~s.shift(1, fill_value=False))
        .fillna(False)
    )

    d["idling_end_flag"] = (
        d.groupby("hashvin")["is_idling"]
        .transform(lambda s: ~s & s.shift(1, fill_value=False))
        .fillna(False)
    )

    d["idling_session_id"] = d.groupby("hashvin")["idling_start_flag"].cumsum()

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
# メイン処理関数
# ======================
def detect_sessions(
    df: pd.DataFrame, params: Optional[SessionParams] = None
) -> pd.DataFrame:
    """
    5つのセッションを判定するメイン関数

    入力:
        df: 時系列データ (hashvin, tsu_current_time, tsu_igon_time,
                        tsu_rawgnss_latitude, tsu_rawgnss_longitude, soc, charge_mode)
        params: セッション判定パラメータ

    出力:
        セッションフラグが付与されたDataFrame
    """
    p = params or SessionParams()

    # Step1-3: 前処理
    print("Step1-3: 基礎データ準備中...")
    d1 = preprocess_data(df, p)

    # 充電セッション処理
    print("充電セッション処理中...")
    d2 = stitch_charging_sessions(d1, p)

    # Step4-6: セッション判定
    print("Step4-6: セッション判定中...")
    d3 = determine_sessions(d2, p)

    # Step7: 充電後移動なし判定
    print("Step7: 充電後移動なし判定中...")
    d4 = check_no_movement_after_charge(d3, p)

    # 放置時間分類
    print("放置時間分類中...")
    d5 = classify_parking_duration(d4, p)

    # セッションタイプ統合
    print("セッションタイプ統合中...")
    d6 = determine_session_type(d5)

    print("セッション判定完了!")
    return d6


# ======================
# 品質チェック関数
# ======================
def validate_sessions(df: pd.DataFrame) -> Dict[str, Any]:
    """セッション判定結果の品質チェック"""
    results: Dict[str, Any] = {
        "total_rows": len(df),
        "session_coverage": {},
        "session_overlaps": {},
        "charge_coverage": 0.0,
        "errors": [],
    }

    # セッションカバレッジチェック
    session_flags = ["is_charging_stitched", "is_moving", "is_idling", "is_parking", "is_inactive"]
    for flag in session_flags:
        if flag in df.columns:
            results["session_coverage"][flag] = int(df[flag].sum())
        else:
            results["errors"].append(f"列が見つかりません: {flag}")

    # セッション重複チェック
    for i, flag1 in enumerate(session_flags):
        if flag1 not in df.columns:
            continue
        for flag2 in session_flags[i + 1 :]:
            if flag2 not in df.columns:
                continue
            overlap = int((df[flag1] & df[flag2]).sum())
            if overlap > 0:
                results["session_overlaps"][f"{flag1}_vs_{flag2}"] = overlap
                results["errors"].append(
                    f"セッション重複: {flag1} vs {flag2} = {overlap}行"
                )

    # 充電イベント捕捉率
    if "charge_mode" in df.columns and "is_charging_stitched" in df.columns:
        charge_mode_count = int(
            df["charge_mode"]
            .isin(["100V charging", "200V charging", "Fast charging"])
            .sum()
        )
        charge_stitched_count = int(df["is_charging_stitched"].sum())
        if charge_mode_count > 0:
            results["charge_coverage"] = float(charge_stitched_count) / float(
                charge_mode_count
            )

    # 全時点カバレッジチェック
    available_flags = [f for f in session_flags if f in df.columns]
    if available_flags:
        total_covered = int(df[available_flags].any(axis=1).sum())
        results["total_coverage"] = float(total_covered) / float(len(df))

        if results["total_coverage"] < 0.95:
            results["errors"].append(
                f"総カバレッジ不足: {results['total_coverage']:.2%}"
            )

    # session_type列のチェック
    if "session_type" in df.columns:
        session_type_counts = df["session_type"].value_counts()
        results["session_type_distribution"] = session_type_counts.to_dict()

        # 未分類（None）の行があるかチェック
        none_count = df["session_type"].isna().sum()
        if none_count > 0:
            results["errors"].append(f"未分類のセッション: {none_count}行")

    return results


# ======================
# セッション詳細データ抽出機能
# ======================
def extract_session_details(df: pd.DataFrame) -> pd.DataFrame:
    """
    各セッションの開始・終了時点での詳細データを抽出
    
    Args:
        df: セッション判定済みのDataFrame
        
    Returns:
        セッション詳細データのDataFrame
        - session_type: セッションタイプ
        - session_id: セッションID
        - event_type: 'start' または 'end'
        - timestamp: 開始・終了時刻
        - soc: SOC値
        - distance_from_prev: 前の時点からの移動距離
        - hashvin: 車両ID
        - latitude, longitude: 位置情報
        - duration_min: セッション時間（終了時点のみ）
        - soc_diff: SOC差分（終了時点のみ）
        - total_distance: セッション中の総移動距離（終了時点のみ）
    """
    result_data = []
    
    # セッションタイプとフラグの対応
    session_configs = [
        {
            'type': 'charging',
            'start_flag': 'charge_start_flag',
            'end_flag': 'charge_end_flag',
            'session_id': 'charge_session_id',
            'active_flag': 'is_charging_stitched'
        },
        {
            'type': 'moving',
            'start_flag': 'movement_start_flag',
            'end_flag': 'movement_end_flag',
            'session_id': 'movement_session_id',
            'active_flag': 'is_moving'
        },
        {
            'type': 'idling',
            'start_flag': 'idling_start_flag',
            'end_flag': 'idling_end_flag',
            'session_id': 'idling_session_id',
            'active_flag': 'is_idling'
        },
        {
            'type': 'parking',
            'start_flag': 'parking_start_flag',
            'end_flag': 'parking_end_flag',
            'session_id': 'parking_session_id',
            'active_flag': 'is_parking'
        }
    ]
    
    for hashvin, vehicle_data in df.groupby("hashvin"):
        vehicle_data = vehicle_data.copy().sort_values("tsu_current_time")
        
        for config in session_configs:
            session_type = config['type']
            start_flag = config['start_flag']
            end_flag = config['end_flag']
            session_id_col = config['session_id']
            active_flag = config['active_flag']
            
            # 必要なカラムが存在するかチェック
            if not all(col in vehicle_data.columns for col in [start_flag, end_flag, session_id_col]):
                continue
                
            # 開始イベントの処理
            start_events = vehicle_data[vehicle_data[start_flag]].copy()
            for _, row in start_events.iterrows():
                result_data.append({
                    'hashvin': hashvin,
                    'session_type': session_type,
                    'session_id': row[session_id_col],
                    'event_type': 'start',
                    'timestamp': row['tsu_current_time'],
                    'soc': row['soc'],
                    'distance_from_prev': row.get('distance_prev_m', 0),
                    'latitude': row.get('tsu_rawgnss_latitude'),
                    'longitude': row.get('tsu_rawgnss_longitude'),
                    'duration_min': None,
                    'soc_diff': None,
                    'total_distance': None
                })
            
            # 終了イベントの処理
            end_events = vehicle_data[vehicle_data[end_flag]].copy()
            for _, row in end_events.iterrows():
                # 対応する開始イベントを探す
                session_id = row[session_id_col]
                session_data = vehicle_data[
                    (vehicle_data[session_id_col] == session_id) & 
                    (vehicle_data[active_flag])
                ]
                
                # セッション期間の計算
                if len(session_data) > 0:
                    start_time = session_data['tsu_current_time'].min()
                    end_time = row['tsu_current_time']
                    duration_min = (end_time - start_time).total_seconds() / 60
                    
                    # SOC差分計算
                    start_soc = session_data['soc'].iloc[0]
                    end_soc = row['soc']
                    soc_diff = end_soc - start_soc
                    
                    # 総移動距離計算
                    total_distance = session_data['distance_prev_m'].sum()
                else:
                    duration_min = 0
                    soc_diff = 0
                    total_distance = 0
                
                result_data.append({
                    'hashvin': hashvin,
                    'session_type': session_type,
                    'session_id': row[session_id_col],
                    'event_type': 'end',
                    'timestamp': row['tsu_current_time'],
                    'soc': row['soc'],
                    'distance_from_prev': row.get('distance_prev_m', 0),
                    'latitude': row.get('tsu_rawgnss_latitude'),
                    'longitude': row.get('tsu_rawgnss_longitude'),
                    'duration_min': duration_min,
                    'soc_diff': soc_diff,
                    'total_distance': total_distance
                })
    
    # 放置セッション（idling + parking統合）の処理
    for hashvin, vehicle_data in df.groupby("hashvin"):
        vehicle_data = vehicle_data.copy().sort_values("tsu_current_time")
        
        if 'inactive_start_flag' in vehicle_data.columns and 'inactive_end_flag' in vehicle_data.columns:
            # 放置開始イベント
            inactive_starts = vehicle_data[vehicle_data['inactive_start_flag']].copy()
            for _, row in inactive_starts.iterrows():
                # 放置の内訳を判定
                breakdown = []
                if row.get('is_idling', False):
                    breakdown.append('idle')
                if row.get('is_parking', False):
                    breakdown.append('parking')
                
                result_data.append({
                    'hashvin': hashvin,
                    'session_type': 'inactive',
                    'session_id': row.get('inactive_session_id', 0),
                    'event_type': 'start',
                    'timestamp': row['tsu_current_time'],
                    'soc': row['soc'],
                    'distance_from_prev': row.get('distance_prev_m', 0),
                    'latitude': row.get('tsu_rawgnss_latitude'),
                    'longitude': row.get('tsu_rawgnss_longitude'),
                    'duration_min': None,
                    'soc_diff': None,
                    'total_distance': None,
                    'inactive_breakdown': '|'.join(breakdown) if breakdown else 'unknown'
                })
            
            # 放置終了イベント
            inactive_ends = vehicle_data[vehicle_data['inactive_end_flag']].copy()
            for _, row in inactive_ends.iterrows():
                session_id = row.get('inactive_session_id', 0)
                
                # セッション期間データを取得
                inactive_data = vehicle_data[
                    (vehicle_data.get('inactive_session_id', 0) == session_id) & 
                    (vehicle_data.get('is_inactive', False))
                ]
                
                # 放置の内訳を判定
                breakdown = []
                if len(inactive_data) > 0:
                    if inactive_data['is_idling'].any():
                        breakdown.append('idle')
                    if inactive_data['is_parking'].any():
                        breakdown.append('parking')
                
                # セッション期間の計算
                if len(inactive_data) > 0:
                    start_time = inactive_data['tsu_current_time'].min()
                    end_time = row['tsu_current_time']
                    duration_min = (end_time - start_time).total_seconds() / 60
                    
                    start_soc = inactive_data['soc'].iloc[0]
                    end_soc = row['soc']
                    soc_diff = end_soc - start_soc
                    
                    total_distance = inactive_data['distance_prev_m'].sum()
                else:
                    duration_min = 0
                    soc_diff = 0
                    total_distance = 0
                
                result_data.append({
                    'hashvin': hashvin,
                    'session_type': 'inactive',
                    'session_id': session_id,
                    'event_type': 'end',
                    'timestamp': row['tsu_current_time'],
                    'soc': row['soc'],
                    'distance_from_prev': row.get('distance_prev_m', 0),
                    'latitude': row.get('tsu_rawgnss_latitude'),
                    'longitude': row.get('tsu_rawgnss_longitude'),
                    'duration_min': duration_min,
                    'soc_diff': soc_diff,
                    'total_distance': total_distance,
                    'inactive_breakdown': '|'.join(breakdown) if breakdown else 'unknown'
                })
    
    result_df = pd.DataFrame(result_data)
    if len(result_df) > 0:
        result_df = result_df.sort_values(['hashvin', 'timestamp']).reset_index(drop=True)
    
    return result_df


def get_session_summary_by_type(df: pd.DataFrame, session_type: Optional[str] = None) -> pd.DataFrame:
    """
    特定のセッションタイプまたは全セッションのサマリーを取得
    
    Args:
        df: セッション判定済みのDataFrame
        session_type: 'charging', 'moving', 'idling', 'parking', 'inactive' または None（全て）
        
    Returns:
        セッションサマリーのDataFrame
    """
    session_details = extract_session_details(df)
    
    if session_type:
        session_details = session_details[session_details['session_type'] == session_type]
    
    if len(session_details) == 0:
        return pd.DataFrame()
    
    # セッション単位でのサマリー作成
    summary_data = []
    
    for (hashvin, s_type, s_id), group in session_details.groupby(['hashvin', 'session_type', 'session_id']):
        start_row = group[group['event_type'] == 'start']
        end_row = group[group['event_type'] == 'end']
        
        if len(start_row) > 0 and len(end_row) > 0:
            start_data = start_row.iloc[0]
            end_data = end_row.iloc[0]
            
            summary_data.append({
                'hashvin': hashvin,
                'session_type': s_type,
                'session_id': s_id,
                'start_timestamp': start_data['timestamp'],
                'end_timestamp': end_data['timestamp'],
                'start_soc': start_data['soc'],
                'end_soc': end_data['soc'],
                'soc_diff': end_data['soc_diff'],
                'duration_min': end_data['duration_min'],
                'total_distance': end_data['total_distance'],
                'start_lat': start_data['latitude'],
                'start_lon': start_data['longitude'],
                'end_lat': end_data['latitude'],
                'end_lon': end_data['longitude'],
                'inactive_breakdown': end_data.get('inactive_breakdown')
            })
    
    return pd.DataFrame(summary_data)


if __name__ == "__main__":
    print("5セッション判定モジュール")
    print("使用方法:")
    print("from session_detector import detect_sessions, SessionParams")
    print("params = SessionParams()")
    print("result = detect_sessions(df, params)")
