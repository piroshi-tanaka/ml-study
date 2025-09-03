# -*- coding: utf-8 -*-
"""
ハイブリッド放置抽出 前処理モジュール
-----------------------------------
本モジュールは、以下の処理を実装します。

1) 充電区間の癒着（is_charging の短い欠損/揺れを連結）
2) IG-ONの更新イベント（tsu_igon_time の更新行）のデバウンス
3) 各 IG-ON アンカー周りでの 双方向拡張（BWD/FWD）
   - A: |ΔSOC| ≤ SOC_TH（SOC安定）
   - B: 距離 ≤ DIST_TH（移動なし）
   ※ 充電中は常に除外（放置対象外）
4) 各 charge_end について、充電後に最初に到来する
   「IG-OFF部分が PARK_TH（6h）以上」の非活動塊（放置）を採用
5) parking_end は IG-ON更新時刻に固定
   parking_start は idling_end と同一点（IG-OFF瞬間は未観測のため近似）
6) 起動後アイドリング（idling_future_*）列は文脈として保持（放置には含めない）
7) 合計の非活動塊（idling_before + parking + idling_after）を total_idle_block_* として出力

※ 中〜大規模データでは車両ごとに処理し、チャンク保存するのが実務的です。
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Tuple


# ======================
# パラメータ定義
# ======================
@dataclass
class Params:
    DIST_TH_m: float = 150.0         # 距離しきい値（移動なし判定の上限[m]）
    SOC_TH_pct: float = 5.0          # SOC変化しきい値（%）
    PARK_TH_min: float = 360.0       # 放置時間しきい値（分）→ 6h
    GAP_MAX_min: float = 15.0        # 充電ブロック間の許容ギャップ（分）
    IGON_DEBOUNCE_min: float = 5.0   # IG-ONイベントのデバウンス（分）
    SMOOTH_WINDOW: int = 5           # 平滑化（rolling median）の窓長（奇数推奨）
    EARTH_R_m: float = 6_371_000.0   # 地球半径（m）


# ======================
# ユーティリティ
# ======================
def haversine_m(lat1, lon1, lat2, lon2, R=6_371_000.0):
    """ベクトル化ハバースイン距離（メートル）。NaN安全ではないので呼び出し側でfillしてください。"""
    lat1, lon1, lat2, lon2 = map(np.deg2rad, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


# ======================
# 前処理（型整備・距離）
# ======================
def preprocess_base(df: pd.DataFrame, p: Params) -> pd.DataFrame:
    """
    最小想定カラム:
      hashvin, tsu_current_time, tsu_igon_time, tsu_latitude, tsu_longitude,
      soc (0-100), is_charging(bool) または charge_mode
    """
    d = df.copy()
    d['hashvin'] = d['hashvin'].astype('string')

    # SOCを数値型に変換、異常値（負の値）はNaNに変換
    d['soc'] = pd.to_numeric(d['soc'], errors='coerce')
    d.loc[d['soc'] < 0, 'soc'] = np.nan

    # is_charging が無い場合は charge_mode から代替生成
    d['is_charging'] = d['charge_mode'].isin(['100V charging', '200V charging', 'Fast charging'])
    d['is_charging'] = d['is_charging'].fillna(False).astype(bool)

    # hashvinごとにtsu_igon_timeが前のデータと比べて更新されていたら、is_igon_time_changeをTrueにする
    d['is_igon_time_change'] = False
    d['is_igon_time_change'] = d.groupby('hashvin')['tsu_igon_time'].apply(lambda s: s.ne(s.shift())).fillna(False)

    # 時系列整列
    d = d.sort_values(['hashvin', 'tsu_current_time']).reset_index(drop=True)

    # 直前地点との距離（m）- 異常値処理を追加
    d['lat_prev'] = d.groupby('hashvin')['tsu_latitude'].shift()
    d['lon_prev'] = d.groupby('hashvin')['tsu_longitude'].shift()
    
    # 異常値チェック（緯度・経度がNaNや異常値の場合は距離計算をスキップ）
    valid_coords = (
        d['tsu_latitude'].notna() & d['tsu_longitude'].notna() & 
        d['lat_prev'].notna() & d['lon_prev'].notna() &
        (d['tsu_latitude'].between(-90, 90)) & (d['tsu_longitude'].between(-180, 180)) &
        (d['lat_prev'].between(-90, 90)) & (d['lon_prev'].between(-180, 180))
    )
    
    d['dist_prev_m'] = np.nan
    d.loc[valid_coords, 'dist_prev_m'] = haversine_m(
        d.loc[valid_coords, 'lat_prev'], 
        d.loc[valid_coords, 'lon_prev'], 
        d.loc[valid_coords, 'tsu_latitude'], 
        d.loc[valid_coords, 'tsu_longitude']
    )
    d['dist_prev_m'] = d['dist_prev_m'].fillna(0.0)

    return d


# ======================
# 充電区間の癒着（連結）
# ======================
def stitch_charging(d: pd.DataFrame, p: Params) -> pd.DataFrame:
    """
    is_charging の短い欠損/反転を連結して、堅牢な charge_start / charge_end を復元。
    出力に 'is_charging_stitched', 'charge_block_id', 'charge_start_flag', 'charge_end_flag' を付与。
    """
    out = d.copy()

    # 生の充電ブロックID（Trueブロックのランレングス）
    out['chg_start_flag'] = (out['is_charging'] & ~out.groupby('hashvin')['is_charging'].shift(fill_value=False))
    out['raw_block_id'] = out.groupby('hashvin')['chg_start_flag'].cumsum()
    out.loc[~out['is_charging'], 'raw_block_id'] = np.nan

    rows = []
    for vid, g in out.groupby('hashvin', sort=False):
        g = g.copy()

        # Trueブロックの開始/終了インデックス抽出
        blocks = []
        cur_id = None
        start_idx = None
        for i, (idx, row) in enumerate(g.iterrows()):
            if row['is_charging'] and pd.notna(row['raw_block_id']):
                if cur_id is None:
                    cur_id = int(row['raw_block_id'])
                    start_idx = idx
            else:
                if cur_id is not None:
                    end_idx = g.index[i-1]
                    blocks.append((start_idx, end_idx))
                    cur_id = None
                    start_idx = None
        if cur_id is not None:
            blocks.append((start_idx, g.index[-1]))

        # 充電ブロック間の短ギャップを条件付きで連結
        stitched = []
        if not blocks:
            # その車両に充電が存在しない
            g['is_charging_stitched'] = False
            g['charge_block_id'] = np.nan
            rows.append(g)
            continue

        s = 0
        while s < len(blocks):
            st, en = blocks[s]
            # 次ブロックとのギャップを確認しながら前方へ連結
            while s + 1 < len(blocks):
                st2, en2 = blocks[s+1]
                # ギャップ時間（分）
                t_gap = (out.loc[st2, 'tsu_current_time'] - out.loc[en, 'tsu_current_time']).total_seconds() / 60.0
                if t_gap > p.GAP_MAX_min:
                    break
                # ギャップ中の移動（端点近似）
                dist_gap = haversine_m(
                    out.loc[en, 'tsu_latitude'], out.loc[en, 'tsu_longitude'],
                    out.loc[st2, 'tsu_latitude'], out.loc[st2, 'tsu_longitude']
                )

                # 2条件（時間と移動距離）を満たせば連結
                if (t_gap <= p.GAP_MAX_min) and (dist_gap <= p.DIST_TH_m):
                    en = en2
                    s += 1
                else:
                    break
            stitched.append((st, en))
            s += 1

        # 連結結果をマーキング
        g['is_charging_stitched'] = False
        g['charge_block_id'] = np.nan
        bid = 0
        for st, en in stitched:
            bid += 1
            g.loc[st:en, 'is_charging_stitched'] = True
            g.loc[st:en, 'charge_block_id'] = bid

        rows.append(g)

    out = pd.concat(rows).sort_values(['hashvin', 'tsu_current_time']).reset_index(drop=True)

    # 連結後の charge_start / charge_end フラグ
    # charge_end_flag: is_charging_stitchedがTrue→FalseとなったFalseの時点
    out['charge_start_flag'] = (
        out['is_charging_stitched'] & ~out.groupby('hashvin')['is_charging_stitched'].shift(fill_value=False)
    )
    out['charge_end_flag'] = (
        ~out['is_charging_stitched'] & out.groupby('hashvin')['is_charging_stitched'].shift(fill_value=False)
    )
    return out


# ======================
# IG-ON更新イベントのデバウンス
# ======================
def debounce_igon_events(d: pd.DataFrame, p: Params) -> pd.DataFrame:
    """
    tsu_igon_time が前行と異なる（更新）行をイベントとみなし、
    連続するTrueの場合は初回のみをigon_event=Trueとし、後続はFalseとする。
    出力に 'igon_event' を付与。
    """
    out = d.copy()

    # tsu_igon_time の変化（NaN安全）
    out['igon_event'] = False
    ig = out.groupby('hashvin')['tsu_igon_time'].apply(lambda s: s.ne(s.shift()))
    ig = ig.fillna(False)
    out.loc[ig.index, 'igon_event'] = ig.values

    # 連続するTrueの場合は初回のみを保持
    def _deb(g):
        idxs = g.index[g['igon_event']].tolist()
        keep = set()
        last_igon_time = None
        for idx in idxs:
            t = g.at[idx, 'tsu_igon_time']
            if pd.isna(t):
                continue
            if (last_igon_time is None) or (t != last_igon_time):
                keep.add(idx)
                last_igon_time = t
        g['igon_event'] = g.index.isin(keep)
        return g

    out = out.groupby('hashvin', group_keys=False).apply(_deb)
    return out


# ======================
# アンカー周りの拡張（BWD/FWD）
# ======================
def _meets_idle_condition(anchor_row, row, p: Params) -> bool:
    """アンカー行に対し、移動距離のみで判定し、かつ充電中でないかを判定。"""
    if row['is_charging_stitched']:
        return False

    # 距離（B条件のみ）
    dist = haversine_m(anchor_row['tsu_latitude'], anchor_row['tsu_longitude'], 
                       row['tsu_latitude'], row['tsu_longitude'])
    return (dist <= p.DIST_TH_m) if pd.notna(dist) else False


def expand_bwd(g: pd.DataFrame, anchor_idx, p: Params) -> Tuple[int, int]:
    """
    起動前アイドリング（過去方向）の拡張。
    戻り値: (idling_start_idx, idling_end_idx)
    ※ idling_end_idx は anchor_idx-1（アンカー直前の行）とする。
    """
    if anchor_idx <= g.index[0]:
        return (anchor_idx, anchor_idx)  # 先頭なら退避

    end_idx = g.index[g.index.get_loc(anchor_idx) - 1]
    start_idx = end_idx
    anchor_row = g.loc[end_idx]  # 直前行をアンカー参照とする（IG-OFF直前の最後の観測点）

    i_pos = g.index.get_loc(end_idx)
    while i_pos > 0:
        prev_idx = g.index[i_pos - 1]
        if _meets_idle_condition(anchor_row, g.loc[prev_idx], p):
            start_idx = prev_idx
            i_pos -= 1
        else:
            break
    return (start_idx, end_idx)


def expand_fwd(g: pd.DataFrame, anchor_idx, p: Params) -> Tuple[int, int]:
    """
    起動後アイドリング（未来方向）の拡張。
    戻り値: (idling_future_start_idx, idling_future_end_idx)
    ※ 放置には含めず、文脈として保持するだけ。
    """
    start_idx = anchor_idx
    end_idx = anchor_idx
    anchor_row = g.loc[anchor_idx]

    i_pos = g.index.get_loc(anchor_idx)
    while i_pos + 1 < len(g.index):
        nxt_idx = g.index[i_pos + 1]
        if _meets_idle_condition(anchor_row, g.loc[nxt_idx], p):
            end_idx = nxt_idx
            i_pos += 1
        else:
            break
    return (start_idx, end_idx)


# ======================
# セッション生成（メイン入口）
# ======================
def build_sessions(df: pd.DataFrame, params: Optional[Params] = None) -> pd.DataFrame:
    """
    入力（最小カラム）:
      hashvin, tsu_current_time, tsu_igon_time, tsu_latitude, tsu_longitude,
      soc (0-100), is_charging(bool) または charge_mode

    出力:
      1レコード = 1つの (充電終了 → 最初の放置(IG-OFF部分が6h以上)) セッション
      仕様に基づく 6イベント + 未来側 + トータル非活動塊 の列を含む DataFrame
    """
    p = params or Params()

    # (1) 前処理
    d0 = preprocess_base(df, p)

    # (2) 充電癒着
    d1 = stitch_charging(d0, p)

    # (3) IG-ONイベントのデバウンス
    d2 = debounce_igon_events(d1, p)

    sessions = []

    # 車両ごとに処理（計算量・メモリ効率のため）
    for vid, g in d2.groupby('hashvin', sort=False):
        g = g.reset_index(drop=False)  # 元の行番号を保持

        # 連結後の charge_end の行インデックス
        chg_end_idxs = g.index[g['charge_end_flag']].tolist()
        if not chg_end_idxs:
            continue

        # IG-ON更新イベントの行インデックス
        igon_idxs = g.index[g['igon_event']].tolist()
        if not igon_idxs:
            continue

        # 各 charge_end ごとに、初回の「IG-OFF部分 >= PARK_TH」を採用
        for ce_idx in chg_end_idxs:
            ce_ts = g.at[ce_idx, 'tsu_current_time']

            # 充電終了後に発生するアンカー候補
            anchors = [ai for ai in igon_idxs if g.at[ai, 'tsu_igon_time'] > ce_ts]
            if not anchors:
                continue

            chosen = False
            for ak in anchors:
                # --- 起動前アイドリング（過去方向） ---
                if ak - 1 < 0:
                    continue
                idle_start_idx, idle_end_idx = expand_bwd(g, ak, p)

                # parking_start は idling_end と同一点（IG-OFF瞬間は未観測のため近似）
                ps_idx = idle_end_idx
                # parking_end は IG-ON更新時刻で固定
                pe_idx = ak

                # IG-OFF部分の継続時間（分）
                dur_min = (g.at[pe_idx, 'tsu_igon_time'] - g.at[ps_idx, 'tsu_current_time']).total_seconds() / 60.0
                if dur_min >= p.PARK_TH_min:
                    # --- 起動後アイドリング（未来方向・保持のみ） ---
                    idf_st_idx, idf_en_idx = expand_fwd(g, ak, p)

                    # この charge ブロックの開始 index（癒着後のblock_idから先頭を逆引き）
                    cbid = g.at[ce_idx, 'charge_block_id']
                    if pd.isna(cbid):
                        break
                    block_mask = (g['charge_block_id'] == cbid)
                    cs_idx = int(g.index[block_mask][0])

                    # セッション行を構築
                    row = {
                        'hashvin': vid,
                        # 充電開始/終了
                        'charge_start_ts': g.at[cs_idx, 'tsu_current_time'], 'charge_start_soc': g.at[cs_idx, 'soc'],
                        'charge_start_lat': g.at[cs_idx, 'tsu_latitude'], 'charge_start_lon': g.at[cs_idx, 'tsu_longitude'],
                        'charge_end_ts': g.at[ce_idx, 'tsu_current_time'], 'charge_end_soc': g.at[ce_idx, 'soc'],
                        'charge_end_lat': g.at[ce_idx, 'tsu_latitude'], 'charge_end_lon': g.at[ce_idx, 'tsu_longitude'],
                        # 起動前アイドリング（BWD）
                        'idling_start_ts': g.at[idle_start_idx, 'tsu_current_time'], 'idling_start_soc': g.at[idle_start_idx, 'soc'],
                        'idling_start_lat': g.at[idle_start_idx, 'tsu_latitude'], 'idling_start_lon': g.at[idle_start_idx, 'tsu_longitude'],
                        'idling_end_ts': g.at[idle_end_idx, 'tsu_current_time'], 'idling_end_soc': g.at[idle_end_idx, 'soc'],
                        'idling_end_lat': g.at[idle_end_idx, 'tsu_latitude'], 'idling_end_lon': g.at[idle_end_idx, 'tsu_longitude'],
                        # 放置（IG-OFF部分）
                        'parking_start_ts': g.at[ps_idx, 'tsu_current_time'], 'parking_start_soc': g.at[ps_idx, 'soc'],
                        'parking_start_lat': g.at[ps_idx, 'tsu_latitude'], 'parking_start_lon': g.at[ps_idx, 'tsu_longitude'],
                        'parking_end_ts': g.at[pe_idx, 'tsu_igon_time'], 'parking_end_soc': g.at[pe_idx, 'soc'],
                        'parking_end_lat': g.at[pe_idx, 'tsu_latitude'], 'parking_end_lon': g.at[pe_idx, 'tsu_longitude'],
                    }

                    # 継続時間（時間）
                    row['idling_duration_h'] = (row['idling_end_ts'] - row['idling_start_ts']).total_seconds() / 3600.0
                    row['parking_duration_h'] = (row['parking_end_ts'] - row['parking_start_ts']).total_seconds() / 3600.0

                    # 起動後アイドリング（保持のみ）
                    row['idling_future_start_ts'] = g.at[idf_st_idx, 'tsu_current_time']
                    row['idling_future_end_ts']   = g.at[idf_en_idx, 'tsu_current_time']
                    row['idling_future_duration_h'] = (row['idling_future_end_ts'] - row['idling_future_start_ts']).total_seconds() / 3600.0
                    row['idling_future_start_lat'] = g.at[idf_st_idx, 'tsu_latitude']
                    row['idling_future_start_lon'] = g.at[idf_st_idx, 'tsu_longitude']
                    row['idling_future_end_lat']   = g.at[idf_en_idx, 'tsu_latitude']
                    row['idling_future_end_lon']   = g.at[idf_en_idx, 'tsu_longitude']
                    row['idling_future_start_soc'] = g.at[idf_st_idx, 'soc']
                    row['idling_future_end_soc']   = g.at[idf_en_idx, 'soc']

                    # トータル非活動塊（レポート用）
                    row['total_idle_block_start_ts'] = row['idling_start_ts']
                    row['total_idle_block_end_ts'] = max(row['idling_future_end_ts'], row['parking_end_ts'])
                    row['total_idle_block_duration_h'] = (row['total_idle_block_end_ts'] - row['total_idle_block_start_ts']).total_seconds() / 3600.0

                    sessions.append(row)
                    chosen = True
                    break  # この charge_end では最初の >=6h を採用
            # 未採用なら次の charge_end へ

    if not sessions:
        return pd.DataFrame()

    out = pd.DataFrame(sessions).sort_values(['hashvin', 'charge_end_ts']).reset_index(drop=True)
    return out
