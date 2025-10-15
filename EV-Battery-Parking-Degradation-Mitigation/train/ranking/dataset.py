"""
EV の「充電→（次の充電までの間に発生する）最初の長時間放置(>=6h)」をランキング予測するための
データセット作成・特徴量生成ユーティリティ群です。

本モジュールは、各充電イベントに対し「候補となる放置クラスタ」を列挙し、
（充電, 候補クラスタ）のペアごとに1行の学習データを作ることで、
AutoGluon の二値分類（正例=実際に発生した放置クラスタ）として学習できるように整形します。
推論時は陽性確率をランキングスコアとして用います。

前提として入力はセッション単位の表で、少なくとも以下の列を持つことを想定します:
  - hashvin, session_cluster, session_type (inactive/charging)
  - start_time, end_time, duration_minutes
  - start_lat, start_lon, end_lat, end_lon

主要な処理ステップ:
  1) prepare_sessions:
     派生列（weekday, start_hour など）を付与し、各充電から「次の充電までの区間」に存在する
     最初の長時間放置（inactive かつ duration_minutes >= 閾値）をリンクします。
  2) build_charge_to_next_long_table:
     充電イベントごとのリンク結果（正解ラベルに相当）を表にまとめます。
  3) compute_cluster_centroids_by_vehicle:
     車両×放置クラスタごとの代表座標（start_lat/lon の平均）を計算します。
  4) build_candidate_pool_per_vehicle:
     車両単位で長時間放置クラスタの滞在時間を合計し、長い順に Top-N の候補プールを作成します。
     ここで得た候補は静的な土台であり、最終候補は `pipeline.generate_candidates` が文脈に応じて組み直します。
  5) build_ranking_training_data:
     各充電イベントに対して候補クラスタを展開し、特徴量を組み立て、
     真の放置クラスタには label=1、それ以外は label=0 を付与した学習用データを返します。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import math
import numpy as np
import pandas as pd


# -----------------------------
# I/O helpers
# -----------------------------

def load_sessions(csv_path: Path | str) -> pd.DataFrame:
    """
    セッション CSV を読み込み、時刻列を pandas の datetime に変換した上で、
    Asia/Tokyo タイムゾーンへ正規化します（tz 付き→変換して tz を落とす／naive→Tokyo でローカライズ）。
    """
    df = pd.read_csv(csv_path)
    # start_time / end_time を datetime 化し、Asia/Tokyo に寄せる
    for col in ["start_time", "end_time"]:
        s = pd.to_datetime(df[col], errors="coerce")
        # tz aware の場合は Tokyo へ変換後に tz 情報を落とす
        if pd.api.types.is_datetime64tz_dtype(s.dtype):
            df[col] = s.dt.tz_convert("Asia/Tokyo").dt.tz_localize(None)
        else:
            # naive の場合は Tokyo でローカライズ
            try:
                df[col] = s.dt.tz_localize(
                    "Asia/Tokyo", nonexistent="shift_forward", ambiguous="NaT"
                ).dt.tz_localize(None)
            except Exception:
                # ローカライズに失敗した場合は、そのまま（tz なし）を採用
                df[col] = s
    return df


# -----------------------------
# Core preprocessing
# -----------------------------

def prepare_sessions(df: pd.DataFrame, long_park_threshold_minutes: int = 360) -> pd.DataFrame:
    """
    前処理と充電→長時間放置リンクを行います。
    - 長時間放置判定: inactive かつ duration_minutes >= long_park_threshold_minutes
    - 車両内の時間順に並べ、prev_*/next_* の文脈列を付与
    - 各充電行に対して、次の充電が始まるまでの区間にある「最初の長時間放置」を探索・リンク
      → next_long_park_* 列として記録。
    """
    df = df.copy()
    df = df.sort_values(["hashvin", "start_time"]).reset_index(drop=True)

    df["duration_minutes"] = pd.to_numeric(df["duration_minutes"], errors="coerce")
    df["weekday"] = df["start_time"].dt.dayofweek
    df["start_hour"] = df["start_time"].dt.hour
    df["date"] = df["start_time"].dt.date

    df["is_long_park"] = (df["session_type"] == "inactive") & (
        df["duration_minutes"].astype(float) >= float(long_park_threshold_minutes)
    )

    g = df.groupby("hashvin", group_keys=False)
    df["prev_session_type"] = g["session_type"].shift(1)
    df["prev_cluster"] = g["session_cluster"].shift(1)
    df["prev_is_long_park"] = g["is_long_park"].shift(1)
    df["next_session_type"] = g["session_type"].shift(-1)
    df["next_cluster"] = g["session_cluster"].shift(-1)
    df["next_is_long_park"] = g["is_long_park"].shift(-1)

    # 充電から見た「最初の長時間放置」リンクの書き込み先を初期化
    df["next_long_park_cluster"] = np.nan
    df["next_long_park_start_time"] = pd.NaT
    df["next_long_park_end_time"] = pd.NaT

    # 車両ごとに時系列を走査し、各充電から次の充電までに発生する「最初の長時間放置」を紐付け
    for hv, sub in df.groupby("hashvin", group_keys=False):
        idxs = sub.index.to_list()
        for i in idxs:
            if df.at[i, "session_type"] != "charging":
                continue
            # 充電行の直後から先へ走査
            for j in (k for k in idxs if k > i):
                stype = df.at[j, "session_type"]
                if stype == "charging":
                    # 次の充電が来たら探索終了（リンクなし）
                    break
                if bool(df.at[j, "is_long_park"]):
                    df.at[i, "next_long_park_cluster"] = df.at[j, "session_cluster"]
                    df.at[i, "next_long_park_start_time"] = df.at[j, "start_time"]
                    df.at[i, "next_long_park_end_time"] = df.at[j, "end_time"]
                    break

    return df


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    2地点間の球面距離（ハーサイン距離, km）を計算します。
    入力が NaN 等で失敗した場合は NaN を返します。
    """
    try:
        R = 6371.0  # 地球半径（km）
        phi1 = math.radians(float(lat1))
        phi2 = math.radians(float(lat2))
        dphi = math.radians(float(lat2) - float(lat1))
        dl = math.radians(float(lon2) - float(lon1))
        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dl / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return float(R * c)
    except Exception:
        return float("nan")


def build_charge_to_next_long_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    充電イベントごとに「次に来る長時間放置（存在すれば）」を1行にまとめた表を構築します。
    途中で別の充電が挟まる場合はリンクしません（＝該当なし）。

    返す列:
      hashvin, weekday, charge_cluster, charge_start_time, charge_start_hour, charge_end_time,
      park_cluster, park_start_time, park_start_hour, park_duration_minutes, gap_minutes,
      dist_charge_to_park_km
    長時間放置が存在しない場合、park_* 列は NaN となります。
    """
    charges = df[df["session_type"] == "charging"].copy()
    if charges.empty:
        return pd.DataFrame(
            columns=[
                "hashvin",
                "weekday",
                "charge_cluster",
                "charge_start_time",
                "charge_start_hour",
                "charge_end_time",
                "park_cluster",
                "park_start_time",
                "park_start_hour",
                "park_duration_minutes",
                "gap_minutes",
                "dist_charge_to_park_km",
            ]
        )

    out: List[dict] = []
    for _, row in charges.iterrows():
        hv = row.get("hashvin")
        c_cluster = row.get("session_cluster")
        c_start = pd.to_datetime(row.get("start_time"))
        c_end = pd.to_datetime(row.get("end_time"))
        weekday = c_start.dayofweek if pd.notnull(c_start) else np.nan
        c_hour = c_start.hour if pd.notnull(c_start) else np.nan

        p_cluster = row.get("next_long_park_cluster")
        p_start = pd.to_datetime(row.get("next_long_park_start_time"))
        p_end = pd.to_datetime(row.get("next_long_park_end_time"))

        if pd.notnull(p_cluster):
            p_hour = p_start.hour if pd.notnull(p_start) else np.nan
            p_dur = (p_end - p_start).total_seconds() / 60.0 if (pd.notnull(p_end) and pd.notnull(p_start)) else np.nan
            gap_min = (p_start - c_end).total_seconds() / 60.0 if (pd.notnull(p_start) and pd.notnull(c_end)) else np.nan
            # 充電終了地点 → 放置開始地点 までの距離（km）
            c_end_lat = float(row.get("end_lat", np.nan))
            c_end_lon = float(row.get("end_lon", np.nan))
            # 放置セッションの開始座標を元 df から取得
            park_row = df[(df["hashvin"] == hv) & (df["start_time"] == p_start) & (df["end_time"] == p_end)]
            if not park_row.empty:
                p_lat = float(park_row.iloc[0].get("start_lat", np.nan))
                p_lon = float(park_row.iloc[0].get("start_lon", np.nan))
            else:
                p_lat = p_lon = float("nan")
            dist_km = _haversine_km(c_end_lat, c_end_lon, p_lat, p_lon)
        else:
            p_hour = np.nan
            p_dur = np.nan
            gap_min = np.nan
            dist_km = np.nan

        out.append(
            {
                "hashvin": hv,
                "weekday": weekday,
                "charge_cluster": c_cluster,
                "charge_start_time": c_start,
                "charge_start_hour": c_hour,
                "charge_end_time": c_end,
                "park_cluster": p_cluster,
                "park_start_time": p_start,
                "park_start_hour": p_hour,
                "park_duration_minutes": p_dur,
                "gap_minutes": gap_min,
                "dist_charge_to_park_km": dist_km,
            }
        )

    return pd.DataFrame(out)


# -----------------------------
# Candidate and feature engineering
# -----------------------------

def compute_cluster_centroids_by_vehicle(df: pd.DataFrame) -> pd.DataFrame:
    """
    車両×放置クラスタごとに、長時間放置セッションの開始座標の平均（重心）を算出します。
    返り値の列: [hashvin, cluster, lat, lon, count]
    """
    parks = df[(df["session_type"] == "inactive") & (df["is_long_park"] == True)].copy()
    if parks.empty:
        return pd.DataFrame(columns=["hashvin", "cluster", "lat", "lon", "count"])  # noqa: E712

    grouped = parks.groupby(["hashvin", "session_cluster"], as_index=False).agg(
        lat=("start_lat", "mean"),
        lon=("start_lon", "mean"),
        count=("session_cluster", "size"),
    )
    grouped = grouped.rename(columns={"session_cluster": "cluster"})[
        ["hashvin", "cluster", "lat", "lon", "count"]
    ]
    return grouped


def _popularity(series: pd.Series) -> pd.Series:
    """
    候補クラスタの「人気度」を同一系列内の相対頻度として計算します。
    戻り値は元の系列と同じ形で、各要素を頻度にマップした Series です。
    """
    counts = series.value_counts(dropna=False)
    return series.map(counts / counts.sum())


def build_candidate_pool_per_vehicle(
    df: pd.DataFrame,
    top_n_per_vehicle: int = 10,
) -> Dict[str, List[int]]:
    """
    車両ごとの長時間放置履歴を集計し、滞在時間の長い順で Top-N 候補プールを作成する。
    ここで得た候補は学習前処理で使う静的な下地であり、充電イベントごとの最終候補は
    `pipeline.generate_candidates` が曜日・時間帯・充電クラスタなどの文脈を踏まえて組み直す。
    戻り値: {hashvin: [cluster_id, ...]}
    """
    parks = df[(df["session_type"] == "inactive") & (df["is_long_park"] == True)].copy()  # noqa: E712
    parks["duration_minutes"] = pd.to_numeric(parks["duration_minutes"], errors="coerce").fillna(0.0)

    per_vehicle: Dict[str, List[int]] = {}
    for hv, g in parks.groupby("hashvin"):
        top = (
            g.groupby("session_cluster")["duration_minutes"].sum()
            .sort_values(ascending=False)
            .head(int(top_n_per_vehicle))
            .index.to_list()
        )
        per_vehicle[str(hv)] = [int(x) for x in top]

    # 履歴が存在しない車両は空リストのまま扱い、generate_candidates が文脈から補完する
    for hv in df["hashvin"].unique():
        hv = str(hv)
        if hv not in per_vehicle:
            per_vehicle[hv] = []

    return per_vehicle


def _cyclical_enc_hour(hour: float | int) -> Tuple[float, float]:
    """
    時刻（0-23）をサイン・コサインの循環表現へ変換します。
    NaN の場合は (0.0, 0.0) を返します。
    """
    if pd.isna(hour):
        return 0.0, 0.0
    rad = 2 * math.pi * (float(hour) % 24.0) / 24.0
    return math.sin(rad), math.cos(rad)


def _cyclic_hour_distance(a: float | int, b: float | int) -> float:
    """
    2つの時刻 a, b（0-23）間の巡回距離（24時間を一周とする短い方の差）を返します。
    いずれかが NaN の場合は NaN。
    """
    if pd.isna(a) or pd.isna(b):
        return float("nan")
    a = float(a) % 24.0
    b = float(b) % 24.0
    d = abs(a - b)
    return min(d, 24.0 - d)


def _hour_to_bin(hour: float | int, bin_size: int) -> int:
    """
    時刻（0-23）を bin_size 時間幅のビン番号（0..(24/bin_size - 1)）に変換。
    NaN の場合は -1 を返す。
    """
    if pd.isna(hour):
        return -1
    hour = float(hour) % 24.0
    b = int(hour // max(1, int(bin_size)))
    max_bin = max(1, 24 // max(1, int(bin_size)))
    return int(min(b, max_bin - 1))


def _build_transition_tables(
    charge_to_long: pd.DataFrame,
    bin_size: int = 3,
    alpha: float = 1.0,
) -> Dict[str, pd.DataFrame]:
    """
    充電→長時間放置のラベル表（charge_to_long）から、過去の遷移回数・確率テーブルを構築します。
    - グローバル / 車両別 の両方を計算
    - 条件は以下の2種類を用意:
      (A) P(c_p | c_c, h_c_bin) … 充電クラスタと充電開始時刻ビンに条件付け
      (B) P(c_p | h_c_bin) … h_c_bin のみで条件付け（疎な場合のフォールバック）

    ラプラス平滑化（alpha）で疎セルを補正します。
    戻り値は lookup 用の DataFrame を dict で返します。
    """
    c2p = charge_to_long.dropna(subset=["park_cluster"]).copy()
    if c2p.empty:
        return {
            "global_cchc": pd.DataFrame(),
            "vehicle_cchc": pd.DataFrame(),
            "global_hc": pd.DataFrame(),
            "vehicle_hc": pd.DataFrame(),
        }

    # 時刻ビンを付与
    c2p["h_c_bin"] = c2p["charge_start_hour"].apply(lambda h: _hour_to_bin(h, bin_size))
    c2p["park_cluster"] = c2p["park_cluster"].astype(int)
    c2p["charge_cluster"] = c2p["charge_cluster"].astype(int)

    # (A) 条件: (charge_cluster, h_c_bin)
    grpA_g = (
        c2p.groupby(["charge_cluster", "h_c_bin", "park_cluster"], as_index=False)
        .size()
        .rename(columns={"size": "cnt"})
    )
    # 正規化用の合計
    denomA_g = grpA_g.groupby(["charge_cluster", "h_c_bin"], as_index=False)["cnt"].sum().rename(columns={"cnt": "denom"})
    tblA_g = grpA_g.merge(denomA_g, on=["charge_cluster", "h_c_bin"], how="left")
    # ラプラス平滑化: (cnt+alpha)/(denom+alpha*K) の K は park_cluster の種類数
    K = c2p["park_cluster"].nunique()
    tblA_g["prob"] = (tblA_g["cnt"] + float(alpha)) / (tblA_g["denom"] + float(alpha) * float(K))

    # 車両別 (A)
    grpA_v = (
        c2p.groupby(["hashvin", "charge_cluster", "h_c_bin", "park_cluster"], as_index=False)
        .size()
        .rename(columns={"size": "cnt"})
    )
    denomA_v = grpA_v.groupby(["hashvin", "charge_cluster", "h_c_bin"], as_index=False)["cnt"].sum().rename(columns={"cnt": "denom"})
    tblA_v = grpA_v.merge(denomA_v, on=["hashvin", "charge_cluster", "h_c_bin"], how="left")
    tblA_v["prob"] = (tblA_v["cnt"] + float(alpha)) / (tblA_v["denom"] + float(alpha) * float(K))

    # (B) 条件: (h_c_bin) のみ
    grpB_g = (
        c2p.groupby(["h_c_bin", "park_cluster"], as_index=False)
        .size()
        .rename(columns={"size": "cnt"})
    )
    denomB_g = grpB_g.groupby(["h_c_bin"], as_index=False)["cnt"].sum().rename(columns={"cnt": "denom"})
    tblB_g = grpB_g.merge(denomB_g, on=["h_c_bin"], how="left")
    tblB_g["prob"] = (tblB_g["cnt"] + float(alpha)) / (tblB_g["denom"] + float(alpha) * float(K))

    grpB_v = (
        c2p.groupby(["hashvin", "h_c_bin", "park_cluster"], as_index=False)
        .size()
        .rename(columns={"size": "cnt"})
    )
    denomB_v = grpB_v.groupby(["hashvin", "h_c_bin"], as_index=False)["cnt"].sum().rename(columns={"cnt": "denom"})
    tblB_v = grpB_v.merge(denomB_v, on=["hashvin", "h_c_bin"], how="left")
    tblB_v["prob"] = (tblB_v["cnt"] + float(alpha)) / (tblB_v["denom"] + float(alpha) * float(K))

    return {
        "global_cchc": tblA_g,
        "vehicle_cchc": tblA_v,
        "global_hc": tblB_g,
        "vehicle_hc": tblB_v,
    }


def build_ranking_training_data(
    df_sessions: pd.DataFrame,
    charge_to_long: pd.DataFrame,
    candidate_pool: Dict[str, List[int]],
    centroids_by_vehicle: pd.DataFrame,
    negative_sample_k: Optional[int] = None,
    hour_bin_size: int = 3,
    alpha_smooth: float = 1.0,
) -> pd.DataFrame:
    """
    各充電イベントに対して、候補クラスタを展開し特徴量を計算して学習用の行を生成します。

    引数:
      - df_sessions: prepare_sessions の出力（文脈情報を含むセッション表）
      - charge_to_long: build_charge_to_next_long_table の出力（真のリンク/ラベル情報）
      - candidate_pool: {hashvin: [候補クラスタ,...]} 形式の辞書
      - centroids_by_vehicle: 車両×クラスタの重心座標テーブル（lat/lon/count）
      - negative_sample_k: 負例（真のクラスタ以外の候補）を各 group で最大 K 件にサンプリング。
        None の場合はサンプリングせず全候補を使用。

    戻り値:
      列 [group_id, label, hashvin, charge_cluster, candidate_cluster, ...特徴量] を持つ DataFrame。
    """
    # Popularity features
    parks = df_sessions[(df_sessions["session_type"] == "inactive") & (df_sessions["is_long_park"] == True)].copy()  # noqa: E712
    parks["pop_global"] = _popularity(parks["session_cluster"])
    parks["pop_vehicle"] = parks.groupby("hashvin")["session_cluster"].transform(_popularity)

    pop_global = (
        parks.groupby("session_cluster")["pop_global"].mean().rename("pop_global").reset_index()
    )
    pop_vehicle = (
        parks.groupby(["hashvin", "session_cluster"])["pop_vehicle"].mean().rename("pop_vehicle").reset_index()
    )

    # 車両×クラスタごとの「代表開始時刻」（平均 start_hour）
    typical_hour = (
        parks.groupby(["hashvin", "session_cluster"]).agg(mean_start_hour=("start_hour", "mean")).reset_index()
    )

    # Prepare centroid lookup
    centroids = centroids_by_vehicle.copy()
    centroids.columns = ["hashvin", "cluster", "centroid_lat", "centroid_lon", "centroid_count"]

    # 過去の「充電→長時間放置」から遷移回数・確率のテーブルを作成
    trans_tbls = _build_transition_tables(charge_to_long, bin_size=int(hour_bin_size), alpha=float(alpha_smooth))

    # 候補行を構築（真のラベルが存在する charge のみ学習用に使用）
    rows: List[dict] = []
    lbl = charge_to_long.dropna(subset=["park_cluster"]).copy()

    # join popularity and typical hour for convenience (not strictly required here)
    for _, r in lbl.iterrows():
        hv = str(r["hashvin"])
        group_id = f"{hv}__{pd.to_datetime(r['charge_start_time']).isoformat()}"
        true_cluster = int(r["park_cluster"]) if pd.notnull(r["park_cluster"]) else None

        candidates = list(map(int, candidate_pool.get(hv, [])))
        # 真のクラスタが候補に含まれていなければ必ず追加
        if true_cluster is not None and true_cluster not in candidates:
            candidates = [true_cluster] + candidates

        # 負例サンプリング（データサイズ制御のため任意）
        if negative_sample_k is not None and len(candidates) > 0 and true_cluster is not None:
            negs = [c for c in candidates if c != true_cluster]
            rng = np.random.default_rng(123)
            negs = rng.choice(negs, size=min(int(negative_sample_k), len(negs)), replace=False).tolist()
            candidates = [true_cluster] + negs

        # 充電側文脈
        c_cluster = int(r.get("charge_cluster")) if pd.notnull(r.get("charge_cluster")) else None
        c_hour = float(r.get("charge_start_hour")) if pd.notnull(r.get("charge_start_hour")) else np.nan
        wday = int(r.get("weekday")) if pd.notnull(r.get("weekday")) else -1
        c_sin, c_cos = _cyclical_enc_hour(c_hour)
        # 充電終了地点の座標（end_lat/lon）を sessions から取得
        _charge_rows = df_sessions[(df_sessions["hashvin"] == hv) & (pd.to_datetime(df_sessions["start_time"]) == pd.to_datetime(r.get("charge_start_time")))]
        if not _charge_rows.empty:
            c_end_lat = float(_charge_rows.iloc[0].get("end_lat", np.nan))
            c_end_lon = float(_charge_rows.iloc[0].get("end_lon", np.nan))
            # 直前セッション（充電直前の行動）情報を取得
            hv_rows = df_sessions[df_sessions["hashvin"] == hv].sort_values("start_time")
            # 充電行の位置
            idx = hv_rows.index.get_indexer_for(_charge_rows.index)[0] if len(_charge_rows.index) else -1
            # index から1つ前の行を探す（見つからなければ None）
            prev_end_lat = prev_end_lon = float("nan")
            prev_cluster = None
            prev_is_long = 0
            if idx != -1:
                pos = list(hv_rows.index).index(_charge_rows.index[0])
                if pos - 1 >= 0:
                    prev_row = hv_rows.iloc[pos - 1]
                    prev_end_lat = float(prev_row.get("end_lat", np.nan))
                    prev_end_lon = float(prev_row.get("end_lon", np.nan))
                    prev_cluster = prev_row.get("session_cluster")
                    prev_is_long = 1 if bool(prev_row.get("is_long_park", False)) else 0
        else:
            c_end_lat = float("nan")
            c_end_lon = float("nan")
            prev_end_lat = float("nan")
            prev_end_lon = float("nan")
            prev_cluster = None
            prev_is_long = 0

        # 充電開始時刻のビン
        h_c_bin = _hour_to_bin(c_hour, int(hour_bin_size))

        for cand in candidates:
            # 候補クラスタ側の統計・特徴
            pv = pop_vehicle[(pop_vehicle["hashvin"] == hv) & (pop_vehicle["session_cluster"] == cand)]
            pv_val = float(pv["pop_vehicle"].values[0]) if not pv.empty else 0.0
            pg = pop_global[pop_global["session_cluster"] == cand]
            pg_val = float(pg["pop_global"].values[0]) if not pg.empty else 0.0

            th = typical_hour[(typical_hour["hashvin"] == hv) & (typical_hour["session_cluster"] == cand)]
            mean_hp = float(th["mean_start_hour"].values[0]) if not th.empty else np.nan
            hour_diff = _cyclic_hour_distance(c_hour, mean_hp)
            # 候補クラスタの重心と充電終了地点の距離
            cent = centroids[(centroids["hashvin"] == hv) & (centroids["cluster"] == cand)]
            if not cent.empty:
                plat = float(cent["centroid_lat"].values[0])
                plon = float(cent["centroid_lon"].values[0])
                dist_km = _haversine_km(c_end_lat, c_end_lon, plat, plon)
                cnt_hist = float(cent["centroid_count"].values[0])
            else:
                dist_km = float("nan")
                cnt_hist = 0.0

            # 直前行動に基づく特徴
            prev_same_cand = 1 if (prev_cluster is not None and str(prev_cluster) == str(cand)) else 0
            prev_to_cand_km = _haversine_km(prev_end_lat, prev_end_lon, plat, plon) if not np.isnan(dist_km) else float("nan")

            # 過去の遷移回数・確率（条件: (charge_cluster, h_c_bin) / (h_c_bin)）
            # グローバル
            def _lookup(tbl: pd.DataFrame, query: dict) -> Tuple[float, float]:
                if tbl.empty:
                    return 0.0, 0.0
                q = tbl
                for k, v in query.items():
                    q = q[q[k] == v]
                if q.empty:
                    return 0.0, 0.0
                return float(q["cnt"].iloc[0]), float(q["prob"].iloc[0])

            cnt_gcchc, prob_gcchc = _lookup(
                trans_tbls["global_cchc"], {"charge_cluster": c_cluster, "h_c_bin": h_c_bin, "park_cluster": cand}
            ) if c_cluster is not None else (0.0, 0.0)
            cnt_vcchc, prob_vcchc = _lookup(
                trans_tbls["vehicle_cchc"], {"hashvin": hv, "charge_cluster": c_cluster, "h_c_bin": h_c_bin, "park_cluster": cand}
            ) if c_cluster is not None else (0.0, 0.0)
            cnt_ghc, prob_ghc = _lookup(trans_tbls["global_hc"], {"h_c_bin": h_c_bin, "park_cluster": cand})
            cnt_vhc, prob_vhc = _lookup(trans_tbls["vehicle_hc"], {"hashvin": hv, "h_c_bin": h_c_bin, "park_cluster": cand})

            rows.append(
                {
                    "group_id": group_id,
                    "label": 1 if (true_cluster is not None and cand == true_cluster) else 0,
                    "hashvin": hv,
                    "charge_cluster": c_cluster,
                    "candidate_cluster": cand,
                    "weekday": wday,
                    "charge_start_hour": c_hour,
                    "charge_hour_sin": c_sin,
                    "charge_hour_cos": c_cos,
                    "cand_pop_vehicle": pv_val,
                    "cand_pop_global": pg_val,
                    "cand_mean_start_hour": mean_hp,
                    "cand_hour_diff": hour_diff,
                    "dist_charge_to_cand_km": dist_km,
                    "cand_hist_count": cnt_hist,
                    "same_as_charge_cluster": 1 if (str(c_cluster) == str(cand)) else 0,
                    # 遷移履歴に基づく特徴
                    "trans_cnt_global_cchc": cnt_gcchc,
                    "trans_prob_global_cchc": prob_gcchc,
                    "trans_cnt_vehicle_cchc": cnt_vcchc,
                    "trans_prob_vehicle_cchc": prob_vcchc,
                    "trans_cnt_global_hc": cnt_ghc,
                    "trans_prob_global_hc": prob_ghc,
                    "trans_cnt_vehicle_hc": cnt_vhc,
                    "trans_prob_vehicle_hc": prob_vhc,
                    # 直前行動の影響
                    "prev_same_as_candidate": prev_same_cand,
                    "prev_is_long_park": prev_is_long,
                    "prev_to_cand_dist_km": prev_to_cand_km,
                }
            )

    return pd.DataFrame(rows)


def get_feature_columns(df_rank: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    学習に用いる特徴量列名と、その中でカテゴリ扱いが望ましい列名の候補を返します。
    AutoGluon は型から自動推定もしますが、参照用に返しておきます。
    """
    cat_cols = ["hashvin", "charge_cluster", "candidate_cluster", "weekday"]
    feat_cols = [
        c
        for c in df_rank.columns
        if c
        not in (
            "group_id",
            "label",
        )
    ]
    return feat_cols, [c for c in cat_cols if c in df_rank.columns]
