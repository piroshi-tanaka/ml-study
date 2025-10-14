"""
EV の充電・放置挙動を可視化・集計するための再利用可能な EDA パイプラインです。

本コードはデータ抽出Step.mdの要件に沿って、以下を実装しています。
- フォント設定はノートブック側で一度だけ行う前提（コード内では設定しません）
- すべての処理を車両ごと（hashvin）に独立実行し、出力も車両ごとにフォルダ分け
- prepare_sessions 内で「充電 c の後、次の長時間放置（途中に充電が入らない最初のもの）」を紐づけ
- 出力先は outdir の直下に hashvin ごとにサブフォルダを作成し、配下に plots/ と tables/ を作成
- 15分スロット展開は行わず、曜日×時間（1時間単位）に重なりがあればカウントする方針に変更
- 週次の母数はデータ期間の最小日〜最大日に含まれる各曜日の出現回数（例: 52〜53 を想定）
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from scipy.spatial.distance import jensenshannon


def setup_plot_style(_: Iterable[str] | None = None) -> None:
    """
    ノートブック側でフォントを一度だけ設定する前提のため、ここではスタイルのみ設定します。
    互換性のためのダミー関数です（過去のノートブックから呼ばれてもエラーにならないように）。
    """
    sns.set_theme(style="whitegrid")


def load_sessions(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Parse datetimes and align to Asia/Tokyo (requirement). If tz-aware, convert; if naive, localize.
    for col in ["start_time", "end_time"]:
        series = pd.to_datetime(df[col], errors="coerce")
        if pd.api.types.is_datetime64tz_dtype(series.dtype):
            df[col] = series.dt.tz_convert("Asia/Tokyo").dt.tz_localize(None)
        else:
            df[col] = series.dt.tz_localize("Asia/Tokyo", nonexistent="shift_forward", ambiguous="NaT").dt.tz_localize(None)
    return df


def prepare_sessions(
    df: pd.DataFrame,
    long_park_threshold_minutes: int = 360,
) -> pd.DataFrame:
    """
    前処理（特徴量生成・セッション間の関係付け）を行います。処理単位は車両（hashvin）です。

    具体的には：
    - 長時間放置の定義: inactive かつ duration_minutes >= threshold（デフォルト 360 分 = 6 時間）
    - 同一車両内での前後関係（prev_*/next_*）を付与
    - 「充電 → 次の長時間放置（途中に充電が挟まらない最初のもの）」の紐付けを作成
      - 充電行に next_long_park_* を格納
      - 紐付けられた長時間放置行には after_charge_first_long = True を付与
    - 互換目的で after_charge（直前が充電の長時間放置）も残置
    """
    df = df.copy()
    df = df.sort_values(["hashvin", "start_time"]).reset_index(drop=True)

    df["duration_minutes"] = df["duration_minutes"].astype(float)
    df["weekday"] = df["start_time"].dt.dayofweek
    df["start_hour"] = df["start_time"].dt.hour
    df["date"] = df["start_time"].dt.date

    df["is_long_park"] = (df["session_type"] == "inactive") & (
        df["duration_minutes"].astype(float) >= float(long_park_threshold_minutes)
    )

    grouped = df.groupby("hashvin", group_keys=False)
    df["prev_session_type"] = grouped["session_type"].shift(1)
    df["prev_cluster"] = grouped["session_cluster"].shift(1)
    df["prev_is_long_park"] = grouped["is_long_park"].shift(1)
    df["next_session_type"] = grouped["session_type"].shift(-1)
    df["next_cluster"] = grouped["session_cluster"].shift(-1)
    df["next_is_long_park"] = grouped["is_long_park"].shift(-1)

    # 充電 → 次の長時間放置（途中に充電が無い場合のみ）を紐付ける列を初期化
    df["next_long_park_cluster"] = np.nan
    df["next_long_park_start_time"] = pd.NaT
    df["next_long_park_end_time"] = pd.NaT
    df["after_charge_first_long"] = False

    for hv, g in df.groupby("hashvin", group_keys=False):
        idxs = g.index.tolist()
        for i in idxs:
            if df.at[i, "session_type"] != "charging":
                continue
            next_idx: Optional[int] = None
            interrupted = False
            for j in [k for k in idxs if k > i]:
                stype = df.at[j, "session_type"]
                if stype == "charging":
                    interrupted = True
                    break
                if bool(df.at[j, "is_long_park"]):
                    next_idx = j
                    break
            if (not interrupted) and (next_idx is not None):
                df.at[i, "next_long_park_cluster"] = df.at[next_idx, "session_cluster"]
                df.at[i, "next_long_park_start_time"] = df.at[next_idx, "start_time"]
                df.at[i, "next_long_park_end_time"] = df.at[next_idx, "end_time"]
                df.at[next_idx, "after_charge_first_long"] = True

    # 互換目的のフラグ（新ヒートマップでは未使用だが残しておく）
    df["after_charge"] = df["is_long_park"] & (df["prev_session_type"] == "charging")

    # Charge-only helper columns
    df["charge_start_hour"] = np.where(
        df["session_type"] == "charging", df["start_hour"], np.nan
    )
    df["charge_cluster"] = np.where(
        df["session_type"] == "charging", df["session_cluster"], np.nan
    )
    return df


def ensure_dirs(output_root: Path) -> Tuple[Path, Path]:
    plots_dir = output_root / "plots"
    tables_dir = output_root / "tables"
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir, tables_dir


# ------------------------------
# Charge -> Next Long-Park mapping (Step 4)
# ------------------------------

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    try:
        import math
        if any(pd.isna(x) for x in [lat1, lon1, lat2, lon2]):
            return float("nan")
        R = 6371.0
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c
    except Exception:
        return float("nan")


def build_charge_to_next_long_table(df: pd.DataFrame) -> pd.DataFrame:
    """Construct per-charge mapping to first subsequent long park without intervening charge.

    Output columns:
    - hashvin, weekday (of charge start), charge_cluster, charge_start_time,
      charge_start_hour, charge_end_time,
      park_cluster, park_start_time, park_start_hour, park_duration_minutes,
      gap_minutes, dist_charge_to_park_km
    """
    charges = df[df["session_type"] == "charging"].copy()
    if charges.empty:
        return pd.DataFrame(
            columns=
            [
                "hashvin","weekday","charge_cluster","charge_start_time","charge_start_hour","charge_end_time",
                "park_cluster","park_start_time","park_start_hour","park_duration_minutes","gap_minutes","dist_charge_to_park_km",
            ]
        )

    # Join next long-park info from current df (prepared in prepare_sessions)
    out_rows: List[Dict] = []
    for i, row in charges.iterrows():
        hv = row["hashvin"]
        charge_cluster = row.get("session_cluster")
        c_start = row.get("start_time")
        c_end = row.get("end_time")
        c_start_hour = pd.to_datetime(c_start).hour if pd.notnull(c_start) else np.nan
        weekday = pd.to_datetime(c_start).dayofweek if pd.notnull(c_start) else np.nan
        park_cluster = row.get("next_long_park_cluster")
        p_start = row.get("next_long_park_start_time")
        p_end = row.get("next_long_park_end_time")
        if pd.notnull(park_cluster):
            park_start_hour = pd.to_datetime(p_start).hour if pd.notnull(p_start) else np.nan
            park_duration = (pd.to_datetime(p_end) - pd.to_datetime(p_start)).total_seconds() / 60.0 if (pd.notnull(p_end) and pd.notnull(p_start)) else np.nan
            gap_minutes = (pd.to_datetime(p_start) - pd.to_datetime(c_end)).total_seconds() / 60.0 if (pd.notnull(p_start) and pd.notnull(c_end)) else np.nan
            # Distance from charge end to park start (uses end_lat/lon of charge and start_lat/lon of park)
            # Find that park row to get its start coords
            park_row = df[(df["hashvin"]==hv) & (df["start_time"]==p_start) & (df["end_time"]==p_end)]
            if not park_row.empty:
                p_lat = float(park_row.iloc[0].get("start_lat", np.nan))
                p_lon = float(park_row.iloc[0].get("start_lon", np.nan))
            else:
                p_lat = p_lon = float("nan")
            c_end_lat = float(row.get("end_lat", np.nan))
            c_end_lon = float(row.get("end_lon", np.nan))
            dist_km = _haversine_km(c_end_lat, c_end_lon, p_lat, p_lon)
        else:
            park_start_hour = np.nan
            park_duration = np.nan
            gap_minutes = np.nan
            dist_km = np.nan

        out_rows.append(
            {
                "hashvin": hv,
                "weekday": weekday,
                "charge_cluster": charge_cluster,
                "charge_start_time": c_start,
                "charge_start_hour": c_start_hour,
                "charge_end_time": c_end,
                "park_cluster": park_cluster,
                "park_start_time": p_start,
                "park_start_hour": park_start_hour,
                "park_duration_minutes": park_duration,
                "gap_minutes": gap_minutes,
                "dist_charge_to_park_km": dist_km,
            }
        )

    return pd.DataFrame(out_rows)


def step1_long_park_distribution(
    df: pd.DataFrame,
    plots_dir: Path,
    tables_dir: Path,
    per_vehicle_top_n: int = 10,
    top_k: int = 5,
) -> List[int]:
    """Bar charts of long-park clusters and return top-K cluster ids."""
    long_df = df[df["is_long_park"]].copy()
    if long_df.empty:
        return []

    cluster_totals = (
        long_df.groupby("session_cluster")["duration_minutes"].sum().sort_values(ascending=False)
    )
    cluster_totals_hours = (cluster_totals / 60).rename("total_hours")
    cluster_totals_hours.to_csv(tables_dir / "long_park_cluster_total_hours.csv", header=True)

    for hashvin, group in long_df.groupby("hashvin"):
        top_series = (
            group.groupby("session_cluster")["duration_minutes"].sum().sort_values(ascending=False).head(per_vehicle_top_n)
            / 60.0
        )
        if top_series.empty:
            continue
        plt.figure(figsize=(10, 4))
        top_series.sort_values().plot(kind="barh", color="#1f77b4")
        plt.xlabel("累積滞在時間 [時間]")
        plt.ylabel("クラスタ ID")
        plt.title(f"{hashvin} の長時間放置クラスタ 上位{per_vehicle_top_n}")
        plt.tight_layout()
        plt.savefig(plots_dir / f"bar_cluster_distribution_{hashvin}.png", dpi=220)
        plt.close()

    focus_clusters = cluster_totals.index.tolist()[:top_k]
    return focus_clusters


def _weekday_denominator_df(start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    """Build denominator counts per weekday-hour based on date range inclusive.

    For each weekday (0-6), denominator equals the number of dates of that
    weekday in [start_date, end_date]. Repeated across all 24 hours.
    """
    all_days = pd.date_range(start=start_date.normalize(), end=end_date.normalize(), freq="D")
    weekday_counts = pd.Series(all_days.weekday).value_counts().reindex(range(7), fill_value=0).sort_index()
    denom = pd.DataFrame({h: weekday_counts.values for h in range(24)})
    denom.index = range(7)
    denom.columns = list(range(24))
    return denom.astype(int)


def _count_overlaps_by_weekday_hour(df_long: pd.DataFrame) -> pd.DataFrame:
    """Count sessions that overlap each weekday-hour bin at least once.

    A session contributes +1 to every (weekday, hour) bin it overlaps with any
    positive duration. Overlap is computed by stepping through hour edges.
    """
    mat = pd.DataFrame(0, index=range(7), columns=range(24), dtype=int)
    if df_long.empty:
        return mat
    for _, r in df_long.iterrows():
        s: pd.Timestamp = r["start_time"]
        e: pd.Timestamp = r["end_time"]
        if pd.isna(s) or pd.isna(e) or e <= s:
            continue
        t = s.floor("H")
        while t < e:
            t_end = t + pd.Timedelta(hours=1)
            overlap_start = max(s, t)
            overlap_end = min(e, t_end)
            if overlap_end > overlap_start:
                mat.at[int(t.weekday()), int(t.hour)] += 1
            t = t_end
    return mat.astype(int)


# ------------------------------
# Mixture stats for charge->park (Step 5)
# ------------------------------

_BINS = [(0,6),(6,9),(9,12),(12,15),(15,18),(18,21),(21,24)]


def _hour_to_bin(h: int) -> str:
    for a,b in _BINS:
        if a <= int(h) < b:
            return f"{a:02d}-{b:02d}"
    return "NA"


def build_mixture_stats(mapping: pd.DataFrame, tables_dir: Path, alpha: float = 1.0) -> None:
    """Compute counts and conditional probabilities for park cluster vs charge/park hours and charge cluster.

    - count(h_c, h_p, c_p) and P(c_p | h_c, h_p)
    - count(c_c, h_c, h_p, c_p) and P(c_p | c_c, h_c, h_p)
    h_c/h_p also binned to coarse bins for robustness.
    """
    if mapping.empty:
        return
    m = mapping.copy()
    # Prepare hour columns and bins
    m["h_c"] = m["charge_start_hour"].astype("Int64")
    m["h_p"] = m["park_start_hour"].astype("Int64")
    m["h_c_bin"] = m["h_c"].apply(lambda x: _hour_to_bin(x) if pd.notnull(x) else "NA")
    m["h_p_bin"] = m["h_p"].apply(lambda x: _hour_to_bin(x) if pd.notnull(x) else "NA")
    m["c_c"] = m["charge_cluster"].astype("Int64")
    m["c_p"] = m["park_cluster"].astype("Int64")
    m["c_p_filled"] = m["c_p"].astype(object).where(m["c_p"].notna(), other="NA")

    # Counts without charge cluster
    counts_hh_cp = m.groupby(["h_c_bin","h_p_bin","c_p_filled"]).size().rename("count").reset_index()
    counts_hh = counts_hh_cp.groupby(["h_c_bin","h_p_bin"])['count'].sum().rename("total").reset_index()
    probs_hh = counts_hh_cp.merge(counts_hh, on=["h_c_bin","h_p_bin"], how="left")
    # Laplace smoothing across categories per (h_c_bin,h_p_bin)
    # Estimate number of categories K as distinct c_p including NA for that hh
    probs_rows = []
    for (hc, hp), sub in counts_hh_cp.groupby(["h_c_bin","h_p_bin"], dropna=False):
        total = int(sub['count'].sum())
        cats = sub['c_p_filled'].astype(str).unique().tolist()
        K = max(len(cats), 1)
        for _, r in sub.iterrows():
            num = int(r['count'])
            p = (num + alpha) / (total + alpha*K)
            probs_rows.append({"h_c_bin":hc, "h_p_bin":hp, "c_p_filled":r['c_p_filled'], "prob":p})
    probs_hh_df = pd.DataFrame(probs_rows)

    counts_hh_cp.to_csv(tables_dir / "counts_hcbin_hpbin_cp.csv", index=False)
    probs_hh_df.to_csv(tables_dir / "probs_cp_given_hcbin_hpbin.csv", index=False)

    # Counts with charge cluster
    counts_cc_hh_cp = m.groupby(["c_c","h_c_bin","h_p_bin","c_p_filled"]).size().rename("count").reset_index()
    counts_cc_hh = counts_cc_hh_cp.groupby(["c_c","h_c_bin","h_p_bin"])['count'].sum().rename("total").reset_index()
    probs_rows2 = []
    for (cc, hc, hp), sub in counts_cc_hh_cp.groupby(["c_c","h_c_bin","h_p_bin"], dropna=False):
        total = int(sub['count'].sum())
        cats = sub['c_p_filled'].astype(str).unique().tolist()
        K = max(len(cats), 1)
        for _, r in sub.iterrows():
            num = int(r['count'])
            p = (num + alpha) / (total + alpha*K)
            probs_rows2.append({"c_c":cc, "h_c_bin":hc, "h_p_bin":hp, "c_p_filled":r['c_p_filled'], "prob":p})
    probs_cc_hh_df = pd.DataFrame(probs_rows2)

    counts_cc_hh_cp.to_csv(tables_dir / "counts_cc_hcbin_hpbin_cp.csv", index=False)
    probs_cc_hh_df.to_csv(tables_dir / "probs_cp_given_cc_hcbin_hpbin.csv", index=False)


def _plot_charge_hour_to_nextpark(mapping: pd.DataFrame, plots_dir: Path) -> None:
    """
    充電開始時刻（時間）× 充電クラスタごとの「次の長時間放置クラスタ」の分布を棒グラフで可視化します。
    - 各（充電クラスタ, 時間）における park_cluster の割合を算出し、facet に充電クラスタを取って描画します。
    - 可視化は参考用であり、詳細な数表は mixture の CSV を参照してください。
    """
    df = mapping.dropna(subset=["park_cluster", "charge_cluster", "charge_start_hour"])  # 欠損は除外
    if df.empty:
        return
    counts = (
        df.groupby(["charge_cluster", "charge_start_hour", "park_cluster"]).size().rename("count").reset_index()
    )
    totals = counts.groupby(["charge_cluster", "charge_start_hour"])['count'].sum().rename("total").reset_index()
    probs = counts.merge(totals, on=["charge_cluster", "charge_start_hour"], how="left")
    probs["probability"] = probs["count"] / probs["total"].replace(0, np.nan)
    probs = probs.dropna(subset=["probability"]).copy()
    probs["park_cluster_label"] = probs["park_cluster"].map(lambda x: f"クラスタ{x}")
    probs["charge_cluster_label"] = probs["charge_cluster"].map(lambda x: f"充電クラスタ{x}")
    g = sns.catplot(
        data=probs,
        kind="bar",
        x="charge_start_hour",
        y="probability",
        hue="park_cluster_label",
        col="charge_cluster_label",
        col_wrap=3,
        height=3.6,
        aspect=1.2,
    )
    g.set_titles("{col_name}")
    g.set_axis_labels("充電開始時刻 (時)", "次の長時間放置クラスタの割合")
    if g._legend is not None:
        g._legend.set_title("次のクラスタ")
    for ax in g.axes.flatten():
        ax.yaxis.set_major_locator(MaxNLocator(5))
    plt.tight_layout()
    plt.savefig(plots_dir / "charge_start_hour_next_long_cluster.png", dpi=220)
    plt.close()


def _plot_h2d_top1_heatmap(
    mapping: pd.DataFrame,
    plots_dir: Path,
    charge_cluster: Optional[int] = None,
) -> None:
    """
    H2D ヒートマップ（h_c × h_p）を描画します。
    - それぞれのセル（h_c=充電開始時刻の時間, h_p=放置開始時刻の時間）において、
      最も多い park_cluster（Top1）を特定し、その Top1 の割合（確率）を着色します。
    - 各セルには注釈として「Top1 のクラスタ ID / 確率 / サンプル数 n」を表示します。
    - charge_cluster を指定すると、その充電クラスタに限定した H2D を個別に出力します。

    分析の意図:
    - 充電開始の時間帯と、その後の放置開始の時間帯の関係性を俯瞰します。
    - セルの色が濃いほど「特定の次放置クラスタに偏る」傾向が強いことを示し、
      行動パターンの時間的な規則性（例: 深夜に充電→翌朝の自宅付近クラスタへ）を把握できます。
    """
    df = mapping.dropna(subset=["park_cluster", "charge_start_hour", "park_start_hour"]).copy()
    if charge_cluster is not None:
        df = df[df["charge_cluster"].astype("Int64") == int(charge_cluster)]
    if df.empty:
        return
    # カウント集計: (h_c, h_p, c_p)
    counts = (
        df.groupby(["charge_start_hour", "park_start_hour", "park_cluster"]).size().rename("count").reset_index()
    )
    # セル合計: (h_c, h_p)
    totals = (
        counts.groupby(["charge_start_hour", "park_start_hour"])['count'].sum().rename("total").reset_index()
    )
    merged = counts.merge(totals, on=["charge_start_hour", "park_start_hour"], how="left")
    merged["prob"] = merged["count"] / merged["total"].replace(0, np.nan)
    # Top1 を抽出
    top1 = merged.sort_values(["charge_start_hour", "park_start_hour", "prob"], ascending=[True, True, False])
    top1 = top1.groupby(["charge_start_hour", "park_start_hour"], as_index=False).first()
    # ピボット化（値=prob）
    prob_mat = top1.pivot(index="charge_start_hour", columns="park_start_hour", values="prob").reindex(index=range(24), columns=range(24))
    # 注釈テキスト（C=クラスタ, p=割合%, n=件数）
    annot = pd.DataFrame("", index=prob_mat.index, columns=prob_mat.columns)
    for _, r in top1.iterrows():
        hc = int(r["charge_start_hour"]); hp = int(r["park_start_hour"]) 
        c = int(r["park_cluster"]) if pd.notnull(r["park_cluster"]) else -1
        p = float(r["prob"]) if pd.notnull(r["prob"]) else 0.0
        n = int(r["total"]) if pd.notnull(r["total"]) else 0
        annot.at[hc, hp] = f"C={c}\n{p*100:.0f}%\n(n={n})"

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        prob_mat * 100.0,
        vmin=0,
        vmax=100,
        cmap="YlOrRd",
        linewidths=0.5,
        linecolor="white",
        annot=annot.values,
        fmt="",
        annot_kws={"fontsize": 7},
        cbar_kws={"label": "Top1 クラスタの割合 [%]"},
    )
    title = (
        f"H2D Heatmap（Top1）: 全体"
        if charge_cluster is None
        else f"H2D Heatmap（Top1）: 充電クラスタ {charge_cluster}"
    )
    plt.title(title)
    plt.xlabel("放置開始時刻 h_p (時)")
    plt.ylabel("充電開始時刻 h_c (時)")
    plt.tight_layout()
    out_name = (
        "h2d_top1_overall.png" if charge_cluster is None else f"h2d_top1_chargecluster_{charge_cluster}.png"
    )
    plt.savefig(plots_dir / out_name, dpi=220)
    plt.close()


def _compute_route_and_charge_specific_metrics(
    df: pd.DataFrame,
    long_df: pd.DataFrame,
    tables_dir: Path,
) -> None:
    """
    追加の評価指標を算出します。
    - route_return_ratio: 充電の「次の長時間放置」が、その車両の『代表クラスタ（最頻の長時間放置クラスタ）』に戻る割合
      分母: 次の長時間放置が特定できた充電イベント数
    - charge_specific_ratio: 充電後最初の長時間放置に偏るクラスタの度合い（lift の平均）
      定義: 各クラスタの lift = (after_share / overall_share)。クラスタごとの after 事象で重み付け平均
    """
    if long_df.empty:
        return
    # 代表クラスタ（最頻の長時間放置クラスタ）
    home_cluster = (
        long_df.groupby("session_cluster").size().sort_values(ascending=False).index.tolist()[0]
    )
    # マッピング（NA 除外）
    mapping = build_charge_to_next_long_table(df)
    m_valid = mapping.dropna(subset=["park_cluster"]).copy()
    if m_valid.empty:
        pd.DataFrame([{"home_cluster": np.nan, "route_return_ratio": np.nan, "charge_specific_ratio": np.nan}]).to_csv(
            tables_dir / "metrics_route_return_and_charge_specific.csv", index=False
        )
        return
    route_return_ratio = float((m_valid["park_cluster"].astype(int) == int(home_cluster)).mean())

    # share を簡便に算出（分母はデータ期間の曜日×時間の総セル数と等価な合計値）
    start_date = long_df["start_time"].min().normalize()
    end_date = long_df["end_time"].max().normalize()
    denom = _weekday_denominator_df(start_date, end_date)
    total_denom = float(denom.values.sum()) if denom.values.size else np.nan
    overall_by_cluster = long_df.groupby("session_cluster").size().rename("n").astype(float)
    after_by_cluster = long_df[long_df["after_charge_first_long"]].groupby("session_cluster").size().rename("n").astype(float)
    all_clusters = sorted(set(overall_by_cluster.index.tolist()) | set(after_by_cluster.index.tolist()))
    ov = overall_by_cluster.reindex(all_clusters, fill_value=0.0)
    af = after_by_cluster.reindex(all_clusters, fill_value=0.0)
    ov_share = ov / total_denom if total_denom and total_denom > 0 else ov * np.nan
    af_share = af / total_denom if total_denom and total_denom > 0 else af * np.nan
    lift = af_share / ov_share.replace(0, np.nan)
    weights = af
    charge_specific_ratio = float((lift.fillna(0.0) * weights).sum() / weights.sum()) if float(weights.sum()) > 0 else float("nan")

    pd.DataFrame(
        [
            {
                "home_cluster": int(home_cluster),
                "route_return_ratio": route_return_ratio,
                "charge_specific_ratio": charge_specific_ratio,
            }
        ]
    ).to_csv(tables_dir / "metrics_route_return_and_charge_specific.csv", index=False)

    pd.DataFrame(
        {
            "cluster": all_clusters,
            "overall_share": ov_share.values,
            "after_share": af_share.values,
            "lift_after_vs_overall": lift.values,
            "after_count": weights.values,
        }
    ).to_csv(tables_dir / "cluster_lift_after_vs_all.csv", index=False)


def _build_percentage_annotation(
    percent_matrix: pd.DataFrame,
    numerator_matrix: pd.DataFrame,
    denominator_matrix: pd.DataFrame,
    value_label: str,
) -> pd.DataFrame:
    annot = percent_matrix.copy().astype(float)
    for idx in annot.index:
        for col in annot.columns:
            value = float(annot.loc[idx, col]) if col in annot.columns else 0.0
            numerator = (
                int(numerator_matrix.loc[idx, col])
                if idx in numerator_matrix.index and col in numerator_matrix.columns
                else 0
            )
            denominator = (
                int(denominator_matrix.loc[idx, col])
                if idx in denominator_matrix.index and col in denominator_matrix.columns
                else 0
            )
            if denominator == 0:
                annot.loc[idx, col] = "0.0%\n(n=0/0)"
            else:
                annot.loc[idx, col] = f"{value:.1f}%\n(n={numerator}/{denominator})"
    annot.attrs["value_label"] = value_label
    return annot


def _build_diff_annotation(
    diff_percent_matrix: pd.DataFrame,
    after_numerators: pd.DataFrame,
    after_denominators: pd.DataFrame,
    all_numerators: pd.DataFrame,
    all_denominators: pd.DataFrame,
) -> pd.DataFrame:
    annot = diff_percent_matrix.copy().astype(float)
    for idx in annot.index:
        for col in annot.columns:
            diff_val = float(annot.loc[idx, col]) if col in annot.columns else 0.0
            after_num = (
                int(after_numerators.loc[idx, col])
                if idx in after_numerators.index and col in after_numerators.columns
                else 0
            )
            after_den = (
                int(after_denominators.loc[idx, col])
                if idx in after_denominators.index and col in after_denominators.columns
                else 0
            )
            all_num = (
                int(all_numerators.loc[idx, col])
                if idx in all_numerators.index and col in all_numerators.columns
                else 0
            )
            all_den = (
                int(all_denominators.loc[idx, col])
                if idx in all_denominators.index and col in all_denominators.columns
                else 0
            )
            annot.loc[idx, col] = f"{diff_val:.1f}pt\n後={after_num}/{after_den}\n全={all_num}/{all_den}"
    annot.attrs["value_label"] = "差分(ポイント)"
    return annot


def _plot_heatmap(
    matrix: pd.DataFrame,
    annot: pd.DataFrame,
    title: str,
    x_label: str,
    y_label: str,
    colorbar_label: str,
    path: Path,
    cmap: str = "YlGnBu",
    center: float | None = None,
) -> None:
    if matrix.empty:
        return
    plt.figure(figsize=(12, 4))
    sns.heatmap(
        matrix,
        cmap=cmap,
        center=center,
        linewidths=0.5,
        linecolor="white",
        annot=annot.values,
        fmt="",
        annot_kws={"fontsize": 8},
        cbar_kws={"label": colorbar_label},
    )
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def step2_heatmaps_and_charge_effects(
    df: pd.DataFrame,
    focus_clusters: Iterable[int],
    plots_dir: Path,
    tables_dir: Path,
    freq_minutes: int | None = None,  # 互換目的のダミー引数（未使用）
) -> None:
    """Heatmaps for weekday×hour shares (overall vs after-charge-first-long)."""
    long_df = df[df["is_long_park"]].copy()
    if long_df.empty or not focus_clusters:
        return

    start_date = long_df["start_time"].min().normalize()
    end_date = long_df["end_time"].max().normalize()
    denom = _weekday_denominator_df(start_date, end_date)

    overall_counts = _count_overlaps_by_weekday_hour(long_df)
    overall_after_counts = _count_overlaps_by_weekday_hour(
        long_df[long_df["after_charge_first_long"]]
    )

    for cluster_id in focus_clusters:
        cluster_slice = long_df[long_df["session_cluster"] == cluster_id]
        if cluster_slice.empty:
            continue

        cluster_counts = _count_overlaps_by_weekday_hour(cluster_slice)
        aligned_cluster_counts, aligned_overall_counts = cluster_counts.align(
            overall_counts, join="right", fill_value=0
        )
        aligned_overall_counts = aligned_overall_counts.astype(int)

        share_all = aligned_cluster_counts.divide(denom.where(denom != 0)).fillna(0.0)

        after_slice = cluster_slice[cluster_slice["after_charge_first_long"]]
        cluster_after_counts = _count_overlaps_by_weekday_hour(after_slice)
        aligned_cluster_after_counts, aligned_overall_after_counts = cluster_after_counts.align(
            overall_after_counts, join="right", fill_value=0
        )
        aligned_overall_after_counts = aligned_overall_after_counts.astype(int)
        share_after = aligned_cluster_after_counts.divide(denom.where(denom != 0)).fillna(0.0)

        aligned_all_share, aligned_after_share = share_all.align(share_after, fill_value=0.0)
        all_percent_aligned = aligned_all_share * 100
        after_percent_aligned = aligned_after_share * 100
        diff_percent = after_percent_aligned - all_percent_aligned

        aligned_all_numerators, aligned_all_denominators = aligned_cluster_counts.align(denom, join="right", fill_value=0)
        aligned_after_numerators, aligned_after_denominators = aligned_cluster_after_counts.align(denom, join="right", fill_value=0)

        _plot_heatmap(
            matrix=all_percent_aligned,
            annot=_build_percentage_annotation(
                all_percent_aligned,
                aligned_all_numerators,
                aligned_all_denominators,
                "滞在比率 (%)",
            ),
            title=f"クラスタ {cluster_id} | 長時間放置全体 滞在比率",
            x_label="開始時刻 (時)",
            y_label="曜日 (0=月)",
            colorbar_label="滞在比率 [%]",
            path=plots_dir / f"heatmap_cluster_{cluster_id}_all.png",
        )
        _plot_heatmap(
            matrix=after_percent_aligned,
            annot=_build_percentage_annotation(
                after_percent_aligned,
                aligned_after_numerators,
                aligned_after_denominators,
                "滞在比率 (%)",
            ),
            title=f"クラスタ {cluster_id} | 充電後最初の長時間放置 滞在比率",
            x_label="開始時刻 (時)",
            y_label="曜日 (0=月)",
            colorbar_label="滞在比率 [%]",
            path=plots_dir / f"heatmap_cluster_{cluster_id}_aftercharge.png",
        )
        _plot_heatmap(
            matrix=diff_percent,
            annot=_build_diff_annotation(
                diff_percent,
                aligned_after_numerators,
                aligned_after_denominators,
                aligned_all_numerators,
                aligned_all_denominators,
            ),
            title=f"クラスタ {cluster_id} | 充電後最初 − 全体 (ポイント差)",
            x_label="開始時刻 (時)",
            y_label="曜日 (0=月)",
            colorbar_label="差分 [pt]",
            path=plots_dir / f"heatmap_cluster_{cluster_id}_diff.png",
            cmap="RdBu_r",
            center=0.0,
        )

        # Save raw matrices (shares are 0-1 range)
        aligned_all_share.to_csv(tables_dir / f"heatmap_cluster_{cluster_id}_all.csv")
        aligned_after_share.to_csv(tables_dir / f"heatmap_cluster_{cluster_id}_aftercharge.csv")
        (aligned_after_share - aligned_all_share).to_csv(
            tables_dir / f"heatmap_cluster_{cluster_id}_diff.csv"
        )


# ------------------------------
# Daily time-bin labeling and edges (Step 6)
# ------------------------------

def _overlap_minutes(a_start: pd.Timestamp, a_end: pd.Timestamp, b_start: pd.Timestamp, b_end: pd.Timestamp) -> float:
    s = max(a_start, b_start)
    e = min(a_end, b_end)
    if pd.isna(s) or pd.isna(e) or e <= s:
        return 0.0
    return float((e - s).total_seconds() / 60.0)


def build_timebin_cluster_tables(df: pd.DataFrame, tables_dir: Path) -> None:
    """For each day, choose cluster per coarse time bin by max overlapped minutes, then count nodes and edges.

    Nodes: counts of (time_bin, cluster)
    Edges: counts of (time_bin_i, cluster_i) -> (time_bin_j, cluster_j) for consecutive bins within the day
    """
    inactive = df[df["session_type"] == "inactive"].copy()
    if inactive.empty:
        return
    # bins in local day time
    bin_labels = [f"{a:02d}-{b:02d}" for a,b in _BINS]
    records = []
    for (hv, date), day_df in inactive.groupby(["hashvin","date"], as_index=False):
        # For each bin, compute overlap minutes per cluster and pick argmax
        assignments = []
        for (a,b), label in zip(_BINS, bin_labels):
            b_start = pd.Timestamp(year=day_df["start_time"].iloc[0].year,
                                   month=day_df["start_time"].iloc[0].month,
                                   day=pd.to_datetime(date).day,
                                   hour=a)
            b_end = b_start + pd.Timedelta(hours=(b-a))
            totals: Dict[int, float] = {}
            for _, r in day_df.iterrows():
                mins = _overlap_minutes(pd.to_datetime(r["start_time"]), pd.to_datetime(r["end_time"]), b_start, b_end)
                if mins > 0:
                    cl = int(r["session_cluster"])
                    totals[cl] = totals.get(cl, 0.0) + mins
            if totals:
                sel_cluster = max(totals.items(), key=lambda kv: kv[1])[0]
                assignments.append((label, sel_cluster))
                records.append({"hashvin": hv, "date": date, "time_bin": label, "cluster": sel_cluster})
        # Build edges between consecutive bins for this day
        for (from_bin, from_cl), (to_bin, to_cl) in zip(assignments, assignments[1:]):
            records.append({
                "hashvin": hv,
                "date": date,
                "from_bin": from_bin,
                "from_cluster": from_cl,
                "to_bin": to_bin,
                "to_cluster": to_cl,
                "_edge": True,
            })

    if not records:
        return
    rec_df = pd.DataFrame(records)
    node_df = rec_df[rec_df["_edge"].isna()].groupby(["time_bin","cluster"]).size().rename("count").reset_index()
    edge_df = rec_df[rec_df["_edge"] == True].groupby(["from_bin","from_cluster","to_bin","to_cluster"]).size().rename("count").reset_index()
    node_df.to_csv(tables_dir / "timebin_cluster_nodes.csv", index=False)
    edge_df.to_csv(tables_dir / "timebin_cluster_edges.csv", index=False)


def _plot_network(
    edges: pd.DataFrame,
    source_col: str,
    target_col: str,
    weight_col: str,
    title: str,
    path: Path,
) -> None:
    if edges.empty:
        return
    graph = nx.DiGraph()
    for _, row in edges.iterrows():
        weight = row[weight_col]
        if weight <= 0:
            continue
        graph.add_edge(row[source_col], row[target_col], weight=weight)
    if not graph.edges:
        return
    edge_weights = np.array([d["weight"] for _, _, d in graph.edges(data=True)])
    max_weight = edge_weights.max() if edge_weights.size else 1.0
    edge_widths = 1.5 + 4.0 * (edge_weights / max_weight)
    pos = nx.spring_layout(graph, seed=42, k=1.5 / np.sqrt(max(graph.number_of_nodes(), 1)))
    plt.figure(figsize=(8, 6))
    nx.draw_networkx_nodes(graph, pos, node_size=1200, node_color="#fee090")
    nx.draw_networkx_labels(graph, pos, font_size=9)
    nx.draw_networkx_edges(
        graph,
        pos,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=15,
        width=edge_widths,
        edge_color="#4575b4",
        connectionstyle="arc3,rad=0.1",
    )
    edge_labels = {(u, v): f"{data['weight']:.0f}" for u, v, data in graph.edges(data=True)}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def step3_transition_analysis(
    df: pd.DataFrame,
    plots_dir: Path,
    tables_dir: Path,
) -> None:
    """Transition analysis before/after charge and daily transitions."""
    charging_df = df[df["session_type"] == "charging"].copy()
    if not charging_df.empty:
        before_edges = (
            charging_df.dropna(subset=["prev_cluster"])
            .groupby(["prev_cluster", "session_cluster"])
            .size()
            .reset_index(name="count")
        )
        before_edges["probability"] = before_edges.groupby("prev_cluster")["count"].transform(lambda x: x / x.sum())
        before_edges.rename(columns={"session_cluster": "charge_cluster"}, inplace=True)
        before_edges.to_csv(tables_dir / "transition_prob_before.csv", index=False)
        _plot_network(
            before_edges,
            "prev_cluster",
            "charge_cluster",
            "count",
            "充電前の遷移ネットワーク (前クラスタ → 充電クラスタ)",
            plots_dir / "network_before_charge.png",
        )

    # After: charge → next long-park (no intervening charge)
    after_charge_df = charging_df.dropna(subset=["next_long_park_cluster"]).copy()
    if not after_charge_df.empty:
        after_edges = (
            after_charge_df.groupby(["session_cluster", "next_long_park_cluster"]).size().reset_index(name="count")
        )
        after_edges["probability"] = after_edges.groupby("session_cluster")["count"].transform(lambda x: x / x.sum())
        after_edges.rename(columns={"session_cluster": "charge_cluster"}, inplace=True)
        after_edges.to_csv(tables_dir / "transition_prob_after.csv", index=False)
        _plot_network(
            after_edges,
            "charge_cluster",
            "next_long_park_cluster",
            "count",
            "充電後の遷移ネットワーク (充電クラスタ → 次長時間放置)",
            plots_dir / "network_after_charge.png",
        )

    # Daily transitions among inactive sessions (same-day)
    inactive_df = df[df["session_type"] == "inactive"].copy()
    if inactive_df.empty:
        return
    daily_group = inactive_df.groupby(["hashvin", "date"], group_keys=False)
    inactive_df["next_cluster_same_day"] = daily_group["session_cluster"].shift(-1)
    transitions = inactive_df.dropna(subset=["next_cluster_same_day"]).copy()

    charge_day_flags = (
        df[df["session_type"] == "charging"].groupby(["hashvin", "date"]).size().reset_index(name="charge_events")
    )
    charge_day_flags["has_charge"] = charge_day_flags["charge_events"] > 0
    transitions = transitions.merge(
        charge_day_flags[["hashvin", "date", "has_charge"]],
        on=["hashvin", "date"],
        how="left",
    )
    transitions["has_charge"] = transitions["has_charge"].fillna(False)

    def _transition_matrix(sub_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        counts = sub_df.groupby(["session_cluster", "next_cluster_same_day"]).size().unstack(fill_value=0)
        probs = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
        return probs, counts

    matrix_charge, counts_charge = _transition_matrix(transitions[transitions["has_charge"]])
    matrix_nocharge, counts_nocharge = _transition_matrix(transitions[~transitions["has_charge"]])

    matrix_charge.to_csv(tables_dir / "transition_matrix_charge_days.csv")
    matrix_nocharge.to_csv(tables_dir / "transition_matrix_nocharge_days.csv")

    aligned_charge, aligned_nocharge = matrix_charge.align(matrix_nocharge, fill_value=0.0, join="outer")
    aligned_counts_charge, aligned_counts_nocharge = counts_charge.align(counts_nocharge, fill_value=0, join="outer")
    diff_matrix = aligned_charge - aligned_nocharge
    diff_matrix.to_csv(tables_dir / "transition_matrix_diff.csv")

    def _annot_transition(percent_matrix: pd.DataFrame, count_matrix: pd.DataFrame) -> pd.DataFrame:
        annot = percent_matrix.copy()
        for idx in annot.index:
            for col in annot.columns:
                val = annot.loc[idx, col]
                cnt = int(count_matrix.loc[idx, col]) if (idx in count_matrix.index and col in count_matrix.columns) else 0
                annot.loc[idx, col] = (f"{val:.1f}%\n(n={cnt})" if cnt > 0 else "0.0%\n(n=0)")
        return annot

    def _annot_transition_diff(diff_percent: pd.DataFrame, counts_charge: pd.DataFrame, counts_nocharge: pd.DataFrame) -> pd.DataFrame:
        annot = diff_percent.copy()
        for idx in annot.index:
            for col in annot.columns:
                charge_n = int(counts_charge.loc[idx, col]) if (idx in counts_charge.index and col in counts_charge.columns) else 0
                nocharge_n = int(counts_nocharge.loc[idx, col]) if (idx in counts_nocharge.index and col in counts_nocharge.columns) else 0
                annot.loc[idx, col] = f"{diff_percent.loc[idx, col]:.1f}pt\n有={charge_n}\n無={nocharge_n}"
        return annot

    for label, matrix, counts in [("charge", aligned_charge * 100, aligned_counts_charge), ("nocharge", aligned_nocharge * 100, aligned_counts_nocharge)]:
        _plot_heatmap(
            matrix=matrix,
            annot=_annot_transition(matrix, counts),
            title=("日次遷移: 充電あり日" if label == "charge" else "日次遷移: 充電なし日"),
            x_label="遷移先クラスタ",
            y_label="遷移元クラスタ",
            colorbar_label="遷移確率 [%]",
            path=plots_dir / f"transition_matrix_{label}.png",
        )

    diff_percent = diff_matrix * 100
    _plot_heatmap(
        matrix=diff_percent,
        annot=_annot_transition_diff(diff_percent, aligned_counts_charge, aligned_counts_nocharge),
        title="日次遷移: 差分 (充電あり − なし)",
        x_label="遷移先クラスタ",
        y_label="遷移元クラスタ",
        colorbar_label="差分 [pt]",
        path=plots_dir / "transition_matrix_diff.png",
        cmap="RdBu_r",
        center=0.0,
    )

    # JS / TV distance per origin cluster
    metrics = []
    for origin_cluster in diff_matrix.index:
        p = aligned_charge.loc[origin_cluster].values
        q = aligned_nocharge.loc[origin_cluster].values
        if not p.size or not q.size:
            continue
        jsd = float(jensenshannon(p, q, base=2))
        tvd = float(0.5 * np.abs(p - q).sum())
        metrics.append({"from_cluster": origin_cluster, "js_distance": jsd, "tv_distance": tvd})
    if metrics:
        pd.DataFrame(metrics).to_csv(tables_dir / "transition_diff_metrics.csv", index=False)


def run_pipeline(
    csv_path: Path,
    output_root: Path,
    per_vehicle_top_n: int = 10,
    focus_clusters: int = 5,
    long_park_threshold_minutes: int = 360,
) -> None:
    """Run full EDA per vehicle (hashvin), writing to per-vehicle folders."""
    df_all = load_sessions(csv_path)
    for hv in sorted(df_all["hashvin"].dropna().unique()):
        sub = df_all[df_all["hashvin"] == hv].copy()
        sub = prepare_sessions(sub, long_park_threshold_minutes=long_park_threshold_minutes)
        hv_root = output_root / str(hv)
        plots_dir, tables_dir = ensure_dirs(hv_root)

        focus_cluster_ids = step1_long_park_distribution(
            sub,
            plots_dir=plots_dir,
            tables_dir=tables_dir,
            per_vehicle_top_n=per_vehicle_top_n,
            top_k=focus_clusters,
        )

        step2_heatmaps_and_charge_effects(
            sub,
            focus_clusters=focus_cluster_ids,
            plots_dir=plots_dir,
            tables_dir=tables_dir,
        )

        step3_transition_analysis(sub, plots_dir=plots_dir, tables_dir=tables_dir)

        # Step 4: charge -> next long park mapping table
        mapping = build_charge_to_next_long_table(sub)
        mapping.to_csv(tables_dir / "charge_to_next_long_mapping.csv", index=False)

        # 可視化: 充電開始時刻 × 充電クラスタ → 次の長時間放置クラスタの割合（facet bar）
        _plot_charge_hour_to_nextpark(mapping, plots_dir=plots_dir)

        # 可視化: H2D ヒートマップ（h_c × h_p の Top1 クラスタ割合）。全体と充電クラスタ別
        _plot_h2d_top1_heatmap(mapping, plots_dir=plots_dir, charge_cluster=None)
        for cc in sorted(pd.Series(mapping["charge_cluster"].dropna().unique()).astype(int)):
            _plot_h2d_top1_heatmap(mapping, plots_dir=plots_dir, charge_cluster=int(cc))

        # Step 5: mixture statistics and conditional probabilities
        build_mixture_stats(mapping, tables_dir=tables_dir, alpha=1.0)

        # Step 6: per-day time bin labeling and edges
        build_timebin_cluster_tables(sub, tables_dir=tables_dir)

        # 評価指標（route_return_ratio / charge_specific_ratio）
        _compute_route_and_charge_specific_metrics(sub, sub[sub["is_long_park"]], tables_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EV 放置挙動 EDA パイプラインを実行します")
    parser.add_argument("--csv-path", type=Path, required=True, help="セッションログ CSV のパス")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="成果物出力ディレクトリ")
    parser.add_argument("--per-vehicle-top-n", type=int, default=10, help="Step1 で車両ごとに描画する上位クラスタ数")
    parser.add_argument("--focus-clusters", type=int, default=5, help="Step2 の詳細対象となるクラスタ数（各車両の上位から）")
    parser.add_argument(
        "--long-park-threshold-minutes",
        type=int,
        default=360,
        help="長時間放置と見なす最小分数（充電後の次長時間放置判定にも使用）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(
        csv_path=args.csv_path,
        output_root=args.output_dir,
        per_vehicle_top_n=args.per_vehicle_top_n,
        focus_clusters=args.focus_clusters,
        long_park_threshold_minutes=args.long_park_threshold_minutes,
    )


if __name__ == "__main__":
    main()
