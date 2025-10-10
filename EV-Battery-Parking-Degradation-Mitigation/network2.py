# day_temporal_network.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import pandas as pd
import numpy as np
import networkx as nx
from dataclasses import dataclass
from typing import Optional, Tuple, List, Literal, Dict

import plotly.graph_objects as go
from plotly.subplots import make_subplots


# =========================================================
# モデル / ユーティリティ
# =========================================================
@dataclass
class TemporalMeta:
    cluster_order: List[str]
    y_map: Dict[str, int]
    other_label: str
    day_start_hour: int

def _ensure_dt(df: pd.DataFrame, col: str) -> pd.Series:
    s = pd.to_datetime(df[col], errors="coerce")
    if s.isna().any():
        raise ValueError(f"[{col}] に日時へ変換できない値が {int(s.isna().sum())} 件あります。")
    return s

def _service_day_floor(ts: pd.Series, day_start_hour: int) -> pd.Series:
    """
    例: day_start_hour=4 なら、毎日 04:00 をその日の開始とみなす。
    04:00〜翌04:00を1サービス日とし、基準日付は「その日の04:00時点の日付」。
    """
    # tsをローカルタイムで扱っている前提（タイムゾーン扱いは外側で調整）
    adj = ts - pd.to_timedelta(day_start_hour, unit="h")
    return adj.dt.normalize()  # 00:00に丸め→それがサービス日の基準日付

def _split_by_service_day(df: pd.DataFrame, start_col: str, end_col: str, day_start_hour: int) -> pd.DataFrame:
    """
    セッションをサービス日境界（毎日 day_start_hour）で分割。
    長い滞在が 04:00 を跨ぐ場合、区切って2レコードにする。
    """
    d = df.copy()
    d[start_col] = _ensure_dt(d, start_col)
    d[end_col]   = _ensure_dt(d, end_col)
    assert (d[end_col] >= d[start_col]).all(), "end_time < start_time の行があります。"

    out = []
    for _, r in d.iterrows():
        s = r[start_col]
        e = r[end_col]
        # このイベントの最初のサービス日境界（次の day_start まで）
        cur_start = s
        while True:
            # 現在のセグメントが属するサービス日の開始時刻
            day0 = _service_day_floor(pd.Series([cur_start]), day_start_hour).iloc[0] + pd.Timedelta(hours=day_start_hour)
            day1 = day0 + pd.Timedelta(days=1)  # 次のサービス日開始
            seg_end = min(e, day1)  # 境界で分割
            newr = r.copy()
            newr[start_col] = cur_start
            newr[end_col]   = seg_end
            out.append(newr)
            if seg_end >= e:
                break
            cur_start = seg_end
    return pd.DataFrame(out).reset_index(drop=True)

def _to_hours_since_day_start(ts: pd.Series, day_start_hour: int) -> pd.Series:
    day0 = _service_day_floor(ts, day_start_hour) + pd.Timedelta(hours=day_start_hour)
    return (ts - day0).dt.total_seconds() / 3600.0  # 0〜24 の範囲（最大で24に近い）



# =========================================================
# ノード/エッジ生成（1日=固定24hビュー用）
# =========================================================
def build_day_temporal_network(
    df: pd.DataFrame,
    *,
    time_col_start: str = "start_time",
    time_col_end: str   = "end_time",
    cluster_col: str    = "cluster_id",
    type_col: str       = "sessionType",         # "inactive", "charging" （movingは前処理で除外済み）
    duration_col: str   = "duration_minutes",
    min_stop_minutes: int = 30,                  # 念のための下限（0で無効）
    long_parking_hours: float = 6.0,             # inactive & >= 6h を放置扱い
    day_start_hour: int = 4,                     # ⬅️ ここが肝：1日の開始を 4:00 に固定
    top_n_clusters: Optional[int] = None,        # 上位N + OTHER
    other_label: str = "OTHER",
    lane_offset_for_charging: float = 0.45,      # 充電を別レーンに
    jitter_std: float = 0.03,
    edge_anchor: Tuple[Literal["start","end"], Literal["start","end"]] = ("end","start")
) -> Tuple[pd.DataFrame, pd.DataFrame, TemporalMeta]:
    """
    - セッションを 4:00 境界で分割し、各セグメントに service_day を付与
    - x=0..24h（4:00→0h 起点）の座標を付与（start_h, end_h, mid_h）
    - 日内の連続イベント（同一 service_day）のみエッジで接続
    """
    need_cols = {type_col, time_col_start, time_col_end, cluster_col}
    missing = need_cols - set(df.columns)
    if missing:
        raise KeyError(f"入力に必要な列が不足: {missing}")

    d0 = df.copy()
    # duration
    if duration_col in d0.columns:
        d0["duration_min"] = pd.to_numeric(d0[duration_col], errors="coerce")
        if d0["duration_min"].isna().any():
            raise ValueError(f"[{duration_col}] に数値変換できない値があります。")
    else:
        d0[time_col_start] = _ensure_dt(d0, time_col_start)
        d0[time_col_end]   = _ensure_dt(d0, time_col_end)
        d0["duration_min"] = (d0[time_col_end] - d0[time_col_start]).dt.total_seconds()/60.0

    # 念のためのフィルタ
    if min_stop_minutes > 0:
        d0 = d0[d0["duration_min"] >= min_stop_minutes].copy()
    d0 = d0[d0[type_col] != "moving"].copy()

    # 日跨ぎ分割（4:00 境界）
    d = _split_by_service_day(d0, time_col_start, time_col_end, day_start_hour)

    # サービス日キー
    d["service_day"] = _service_day_floor(d[time_col_start], day_start_hour)  # その日の 00:00（ただし 04:00起点前提の基準日付）
    d["weekday"]     = (d["service_day"]).dt.weekday    # 0=Mon..6=Sun（サービス日に依存）
    d["month"]       = (d["service_day"]).dt.month

    # 0〜24h の時間座標
    d["start_h"] = _to_hours_since_day_start(d[time_col_start], day_start_hour)
    d["end_h"]   = _to_hours_since_day_start(d[time_col_end], day_start_hour)
    d["mid_h"]   = (d["start_h"] + d["end_h"]) / 2.0

    # 放置フラグ
    d["duration_h"] = d["duration_min"] / 60.0
    d["is_long_parking"] = (d[type_col].eq("inactive")) & (d["duration_h"] >= long_parking_hours)

    # クラスタ順（総滞在分で並べ替え）
    clus_order = (
        d.groupby(cluster_col)["duration_min"].sum().sort_values(ascending=False).index.tolist()
    )
    if top_n_clusters and len(clus_order) > top_n_clusters:
        keep = set(clus_order[:top_n_clusters])
        d[cluster_col] = np.where(d[cluster_col].isin(keep), d[cluster_col], other_label)
        clus_order = (
            d.groupby(cluster_col)["duration_min"].sum().sort_values(ascending=False).index.tolist()
        )

    y_map = {cid: i for i, cid in enumerate(clus_order)}
    d["y_base"] = d[cluster_col].map(y_map).astype(float)
    d["y"]      = d["y_base"] + np.where(d[type_col].eq("charging"), lane_offset_for_charging, 0.0)
    rng = np.random.default_rng(42)
    d["y_jitter"] = d["y"] + rng.normal(0, jitter_std, size=len(d))

    # 並べ替え & node_id
    d = d.sort_values(["service_day", time_col_start, time_col_end]).reset_index(drop=True)
    d["node_id"] = np.arange(len(d))

    # 日内エッジの構築（同一 service_day 内で連続するイベントを接続）
    edges = []
    src_anchor, dst_anchor = edge_anchor
    for _, g in d.groupby("service_day", sort=True):
        g = g.sort_values(time_col_start)
        ids = g["node_id"].tolist()
        for p, c in zip(ids[:-1], ids[1:]):
            edges.append((int(p), int(c)))
    e = pd.DataFrame(edges, columns=["src", "dst"])

    # アンカー（start/end）に応じた x 座標（0〜24h）
    cols = ["node_id", "start_h", "end_h", "mid_h", "y_jitter", cluster_col, type_col,
            "duration_min", "duration_h", "is_long_parking", "service_day", "weekday", "month"]
    e = (
        e.merge(d[cols], left_on="src", right_on="node_id", how="left")
         .drop(columns=["node_id"])
         .rename(columns={"start_h":"src_start_h","end_h":"src_end_h","mid_h":"src_mid_h",
                          "y_jitter":"y_src", cluster_col:"cluster_src", type_col:"type_src"})
    )
    e = (
        e.merge(d[cols], left_on="dst", right_on="node_id", how="left")
         .drop(columns=["node_id"])
         .rename(columns={"start_h":"dst_start_h","end_h":"dst_end_h","mid_h":"dst_mid_h",
                          "y_jitter":"y_dst", cluster_col:"cluster_dst", type_col:"type_dst"})
    )
    e["x_src"] = np.where(src_anchor == "start", e["src_start_h"], e["src_end_h"])
    e["x_dst"] = np.where(dst_anchor == "start", e["dst_start_h"], e["dst_end_h"])

    # 出力 nodes
    nodes = d.rename(columns={cluster_col:"cluster_id", type_col:"sessionType"})[
        ["node_id","service_day","weekday","month",
         "start_h","end_h","mid_h","y_jitter",
         "cluster_id","sessionType",
         "duration_min","duration_h","is_long_parking"]
    ].copy()

    edges = e.copy()
    meta  = TemporalMeta(cluster_order=clus_order, y_map=y_map, other_label=other_label, day_start_hour=day_start_hour)
    return nodes, edges, meta


# =========================================================
# 可視化（x=0..24h 固定）
# =========================================================
def plot_day_temporal_network(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    service_day: Optional[pd.Timestamp] = None,   # Noneなら全日合算（多すぎると重いので通常は日指定）
    figsize: Tuple[int,int] = (1100, 420),
    show_duration_bar: bool = True,
    bar_alpha_inactive: float = 0.4,
    bar_alpha_charging: float = 0.35,
    node_size_scale: float = 28.0,
    highlight_long_parking: bool = True,
    title: Optional[str] = None
) -> go.Figure:
    """
    指定日の 0..24h（4:00起点）を1枚に描く。service_dayは build関数が付与した基準日付（00:00）です。
    """
    if service_day is not None:
        nd = nodes[nodes["service_day"].eq(pd.to_datetime(service_day))]
        ids = set(nd["node_id"])
        ed = edges[edges["src"].isin(ids) & edges["dst"].isin(ids)]
    else:
        nd = nodes.copy()
        ed = edges.copy()

    fig = go.Figure()

    # 1) 遷移エッジ（薄い線）
    if len(ed) > 0:
        fig.add_trace(go.Scattergl(
            x=np.ravel(ed[["x_src","x_dst"]].values.T),
            y=np.ravel(ed[["y_src","y_dst"]].values.T),
            mode="lines", line=dict(width=0.8), opacity=0.18,
            name="transition", hoverinfo="skip", showlegend=False
        ))

    # 2) 横棒（start_h→end_h）：放置は太め、充電も可視化
    if show_duration_bar:
        for sess, g in nd.groupby("sessionType"):
            width = 6 if sess=="inactive" else (4 if sess=="charging" else 3)
            alpha = bar_alpha_inactive if sess=="inactive" else (bar_alpha_charging if sess=="charging" else 0.25)
            fig.add_trace(go.Scattergl(
                x=np.ravel(g[["start_h","end_h"]].values.T),
                y=np.ravel(g[["y_jitter","y_jitter"]].values.T),
                mode="lines", line=dict(width=width), opacity=alpha,
                name=f"{sess}_bar", hoverinfo="skip", showlegend=False
            ))

    # 3) ノード：充電は三角で少し大きく、放置は丸
    for sess, g in nd.groupby("sessionType"):
        sizes = np.clip(g["duration_h"] * node_size_scale, 12, 300)
        symbol = "triangle-up" if sess=="charging" else "circle"
        fig.add_trace(go.Scattergl(
            x=g["mid_h"], y=g["y_jitter"], mode="markers",
            marker=dict(size=sizes, symbol=symbol, line=dict(width=0.8 if sess=="charging" else 0)),
            name=f"{sess} (n={len(g)})", opacity=0.85 if sess=="charging" else 0.7,
            customdata=np.stack([g["cluster_id"], g["duration_min"]], axis=1),
            hovertemplate=(
                "t=%{x:.2f}h<br>y=%{y:.2f}<br>cluster=%{customdata[0]}<br>"
                "dur_min=%{customdata[1]:.0f}<extra>%{fullData.name}</extra>"
            )
        ))

    # 4) 長期放置の強調枠
    if highlight_long_parking:
        lp = nd[nd["is_long_parking"]]
        if len(lp) > 0:
            sizes = np.clip(lp["duration_h"] * node_size_scale, 28, 340)
            fig.add_trace(go.Scattergl(
                x=lp["mid_h"], y=lp["y_jitter"], mode="markers",
                marker=dict(size=sizes, symbol="circle-open", line=dict(width=1.5)),
                name="long_parking(>=6h)", hoverinfo="skip", opacity=0.95
            ))

    fig.update_layout(
        title=title or (f"Temporal Network (service_day={service_day.date()})" if service_day is not None else "Temporal Network — Day View"),
        width=figsize[0], height=figsize[1],
        plot_bgcolor="white",
        legend=dict(orientation="h", x=1, xanchor="right", y=1.08),
        margin=dict(l=60, r=20, t=60, b=50)
    )
    # x=0..24 固定
    fig.update_xaxes(range=[0, 24], title="time of day (h from 04:00)", showgrid=True, gridcolor="rgba(0,0,0,0.08)")
    # yはクラスタ段
    # 簡易：四捨五入段ごとに代表クラスタ名
    y_round = nd["y_jitter"].round()
    ticks = np.sort(y_round.unique())
    labels = []
    tmp = nd.copy()
    tmp["y_round"] = y_round
    for yv in ticks:
        labels.append(tmp.loc[tmp["y_round"].eq(yv), "cluster_id"].value_counts().index[0])
    fig.update_yaxes(tickmode="array", tickvals=ticks, ticktext=labels, title="cluster (charging lane offset)")

    return fig


def facet_day_temporal_network(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    service_days: Optional[List[pd.Timestamp]] = None,   # 指定しなければ全日（多いと重い）
    max_cols: int = 3,
    figsize_per: Tuple[int,int] = (900, 320),
    title: str = "Temporal Network — Daily Facets"
) -> go.Figure:
    if service_days is None:
        service_days = sorted(nodes["service_day"].drop_duplicates().tolist())
    n = len(service_days)
    ncols = min(max_cols, n)
    nrows = int(np.ceil(n / ncols))

    fig = make_subplots(rows=nrows, cols=ncols,
                        shared_xaxes=False, shared_yaxes=False,
                        subplot_titles=[str(pd.to_datetime(d).date()) for d in service_days],
                        horizontal_spacing=0.04, vertical_spacing=0.08)

    r = c = 1
    for day in service_days:
        nd = nodes[nodes["service_day"].eq(pd.to_datetime(day))]
        ids = set(nd["node_id"])
        ed = edges[edges["src"].isin(ids) & edges["dst"].isin(ids)]

        sub = plot_day_temporal_network(nd, ed, service_day=day, figsize=figsize_per, title=None)
        for tr in sub.data:
            fig.add_trace(tr, row=r, col=c)

        # 軸：0..24固定
        fig.update_xaxes(range=[0,24], showgrid=True, gridcolor="rgba(0,0,0,0.07)", row=r, col=c)
        fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.07)", row=r, col=c)

        c += 1
        if c > ncols:
            c = 1
            r += 1

    fig.update_layout(
        title=title,
        width=figsize_per[0]*ncols,
        height=figsize_per[1]*nrows + 60,
        showlegend=False,
        plot_bgcolor="white",
        margin=dict(l=60, r=20, t=80, b=50)
    )
    return fig


# =========================================================
# 定量評価（1日ビューに対応）
# =========================================================
def compute_day_transition_stats(
    nodes: pd.DataFrame,
    edges: pd.DataFrame
) -> pd.DataFrame:
    """
    サービス日ごとの遷移統計を返す（遷移行列・H(next|prev)・充電直後の長期放置率）。
    出力: service_day, p_long_after_charge, p_long_after_noncharge, diff, entropy_mean
    """
    out = []
    for day, nd in nodes.groupby("service_day"):
        ids = set(nd["node_id"])
        ed = edges[edges["src"].isin(ids) & edges["dst"].isin(ids)]

        # 遷移レコード作成
        e = (
            ed.merge(nd[["node_id","cluster_id","sessionType","is_long_parking"]],
                     left_on="src", right_on="node_id", how="left")
              .rename(columns={"cluster_id":"cluster_src","sessionType":"type_src","is_long_parking":"long_src"})
              .drop(columns=["node_id"])
        )
        e = (
            e.merge(nd[["node_id","cluster_id","sessionType","is_long_parking"]],
                    left_on="dst", right_on="node_id", how="left")
              .rename(columns={"cluster_id":"cluster_dst","sessionType":"type_dst","is_long_parking":"long_dst"})
              .drop(columns=["node_id"])
        )
        if len(e) == 0:
            out.append({"service_day": day, "p_long_after_charge": np.nan,
                        "p_long_after_noncharge": np.nan, "diff": np.nan, "entropy_mean": np.nan})
            continue

        # 条件付き確率
        counts = e.groupby(["cluster_src","cluster_dst"]).size().rename("n").reset_index()
        totals = counts.groupby("cluster_src")["n"].sum().rename("row_total").reset_index()
        mat = counts.merge(totals, on="cluster_src", how="left")
        mat["p"] = mat["n"] / mat["row_total"]

        # H(next|prev) の日内平均
        ent = (
            mat.groupby("cluster_src")["p"]
               .apply(lambda s: float(-(s*np.log2(np.clip(s,1e-12,1))).sum()))
               .rename("H").reset_index()
        )
        entropy_mean = float(ent["H"].mean()) if len(ent) > 0 else np.nan

        # 充電直後に長期放置？
        ce = e[e["type_src"].eq("charging")]
        nce = e[~e["type_src"].eq("charging")]
        p_long_after_charge = float(ce["long_dst"].mean()) if len(ce) > 0 else np.nan
        p_long_after_noncharge = float(nce["long_dst"].mean()) if len(nce) > 0 else np.nan

        out.append({
            "service_day": day,
            "p_long_after_charge": p_long_after_charge,
            "p_long_after_noncharge": p_long_after_noncharge,
            "diff": (p_long_after_charge - p_long_after_noncharge),
            "entropy_mean": entropy_mean
        })

    return pd.DataFrame(out).sort_values("service_day").reset_index(drop=True)