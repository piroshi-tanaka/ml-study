# day_grid_view.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Literal

import plotly.graph_objects as go
import plotly.express as px


# =========================
# モデル / ユーティリティ
# =========================
@dataclass
class DayMeta:
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
    """4:00起点のサービス日を決める基準日（00:00）を返す。"""
    adj = ts - pd.to_timedelta(day_start_hour, unit="h")
    return adj.dt.normalize()

def _split_by_service_day(df: pd.DataFrame, start_col: str, end_col: str, day_start_hour: int) -> pd.DataFrame:
    """04:00境界で日跨ぎセッションを分割。"""
    d = df.copy()
    d[start_col] = _ensure_dt(d, start_col)
    d[end_col]   = _ensure_dt(d, end_col)
    assert (d[end_col] >= d[start_col]).all(), "end_time < start_time の行があります。"

    out = []
    for _, r in d.iterrows():
        s = r[start_col]
        e = r[end_col]
        cur_start = s
        while True:
            day0 = _service_day_floor(pd.Series([cur_start]), day_start_hour).iloc[0] + pd.Timedelta(hours=day_start_hour)
            next_day = day0 + pd.Timedelta(days=1)
            seg_end = min(e, next_day)
            nr = r.copy()
            nr[start_col] = cur_start
            nr[end_col]   = seg_end
            out.append(nr)
            if seg_end >= e:
                break
            cur_start = seg_end
    return pd.DataFrame(out).reset_index(drop=True)


# =========================
# データ構築（x=日付）
# =========================
def build_day_grid_nodes(
    df: pd.DataFrame,
    *,
    time_col_start: str = "start_time",
    time_col_end: str   = "end_time",
    cluster_col: str    = "cluster_id",
    type_col: str       = "sessionType",       # "inactive"/"charging"（movingは除外前提）
    duration_col: str   = "duration_minutes",
    min_stop_minutes: int = 30,
    long_parking_hours: float = 6.0,
    day_start_hour: int = 4,                   # サービス日 4:00 起点
    top_n_clusters: Optional[int] = 12,        # y軸は上位N + OTHER
    other_label: str = "OTHER",
    lane_offset_for_charging: float = 0.35,    # 充電を別レーンに少しずらす
    jitter_std: float = 0.04
) -> Tuple[pd.DataFrame, DayMeta]:
    """
    全期間を “日付×クラスタ” で可視化する散布用ノードを作る。
    出力: nodes（1行=分割済みイベント）, meta
    """
    need = {time_col_start, time_col_end, cluster_col, type_col}
    if not need.issubset(df.columns):
        raise KeyError(f"必要列が不足: {need - set(df.columns)}")

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

    # 4:00 境界で分割
    d = _split_by_service_day(d0, time_col_start, time_col_end, day_start_hour)

    # サービス日（基準00:00）と表示用 “日付”（YYYY-MM-DD）
    d["service_day"] = _service_day_floor(d[time_col_start], day_start_hour)
    d["date"]        = pd.to_datetime(d["service_day"].dt.date)  # x軸に使う

    # 付随情報
    d["duration_h"] = d["duration_min"] / 60.0
    d["is_long_parking"] = (d[type_col].eq("inactive")) & (d["duration_h"] >= long_parking_hours)

    # クラスタ順（総滞在分でソート）
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

    # 散布描画で使う列だけ
    nodes = d.rename(columns={cluster_col:"cluster_id", type_col:"sessionType"})[
        ["date", "y_jitter", "cluster_id", "sessionType",
         "duration_min", "duration_h", "is_long_parking"]
    ].copy()

    meta = DayMeta(cluster_order=clus_order, y_map=y_map, other_label=other_label, day_start_hour=day_start_hour)
    return nodes, meta


# =========================
# 散布図（全期間：x=日付, y=クラスタ）
# =========================
def plot_day_grid_scatter(
    nodes: pd.DataFrame,
    *,
    figsize: Tuple[int,int] = (1200, 500),
    node_size_scale: float = 26.0,
    color_by: Literal["cluster","type"] = "cluster",   # 色をクラスタかタイプで切替
    highlight_long_parking: bool = True,
    title: Optional[str] = "Temporal Grid — All Days (x=date, y=cluster)"
) -> go.Figure:
    fig = go.Figure()

    # 色設定
    if color_by == "cluster":
        # クラスタごとにトレース（色分け）
        clusters = nodes["cluster_id"].unique().tolist()
        palette = px.colors.qualitative.Safe + px.colors.qualitative.Set2 + px.colors.qualitative.Pastel
        color_map = {c: palette[i % len(palette)] for i, c in enumerate(clusters)}
        for c in clusters:
            g = nodes[nodes["cluster_id"] == c]
            sizes = np.clip(g["duration_h"] * node_size_scale, 10, 260)
            symbol = np.where(g["sessionType"].eq("charging"), "triangle-up", "circle")
            fig.add_trace(go.Scattergl(
                x=g["date"], y=g["y_jitter"],
                mode="markers",
                marker=dict(size=sizes, color=color_map[c], symbol=symbol, line=dict(width=0.7)),
                name=f"{c} (n={len(g)})",
                opacity=0.8,
                customdata=np.stack([g["cluster_id"], g["sessionType"], g["duration_min"]], axis=1),
                hovertemplate="date=%{x|%Y-%m-%d}<br>cluster=%{customdata[0]}<br>type=%{customdata[1]}<br>dur_min=%{customdata[2]:.0f}<extra></extra>"
            ))
    else:
        # タイプごとにトレース（色2〜3色でスッキリ）
        for sess, g in nodes.groupby("sessionType"):
            sizes = np.clip(g["duration_h"] * node_size_scale, 10, 260)
            symbol = "triangle-up" if sess == "charging" else "circle"
            fig.add_trace(go.Scattergl(
                x=g["date"], y=g["y_jitter"],
                mode="markers",
                marker=dict(size=sizes, symbol=symbol),
                name=f"{sess} (n={len(g)})",
                opacity=0.8,
                customdata=np.stack([g["cluster_id"], g["duration_min"]], axis=1),
                hovertemplate="date=%{x|%Y-%m-%d}<br>cluster=%{customdata[0]}<br>dur_min=%{customdata[1]:.0f}<extra></extra>"
            ))

    # 長期放置の強調（枠だけ重ねる）
    if highlight_long_parking and "is_long_parking" in nodes.columns:
        lp = nodes[nodes["is_long_parking"]]
        if len(lp) > 0:
            sizes = np.clip(lp["duration_h"] * node_size_scale, 24, 320)
            fig.add_trace(go.Scattergl(
                x=lp["date"], y=lp["y_jitter"],
                mode="markers",
                marker=dict(size=sizes, symbol="circle-open", line=dict(width=1.4)),
                name="long_parking(>=6h)",
                hoverinfo="skip",
                opacity=0.95
            ))

    fig.update_layout(
        title=title,
        width=figsize[0], height=figsize[1],
        plot_bgcolor="white",
        legend=dict(orientation="h", x=1, xanchor="right", y=1.08),
        margin=dict(l=60, r=20, t=60, b=50)
    )
    fig.update_xaxes(title="service day (date, 04:00 start)", showgrid=True, gridcolor="rgba(0,0,0,0.08)")
    # y軸はクラスタ段。丸め段ごとの代表ラベルを付ける
    y_round = nodes["y_jitter"].round()
    ticks = np.sort(y_round.unique())
    labels = []
    tmp = nodes.copy()
    tmp["y_round"] = y_round
    for yv in ticks:
        labels.append(tmp.loc[tmp["y_round"].eq(yv), "cluster_id"].value_counts().index[0])
    fig.update_yaxes(tickmode="array", tickvals=ticks, ticktext=labels, title="cluster (charging lane offset)")

    return fig


# =========================
# ヒートマップ（合計滞在時間：日×クラスタ）
# =========================
def plot_day_cluster_heatmap(
    nodes: pd.DataFrame,
    *,
    session_filter: Optional[Literal["inactive","charging"]] = None,  # None=両方合算
    agg: Literal["duration_h","count"] = "duration_h",
    figsize: Tuple[int,int] = (1200, 520),
    title: Optional[str] = None
) -> go.Figure:
    x = nodes.copy()
    if session_filter:
        x = x[x["sessionType"].eq(session_filter)]
    # ピボット：date × cluster へ
    if agg == "duration_h":
        dat = x.pivot_table(index="cluster_id", columns="date", values="duration_h", aggfunc="sum", fill_value=0.0)
        ztitle = "total hours"
        title  = title or f"Heatmap — total hours per day/cluster ({session_filter or 'all'})"
    else:
        dat = x.pivot_table(index="cluster_id", columns="date", values="duration_h", aggfunc="count", fill_value=0)
        ztitle = "events (count)"
        title  = title or f"Heatmap — events per day/cluster ({session_filter or 'all'})"

    dat = dat.sort_index()  # yに安定順序
    fig = go.Figure(data=go.Heatmap(
        z=dat.values,
        x=dat.columns,
        y=dat.index,
        colorscale="YlGnBu",
        colorbar=dict(title=ztitle)
    ))
    fig.update_layout(
        title=title,
        width=figsize[0], height=figsize[1],
        plot_bgcolor="white",
        margin=dict(l=80, r=30, t=60, b=50)
    )
    fig.update_xaxes(title="service day (date)")
    fig.update_yaxes(title="cluster")
    return fig
    
# df: 事前整形済み (movingと30分未満は除外済みでもOK)
# 必須列: sessionType(inactive/charging), start_time, end_time, cluster_id
# 任意列: duration_minutes（無ければ自動算出）

from day_grid_view import build_day_grid_nodes, plot_day_grid_scatter, plot_day_cluster_heatmap

nodes, meta = build_day_grid_nodes(
    df,
    day_start_hour=4,          # 4:00起点で日付を切る
    top_n_clusters=12,         # yは上位クラスタ+OTHER
    lane_offset_for_charging=0.35
)

# 散布（全期間、x=日付, y=クラスタ、点サイズ=滞在時間、充電は三角）
fig = plot_day_grid_scatter(nodes, color_by="cluster")   # or color_by="type"
fig.show()

# ヒートマップ（例：放置のみ、合計滞在時間）
fig2 = plot_day_cluster_heatmap(nodes, session_filter="inactive", agg="duration_h")
fig2.show()
