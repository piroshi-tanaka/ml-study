=# tod_transition_network.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import pandas as pd, numpy as np
from typing import List, Tuple, Optional, Dict
import plotly.graph_objects as go
import networkx as nx
import colorsys

# =========================
# 小ユーティリティ
# =========================
def _ensure_dt(s: pd.Series, name: str) -> pd.Series:
    out = pd.to_datetime(s, errors="coerce")
    if out.isna().any():
        raise ValueError(f"[{name}] に日時変換できない値があります（{int(out.isna().sum())}件）。")
    return out

def _hour_from_day_start(ts: pd.Series, day_start_hour: int) -> pd.Series:
    h = ts.dt.hour + ts.dt.minute/60.0 + ts.dt.second/3600.0
    return (h - day_start_hour) % 24

def _service_day(ts: pd.Series, day_start_hour: int) -> pd.Series:
    adj = ts - pd.to_timedelta(day_start_hour, unit="h")
    return adj.dt.normalize()

def _default_bins() -> List[Tuple[int,int]]:
    return [(0,6),(6,10),(10,16),(16,20),(20,24)]

def _bin_label(a: int, b: int) -> str:
    return f"[{a}-{b})"

def _hash_color(key: str) -> str:
    # クラスタID→色（安定）
    h = (hash(str(key)) % 360) / 360.0
    s, v = 0.55, 0.85
    r,g,b = colorsys.hsv_to_rgb(h, s, v)
    return f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"

# =========================
# 前処理 & 集約
# =========================
def build_time_binned_bundle(
    df: pd.DataFrame,
    *,
    col_type: str = "sessionType",         # "inactive"/"charging"/"moving"
    col_start: str = "start_time",
    col_end: str   = "end_time",
    col_cluster: str = "cluster_id",
    col_duration_min: str = "duration_min",
    min_stop_minutes: int = 30,
    drop_moving: bool = True,
    day_start_hour: int = 4,
    hour_bins: Optional[List[Tuple[int,int]]] = None
) -> Dict:
    hour_bins = hour_bins or _default_bins()

    x = df.copy()
    x[col_start] = _ensure_dt(x[col_start], col_start)
    x[col_end]   = _ensure_dt(x[col_end], col_end)

    # duration（無ければ算出）
    if col_duration_min in x.columns:
        x["duration_min"] = pd.to_numeric(x[col_duration_min], errors="coerce")
    else:
        x["duration_min"] = (x[col_end] - x[col_start]).dt.total_seconds()/60.0

    # フィルタ
    if min_stop_minutes:
        x = x[x["duration_min"] >= min_stop_minutes].copy()
    if drop_moving and col_type in x.columns:
        x = x[x[col_type].astype(str).str.lower() != "moving"].copy()

    # サービス日 & 時刻
    x["service_day"] = _service_day(x[col_start], day_start_hour)
    x["hour24"]      = _hour_from_day_start(x[col_start], day_start_hour)

    # ソースイベントの時間帯バケット
    def _bin(h):
        for a,b in hour_bins:
            if (h >= a) and (h < b): return _bin_label(a,b)
        return _bin_label(*hour_bins[-1])
    x["hour_bin"] = x["hour24"].apply(_bin)

    # 連続イベントの遷移（単系列想定）
    x = x.sort_values([col_start, col_end]).reset_index(drop=True)
    prev = x.shift(1)

    edges = pd.DataFrame({
        "prev": prev[col_cluster].astype(str),
        "next": x[col_cluster].astype(str),
        "prev_hour": prev["hour24"],
        "prev_bin": prev["hour_bin"],
        "same_day": (prev["service_day"] == x["service_day"])
    }).dropna(subset=["prev","next","prev_hour","prev_bin"])

    # 時間帯ごとに集計
    edges_by_bin = {}
    nodes_by_bin = {}
    for a,b in hour_bins:
        lab = _bin_label(a,b)
        ebin = edges[edges["prev_bin"] == lab]
        # エッジ集計
        c = (ebin.groupby(["prev","next"]).size().rename("n").reset_index())
        if len(c) == 0:
            edges_by_bin[lab] = pd.DataFrame(columns=["prev","next","n","p"])
            nodes_by_bin[lab] = pd.DataFrame(columns=["cluster_id","count","dwell_min"])
            continue
        c["row_total"] = c.groupby("prev")["n"].transform("sum")
        c["p"] = c["n"] / c["row_total"]
        edges_by_bin[lab] = c.sort_values("n", ascending=False).reset_index(drop=True)
        # ノード（その時間帯に出現したイベント）
        s = x[x["hour_bin"] == lab]
        nodes_by_bin[lab] = (
            s.groupby(col_cluster)
             .agg(count=(col_cluster,"size"), dwell_min=("duration_min","sum"))
             .reset_index().rename(columns={col_cluster:"cluster_id"})
             .sort_values(["count","dwell_min"], ascending=False)
        )

    # 全期間（レイアウト用）
    global_edges = (
        edges.groupby(["prev","next"]).size().rename("n").reset_index()
    )
    global_nodes = (
        x.groupby(col_cluster).agg(count=(col_cluster,"size"), dwell_min=("duration_min","sum"))
         .reset_index().rename(columns={col_cluster:"cluster_id"})
    )

    # 色
    color_map = {str(cid): _hash_color(str(cid)) for cid in global_nodes["cluster_id"].astype(str).tolist()}

    return dict(
        bins=[_bin_label(a,b) for a,b in hour_bins],
        edges_by_bin=edges_by_bin,
        nodes_by_bin=nodes_by_bin,
        global_edges=global_edges,
        global_nodes=global_nodes,
        color_map=color_map
    )

# =========================
# レイアウト（固定）
# =========================
def compute_global_layout(global_edges: pd.DataFrame, seed: int = 42) -> Dict[str, tuple[float,float]]:
    G = nx.DiGraph()
    for _, r in global_edges.iterrows():
        G.add_edge(str(r["prev"]), str(r["next"]), weight=float(r["n"]))
    pos = nx.spring_layout(G.to_undirected(), weight="weight", k=None, seed=seed)
    return {str(k):(float(v[0]), float(v[1])) for k,v in pos.items()}

# =========================
# プロット（時間帯切り替え）
# =========================
def plot_timeband_network(
    bundle: Dict,
    layout_xy: Dict[str, tuple[float,float]],
    *,
    top_k_nodes: int = 18,
    edge_min_count: int = 3,
    edge_min_prob: float = 0.05,
    title: str = "時間帯ごとの遷移ネットワーク（位置固定）",
    width: int = 1100, height: int = 800
) -> go.Figure:
    bins = bundle["bins"]
    edges_by_bin = bundle["edges_by_bin"]
    nodes_by_bin = bundle["nodes_by_bin"]
    color_map = bundle["color_map"]

    frames = []
    for hb in bins:
        nb = nodes_by_bin[hb].copy()
        eb = edges_by_bin[hb].copy()

        # ノード上位だけ残す
        keep = set(nb["cluster_id"].astype(str).head(top_k_nodes))
        nb = nb[nb["cluster_id"].astype(str).isin(keep)].copy()
        eb = eb[eb["prev"].astype(str).isin(keep) & eb["next"].astype(str).isin(keep)].copy()

        # 弱いエッジを間引き
        eb = eb[(eb["n"] >= edge_min_count) & (eb["p"] >= edge_min_prob)].copy()

        # ノードトレース
        xs, ys, cs, sizes, texts, hovers = [], [], [], [], [], []
        for _, r in nb.iterrows():
            cid = str(r["cluster_id"])
            if cid not in layout_xy: continue
            x,y = layout_xy[cid]
            xs.append(x); ys.append(y)
            cs.append(color_map.get(cid, "gray"))
            sizes.append(np.clip(8 + r["dwell_min"]/30.0, 10, 40))  # 30分ごとに拡大
            texts.append(cid)
            hovers.append(f"cluster={cid}<br>count={int(r['count'])}<br>dwell={r['dwell_min']:.0f} min")

        node_trace = go.Scattergl(
            x=xs, y=ys, mode="markers+text",
            text=texts, textposition="top center",
            marker=dict(size=sizes, color=cs, line=dict(width=0.8, color="rgba(0,0,0,0.35)")),
            hovertext=hovers, hoverinfo="text",
            name=f"nodes {hb}"
        )

        # エッジトレース（まとめて1本）
        ex, ey = [], []
        if len(eb) > 0:
            nmax = eb["n"].max()
            for _, r in eb.iterrows():
                s,t = str(r["prev"]), str(r["next"])
                if (s in layout_xy) and (t in layout_xy):
                    x0,y0 = layout_xy[s]; x1,y1 = layout_xy[t]
                    ex += [x0, x1, None]; ey += [y0, y1, None]
            edge_opacity = 0.25 + 0.6 * float(min(1.0, nmax / (nmax if nmax>0 else 1)))
        else:
            edge_opacity = 0.15

        edge_trace = go.Scattergl(
            x=ex, y=ey, mode="lines",
            line=dict(width=2.0, color="rgba(60,60,60,0.6)"),
            opacity=edge_opacity,
            hoverinfo="skip",
            name=f"edges {hb}"
        )

        frames.append(dict(name=hb, data=[edge_trace, node_trace]))

    fig = go.Figure(frames=frames)
    fig.add_traces(frames[0]["data"])
    fig.update_layout(
        title=f"{title} — {bins[0]}",
        width=width, height=height,
        plot_bgcolor="white", showlegend=False,
        margin=dict(l=40,r=40,t=70,b=40),
        updatemenus=[dict(
            type="dropdown", x=1, xanchor="right", y=1.12, yanchor="top",
            buttons=[dict(label=fr["name"], method="update",
                          args=[{"visible":[True, True]},
                                {"title":f"{title} — {fr['name']}"}]) for fr in frames]
        )]
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return from tod_transition_network import (
    build_time_binned_bundle, compute_global_layout, plot_timeband_network
)

# df: 必須列 → start_time, end_time, sessionType, cluster_id, duration_min
bundle = build_time_binned_bundle(
    df,
    col_type="sessionType",
    col_start="start_time",
    col_end="end_time",
    col_cluster="cluster_id",
    col_duration_min="duration_min",
    min_stop_minutes=30,       # 30分未満は除外
    drop_moving=True,          # movingは除外
    day_start_hour=4,          # 04:00起点
    hour_bins=[(0,6),(6,10),(10,16),(16,20),(20,24)]  # 変更可
)

layout_xy = compute_global_layout(bundle["global_edges"], seed=42)

fig = plot_timeband_network(
    bundle, layout_xy,
    top_k_nodes=18,       # 時間帯ごとに上位クラスタだけ表示
    edge_min_count=3,     # 回数が少ない遷移は非表示
    edge_min_prob=0.05    # 条件付確率が低い遷移は非表示
)
fig.show()
