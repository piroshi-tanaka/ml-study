# temporal_network.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import pandas as pd
import numpy as np
import networkx as nx
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Literal

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =========================================================
# データモデル
# =========================================================
@dataclass
class TemporalMeta:
    cluster_order: List[str]
    y_map: Dict[str, int]
    other_label: str


# =========================================================
# ユーティリティ
# =========================================================
def _ensure_dt(df: pd.DataFrame, col: str) -> pd.Series:
    s = pd.to_datetime(df[col], errors="coerce")
    if s.isna().any():
        bad = s.isna().sum()
        raise ValueError(f"[{col}] に日時へ変換できない値が {bad} 件あります。")
    return s

def _sin_cos_hour(ts: pd.Series) -> pd.DataFrame:
    # 任意：時間帯を連続表現にしたいときに使える（今回は未使用）
    hour = ts.dt.hour + ts.dt.minute/60.0
    rad = 2*np.pi*hour/24.0
    return pd.DataFrame({"hour_sin": np.sin(rad), "hour_cos": np.cos(rad)})

# =========================================================
# ノード/エッジ生成
# =========================================================
def build_temporal_network(
    df: pd.DataFrame,
    *,
    time_col_start: str = "start_time",
    time_col_end: str = "end_time",
    cluster_col: str = "cluster_id",
    type_col: str = "sessionType",              # "inactive" / "charging" / (movingは除外前提)
    duration_col: str = "duration_minutes",
    min_stop_minutes: int = 30,                 # 念のための下限フィルタ（0でオフ）
    long_parking_hours: float = 6.0,            # 放置判定（inactive & >= 6h）
    lane_offset_for_charging: float = 0.45,     # 充電をy方向にオフセット
    top_n_clusters: Optional[int] = None,       # y軸は上位N + OTHER
    other_label: str = "OTHER",
    edge_stride: int = 1,                       # 遷移エッジ間引き
    edge_anchor: Tuple[Literal["start","end"], Literal["start","end"]] = ("end", "start"),
    jitter_std: float = 0.05
) -> Tuple[pd.DataFrame, pd.DataFrame, TemporalMeta]:
    """
    入力 df 必須列: [sessionType, start_time, end_time, cluster_id] + duration_minutes
      - moving と 30分未満は既に除外済み想定だが、min_stop_minutes>0 なら再フィルタ
    返り値:
      nodes: 1行=イベント（滞在 or 充電）
      edges: 連続イベント間の遷移（前の終了→次の開始等）
      meta : y配置の順序など
    """
    d = df.copy()
    d[time_col_start] = _ensure_dt(d, time_col_start)
    d[time_col_end]   = _ensure_dt(d, time_col_end)

    # duration
    if duration_col in d.columns:
        d["duration_min"] = pd.to_numeric(d[duration_col], errors="coerce")
        if d["duration_min"].isna().any():
            raise ValueError(f"[{duration_col}] に数値変換できない値があります。")
    else:
        d["duration_min"] = (d[time_col_end] - d[time_col_start]).dt.total_seconds()/60.0

    d["duration_h"] = d["duration_min"]/60.0

    # 念のためのフィルタ
    if min_stop_minutes > 0:
        d = d[d["duration_min"] >= min_stop_minutes].copy()
    d = d[d[type_col] != "moving"].copy()

    # 時間属性
    d["weekday"] = d[time_col_start].dt.weekday   # 0=Mon .. 6=Sun
    d["month"]   = d[time_col_start].dt.month
    d["t_mid"]   = d[time_col_start] + (d[time_col_end] - d[time_col_start])/2

    # 放置フラグ
    d["is_long_parking"] = (d[type_col].eq("inactive")) & (d["duration_h"] >= long_parking_hours)

    # y軸順序（総滞在時間でソート）
    clus_order = (
        d.groupby(cluster_col)["duration_min"]
         .sum()
         .sort_values(ascending=False)
         .index.tolist()
    )
    if top_n_clusters and len(clus_order) > top_n_clusters:
        keep = set(clus_order[:top_n_clusters])
        d[cluster_col] = np.where(d[cluster_col].isin(keep), d[cluster_col], other_label)
        clus_order = (
            d.groupby(cluster_col)["duration_min"]
             .sum()
             .sort_values(ascending=False)
             .index.tolist()
        )

    y_map = {cid: i for i, cid in enumerate(clus_order)}
    d["y_base"] = d[cluster_col].map(y_map).astype(float)
    d["y"]      = d["y_base"] + np.where(d[type_col].eq("charging"), lane_offset_for_charging, 0.0)
    rng = np.random.default_rng(42)
    d["y_jitter"] = d["y"] + rng.normal(0, jitter_std, size=len(d))

    # 並べ替え・ID
    d = d.sort_values([time_col_start, time_col_end]).reset_index(drop=True)
    d["node_id"] = np.arange(len(d))

    # エッジ作成
    edges: List[Tuple[int,int]] = []
    for i in range(1, len(d), edge_stride):
        edges.append((int(d.loc[i-1, "node_id"]), int(d.loc[i, "node_id"])))
    e = pd.DataFrame(edges, columns=["src", "dst"])

    # アンカー時刻（終了→開始など）
    src_anchor, dst_anchor = edge_anchor
    e = (
        e.merge(d[["node_id", time_col_start, time_col_end, "y_jitter", cluster_col, type_col]],
                left_on="src", right_on="node_id", how="left")
         .drop(columns=["node_id"])
         .rename(columns={time_col_start:"src_start", time_col_end:"src_end",
                          "y_jitter":"y_src", cluster_col:"cluster_src", type_col:"type_src"})
    )
    e = (
        e.merge(d[["node_id", time_col_start, time_col_end, "y_jitter", cluster_col, type_col]],
                left_on="dst", right_on="node_id", how="left")
         .drop(columns=["node_id"])
         .rename(columns={time_col_start:"dst_start", time_col_end:"dst_end",
                          "y_jitter":"y_dst", cluster_col:"cluster_dst", type_col:"type_dst"})
    )
    e["t_src"] = np.where(src_anchor == "start", e["src_start"], e["src_end"])
    e["t_dst"] = np.where(dst_anchor == "start", e["dst_start"], e["dst_end"])

    # nodes 出力
    nodes = (
        d[["node_id", time_col_start, time_col_end, "t_mid", "y_jitter",
           cluster_col, type_col, "duration_min", "duration_h",
           "is_long_parking", "weekday", "month"]]
        .rename(columns={time_col_start:"start_time", time_col_end:"end_time",
                         cluster_col:"cluster_id", type_col:"sessionType"})
        .copy()
    )
    edges = e.copy()
    meta  = TemporalMeta(cluster_order=clus_order, y_map=y_map, other_label=other_label)
    return nodes, edges, meta


# =========================================================
# Plotly 可視化（共通：単一図）
# =========================================================
def plot_temporal_network(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    figsize: Tuple[int,int] = (1200, 450),
    show_duration_bar: bool = True,
    bar_alpha_inactive: float = 0.35,
    bar_alpha_charging: float = 0.3,
    node_size_scale: float = 22.0,
    highlight_long_parking: bool = True,
    ytick_compact: bool = True,
    title: Optional[str] = None
) -> go.Figure:
    """
    x=時間, y=クラスタ（chargingはオフセット済み）
    - 横棒: start→end（inactiveは太め）
    - ノード: 点（chargingは三角 ^ で強調）
    - エッジ: 微細な線（大量データ向けに省略）
    """
    fig = go.Figure()

    # 1) エッジ（軽くグレー）
    if len(edges) > 0:
        fig.add_trace(go.Scattergl(
            x=np.ravel(edges[["t_src","t_dst"]].values.T),
            y=np.ravel(edges[["y_src","y_dst"]].values.T),
            mode="lines",
            line=dict(width=0.7),
            opacity=0.15,
            name="transition",
            hoverinfo="skip",
            showlegend=False
        ))

    # 2) 滞在バー（start→end）
    if show_duration_bar:
        for sess, g in nodes.groupby("sessionType"):
            if sess == "inactive":
                width = 6
                alpha = bar_alpha_inactive
            elif sess == "charging":
                width = 4
                alpha = bar_alpha_charging
            else:
                width = 3
                alpha = 0.25

            fig.add_trace(go.Scattergl(
                x=np.ravel(g[["start_time","end_time"]].values.T),
                y=np.ravel(g[["y_jitter","y_jitter"]].values.T),
                mode="lines",
                line=dict(width=width),
                opacity=alpha,
                name=f"{sess}_bar",
                hoverinfo="skip",
                showlegend=False
            ))

    # 3) ノード（点）
    for sess, g in nodes.groupby("sessionType"):
        sizes = np.clip(g["duration_h"] * node_size_scale, 12, 260)
        marker = dict(size=sizes)
        symbol = "circle"
        if sess == "charging":
            symbol = "triangle-up"
            marker.update(line=dict(width=0.8))

        fig.add_trace(go.Scattergl(
            x=g["t_mid"],
            y=g["y_jitter"],
            mode="markers",
            marker=marker,
            name=f"{sess} (n={len(g)})",
            customdata=np.stack([g["cluster_id"], g["duration_min"]], axis=1),
            hovertemplate=(
                "time=%{x}<br>y=%{y:.2f}<br>cluster=%{customdata[0]}<br>"
                "dur_min=%{customdata[1]:.0f}<extra>%{fullData.name}</extra>"
            ),
            marker_symbol=symbol,
            opacity=0.75 if sess!="charging" else 0.9
        ))

    # 4) 長期放置の強調枠
    if highlight_long_parking and "is_long_parking" in nodes.columns:
        gp = nodes[nodes["is_long_parking"]]
        if len(gp) > 0:
            sizes = np.clip(gp["duration_h"] * node_size_scale, 28, 320)
            fig.add_trace(go.Scattergl(
                x=gp["t_mid"], y=gp["y_jitter"],
                mode="markers",
                marker=dict(size=sizes, symbol="circle-open", line=dict(width=1.4)),
                name="long_parking(>=6h)",
                hoverinfo="skip",
                opacity=0.95
            ))

    # 軸・レイアウト
    fig.update_layout(
        title=title or "Temporal Network",
        width=figsize[0], height=figsize[1],
        plot_bgcolor="white",
        legend=dict(orientation="h", x=1, xanchor="right", y=1.1),
        margin=dict(l=60, r=20, t=60, b=40)
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)",
                     title_text="cluster (lane offset for charging)")

    # ytick を整数段に丸めてラベル（代表クラスタ名）
    if ytick_compact:
        y_round = nodes["y_jitter"].round()
        ticks = np.sort(y_round.unique())
        labels = []
        tmp = nodes.copy()
        tmp["y_round"] = y_round
        for yv in ticks:
            labels.append(tmp.loc[tmp["y_round"].eq(yv), "cluster_id"].value_counts().index[0])
        fig.update_yaxes(tickmode="array", tickvals=ticks, ticktext=labels)

    return fig


# =========================================================
# ファセット（曜日 / 月）— 同一描画関数を使い回し
# =========================================================
def facet_temporal_network(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    by: Literal["weekday","month"] = "weekday",
    order: Optional[List[int]] = None,
    max_cols: int = 3,
    figsize_per: Tuple[int,int] = (900, 320),
    title_prefix: str = ""
) -> go.Figure:
    keys = sorted(nodes[by].unique().tolist()) if order is None else order
    n = len(keys)
    ncols = min(max_cols, n)
    nrows = int(np.ceil(n / ncols))

    fig = make_subplots(rows=nrows, cols=ncols, shared_xaxes=False, shared_yaxes=False,
                        horizontal_spacing=0.04, vertical_spacing=0.08,
                        subplot_titles=[f"{by}={k}" for k in keys])

    r = c = 1
    for i, k in enumerate(keys):
        n_sub = nodes[nodes[by].eq(k)]
        ids = set(n_sub["node_id"])
        e_sub = edges[edges["src"].isin(ids) & edges["dst"].isin(ids)]
        sub = plot_temporal_network(n_sub, e_sub, figsize=figsize_per, title=None)

        # 各トレースをサブプロットへ移送
        for tr in sub.data:
            fig.add_trace(tr, row=r, col=c)

        # 軸体裁（軽め）
        fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.07)", row=r, col=c)
        fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.07)", row=r, col=c)

        c += 1
        if c > ncols:
            c = 1
            r += 1

    fig.update_layout(
        title=title_prefix + f"Facet by {by}",
        width=figsize_per[0]*ncols,
        height=figsize_per[1]*nrows + 60,
        showlegend=False,
        plot_bgcolor="white",
        margin=dict(l=60, r=20, t=80, b=40)
    )
    return fig


# =========================================================
# 定量評価：遷移確率・エントロピー・中心性・条件付き放置
# =========================================================
def compute_transition_stats(
    nodes: pd.DataFrame,
    edges: pd.DataFrame
) -> Dict[str, pd.DataFrame | float]:
    """
    出力:
      - trans_matrix: prev x next の 回数/確率
      - entropy_by_prev: H(next|prev)
      - p_long_after_charge / p_long_after_noncharge
      - centrality: 遷移グラフの中心性（入次数/出次数/PageRank など）
      - top_transitions: 上位遷移
    """
    e = (
        edges.merge(nodes[["node_id","cluster_id","sessionType","is_long_parking"]],
                    left_on="src", right_on="node_id", how="left")
             .rename(columns={"cluster_id":"cluster_src","sessionType":"type_src","is_long_parking":"long_src"})
             .drop(columns=["node_id"])
    )
    e = (
        e.merge(nodes[["node_id","cluster_id","sessionType","is_long_parking"]],
                left_on="dst", right_on="node_id", how="left")
             .rename(columns={"cluster_id":"cluster_dst","sessionType":"type_dst","is_long_parking":"long_dst"})
             .drop(columns=["node_id"])
    )

    # 遷移行列
    counts = e.groupby(["cluster_src","cluster_dst"]).size().rename("n").reset_index()
    totals = counts.groupby("cluster_src")["n"].sum().rename("row_total").reset_index()
    mat = counts.merge(totals, on="cluster_src", how="left")
    mat["p"] = mat["n"]/mat["row_total"]

    # エントロピー H(next|prev)
    entropy_df = (
        mat.groupby("cluster_src")["p"]
           .apply(lambda s: float(-(s*np.log2(np.clip(s, 1e-12, 1))).sum()))
           .rename("H_next_given_prev").reset_index()
    )

    # 充電直後に long parking になる確率
    charge_edges = e[e["type_src"].eq("charging")]
    noncharge_edges = e[~e["type_src"].eq("charging")]
    p_long_after_charge = float(charge_edges["long_dst"].mean()) if len(charge_edges)>0 else np.nan
    p_long_after_noncharge = float(noncharge_edges["long_dst"].mean()) if len(noncharge_edges)>0 else np.nan

    # グラフ中心性（重み=回数）
    G = nx.DiGraph()
    for _, r in counts.iterrows():
        G.add_edge(r["cluster_src"], r["cluster_dst"], weight=int(r["n"]))
    in_deg  = pd.Series(dict(G.in_degree(weight="weight")),  name="in_degree_w")
    out_deg = pd.Series(dict(G.out_degree(weight="weight")), name="out_degree_w")
    try:
        pr = pd.Series(nx.pagerank(G, weight="weight"), name="pagerank")
    except Exception:
        pr = pd.Series(dtype=float, name="pagerank")

    centrality = (
        pd.concat([in_deg, out_deg, pr], axis=1)
          .fillna(0.0)
          .sort_values("pagerank", ascending=False)
          .reset_index().rename(columns={"index":"cluster_id"})
    )

    top_transitions = mat.sort_values("n", ascending=False).head(30).reset_index(drop=True)

    return {
        "trans_matrix": mat,
        "entropy_by_prev": entropy_df,
        "p_long_after_charge": p_long_after_charge,
        "p_long_after_noncharge": p_long_after_noncharge,
        "centrality": centrality,
        "top_transitions": top_transitions
    }


def compute_transition_stats_by(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    by: Literal["weekday","month"] = "weekday"
) -> pd.DataFrame:
    """
    曜日・月などで条件分割したときの p_long_after_charge などを算出。
    返り値: key, p_long_after_charge, p_long_after_noncharge, diff
    """
    out = []
    for key, n_sub in nodes.groupby(by):
        ids = set(n_sub["node_id"])
        e_sub = edges[edges["src"].isin(ids) & edges["dst"].isin(ids)]
        st = compute_transition_stats(n_sub, e_sub)
        out.append({
            by: key,
            "p_long_after_charge": st["p_long_after_charge"],
            "p_long_after_noncharge": st["p_long_after_noncharge"],
            "diff": (st["p_long_after_charge"] - st["p_long_after_noncharge"])
        })
    return pd.DataFrame(out).sort_values(by).reset_index(drop=True)


# =========================================================
# 使い方（サンプル）
# =========================================================
if __name__ == "__main__":
    # 1) セッションデータを読み込む
    # df = pd.read_parquet("sessions.parquet")
    # 必須列: sessionType, start_time, end_time, cluster_id, duration_minutes

    # # 2) ノード/エッジ生成
    # nodes, edges, meta = build_temporal_network(
    #     df,
    #     top_n_clusters=12,
    #     lane_offset_for_charging=0.5,
    #     edge_stride=1,
    #     edge_anchor=("end","start")  # 前の終了 → 次の開始で接続（推奨）
    # )

    # # 3) 全期間の図
    # fig = plot_temporal_network(nodes, edges, title="All Period")
    # fig.show()

    # # 4) 曜日ファセット
    # fig_w = facet_temporal_network(nodes, edges, by="weekday", order=[0,1,2,3,4,5,6],
    #                                title_prefix="User A — ")
    # fig_w.show()

    # # 5) 月別ファセット
    # fig_m = facet_temporal_network(nodes, edges, by="month", order=list(range(1,13)),
    #                                title_prefix="User A — ")
    # fig_m.show()

    # # 6) 定量評価
    # stats = compute_transition_stats(nodes, edges)
    # print(stats["centrality"].head(10))
    # print(stats["entropy_by_prev"].sort_values("H_next_given_prev").head(10))
    # print("p_long_after_charge =", stats["p_long_after_charge"])
    # print("p_long_after_noncharge =", stats["p_long_after_noncharge"])

    pass