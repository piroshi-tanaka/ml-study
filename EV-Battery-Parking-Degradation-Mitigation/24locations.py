# cluster_gantt_24h_transitions.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Optional, Tuple
import plotly.graph_objects as go

# ===== util =====
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

def _split_by_service_day(df, col_start, col_end, day_start_hour):
    d = df.copy()
    d[col_start] = _ensure_dt(d[col_start], col_start)
    d[col_end]   = _ensure_dt(d[col_end], col_end)
    assert (d[col_end] >= d[col_start]).all(), "end_time < start_time の行があります。"

    out = []
    for _, r in d.iterrows():
        s = r[col_start]; e = r[col_end]
        cur = s
        while True:
            day0 = _service_day(pd.Series([cur]), day_start_hour).iloc[0] + pd.Timedelta(hours=day_start_hour)
            day1 = day0 + pd.Timedelta(days=1)
            seg_end = min(e, day1)
            rr = r.copy()
            rr[col_start] = cur
            rr[col_end]   = seg_end
            out.append(rr)
            if seg_end >= e:
                break
            cur = seg_end
    return pd.DataFrame(out).reset_index(drop=True)

def _bucket_duration_minutes(v: pd.Series,
                             bins: Tuple[int,int,int]=(60,180,360)) -> pd.Series:
    b1,b2,b3 = bins
    lab = np.empty(len(v), dtype=object)
    arr = v.values.astype(float)
    lab[arr <= b1] = "≤1h"
    lab[(arr > b1) & (arr <= b2)] = "1–3h"
    lab[(arr > b2) & (arr <= b3)] = "3–6h"
    lab[arr > b3] = ">6h"
    return pd.Series(lab, index=v.index, dtype="object")

# ===== main =====
def plot_cluster_gantt_24h_with_transitions(
    df: pd.DataFrame,
    *,
    # 列名
    col_start: str = "start_time",
    col_end: str = "end_time",
    col_cluster: str = "cluster_id",
    col_type: str = "sessionType",            # "inactive"/"charging"
    col_duration_min: str = "duration_min",
    # 表示・ロジック
    day_start_hour: int = 4,                  # 04:00起点で 0..24h
    min_stop_minutes: int = 30,               # 30分未満は除外（0で無効）
    duration_bins: Tuple[int,int,int] = (60,180,360),
    charging_label: str = "charging",
    inactive_label: str = "inactive",
    # 見た目
    cluster_spacing: float = 1.8,             # ← クラスタ段の“縦間隔”を拡大
    lane_gap: float = 0.10,                   # 同クラスタ内で日を少し縦ずらし
    line_color: str = "rgba(0,0,0,0.3)",      # 遷移線は固定色
    line_width: float = 1.2,
    title: Optional[str] = None
) -> go.Figure:
    x = df.copy()
    x[col_start] = _ensure_dt(x[col_start], col_start)
    x[col_end]   = _ensure_dt(x[col_end], col_end)

    # duration
    if col_duration_min in x.columns:
        x["duration_min"] = pd.to_numeric(x[col_duration_min], errors="coerce")
    else:
        x["duration_min"] = (x[col_end] - x[col_start]).dt.total_seconds()/60.0
    if min_stop_minutes:
        x = x[x["duration_min"] >= min_stop_minutes].copy()
    x["duration_h"] = x["duration_min"]/60.0

    # 04:00境界で分割 → 0..24h へ
    seg = _split_by_service_day(x, col_start, col_end, day_start_hour)
    seg["service_day"] = _service_day(seg[col_start], day_start_hour)
    seg["start_h"] = _hour_from_day_start(seg[col_start], day_start_hour)
    seg["end_h"]   = _hour_from_day_start(seg[col_end], day_start_hour)
    seg["mid_h"]   = (seg["start_h"] + seg["end_h"]) / 2.0
    seg["duration_min"] = pd.to_numeric(seg["duration_min"], errors="coerce")
    seg["duration_h"]   = seg["duration_min"] / 60.0

    # 色ラベル（点の色）：chargingは固定、inactiveは時間バケット
    is_chg = seg[col_type].astype(str).str.lower().eq(charging_label)
    is_in  = seg[col_type].astype(str).str.lower().eq(inactive_label)
    seg.loc[is_in,  "dot_color_cat"] = _bucket_duration_minutes(seg.loc[is_in, "duration_min"], duration_bins)
    seg.loc[is_chg, "dot_color_cat"] = "charging"
    cat_order = ["charging", "≤1h", "1–3h", "3–6h", ">6h"]
    seg["dot_color_cat"] = pd.Categorical(seg["dot_color_cat"], categories=cat_order, ordered=True)

    # y座標：クラスタ段を“拡大”
    clusters = seg[col_cluster].astype(str).value_counts().index.tolist()
    c_to_idx = {c:i for i,c in enumerate(clusters)}
    # 基本段（クラスタ位置）
    seg["y_base"] = seg[col_cluster].astype(str).map(c_to_idx).astype(float) * cluster_spacing
    # 同クラスタ内でサービス日ごとに少しずらす（重なり軽減）
    seg = seg.sort_values([col_cluster, "service_day", "start_h"]).reset_index(drop=True)
    seg["day_rank_in_cluster"] = seg.groupby(col_cluster)["service_day"].rank("dense").astype(int) - 1
    seg["y_num"] = seg["y_base"] + seg["day_rank_in_cluster"] * lane_gap

    # パレット（点の色）
    palette = {
        "charging": "#E64A19",
        "≤1h": "#9E9E9E",
        "1–3h": "#64B5F6",
        "3–6h": "#1976D2",
        ">6h": "#0D47A1"
    }

    fig = go.Figure()

    # 1) ガント棒（start_h→end_h）：薄め
    for key in cat_order:
        d = seg[seg["dot_color_cat"] == key]
        if len(d) == 0:
            continue
        x_pairs = np.ravel(d[["start_h","end_h"]].values.T)
        y_pairs = np.ravel(np.column_stack([d["y_num"], d["y_num"]]).T)
        fig.add_trace(go.Scattergl(
            x=x_pairs, y=y_pairs, mode="lines",
            name=f"{key} bar",
            line=dict(width=(6 if key!="charging" else 5), color=palette.get(key, "#666")),
            opacity=(0.60 if key!="charging" else 0.75),
            hoverinfo="skip",
            showlegend=False
        ))

    # 2) 1日ごとの“遷移線”を接続（固定色）
    #    各 service_day 内で start_h 昇順に並べ、(mid_h, y_num) を折れ線で繋ぐ
    xs, ys = [], []
    for day, g in seg.groupby("service_day", sort=True):
        g = g.sort_values("start_h")
        xs.extend(g["mid_h"].tolist()); ys.extend(g["y_num"].tolist())
        xs.append(None); ys.append(None)  # 日ごとの切れ目
    fig.add_trace(go.Scattergl(
        x=xs, y=ys, mode="lines",
        name="daily transitions",
        line=dict(width=line_width, color=line_color),
        hoverinfo="skip",
        showlegend=True
    ))

    # 3) 中点の“丸（or 三角）”＝色はバケットで、形はchargingのみ三角
    for key in cat_order:
        d = seg[seg["dot_color_cat"] == key]
        if len(d) == 0:
            continue
        sym = "triangle-up" if key == "charging" else "circle"
        fig.add_trace(go.Scattergl(
            x=d["mid_h"], y=d["y_num"],
            mode="markers",
            name=key,
            marker=dict(
                size=np.clip(d["duration_h"]*24, 8, 26),
                color=palette.get(key, "#555"),
                symbol=sym,
                line=dict(width=(0.8 if key=="charging" else 0))
            ),
            opacity=(0.95 if key=="charging" else 0.80),
            customdata=np.stack([
                d[col_cluster].astype(str),
                d["service_day"].dt.strftime("%Y-%m-%d"),
                d["start_h"].round(2),
                d["end_h"].round(2),
                d["duration_min"].round(0)
            ], axis=1),
            hovertemplate=(
                "cluster=%{customdata[0]}<br>"
                "day=%{customdata[1]}<br>"
                "start=%{customdata[2]}h → end=%{customdata[3]}h<br>"
                "dur=%{customdata[4]} min"
                "<extra>%{fullData.name}</extra>"
            )
        ))

    # 軸/レイアウト
    fig.update_layout(
        title=title or f"Gantt (clusters vs 24h) + Daily Transition Lines  (day_start={day_start_hour:02d}:00)",
        width=1150,
        height=max(520, min(1100, int(len(clusters)*38))),  # クラスタ数に応じて高さ調整
        plot_bgcolor="white",
        legend=dict(orientation="h", x=1, xanchor="right", y=1.06),
        margin=dict(l=140, r=28, t=70, b=60)
    )
    fig.update_xaxes(
        title=f"time of day (h from {day_start_hour:02d}:00)",
        range=[0,24], tick0=0, dtick=2,
        showgrid=True, gridcolor="rgba(0,0,0,0.08)"
    )
    # y軸は“数値座標”だが目盛りをクラスタ名に置換
    tick_vals = []
    tick_text = []
    for c, idx in c_to_idx.items():
        tick_vals.append(idx*cluster_spacing + 0.5*lane_gap)  # 段の代表位置
        tick_text.append(str(c))
    fig.update_yaxes(
        title="cluster_id",
        tickmode="array", tickvals=tick_vals, ticktext=tick_text,
        showgrid=True, gridcolor="rgba(0,0,0,0.10)"
    )
    return fig
    
from cluster_gantt_24h_transitions import plot_cluster_gantt_24h_with_transitions

fig = plot_cluster_gantt_24h_with_transitions(
    df,
    col_start="start_time",
    col_end="end_time",
    col_cluster="cluster_id",
    col_type="sessionType",         # "inactive"/"charging"
    col_duration_min="duration_min",
    day_start_hour=4,               # 04:00起点
    cluster_spacing=2.2,            # ← クラスタ段の縦間隔をさらに広く
    lane_gap=0.12,                  # 同クラスタ内の“日”のズレ幅
    line_color="rgba(33,150,243,0.5)",  # 線は固定色（例：青透明）
    line_width=1.5
)
fig.show()