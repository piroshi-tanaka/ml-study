# cluster_gantt_24h.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Optional, Tuple, Literal
import plotly.graph_objects as go

# ========= ユーティリティ =========
def _ensure_dt(s: pd.Series, name: str) -> pd.Series:
    out = pd.to_datetime(s, errors="coerce")
    if out.isna().any():
        raise ValueError(f"[{name}] に日時変換できない値があります（{int(out.isna().sum())}件）。")
    return out

def _hour_from_day_start(ts: pd.Series, day_start_hour: int) -> pd.Series:
    # 例: day_start_hour=4 → 04:00 を 0 として 0..24 に正規化
    h = ts.dt.hour + ts.dt.minute/60.0 + ts.dt.second/3600.0
    return (h - day_start_hour) % 24

def _service_day(ts: pd.Series, day_start_hour: int) -> pd.Series:
    adj = ts - pd.to_timedelta(day_start_hour, unit="h")
    return adj.dt.normalize()

def _split_by_service_day(df, col_start, col_end, day_start_hour):
    """04:00境界でイベントを分割（ガントが0..24に収まるように）。"""
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

# ========= メイン：ガント風 24h × cluster =========
def plot_cluster_gantt_24h(
    df: pd.DataFrame,
    *,
    # 列名
    col_start: str = "start_time",
    col_end: str = "end_time",
    col_cluster: str = "cluster_id",
    col_type: str = "sessionType",         # "inactive"/"charging" を想定
    col_duration_min: str = "duration_min",
    # 表示・ロジック
    day_start_hour: int = 4,               # 1日の起点（04:00推奨）
    min_stop_minutes: int = 30,            # 30分未満は表示しない（0で無効）
    duration_bins: Tuple[int,int,int] = (60,180,360),  # ≤1h / ≤3h / ≤6h / >6h
    charging_label: str = "charging",
    inactive_label: str = "inactive",
    lane_gap: float = 0.08,                # 同一cluster内で日を少し縦ずらし（重なり軽減）
    title: Optional[str] = None
) -> go.Figure:
    """
    x=0..24h固定, y=cluster_id, 横棒= [start_h, end_h]
    inactiveは放置時間の階級で色分け、chargingは別色・三角マーカーも重ねられるようにする。
    事前に期間は絞り込んでから渡してください（長期だと重なり多くなります）。
    """
    x = df.copy()
    # 基本型
    x[col_start] = _ensure_dt(x[col_start], col_start)
    x[col_end]   = _ensure_dt(x[col_end], col_end)

    # しきい値フィルタ
    if col_duration_min in x.columns:
        x["duration_min"] = pd.to_numeric(x[col_duration_min], errors="coerce")
    else:
        x["duration_min"] = (x[col_end] - x[col_start]).dt.total_seconds()/60.0
    if min_stop_minutes:
        x = x[x["duration_min"] >= min_stop_minutes]

    # 04:00境界で分割
    seg = _split_by_service_day(x, col_start, col_end, day_start_hour)

    # 時刻→0..24h
    seg["start_h"] = _hour_from_day_start(seg[col_start], day_start_hour)
    seg["end_h"]   = _hour_from_day_start(seg[col_end], day_start_hour)
    seg["service_day"] = _service_day(seg[col_start], day_start_hour)
    seg["duration_min"] = pd.to_numeric(seg["duration_min"], errors="coerce")
    seg["duration_h"]   = seg["duration_min"]/60.0

    # 充電/放置で分ける & 放置は時間バケット
    is_chg = seg[col_type].astype(str).str.lower().eq(charging_label)
    is_in  = seg[col_type].astype(str).str.lower().eq(inactive_label)

    seg.loc[is_in, "dur_bucket"] = _bucket_duration_minutes(seg.loc[is_in, "duration_min"], duration_bins)
    seg.loc[is_chg, "dur_bucket"] = "charging"
    cat_order = ["charging", "≤1h", "1–3h", "3–6h", ">6h"]
    seg["dur_bucket"] = pd.Categorical(seg["dur_bucket"], categories=cat_order, ordered=True)

    # 同一cluster内で日ごとに少し縦ずらし（重なり軽減）
    seg = seg.sort_values([col_cluster, "service_day", "start_h"]).reset_index(drop=True)
    seg["cluster_y"] = seg[col_cluster].astype(str)
    # 行毎にサービス日の順序を割り当ててオフセット
    seg["day_rank_in_cluster"] = (
        seg.groupby(col_cluster)["service_day"]
           .rank(method="dense").astype(int) - 1
    )
    # オフセットは 0, +lane_gap, +2*lane_gap, ...（視認性の範囲で）
    seg["y_pos"] = seg["cluster_y"] + ""  # dummy to keep as categorical later
    # Plotlyのcategory軸にオフセットを直接足せないので、テキスト側にはそのまま、
    # 実座標は数値に変換してからオフセットを足すテクニックを使う:
    # 1) 全クラスタの並びを確定
    clusters = seg[col_cluster].astype(str).value_counts().index.tolist()
    cluster_index = {c:i for i,c in enumerate(clusters)}
    seg["y_num"] = seg[col_cluster].astype(str).map(cluster_index).astype(float) + seg["day_rank_in_cluster"]*lane_gap

    # カラーパレット
    palette = {
        "charging": "#E64A19",
        "≤1h": "#9E9E9E",
        "1–3h": "#64B5F6",
        "3–6h": "#1976D2",
        ">6h": "#0D47A1"
    }

    fig = go.Figure()

    # 横棒（hlines）を dur_bucket ごとに描画（Scatterglのlineで表現）
    for key in cat_order:
        d = seg[seg["dur_bucket"] == key]
        if len(d) == 0:
            continue
        # 1イベント=1本の横線（start_h→end_h）
        # まとめて描くために2点ずつ並べる
        x_pairs = np.ravel(d[["start_h","end_h"]].values.T)
        y_pairs = np.ravel(np.column_stack([d["y_num"], d["y_num"]]).T)

        fig.add_trace(go.Scattergl(
            x=x_pairs,
            y=y_pairs,
            mode="lines",
            name=key,
            line=dict(width=(6 if key!="charging" else 5), color=palette.get(key, "#555")),
            opacity=(0.85 if key=="charging" else 0.65),
            hoverinfo="skip"  # 棒自体のホバーは煩雑になりがちなのでオフ（下でマーカーで補う）
        ))

    # 中点マーカー（hover情報用、chargingは三角で少し強調）
    seg["mid_h"] = (seg["start_h"] + seg["end_h"]) / 2.0
    for key in cat_order:
        d = seg[seg["dur_bucket"] == key]
        if len(d) == 0:
            continue
        sym = "triangle-up" if key == "charging" else "circle"
        fig.add_trace(go.Scattergl(
            x=d["mid_h"],
            y=d["y_num"],
            mode="markers",
            name=f"{key} (points)",
            marker=dict(
                size=np.clip(d["duration_h"]*24, 8, 24),
                color=palette.get(key, "#555"),
                symbol=sym,
                line=dict(width=(0.8 if key=="charging" else 0))
            ),
            opacity=(0.95 if key=="charging" else 0.75),
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
            ),
            showlegend=False
        ))

    # レイアウト
    fig.update_layout(
        title=title or f"Gantt-like view: clusters vs 24h (start/end bars, day_start={day_start_hour:02d}:00)",
        width=1100, height=max(480, min(900, int(len(clusters)*28))),
        plot_bgcolor="white",
        legend=dict(orientation="h", x=1, xanchor="right", y=1.05),
        margin=dict(l=120, r=30, t=70, b=60)
    )
    fig.update_xaxes(
        title=f"time of day (h from {day_start_hour:02d}:00)",
        range=[0,24],
        tick0=0, dtick=2,
        showgrid=True, gridcolor="rgba(0,0,0,0.08)"
    )
    # y軸：数値軸→カテゴリラベルに変換（オフセットを保ったままクラスター名表示）
    tick_vals = []
    tick_text = []
    for c, idx in cluster_index.items():
        # 代表位置（オフセット群の中央値付近）に主目盛り
        tick_vals.append(idx + 0.5*lane_gap)  # ちょい上に
        tick_text.append(c)
    fig.update_yaxes(
        title="cluster_id",
        tickmode="array",
        tickvals=tick_vals,
        ticktext=tick_text,
        showgrid=True, gridcolor="rgba(0,0,0,0.10)"
    )
    return fig
    
    
from cluster_gantt_24h import plot_cluster_gantt_24h

# df は事前に期間を絞り込んでおく（例：直近30日）
fig = plot_cluster_gantt_24h(
    df,
    col_start="start_time",
    col_end="end_time",
    col_cluster="cluster_id",
    col_type="sessionType",       # "inactive"/"charging"
    col_duration_min="duration_min",
    day_start_hour=4,             # 04:00起点
    min_stop_minutes=30,          # 30分未満除外（0で無効）
    duration_bins=(60,180,360)    # ≤1h, ≤3h, ≤6h, >6h
)
fig.show()