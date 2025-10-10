# spatial_time_views.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Optional, Tuple, Literal, Dict
import plotly.graph_objects as go
import plotly.express as px

# ---------------------------
# ユーティリティ
# ---------------------------
def _ensure_dt(s: pd.Series) -> pd.Series:
    out = pd.to_datetime(s, errors="coerce")
    if out.isna().any():
        raise ValueError("時刻列に日時変換できない値があります。")
    return out

def _hour_from_day_start(ts: pd.Series, day_start_hour: int) -> pd.Series:
    h = ts.dt.hour + ts.dt.minute/60.0 + ts.dt.second/3600.0
    return (h - day_start_hour) % 24

def _to_relative_xy(lat: pd.Series, lon: pd.Series, center: Literal["median","mean"]="median") -> tuple[pd.Series, pd.Series, float, float]:
    lat = lat.astype(float); lon = lon.astype(float)
    lat0 = lat.median() if center=="median" else lat.mean()
    lon0 = lon.median() if center=="median" else lon.mean()
    x_m = (lon - lon0) * 111_320 * np.cos(np.deg2rad(lat0))   # 東西[m]
    y_m = (lat - lat0) * 110_540                               # 南北[m]
    return x_m, y_m, float(lat0), float(lon0)

def _bucket_duration_minutes(duration_min: pd.Series,
                             bins: Tuple[int,int,int]=(60,180,360)) -> pd.Series:
    b1,b2,b3 = bins
    lab = np.empty(len(duration_min), dtype=object)
    v = duration_min.values.astype(float)
    lab[v <= b1] = "≤1h"
    lab[(v > b1) & (v <= b2)] = "1–3h"
    lab[(v > b2) & (v <= b3)] = "3–6h"
    lab[v > b3] = ">6h"
    return pd.Series(lab, index=duration_min.index, dtype="object")

# =========================================================
# 1) 空間×時間色相マップ（位置固定 / 時刻で着色）
# =========================================================
def plot_spatial_time_scatter(
    df: pd.DataFrame,
    *,
    col_start_time: str = "start_time",
    col_lat: str = "lat",
    col_lon: str = "lon",
    col_session_type: str = "sessionType",   # "inactive"/"charging"
    col_duration_min: str = "duration_min",
    day_start_hour: int = 4,
    center: Literal["median","mean"]="median",
    size_scale: float = 0.12,                # バブル直径係数（分に掛ける）
    duration_bins: Tuple[int,int,int]=(60,180,360),
    charging_label: str = "charging",
    inactive_label: str = "inactive",
    title: Optional[str] = None
) -> go.Figure:
    x = df.copy()
    # 必須列チェック
    need = {col_start_time, col_lat, col_lon, col_session_type, col_duration_min}
    miss = need - set(x.columns)
    if miss:
        raise KeyError(f"必要列が不足: {miss}")

    # 型整形
    x[col_start_time] = _ensure_dt(x[col_start_time])
    x["hour24"] = _hour_from_day_start(x[col_start_time], day_start_hour)
    x["duration_min"] = pd.to_numeric(x[col_duration_min], errors="coerce")
    if x["duration_min"].isna().any():
        raise ValueError(f"{col_duration_min} に数値変換できない値があります。")

    # 位置を固定座標へ（中央値/平均基準）
    x["x_m"], x["y_m"], lat0, lon0 = _to_relative_xy(x[col_lat], x[col_lon], center=center)

    # マーカー設定
    x["size"] = np.clip(x["duration_min"] * size_scale, 6, 36)

    # カラー：inactive=時刻Hue、charging=別色（形状三角）
    is_chg = x[col_session_type].astype(str).str.lower().eq(charging_label)
    is_in  = x[col_session_type].astype(str).str.lower().eq(inactive_label)

    fig = go.Figure()

    # inactive: 色=時刻（HSVみたいな循環に近いカラースケールを使用）
    if is_in.any():
        g = x[is_in]
        fig.add_trace(go.Scattergl(
            x=g["x_m"], y=g["y_m"], mode="markers",
            name="inactive (color=hour)",
            marker=dict(
                size=g["size"],
                color=g["hour24"],                 # 0..24
                colorscale="Turbo",                # 視認性の高い循環寄りスケール
                cmin=0, cmax=24,
                colorbar=dict(title="hour (0–24)")
            ),
            customdata=np.stack([
                g["hour24"].round(2),
                g["duration_min"].round(0)
            ], axis=1),
            hovertemplate=(
                "x=%{x:.0f} m, y=%{y:.0f} m<br>"
                "hour=%{customdata[0]} h<br>"
                "dur=%{customdata[1]} min"
                "<extra>inactive</extra>"
            ),
            opacity=0.80
        ))

    # charging: 固定色＋三角、点を少し大きく
    if is_chg.any():
        g = x[is_chg].copy()
        # 充電イベントも時間ヒント欲しければ色を固定せず別トレースで色=hourにしてもOK
        fig.add_trace(go.Scattergl(
            x=g["x_m"], y=g["y_m"], mode="markers",
            name="charging",
            marker=dict(
                size=np.clip(g["size"]*1.2, 6, 42),
                color="#E64A19",
                symbol="triangle-up",
                line=dict(width=0.8)
            ),
            customdata=np.stack([
                g["hour24"].round(2),
                g["duration_min"].round(0)
            ], axis=1),
            hovertemplate=(
                "x=%{x:.0f} m, y=%{y:.0f} m<br>"
                "hour=%{customdata[0]} h<br>"
                "dur=%{customdata[1]} min"
                "<extra>charging</extra>"
            ),
            opacity=0.95
        ))

    fig.update_layout(
        title=title or f"位置固定（中央値基準）× 時刻色相マップ（day_start={day_start_hour:02d}:00）",
        width=900, height=700,
        plot_bgcolor="white",
        legend=dict(orientation="h", x=1, xanchor="right", y=1.08),
        margin=dict(l=50, r=30, t=70, b=50)
    )
    fig.update_xaxes(title="East-West [m] (centered)", showgrid=True, zeroline=True, zerolinecolor="rgba(0,0,0,0.2)")
    fig.update_yaxes(title="South-North [m] (centered)", showgrid=True, zeroline=True, zerolinecolor="rgba(0,0,0,0.2)")
    return fig

# =========================================================
# 2) 固定グリッド × 時間ヒートマップ（セルごとの利用時刻）
# =========================================================
def plot_grid_hour_heatmap(
    df: pd.DataFrame,
    *,
    col_start_time: str = "start_time",
    col_lat: str = "lat",
    col_lon: str = "lon",
    col_session_type: str = "sessionType",
    col_duration_min: str = "duration_min",
    day_start_hour: int = 4,
    center: Literal["median","mean"]="median",
    cell_size_m: float = 250.0,       # メッシュ1セルの一辺[m]
    agg: Literal["count","duration_h"] = "duration_h",
    filter_session: Optional[Literal["inactive","charging"]] = None,
    title: Optional[str] = None
) -> go.Figure:
    """
    位置を等間隔グリッドに割って、各セル×時刻(0..23h)の利用をヒートマップに。
    - 行: 空間セル（北→南、同緯度内は西→東）でソート
    - 列: 時(0..23)
    - 値: 件数 or 滞在時間合計(時間)
    """
    x = df.copy()
    # 型
    x[col_start_time] = _ensure_dt(x[col_start_time])
    x["hour24"] = _hour_from_day_start(x[col_start_time], day_start_hour).astype(int)
    x["duration_min"] = pd.to_numeric(x[col_duration_min], errors="coerce")
    if x["duration_min"].isna().any():
        raise ValueError(f"{col_duration_min} に数値変換できない値があります。")
    x["duration_h"] = x["duration_min"] / 60.0

    # 位置固定
    x["x_m"], x["y_m"], _, _ = _to_relative_xy(x[col_lat], x[col_lon], center=center)

    # セッション絞り込み（None=両方）
    if filter_session:
        x = x[x[col_session_type].astype(str).str.lower().eq(filter_session)]

    # グリッド割り当て
    cx = np.floor(x["x_m"] / cell_size_m).astype(int)
    cy = np.floor(x["y_m"] / cell_size_m).astype(int)
    x["cell_id"] = list(zip(cy, cx))  # 行優先：北(大y)→南(小y)で並べやすい

    # ピボット
    if agg == "count":
        mat = x.groupby(["cell_id","hour24"]).size().rename("val").reset_index()
    else:
        mat = x.groupby(["cell_id","hour24"])["duration_h"].sum().rename("val").reset_index()

    # セル順序（北→南, 西→東）
    cells = pd.DataFrame(mat["cell_id"].unique(), columns=["cell"]).assign(
        cy=lambda d: d["cell"].apply(lambda t: t[0]),
        cx=lambda d: d["cell"].apply(lambda t: t[1])
    ).sort_values(["cy","cx"], ascending=[False, True])  # 北ほど行が上
    cells["row_id"] = np.arange(len(cells))
    mat = mat.merge(cells[["cell","row_id"]], left_on="cell_id", right_on="cell", how="left")

    # ピボット（行=cell、列=hour）
    pivot = mat.pivot(index="row_id", columns="hour24", values="val").fillna(0.0)
    pivot = pivot.sort_index()

    # ラベル（左側におおよその座標を付けたい場合）
    row_labels = cells.sort_values("row_id")["cell"].apply(lambda t: f"y{t[0]} x{t[1]}")

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=row_labels,
        colorscale="YlOrRd",
        colorbar=dict(title=("count" if agg=="count" else "hours"))
    ))
    fig.update_layout(
        title=title or f"固定グリッド×時間ヒートマップ（cell={cell_size_m:.0f}m, agg={agg}, filter={filter_session or 'both'}）",
        width=900, height=max(400, min(900, 20*len(row_labels))),
        plot_bgcolor="white",
        margin=dict(l=120, r=30, t=60, b=50)
    )
    fig.update_xaxes(title="hour (0–23, from day_start)")
    fig.update_yaxes(title="spatial cells (north→south, west→east)")
    return fig

from spatial_time_views import plot_spatial_time_scatter, plot_grid_hour_heatmap

# 位置固定 + 時刻色相（充電は三角・別色）
fig1 = plot_spatial_time_scatter(
    df,
    col_start_time="start_time",
    col_lat="lat",
    col_lon="lon",
    col_session_type="sessionType",
    col_duration_min="duration_min",
    day_start_hour=4,
    center="median",                 # 全体の中央値を原点に
    size_scale=0.12                  # 分ベースの点サイズ係数
)
fig1.show()

# 固定グリッド × 時間ヒートマップ（放置のみ/滞在時間合計）
fig2 = plot_grid_hour_heatmap(
    df,
    col_start_time="start_time",
    col_lat="lat",
    col_lon="lon",
    col_session_type="sessionType",
    col_duration_min="duration_min",
    day_start_hour=4,
    cell_size_m=250,                 # メッシュサイズ（m）
    agg="duration_h",                # or "count"
    filter_session="inactive"        # "charging" も可、Noneで両方
)
fig2.show()