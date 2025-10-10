# stay_time_24h_plot.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict
from sklearn.decomposition import PCA
import plotly.graph_objects as go

# ---------------------------
# 前処理：座標→相対m座標 & PCA軸
# ---------------------------
def _to_relative_meters(lat: pd.Series, lon: pd.Series) -> Tuple[pd.Series, pd.Series, float, float]:
    lat0, lon0 = lat.median(), lon.median()  # 全体の中央値を基準
    x_m = (lon - lon0) * 111_320 * np.cos(np.deg2rad(lat0))
    y_m = (lat - lat0) * 110_540
    return x_m, y_m, lat0, lon0

def _pca_axis(x_m: pd.Series, y_m: pd.Series) -> pd.Series:
    pca = PCA(n_components=1)
    axis = pca.fit_transform(pd.DataFrame({"x": x_m, "y": y_m}))
    # シリーズで返す
    return pd.Series(axis.reshape(-1), index=x_m.index)

def _hour_from_day_start(ts: pd.Series, day_start_hour: int) -> pd.Series:
    # 例: day_start_hour=4 なら 04:00→0h として 0..24 に正規化
    h = ts.dt.hour + ts.dt.minute/60.0 + ts.dt.second/3600.0
    h = (h - day_start_hour) % 24
    return h

# ---------------------------
# 放置時間の階級化（充電は別扱い）
# ---------------------------
def _bucket_duration_minutes(
    duration_min: pd.Series,
    bins: Tuple[int, int, int] = (60, 180, 360)
) -> pd.Series:
    """
    bins = (<=1h, <=3h, <=6h, >6h)
    """
    b1, b2, b3 = bins
    labels = []
    for v in duration_min:
        if v <= b1:
            labels.append("≤1h")
        elif v <= b2:
            labels.append("1–3h")
        elif v <= b3:
            labels.append("3–6h")
        else:
            labels.append(">6h")
    return pd.Series(labels, index=duration_min.index, dtype="category")

# ---------------------------
# メイン：24h×位置PCA軸の散布図
# ---------------------------
def plot_stay_area_24h(
    df: pd.DataFrame,
    *,
    # 列名の指定
    col_start_time: str = "start_time",
    col_lat: str = "lat",
    col_lon: str = "lon",
    col_session_type: str = "sessionType",     # "inactive" / "charging"
    col_duration_min: str = "duration_min",
    # 表示設定
    day_start_hour: int = 4,                   # 4:00起点で 0..24h
    size_scale: float = 20.0,                  # バブルサイズ = duration_h * size_scale
    duration_bins: Tuple[int, int, int] = (60, 180, 360),  # 1h/3h/6hの境界（分）
    charging_label: str = "charging",
    inactive_label: str = "inactive",
    title: Optional[str] = None
) -> go.Figure:
    """
    目的：
      x = 時刻(0..24h fixed), y = 位置PCA軸（全体中央値基準）
      色 = 放置時間の階級（inactive） / 充電は別色
      マーカー形状 = chargingを三角、inactiveを丸
      サイズ = 滞在時間（分）に比例

    使い方：
      fig = plot_stay_area_24h(df, col_start_time="start", col_lat="latitude", col_lon="longitude", ...)
      fig.show()
    """
    x = df.copy()

    # 必須列チェック
    need = {col_start_time, col_lat, col_lon, col_session_type, col_duration_min}
    miss = need - set(x.columns)
    if miss:
        raise KeyError(f"必要列が不足しています: {miss}")

    # 型整形
    x[col_start_time] = pd.to_datetime(x[col_start_time], errors="coerce")
    if x[col_start_time].isna().any():
        raise ValueError(f"{col_start_time} に日時変換できない値があります。")

    # 1) 時刻→0..24h
    x["hour24"] = _hour_from_day_start(x[col_start_time], day_start_hour)

    # 2) 相対メートル座標 & PCA軸
    x_m, y_m, _, _ = _to_relative_meters(x[col_lat].astype(float), x[col_lon].astype(float))
    x["pos_axis"] = _pca_axis(x_m, y_m)

    # 3) 放置時間の階級（inactiveのみ色分け）
    x["duration_min"] = pd.to_numeric(x[col_duration_min], errors="coerce")
    if x["duration_min"].isna().any():
        raise ValueError(f"{col_duration_min} に数値変換できない値があります。")

    # charging / inactive を分ける
    is_chg = x[col_session_type].astype(str).str.lower().eq(charging_label)
    is_inact = x[col_session_type].astype(str).str.lower().eq(inactive_label)

    # バブルサイズ（滞在時間→時間に換算）
    x["duration_h"] = x["duration_min"] / 60.0
    x["size"] = np.clip(x["duration_h"] * size_scale, 10, 260)

    # 色カテゴリ（inactiveのみ）
    x.loc[is_inact, "dur_bucket"] = _bucket_duration_minutes(x.loc[is_inact, "duration_min"], duration_bins)
    # charging は固定ラベル
    x.loc[is_chg, "dur_bucket"] = "charging"

    # 表示順（凡例順）
    cat_order = ["charging", "≤1h", "1–3h", "3–6h", ">6h"]
    x["dur_bucket"] = pd.Categorical(x["dur_bucket"], categories=cat_order, ordered=True)

    # 4) Plotlyトレースを積む（charging→inactive順）
    fig = go.Figure()
    palette = {
        "charging": "#E64A19",  # 濃いオレンジ（固定）
        "≤1h": "#9E9E9E",       # グレー
        "1–3h": "#64B5F6",      # ライトブルー
        "3–6h": "#1976D2",      # ブルー
        ">6h": "#0D47A1"        # 濃いブルー
    }

    for key in cat_order:
        d = x[x["dur_bucket"] == key]
        if len(d) == 0:
            continue
        symbol = "triangle-up" if key == "charging" else "circle"
        fig.add_trace(go.Scattergl(
            x=d["hour24"],
            y=d["pos_axis"],
            mode="markers",
            name=key,
            marker=dict(
                size=d["size"],
                color=palette.get(key, "#555"),
                symbol=symbol,
                line=dict(width=0.8 if key == "charging" else 0)
            ),
            opacity=0.85 if key == "charging" else 0.75,
            customdata=np.stack([
                d[col_session_type].astype(str),
                d["duration_min"].astype(float).round(0)
            ], axis=1),
            hovertemplate="time=%{x:.2f}h<br>pos=%.2f<br>type=%{customdata[0]}<br>dur=%{customdata[1]} min<extra>%{fullData.name}</extra>" % 0
        ))

    fig.update_layout(
        title=title or "24h × 滞在エリア（PCA軸）— 放置時間で色分け / 充電は別色",
        width=1100, height=520,
        plot_bgcolor="white",
        legend=dict(orientation="h", x=1, xanchor="right", y=1.08),
        margin=dict(l=60, r=20, t=60, b=50)
    )
    fig.update_xaxes(
        title=f"time of day (h from {day_start_hour:02d}:00)",
        range=[0, 24],
        showgrid=True, gridcolor="rgba(0,0,0,0.08)"
    )
    fig.update_yaxes(
        title="position axis (PCA, median-centered)",
        showgrid=True, gridcolor="rgba(0,0,0,0.08)"
    )
    return fig
    
# df は 30分未満/moving 除外済みを想定
# 必須列: start_time, lat, lon, sessionType, duration_min
from stay_time_24h_plot import plot_stay_area_24h

fig = plot_stay_area_24h(
    df,
    col_start_time="start_time",
    col_lat="lat",
    col_lon="lon",
    col_session_type="sessionType",   # "inactive" / "charging"
    col_duration_min="duration_min",
    day_start_hour=4,                 # 04:00起点で 0..24h
    duration_bins=(60, 180, 360)      # ≤1h, ≤3h, ≤6h, >6h
)
fig.show()