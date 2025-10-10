# timebin_occupancy_and_transitions.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Optional, Tuple, List, Dict, Literal
import plotly.graph_objects as go

# =============== 基本ユーティリティ ===============
def _ensure_dt(s: pd.Series, name: str) -> pd.Series:
    out = pd.to_datetime(s, errors="coerce")
    if out.isna().any():
        raise ValueError(f"[{name}] に日時変換できない値があります（{int(out.isna().sum())}件）。")
    return out

def _service_day(ts: pd.Series, day_start_hour: int) -> pd.Series:
    # 04:00 起点の“サービス日”キー（その日の00:00を返す）
    adj = ts - pd.to_timedelta(day_start_hour, unit="h")
    return adj.dt.normalize()

def _hour_from_day_start(ts: pd.Series, day_start_hour: int) -> pd.Series:
    h = ts.dt.hour + ts.dt.minute/60.0 + ts.dt.second/3600.0
    return (h - day_start_hour) % 24

def _split_by_service_day(df, col_start, col_end, day_start_hour):
    """セッションをサービス日境界（day_start_hour）で分割。"""
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

# =============== 24hをbin化して滞在を集計 ===============
def bin_occupancy_and_transitions(
    df: pd.DataFrame,
    *,
    col_start: str = "start_time",
    col_end: str = "end_time",
    col_cluster: str = "cluster_id",
    col_type: str = "sessionType",            # "inactive"/"charging"/"moving" 等
    col_duration_min: Optional[str] = "duration_min",
    day_start_hour: int = 4,
    bin_minutes: int = 30,                    # 15 or 30 がおすすめ
    min_stop_minutes: int = 30,               # 30分未満を除外（0で無効）
    drop_moving: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    出力:
      - occ:  timebin出現（service_day, bin_idx, bin_start_h, cluster_id, dwell_min, count）
      - trans: timebin遷移（bin_idx, prev_cluster, next_cluster, n, p）
      - meta: bins（bin境界hの配列）, nbins
    """
    assert 1440 % bin_minutes == 0, "bin_minutesは24hを割り切れる値にしてください（例: 15, 30, 60）"
    nbins = 1440 // bin_minutes
    bin_edges_min = np.arange(0, 1440+1, bin_minutes)
    bin_edges_h   = bin_edges_min / 60.0
    # 入力整形
    x = df.copy()
    x[col_start] = _ensure_dt(x[col_start], col_start)
    x[col_end]   = _ensure_dt(x[col_end], col_end)
    # duration
    if col_duration_min and (col_duration_min in x.columns):
        x["duration_min"] = pd.to_numeric(x[col_duration_min], errors="coerce")
    else:
        x["duration_min"] = (x[col_end] - x[col_start]).dt.total_seconds()/60.0
    if min_stop_minutes:
        x = x[x["duration_min"] >= min_stop_minutes].copy()
    if drop_moving and (col_type in x.columns):
        x = x[x[col_type].astype(str).str.lower() != "moving"].copy()

    # 04:00境界で分割
    seg = _split_by_service_day(x, col_start, col_end, day_start_hour)
    # 0..24h に変換（分単位）
    seg["start_h"] = _hour_from_day_start(seg[col_start], day_start_hour)
    seg["end_h"]   = _hour_from_day_start(seg[col_end], day_start_hour)
    seg["service_day"] = _service_day(seg[col_start], day_start_hour)
    seg["start_min"] = (seg["start_h"] * 60).astype(float)
    seg["end_min"]   = (seg["end_h"]   * 60).astype(float)

    # bin按分：各セッションを含むbinに「重なり分の分数」を配分
    rows = []
    for _, r in seg.iterrows():
        s = float(r["start_min"]); e = float(r["end_min"])
        if e <= s:  # 0分イベントは無視（または微小なら1分扱い可）
            continue
        # bin index 範囲
        b0 = int(np.floor(s / bin_minutes))
        b1 = int(np.floor((e-1e-9) / bin_minutes))
        for b in range(b0, b1+1):
            left  = b * bin_minutes
            right = (b+1) * bin_minutes
            ov = max(0.0, min(e, right) - max(s, left))  # overlap in minutes
            if ov <= 0: continue
            rows.append((
                r["service_day"], b, bin_edges_h[b],  # bin開始のh
                str(r[col_cluster]),
                ov,  # dwell_min
                1    # count（発生フラグとして）
            ))
    if not rows:
        raise ValueError("該当データがありません（フィルタが厳しすぎる可能性）。")

    occ = pd.DataFrame(rows, columns=["service_day","bin_idx","bin_start_h","cluster_id","dwell_min","count"])
    # 同一(service_day, bin, cluster)で集約
    occ = (occ.groupby(["service_day","bin_idx","bin_start_h","cluster_id"])
              .agg(dwell_min=("dwell_min","sum"), count=("count","sum"))
              .reset_index())

    # ---- bin時系列から遷移を抽出（サービス日内で隣接binのclusterを比較）----
    # 各(service_day, bin_idx)で「最も滞在分が多いクラスタ」を代表にする（並列イベントが稀にある場合のtie-break）
    rep = (occ.sort_values(["service_day","bin_idx","dwell_min"], ascending=[True, True, False])
              .drop_duplicates(["service_day","bin_idx"])
              .sort_values(["service_day","bin_idx"])
              .reset_index(drop=True))
    # rep: service_day, bin_idx, bin_start_h, cluster_id
    rep["next_cluster"] = rep.groupby("service_day")["cluster_id"].shift(-1)
    rep["prev_cluster"] = rep.groupby("service_day")["cluster_id"].shift(+1)

    # 遷移カウント（隣接binの prev→next）
    trans_pairs = rep.dropna(subset=["cluster_id","next_cluster"]).copy()
    trans_pairs = trans_pairs[trans_pairs["cluster_id"] != trans_pairs["next_cluster"]]
    trans_pairs = trans_pairs.rename(columns={"cluster_id":"prev_cluster"})
    trans_pairs["next_cluster"] = trans_pairs["next_cluster"].astype(str)

    # timebinごとに prev→next を集計（bin_idxは prev のbin）
    trans = (trans_pairs.groupby(["bin_idx","prev_cluster","next_cluster"])
                        .size().rename("n").reset_index())
    trans["row_total"] = trans.groupby(["bin_idx","prev_cluster"])["n"].transform("sum")
    trans["p"] = trans["n"] / trans["row_total"]

    meta = dict(nbins=nbins, bin_minutes=bin_minutes, bin_edges_h=bin_edges_h)
    return dict(occ=occ, trans=trans, rep=rep, meta=meta)

# =============== 可視化：出現ヒートマップ ===============
def plot_occ_heatmap(
    occ: pd.DataFrame,
    *,
    top_k_clusters: Optional[int] = 20,
    agg: Literal["dwell_min","count"]="dwell_min",
    title: Optional[str] = None
) -> go.Figure:
    x = occ.copy()
    # クラスタの上位だけ残して見やすく
    if top_k_clusters:
        keep = x.groupby("cluster_id")[agg].sum().sort_values(ascending=False).head(top_k_clusters).index
        x = x[x["cluster_id"].isin(keep)]
    pivot = (x.groupby(["cluster_id","bin_idx"])[agg].sum()
               .unstack(fill_value=0.0))
    # bin_idx → 時刻ラベル（h）
    cols = pivot.columns
    tick_h = (cols.values * (x["bin_start_h"].unique()[1]-x["bin_start_h"].unique()[0])) if "bin_start_h" in x.columns else cols
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=cols, y=pivot.index,
        colorscale="YlGnBu",
        colorbar=dict(title=agg)
    ))
    fig.update_layout(
        title=title or f"クラスタ出現ヒートマップ（値={agg}）",
        width=1000, height=max(450, min(900, 26*len(pivot))),
        margin=dict(l=140, r=40, t=60, b=50),
        plot_bgcolor="white"
    )
    fig.update_xaxes(title="time bin (0..24h / bin index)", tickmode="auto")
    fig.update_yaxes(title="cluster_id")
    return fig

# =============== 可視化：時間帯別の遷移確率（ヒートマップ） ===============
def plot_transition_heatmap_by_timebin(
    trans: pd.DataFrame,
    *,
    bin_minutes: int,
    bin_window: int = 4,  # 例: 4なら2時間幅（30分binなら2h）
    min_count: int = 3,
    top_k_prev: int = 12,
    top_k_next: int = 12,
    title: Optional[str] = None
) -> go.Figure:
    """
    timebinを中心に±(bin_window//2)の範囲をまとめて、遷移確率行列を作る。
    → 連続binのゆらぎを吸収して読みやすく。
    """
    t = trans.copy()
    figs = []
    frames = []

    nbins = int(1440 // bin_minutes)
    # 可読ラベル
    def bin_label(b):
        h0 = (b * bin_minutes) / 60.0
        h1 = ((b+1) * bin_minutes) / 60.0
        return f"[{h0:.1f}-{h1:.1f})h"

    for b in range(nbins):
        # 窓で抽出
        half = bin_window // 2
        sel = list(range(max(0,b-half), min(nbins, b+half+1)))
        sub = t[t["bin_idx"].isin(sel)].copy()
        if len(sub)==0:
            frames.append(dict(name=bin_label(b), data=[]))
            continue
        # 上位 prev/next 限定
        keep_prev = sub.groupby("prev_cluster")["n"].sum().sort_values(ascending=False).head(top_k_prev).index
        keep_next = sub.groupby("next_cluster")["n"].sum().sort_values(ascending=False).head(top_k_next).index
        sub = sub[sub["prev_cluster"].isin(keep_prev) & sub["next_cluster"].isin(keep_next)]
        sub = sub[sub["n"] >= min_count]

        if len(sub)==0:
            frames.append(dict(name=bin_label(b), data=[]))
            continue

        # 条件付き確率 p(next|prev)
        mat = (sub.groupby(["prev_cluster","next_cluster"])["n"].sum()
                    .reset_index())
        mat["row_total"] = mat.groupby("prev_cluster")["n"].transform("sum")
        mat["p"] = mat["n"] / mat["row_total"]
        pivot = mat.pivot(index="prev_cluster", columns="next_cluster", values="p").fillna(0.0)

        fr = go.Heatmap(
            z=pivot.values, x=pivot.columns, y=pivot.index,
            zmin=0, zmax=1, colorscale="Blues", colorbar=dict(title="p(next|prev)")
        )
        frames.append(dict(name=bin_label(b), data=[fr]))

    # 初期フレーム
    label0 = bin_label(0)
    fig = go.Figure(frames=frames)
    if frames and frames[0]["data"]:
        fig.add_traces(frames[0]["data"])

    fig.update_layout(
        title=title or f"時間binごとの遷移確率（p(next|prev)） — {label0}",
        width=1000, height=700,
        margin=dict(l=160, r=40, t=70, b=50),
        updatemenus=[dict(
            type="dropdown", x=1, xanchor="right", y=1.12, yanchor="top",
            buttons=[dict(label=fr["name"], method="update",
                          args=[{"visible":[True]},
                                {"title":f"時間binごとの遷移確率（p(next|prev)） — {fr['name']}"}]) for fr in frames]
        )],
        plot_bgcolor="white"
    )
    fig.update_xaxes(title="next_cluster")
    fig.update_yaxes(title="prev_cluster")
    return fig

# =============== 可視化：時間bin別 Sankey（主要フロー） ===============
def plot_timebin_sankey(
    trans: pd.DataFrame,
    *,
    bin_idx: int,
    bin_minutes: int,
    top_k_prev: int = 8,
    top_k_next: int = 8,
    min_count: int = 3,
    title: Optional[str] = None
) -> go.Figure:
    h0 = (bin_idx * bin_minutes)/60.0
    h1 = ((bin_idx+1) * bin_minutes)/60.0
    label = f"[{h0:.1f}-{h1:.1f})h"
    sub = trans[trans["bin_idx"] == bin_idx].copy()
    if len(sub)==0:
        raise ValueError("該当binの遷移がありません。")
    prev_top = sub.groupby("prev_cluster")["n"].sum().sort_values(ascending=False).head(top_k_prev).index
    next_top = sub.groupby("next_cluster")["n"].sum().sort_values(ascending=False).head(top_k_next).index
    sub = sub[sub["prev_cluster"].isin(prev_top) & sub["next_cluster"].isin(next_top)]
    sub = sub[sub["n"] >= min_count]
    if len(sub)==0:
        raise ValueError("フィルタ後に表示可能な遷移がありません（top_kやmin_countを緩めてください）。")

    prevs = sub["prev_cluster"].unique().tolist()
    nexts = sub["next_cluster"].unique().tolist()
    nodes = prevs + nexts
    idx = {n:i for i,n in enumerate(nodes)}
    link = dict(
        source=[idx[p] for p in sub["prev_cluster"]],
        target=[idx[n] for n in sub["next_cluster"]],
        value=sub["n"].tolist()
    )
    fig = go.Figure(data=[go.Sankey(node=dict(label=nodes, pad=12, thickness=16), link=link)])
    fig.update_layout(
        title=title or f"Sankey of transitions at {label}",
        width=1000, height=600,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig


from timebin_occupancy_and_transitions import (
    bin_occupancy_and_transitions,
    plot_occ_heatmap,
    plot_transition_heatmap_by_timebin,
    plot_timebin_sankey
)

# 1) 15分binで集約（04:00起点）
bundle = bin_occupancy_and_transitions(
    df,
    col_start="start_time",
    col_end="end_time",
    col_cluster="cluster_id",
    col_type="sessionType",
    col_duration_min="duration_min",
    day_start_hour=4,
    bin_minutes=15,       # ← 15 or 30
    min_stop_minutes=30,
    drop_moving=True
)
occ   = bundle["occ"]
trans = bundle["trans"]
meta  = bundle["meta"]

# 2) クラスタ出現ヒートマップ（値=滞在分）
fig_occ = plot_occ_heatmap(occ, top_k_clusters=20, agg="dwell_min")
fig_occ.show()

# 3) 時間binごとの遷移確率（ドロップダウンでbin切替）
fig_tr = plot_transition_heatmap_by_timebin(
    trans, bin_minutes=meta["bin_minutes"], bin_window=4,  # 例: 15分bin×4=1時間窓
    min_count=3, top_k_prev=12, top_k_next=12
)
fig_tr.show()

# 4) 任意の時間binの主要フローをSankeyで
fig_sk = plot_timebin_sankey(trans, bin_idx=40, bin_minutes=meta["bin_minutes"],
                             top_k_prev=8, top_k_next=8, min_count=3)
fig_sk.show()
