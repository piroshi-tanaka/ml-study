# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# =========================================================
# 1) ノード・エッジ生成（改良版）
# =========================================================
def build_temporal_network(
    df,
    time_col_start="start_time",
    time_col_end="end_time",
    cluster_col="cluster_id",
    type_col="sessionType",
    min_stop_minutes=30,
    long_parking_hours=6,
    lane_offset_for_charging=0.35,     # chargingを上下に少しずらして見やすく
    top_n_clusters=None,               # y軸には上位Nクラスタ + "OTHER"
    other_label="OTHER",
    assume_single_vehicle=True,
    vehicle_col=None,
    edge_stride=1                      # エッジ間引き: 1=全部, 2=1/2本, 3=1/3本...
):
    """
    入力df 必須列: [sessionType, start_time, end_time, cluster_id]
    出力: nodes_df, edges_df, meta
    """
    d = df.copy()
    d[time_col_start] = pd.to_datetime(d[time_col_start])
    d[time_col_end]   = pd.to_datetime(d[time_col_end])
    d["duration_min"] = (d[time_col_end] - d[time_col_start]).dt.total_seconds() / 60.0
    d["duration_h"]   = d["duration_min"] / 60.0
    d = d[d["duration_min"] >= min_stop_minutes].copy()
    d = d[d[type_col] != "moving"].copy()  # moving除去済みならそのまま

    # 追加の時間属性
    d["weekday"] = d[time_col_start].dt.weekday   # 0=Mon ... 6=Sun
    d["month"]   = d[time_col_start].dt.month     # 1..12

    # 長期放置フラグ
    d["is_long_parking"] = (d["duration_h"] >= long_parking_hours) & (d[type_col] == "inactive")

    # x軸はイベントの中点
    d["t_mid"] = d[time_col_start] + (d[time_col_end] - d[time_col_start]) / 2

    # クラスタ出現量で並べ替え（上位N適用のため）
    clus_order = d.groupby(cluster_col)["duration_min"].sum().sort_values(ascending=False).index.tolist()

    if top_n_clusters is not None and top_n_clusters > 0 and len(clus_order) > top_n_clusters:
        top_set = set(clus_order[:top_n_clusters])
        d[cluster_col] = d[cluster_col].where(d[cluster_col].isin(top_set), other_label)
        # 再計算
        clus_order = d.groupby(cluster_col)["duration_min"].sum().sort_values(ascending=False).index.tolist()

    # y 位置マップ
    y_map = {cid: i for i, cid in enumerate(clus_order)}
    d["y_base"] = d[cluster_col].map(y_map).astype(float)

    # chargingだけ上下にオフセットしてレーンを分けて視認性UP
    is_chg = (d[type_col] == "charging").astype(float)
    d["y"] = d["y_base"] + is_chg * lane_offset_for_charging

    # 軽いジッター（重なり低減）
    rng = np.random.default_rng(42)
    d["y_jitter"] = d["y"] + rng.normal(0, 0.05, size=len(d))

    # 並べる・ID付与
    d = d.sort_values("t_mid").reset_index(drop=True)
    d["node_id"] = np.arange(len(d))

    # エッジ（連続イベント間）
    edges = []
    if assume_single_vehicle or vehicle_col is None:
        # 前後接続
        for i in range(1, len(d), edge_stride):
            edges.append((d.loc[i-1, "node_id"], d.loc[i, "node_id"]))
    else:
        # 複数車両時は車両内で接続
        for _, g in d.sort_values("t_mid").groupby(vehicle_col):
            idx = g.index.to_list()
            for p, c in zip(idx[:-1:edge_stride], idx[1::edge_stride]):
                edges.append((d.loc[p, "node_id"], d.loc[c, "node_id"]))

    edges_df = pd.DataFrame(edges, columns=["src", "dst"])
    # 位置情報を結合
    edges_df = edges_df.merge(
        d[["node_id", "t_mid", "y_jitter", cluster_col, type_col]],
        left_on="src", right_on="node_id", how="left"
    ).rename(columns={"t_mid":"t_src", "y_jitter":"y_src",
                      cluster_col:"cluster_src", type_col:"type_src"}).drop(columns=["node_id"])
    edges_df = edges_df.merge(
        d[["node_id", "t_mid", "y_jitter", cluster_col, type_col]],
        left_on="dst", right_on="node_id", how="left"
    ).rename(columns={"t_mid":"t_dst", "y_jitter":"y_dst",
                      cluster_col:"cluster_dst", type_col:"type_dst"}).drop(columns=["node_id"])

    # 出力ノード（描画最小限）
    nodes_df = d[[
        "node_id", "t_mid", "y_jitter", cluster_col, type_col,
        "duration_h", "is_long_parking", "weekday", "month"
    ]].copy()

    meta = {"cluster_order": clus_order, "y_map": y_map, "other_label": other_label}
    return nodes_df, edges_df, meta


# =========================================================
# 2) 共通描画関数（1枚描画）
# =========================================================
def plot_temporal_network(
    nodes,
    edges,
    cluster_col="cluster_id",
    type_col="sessionType",
    figsize=(14, 6),
    alpha_nodes=0.6,
    alpha_edges=0.12,
    size_scale=24.0,
    highlight_long_parking=True,
    month_locator=True
):
    """
    x=時間, y=クラスタ（chargingは少しオフセット）
    点サイズ=滞在時間, 点色=セッション種別, 線=遷移
    """
    fig, ax = plt.subplots(figsize=figsize)

    # エッジ（軽量）
    if len(edges) > 0:
        ax.plot(
            np.ravel(edges[["t_src","t_dst"]].values.T),
            np.ravel(edges[["y_src","y_dst"]].values.T),
            linewidth=0.7, alpha=alpha_edges
        )

    # ノード
    for s_type, g in nodes.groupby(type_col):
        sizes = np.clip(g["duration_h"] * size_scale, 10, 220)
        ax.scatter(g["t_mid"], g["y_jitter"], s=sizes, alpha=alpha_nodes, label=f"{s_type} (n={len(g)})")

    # 長期放置強調
    if highlight_long_parking and "is_long_parking" in nodes.columns:
        gp = nodes[nodes["is_long_parking"]]
        if len(gp) > 0:
            ax.scatter(
                gp["t_mid"], gp["y_jitter"],
                s=np.clip(gp["duration_h"] * size_scale, 30, 280),
                facecolors="none", edgecolors="black", linewidths=1.2, alpha=0.95,
                label="long_parking(>=6h)"
            )

    # 軸体裁
    ax.set_ylabel("cluster")
    ax.set_xlabel("time")
    ax.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.5)
    if month_locator:
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        fig.autofmt_xdate()

    # ytickは概ね整数段に（代表ラベルを自動推定）
    y_round = nodes["y_jitter"].round()
    y_ticks = np.sort(y_round.unique())
    ax.set_yticks(y_ticks)

    labels = []
    tmp = nodes.copy()
    tmp["y_round"] = y_round
    for yv in y_ticks:
        lab = tmp.loc[tmp["y_round"] == yv, cluster_col].value_counts().index[0]
        labels.append(lab)
    ax.set_yticklabels(labels)

    ax.legend(loc="upper right", frameon=False, ncol=2)
    plt.tight_layout()
    return fig, ax


# =========================================================
# 3) ファセット描画（曜日別・月別）
# =========================================================
def facet_temporal_network_by(
    nodes,
    edges,
    by="weekday",                 # "weekday" or "month"
    order=None,                   # 表示順（例：weekdayなら [0,1,2,3,4,5,6]）
    max_cols=3,
    cluster_col="cluster_id",
    type_col="sessionType",
    figsize_per_facet=(12, 4),
    **plot_kwargs
):
    """
    nodes/edgesをフィルタして小 multiples（ファセット）を描く。
    例：by='weekday' で 7枚、by='month' で 12枚。
    """
    valid_by = {"weekday", "month"}
    assert by in valid_by, f"`by` must be one of {valid_by}"

    keys = sorted(nodes[by].dropna().unique().tolist()) if order is None else order
    n = len(keys)
    ncols = min(max_cols, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_per_facet[0]*ncols, figsize_per_facet[1]*nrows), squeeze=False)
    axes = axes.ravel()

    for i, key in enumerate(keys):
        ax = axes[i]
        # フィルタ
        n_sub = nodes[nodes[by] == key]
        if len(n_sub) == 0:
            ax.axis("off")
            continue
        # エッジも対応するノードに限定
        node_ids = set(n_sub["node_id"].tolist())
        e_sub = edges[edges["src"].isin(node_ids) & edges["dst"].isin(node_ids)].copy()

        # 個別描画（凡例は最初だけ）
        show_legend = (i == 0)
        _plot_temporal_on_ax(
            ax, n_sub, e_sub, cluster_col=cluster_col, type_col=type_col,
            show_legend=show_legend, **plot_kwargs
        )
        title = f"{by}={key}"
        if by == "weekday":
            # 0=Mon..6=Sun を見やすく
            title = f"weekday={key} (Mon=0)"
        ax.set_title(title)

    # 余白の空きaxを消す
    for j in range(i+1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    return fig, axes


def _plot_temporal_on_ax(
    ax, nodes, edges,
    cluster_col="cluster_id",
    type_col="sessionType",
    alpha_nodes=0.6,
    alpha_edges=0.12,
    size_scale=24.0,
    highlight_long_parking=True,
    month_locator=False
):
    # エッジ
    if len(edges) > 0:
        ax.plot(
            np.ravel(edges[["t_src","t_dst"]].values.T),
            np.ravel(edges[["y_src","y_dst"]].values.T),
            linewidth=0.6, alpha=alpha_edges
        )
    # ノード
    for s_type, g in nodes.groupby(type_col):
        sizes = np.clip(g["duration_h"] * size_scale, 10, 220)
        ax.scatter(g["t_mid"], g["y_jitter"], s=sizes, alpha=alpha_nodes, label=f"{s_type} (n={len(g)})")
    # 長期放置
    if highlight_long_parking and "is_long_parking" in nodes.columns:
        gp = nodes[nodes["is_long_parking"]]
        if len(gp) > 0:
            ax.scatter(
                gp["t_mid"], gp["y_jitter"],
                s=np.clip(gp["duration_h"] * size_scale, 30, 280),
                facecolors="none", edgecolors="black", linewidths=1.1, alpha=0.95
            )

    ax.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.5)
    # x軸フォーマット（ファセットはラベル密度を抑える）
    if month_locator:
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    else:
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))

    # ytick
    y_round = nodes["y_jitter"].round()
    y_ticks = np.sort(y_round.unique())
    ax.set_yticks(y_ticks)
    labels = []
    tmp = nodes.copy()
    tmp["y_round"] = y_round
    for yv in y_ticks:
        lab = tmp.loc[tmp["y_round"] == yv, cluster_col].value_counts().index[0]
        labels.append(lab)
    ax.set_yticklabels(labels)

    # 凡例は必要なときだけ（外でまとめるなら非表示）
    # ax.legend(loc="upper right", frameon=False, ncol=2)


# =========================================================
# 4) 使い方例
# =========================================================
# sessions: あなたの前処理済みデータ
#   必須列: sessionType(moving/charging/inactive), start_time, end_time, cluster_id
#
# nodes, edges, meta = build_temporal_network(
#     sessions,
#     top_n_clusters=10,            # y軸に上位10クラスタ+OTHER
#     lane_offset_for_charging=0.35,
#     edge_stride=2                 # エッジを半分に間引き
# )
#
# # 全期間1枚
# fig, ax = plot_temporal_network(nodes, edges)
#
# # 曜日ファセット
# fig2, axes2 = facet_temporal_network_by(nodes, edges, by="weekday", order=[0,1,2,3,4,5,6], max_cols=3)
#
# # 月別ファセット
# fig3, axes3 = facet_temporal_network_by(nodes, edges, by="month", order=list(range(1,13)), max_cols=3)
