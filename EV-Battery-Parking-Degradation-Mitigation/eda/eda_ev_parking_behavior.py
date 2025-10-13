"""
EV バッテリ劣化抑制・放置予測のための再利用可能な EDA パイプライン。

`データ抽出Step.md` で定義された分析要件を満たすことを目的とし、
CLI からの実行・Jupyter Notebook からの `%run` 双方を想定している。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from scipy.spatial.distance import jensenshannon

# 15 分粒度でタイムスロット展開を行う（仕様より）。
DEFAULT_FREQ_MINUTES = 15

# 日本語フォントの候補。環境に存在するものが自動的に利用される。
JAPANESE_FONT_CANDIDATES = [
    "IPAexGothic",
    "IPAPGothic",
    "Hiragino Sans",
    "Yu Gothic",
    "YuGothic",
    "Meiryo",
    "Noto Sans CJK JP",
    "Noto Sans JP",
    "TakaoPGothic",
    "MS Gothic",
    "sans-serif",
]


def setup_plot_style(font_candidates: Iterable[str] | None = None) -> None:
    """Configure seaborn style and Japanese-font fallbacks for plots."""
    sns.set_theme(style="whitegrid")
    candidates = font_candidates or JAPANESE_FONT_CANDIDATES
    plt.rcParams["font.family"] = candidates
    plt.rcParams["axes.unicode_minus"] = False  # マイナス記号の豆腐化を防ぐ

# Backward compatibility for prior notebook code
_setup_plot_style = setup_plot_style


def load_sessions(csv_path: Path) -> pd.DataFrame:
    """
    セッション CSV を読み込み、時刻列を `datetime` に変換する。

    Returns
    -------
    pandas.DataFrame
        仕様で定義された全列を含む DataFrame。
    """
    df = pd.read_csv(csv_path)
    for col in ["start_time", "end_time"]:
        series = pd.to_datetime(df[col], errors="coerce")
        # タイムゾーン付きの場合は単純化のためローカル時刻に変換する。
        if pd.api.types.is_datetime64tz_dtype(series.dtype):
            df[col] = series.dt.tz_localize(None)
        else:
            df[col] = series
    return df


def prepare_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """
    解析用の派生列（曜日、時間帯、前後セッション、充電後フラグなど）を作成する。
    """
    df = df.copy()
    df = df.sort_values(["hashvin", "start_time"]).reset_index(drop=True)

    # 時間情報から曜日・時刻などの基本特徴量を生成。
    df["duration_minutes"] = df["duration_minutes"].astype(float)
    df["weekday"] = df["start_time"].dt.dayofweek
    df["start_hour"] = df["start_time"].dt.hour
    df["date"] = df["start_time"].dt.date

    # 長時間放置を「inactive かつ 6 時間以上」として定義。
    df["is_long_park"] = (df["session_type"] == "inactive") & (
        df["duration_minutes"] >= 360
    )

    # 同一車両内での前後セッション情報を付与。
    grouped = df.groupby("hashvin", group_keys=False)
    df["prev_session_type"] = grouped["session_type"].shift(1)
    df["prev_cluster"] = grouped["session_cluster"].shift(1)
    df["prev_is_long_park"] = grouped["is_long_park"].shift(1)
    df["next_session_type"] = grouped["session_type"].shift(-1)
    df["next_cluster"] = grouped["session_cluster"].shift(-1)
    df["next_is_long_park"] = grouped["is_long_park"].shift(-1)

    # 充電直後の長時間放置を識別するフラグ。
    df["after_charge"] = df["is_long_park"] & (df["prev_session_type"] == "charging")

    # 充電セッションでのみ利用する補助列。
    df["charge_start_hour"] = np.where(
        df["session_type"] == "charging", df["start_hour"], np.nan
    )
    df["charge_cluster"] = np.where(
        df["session_type"] == "charging", df["session_cluster"], np.nan
    )
    return df


def ensure_dirs(output_root: Path) -> Tuple[Path, Path]:
    """
    グラフ／表の保存先ディレクトリを用意する。
    """
    plots_dir = output_root / "plots"
    tables_dir = output_root / "tables"
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir, tables_dir


def step1_long_park_distribution(
    df: pd.DataFrame,
    plots_dir: Path,
    tables_dir: Path,
    per_vehicle_top_n: int = 10,
    global_top_k: int = 5,
) -> List[int]:
    """
    Step1: 長時間放置（6h 以上）の滞在クラスタ分布を棒グラフで可視化。

    Returns
    -------
    list of int
        Step2 で掘り下げる上位クラスタ ID。
    """
    long_df = df[df["is_long_park"]].copy()
    if long_df.empty:
        return []

    # 全体での滞在時間（分）を集計し、CSV 出力。
    cluster_totals = (
        long_df.groupby("session_cluster")["duration_minutes"]
        .sum()
        .sort_values(ascending=False)
    )
    cluster_totals_hours = (cluster_totals / 60).rename("total_hours")
    cluster_totals_hours.to_csv(
        tables_dir / "long_park_cluster_total_hours.csv", header=True
    )

    # 車両ごとに上位クラスタを棒グラフで保存。
    for hashvin, group in long_df.groupby("hashvin"):
        top_series = (
            group.groupby("session_cluster")["duration_minutes"]
            .sum()
            .sort_values(ascending=False)
            .head(per_vehicle_top_n)
            / 60.0
        )
        if top_series.empty:
            continue

        plt.figure(figsize=(10, 4))
        top_series.sort_values().plot(kind="barh", color="#1f77b4")
        plt.xlabel("累積滞在時間 [時間]")
        plt.ylabel("クラスタ ID")
        plt.title(f"{hashvin} の長時間放置クラスタ上位 {per_vehicle_top_n}")
        plt.tight_layout()
        plt.savefig(plots_dir / f"bar_cluster_distribution_{hashvin}.png", dpi=220)
        plt.close()

    # 全体上位クラスタ（Step2 の対象）を選定。
    focus_clusters = cluster_totals.index.tolist()[:global_top_k]
    return focus_clusters


def _explode_time_slots(
    df: pd.DataFrame,
    freq_minutes: int = DEFAULT_FREQ_MINUTES,
) -> pd.DataFrame:
    """
    長時間放置セッションを指定粒度（既定 15 分）のスロット単位に展開する。
    """
    df = df.copy()
    freq_str = f"{freq_minutes}T"
    df["time_slots"] = df.apply(
        lambda row: pd.date_range(
            start=row["start_time"],
            end=row["end_time"],
            freq=freq_str,
            inclusive="left",
        )
        if pd.notnull(row["start_time"]) and pd.notnull(row["end_time"])
        else pd.NaT,
        axis=1,
    )
    expanded = df.explode("time_slots").dropna(subset=["time_slots"])
    expanded["slot_weekday"] = expanded["time_slots"].dt.dayofweek
    expanded["slot_hour"] = expanded["time_slots"].dt.hour
    expanded["slot_quarter"] = (
        expanded["time_slots"].dt.hour + expanded["time_slots"].dt.minute / 60.0
    )
    return expanded


def _build_percentage_annotation(
    percent_matrix: pd.DataFrame,
    numerator_matrix: pd.DataFrame,
    denominator_matrix: pd.DataFrame,
    value_label: str,
) -> pd.DataFrame:
    """割合（%）と分子／母数を併記する注釈 DataFrame を生成する。"""
    annot = percent_matrix.copy().astype(float)
    for idx in annot.index:
        for col in annot.columns:
            value = float(annot.loc[idx, col]) if col in annot.columns else 0.0
            numerator = (
                int(numerator_matrix.loc[idx, col])
                if idx in numerator_matrix.index and col in numerator_matrix.columns
                else 0
            )
            denominator = (
                int(denominator_matrix.loc[idx, col])
                if idx in denominator_matrix.index and col in denominator_matrix.columns
                else 0
            )
            if denominator == 0:
                annot.loc[idx, col] = "0.0%\n(分子=0,母数=0)"
            else:
                annot.loc[idx, col] = f"{value:.1f}%\n(分子={numerator},母数={denominator})"
    annot.attrs["value_label"] = value_label
    return annot


def _build_diff_annotation(
    diff_percent_matrix: pd.DataFrame,
    after_numerators: pd.DataFrame,
    after_denominators: pd.DataFrame,
    all_numerators: pd.DataFrame,
    all_denominators: pd.DataFrame,
) -> pd.DataFrame:
    """差分ヒートマップ用の注釈（%ポイント＋分子／母数）を生成する。"""
    annot = diff_percent_matrix.copy().astype(float)
    for idx in annot.index:
        for col in annot.columns:
            diff_val = float(annot.loc[idx, col]) if col in annot.columns else 0.0
            after_num = (
                int(after_numerators.loc[idx, col])
                if idx in after_numerators.index and col in after_numerators.columns
                else 0
            )
            after_den = (
                int(after_denominators.loc[idx, col])
                if idx in after_denominators.index and col in after_denominators.columns
                else 0
            )
            all_num = (
                int(all_numerators.loc[idx, col])
                if idx in all_numerators.index and col in all_numerators.columns
                else 0
            )
            all_den = (
                int(all_denominators.loc[idx, col])
                if idx in all_denominators.index and col in all_denominators.columns
                else 0
            )
            annot.loc[idx, col] = (
                f"{diff_val:.1f}pt\n後={after_num}/{after_den}\n全={all_num}/{all_den}"
            )
    annot.attrs["value_label"] = "差分 (ポイント)"
    return annot


def _plot_heatmap(
    matrix: pd.DataFrame,
    annot: pd.DataFrame,
    title: str,
    x_label: str,
    y_label: str,
    colorbar_label: str,
    path: Path,
    cmap: str = "YlGnBu",
    center: float | None = None,
) -> None:
    """共通のヒートマップ描画ロジック（数値注釈付き）。"""
    if matrix.empty:
        return

    plt.figure(figsize=(12, 4))
    sns.heatmap(
        matrix,
        cmap=cmap,
        center=center,
        linewidths=0.5,
        linecolor="white",
        annot=annot.values,
        fmt="",
        annot_kws={"fontsize": 8},
        cbar_kws={"label": colorbar_label},
    )
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def step2_heatmaps_and_charge_effects(
    df: pd.DataFrame,
    focus_clusters: Iterable[int],
    plots_dir: Path,
    tables_dir: Path,
    freq_minutes: int = DEFAULT_FREQ_MINUTES,
) -> None:
    """
    Step2: 上位クラスタの曜日×時間帯ヒートマップと、充電開始時刻→次放置先の条件付き分布。
    """
    long_df = df[df["is_long_park"]].copy()
    if long_df.empty or not focus_clusters:
        return

    expanded = _explode_time_slots(long_df, freq_minutes=freq_minutes)
    slot_length_hours = freq_minutes / 60.0

    # 母数（全クラスタ）を算出
    overall_counts = (
        expanded.pivot_table(
            index="slot_weekday",
            columns="slot_hour",
            values="time_slots",
            aggfunc="count",
            fill_value=0,
        )
        .astype(int)
    )

    overall_after_counts = (
        expanded[expanded["after_charge"]]
        .pivot_table(
            index="slot_weekday",
            columns="slot_hour",
            values="time_slots",
            aggfunc="count",
            fill_value=0,
        )
        .astype(int)
        if not expanded[expanded["after_charge"]].empty
        else pd.DataFrame(dtype=int)
    )

    heatmap_records = []
    for cluster_id in focus_clusters:
        cluster_slice = expanded[expanded["session_cluster"] == cluster_id]
        if cluster_slice.empty:
            continue

        # 分子（対象クラスタ）のスロット数
        cluster_counts = (
            cluster_slice.pivot_table(
                index="slot_weekday",
                columns="slot_hour",
                values="time_slots",
                aggfunc="count",
                fill_value=0,
            )
            .astype(int)
        )
        aligned_cluster_counts, aligned_overall_counts = cluster_counts.align(
            overall_counts, join="right", fill_value=0
        )
        aligned_overall_counts = aligned_overall_counts.astype(int)

        # 滞在比率 = 対象クラスタの観測数 / 全クラスタの観測数
        share_all = aligned_cluster_counts.divide(
            aligned_overall_counts.where(aligned_overall_counts != 0),
        )
        share_all = share_all.fillna(0.0)

        # 充電直後バージョン
        after_slice = cluster_slice[cluster_slice["after_charge"]]
        cluster_after_counts = (
            after_slice.pivot_table(
                index="slot_weekday",
                columns="slot_hour",
                values="time_slots",
                aggfunc="count",
                fill_value=0,
            )
            .astype(int)
        )
        aligned_cluster_after_counts, aligned_overall_after_counts = cluster_after_counts.align(
            overall_after_counts, join="right", fill_value=0
        )
        aligned_overall_after_counts = aligned_overall_after_counts.astype(int)
        share_after = aligned_cluster_after_counts.divide(
            aligned_overall_after_counts.where(aligned_overall_after_counts != 0),
        ).fillna(0.0)

        # シェア・スロット数を揃えてからパーセント／差分を計算。
        aligned_all_share, aligned_after_share = share_all.align(
            share_after, fill_value=0.0
        )
        all_percent_aligned = aligned_all_share * 100
        after_percent_aligned = aligned_after_share * 100
        diff_percent = after_percent_aligned - all_percent_aligned

        # 分子・母数のアライン（注釈用）
        aligned_all_numerators, aligned_all_denominators = aligned_cluster_counts.align(
            aligned_overall_counts, join="right", fill_value=0
        )
        aligned_after_numerators, aligned_after_denominators = aligned_cluster_after_counts.align(
            aligned_overall_after_counts, join="right", fill_value=0
        )

        _plot_heatmap(
            matrix=all_percent_aligned,
            annot=_build_percentage_annotation(
                all_percent_aligned,
                aligned_all_numerators,
                aligned_all_denominators,
                "滞在比率 (%)",
            ),
            title=f"クラスタ {cluster_id} | 長時間放置全体の滞在分布",
            x_label="開始時刻 (時)",
            y_label="曜日 (0=月)",
            colorbar_label="滞在比率 [%]",
            path=plots_dir / f"heatmap_cluster_{cluster_id}_all.png",
        )
        _plot_heatmap(
            matrix=after_percent_aligned,
            annot=_build_percentage_annotation(
                after_percent_aligned,
                aligned_after_numerators,
                aligned_after_denominators,
                "滞在比率 (%)",
            ),
            title=f"クラスタ {cluster_id} | 充電直後の滞在分布",
            x_label="開始時刻 (時)",
            y_label="曜日 (0=月)",
            colorbar_label="滞在比率 [%]",
            path=plots_dir / f"heatmap_cluster_{cluster_id}_aftercharge.png",
        )
        _plot_heatmap(
            matrix=diff_percent,
            annot=_build_diff_annotation(
                diff_percent,
                aligned_after_numerators,
                aligned_after_denominators,
                aligned_all_numerators,
                aligned_all_denominators,
            ),
            title=f"クラスタ {cluster_id} | 充電直後 − 全体 (ポイント差)",
            x_label="開始時刻 (時)",
            y_label="曜日 (0=月)",
            colorbar_label="差分 [ポイント]",
            path=plots_dir / f"heatmap_cluster_{cluster_id}_diff.png",
            cmap="RdBu_r",
            center=0.0,
        )

        # 生データは CSV に保存（比率は 0-1 のまま保持）。
        aligned_all_share.to_csv(
            tables_dir / f"heatmap_cluster_{cluster_id}_all.csv"
        )
        aligned_after_share.to_csv(
            tables_dir / f"heatmap_cluster_{cluster_id}_aftercharge.csv"
        )
        (aligned_after_share - aligned_all_share).to_csv(
            tables_dir / f"heatmap_cluster_{cluster_id}_diff.csv"
        )

        heatmap_records.append(
            {
                "cluster": cluster_id,
                "total_hours_all": float(aligned_cluster_counts.values.sum() * slot_length_hours),
                "total_hours_after": float(aligned_cluster_after_counts.values.sum() * slot_length_hours),
                "slot_count_all": int(aligned_cluster_counts.values.sum()),
                "slot_count_after": int(aligned_cluster_after_counts.values.sum()),
            }
        )

    if heatmap_records:
        pd.DataFrame(heatmap_records).to_csv(
            tables_dir / "focus_cluster_slot_hours.csv", index=False
        )

    # 充電開始時刻ごとの次長時間放置クラスタの条件付き確率。
    charge_df = df[
        (df["session_type"] == "charging")
        & (df["next_session_type"] == "inactive")
        & (df["next_is_long_park"])
    ].copy()
    if charge_df.empty:
        return

    charge_df["charge_start_hour"] = charge_df["charge_start_hour"].astype(int)
    cond_counts = (
        charge_df.groupby(
            ["charge_cluster", "charge_start_hour", "next_cluster"],
            dropna=True,
        )
        .size()
        .reset_index(name="count")
    )
    cond_counts["probability"] = cond_counts.groupby(
        ["charge_cluster", "charge_start_hour"]
    )["count"].transform(lambda x: x / x.sum())
    cond_counts.to_csv(
        tables_dir / "charge_start_to_next_long_cluster.csv", index=False
    )

    if not cond_counts.empty:
        cond_counts["charge_cluster_label"] = cond_counts["charge_cluster"].apply(
            lambda x: f"クラスタ{x}"
        )
        cond_counts["next_cluster_label"] = cond_counts["next_cluster"].apply(
            lambda x: f"クラスタ{x}"
        )
        g = sns.catplot(
            data=cond_counts,
            kind="bar",
            x="charge_start_hour",
            y="probability",
            hue="next_cluster_label",
            col="charge_cluster_label",
            col_wrap=3,
            height=3.5,
            aspect=1.2,
        )
        g.set_titles("充電クラスタ {col_name}")
        g.set_axis_labels("充電開始時刻 (時)", "次の長時間放置クラスタの確率")
        if g._legend is not None:
            g._legend.set_title("次のクラスタ")
        for ax in g.axes.flatten():
            ax.yaxis.set_major_locator(MaxNLocator(5))
        plt.tight_layout()
        plt.savefig(
            plots_dir / "charge_start_hour_next_long_cluster.png",
            dpi=220,
        )
        plt.close()


def _plot_network(
    edges: pd.DataFrame,
    source_col: str,
    target_col: str,
    weight_col: str,
    title: str,
    path: Path,
) -> None:
    """Step3 で用いる遷移ネットワーク図を描画する。"""
    if edges.empty:
        return

    graph = nx.DiGraph()
    for _, row in edges.iterrows():
        weight = row[weight_col]
        if weight <= 0:
            continue
        graph.add_edge(row[source_col], row[target_col], weight=weight)

    if not graph.edges:
        return

    edge_weights = np.array([d["weight"] for _, _, d in graph.edges(data=True)])
    max_weight = edge_weights.max() if edge_weights.size else 1.0
    edge_widths = 1.5 + 4.0 * (edge_weights / max_weight)

    pos = nx.spring_layout(graph, seed=42, k=1.5 / np.sqrt(max(graph.number_of_nodes(), 1)))
    plt.figure(figsize=(8, 6))
    nx.draw_networkx_nodes(graph, pos, node_size=1200, node_color="#fee090")
    nx.draw_networkx_labels(graph, pos, font_size=9)
    nx.draw_networkx_edges(
        graph,
        pos,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=15,
        width=edge_widths,
        edge_color="#4575b4",
        connectionstyle="arc3,rad=0.1",
    )
    edge_labels = {
        (u, v): f"{data['weight']:.0f}" for u, v, data in graph.edges(data=True)
    }
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def step3_transition_analysis(
    df: pd.DataFrame,
    plots_dir: Path,
    tables_dir: Path,
) -> None:
    """
    Step3: 充電前後の遷移確率・日次遷移行列（充電日 vs 非充電日）・距離指標を算出。
    """
    charging_df = df[df["session_type"] == "charging"].copy()
    if not charging_df.empty:
        before_edges = (
            charging_df.dropna(subset=["prev_cluster"])
            .groupby(["prev_cluster", "session_cluster"])
            .size()
            .reset_index(name="count")
        )
        before_edges["probability"] = before_edges.groupby("prev_cluster")[
            "count"
        ].transform(lambda x: x / x.sum())
        before_edges.rename(
            columns={"session_cluster": "charge_cluster"},
            inplace=True,
        )
        before_edges.to_csv(
            tables_dir / "transition_prob_before.csv", index=False
        )
        _plot_network(
            before_edges,
            "prev_cluster",
            "charge_cluster",
            "count",
            "充電前の遷移ネットワーク (前クラスタ → 充電クラスタ)",
            plots_dir / "network_before_charge.png",
        )

    after_charge_df = charging_df[
        (charging_df["next_session_type"] == "inactive")
        & (charging_df["next_is_long_park"])
    ].copy()
    if not after_charge_df.empty:
        after_edges = (
            after_charge_df.groupby(["session_cluster", "next_cluster"])
            .size()
            .reset_index(name="count")
        )
        after_edges["probability"] = after_edges.groupby("session_cluster")[
            "count"
        ].transform(lambda x: x / x.sum())
        after_edges.rename(
            columns={
                "session_cluster": "charge_cluster",
                "next_cluster": "next_long_park_cluster",
            },
            inplace=True,
        )
        after_edges.to_csv(
            tables_dir / "transition_prob_after.csv", index=False
        )
        _plot_network(
            after_edges,
            "charge_cluster",
            "next_long_park_cluster",
            "count",
            "充電後の遷移ネットワーク (充電クラスタ → 次長時間放置)",
            plots_dir / "network_after_charge.png",
        )

    # 日次レベルの遷移行列（充電あり日／なし日）。
    inactive_df = df[df["session_type"] == "inactive"].copy()
    if inactive_df.empty:
        return

    daily_group = inactive_df.groupby(["hashvin", "date"], group_keys=False)
    inactive_df["next_cluster_same_day"] = daily_group["session_cluster"].shift(-1)
    transitions = inactive_df.dropna(subset=["next_cluster_same_day"]).copy()

    charge_day_flags = (
        df[df["session_type"] == "charging"]
        .groupby(["hashvin", "date"])
        .size()
        .reset_index(name="charge_events")
    )
    charge_day_flags["has_charge"] = charge_day_flags["charge_events"] > 0
    transitions = transitions.merge(
        charge_day_flags[["hashvin", "date", "has_charge"]],
        on=["hashvin", "date"],
        how="left",
    )
    transitions["has_charge"] = transitions["has_charge"].fillna(False)

    def _transition_matrix(
        sub_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """遷移確率行列と元となるカウント行列を同時に返す。"""
        counts = (
            sub_df.groupby(["session_cluster", "next_cluster_same_day"])
            .size()
            .unstack(fill_value=0)
        )
        probs = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0)
        probs = probs.fillna(0.0)
        return probs, counts

    matrix_charge, counts_charge = _transition_matrix(
        transitions[transitions["has_charge"]]
    )
    matrix_nocharge, counts_nocharge = _transition_matrix(
        transitions[~transitions["has_charge"]]
    )

    matrix_charge.to_csv(
        tables_dir / "transition_matrix_charge_days.csv"
    )
    matrix_nocharge.to_csv(
        tables_dir / "transition_matrix_nocharge_days.csv"
    )

    aligned_charge, aligned_nocharge = matrix_charge.align(
        matrix_nocharge, fill_value=0.0, join="outer"
    )
    aligned_counts_charge, aligned_counts_nocharge = counts_charge.align(
        counts_nocharge, fill_value=0, join="outer"
    )
    diff_matrix = aligned_charge - aligned_nocharge
    diff_matrix.to_csv(tables_dir / "transition_matrix_diff.csv")

    def _annot_transition(
        percent_matrix: pd.DataFrame,
        count_matrix: pd.DataFrame,
    ) -> pd.DataFrame:
        annot = percent_matrix.copy()
        for idx in annot.index:
            for col in annot.columns:
                val = annot.loc[idx, col]
                cnt = int(count_matrix.loc[idx, col]) if (
                    idx in count_matrix.index and col in count_matrix.columns
                ) else 0
                annot.loc[idx, col] = (
                    f"{val:.1f}%\n(n={cnt})" if cnt > 0 else "0.0%\n(n=0)"
                )
        return annot

    def _annot_transition_diff(
        diff_percent: pd.DataFrame,
        counts_charge: pd.DataFrame,
        counts_nocharge: pd.DataFrame,
    ) -> pd.DataFrame:
        annot = diff_percent.copy()
        for idx in annot.index:
            for col in annot.columns:
                charge_n = int(counts_charge.loc[idx, col]) if (
                    idx in counts_charge.index and col in counts_charge.columns
                ) else 0
                nocharge_n = int(counts_nocharge.loc[idx, col]) if (
                    idx in counts_nocharge.index and col in counts_nocharge.columns
                ) else 0
                annot.loc[idx, col] = (
                    f"{diff_percent.loc[idx, col]:.1f}pt\n充={charge_n}\n無={nocharge_n}"
                )
        return annot

    for label, matrix, counts in [
        ("charge", aligned_charge * 100, aligned_counts_charge),
        ("nocharge", aligned_nocharge * 100, aligned_counts_nocharge),
    ]:
        _plot_heatmap(
            matrix=matrix,
            annot=_annot_transition(matrix, counts),
            title="日次遷移（充電あり日）" if label == "charge" else "日次遷移（充電なし日）",
            x_label="遷移先クラスタ",
            y_label="遷移元クラスタ",
            colorbar_label="遷移確率 [%]",
            path=plots_dir / f"transition_matrix_{label}.png",
        )

    diff_percent = diff_matrix * 100
    _plot_heatmap(
        matrix=diff_percent,
        annot=_annot_transition_diff(
            diff_percent, aligned_counts_charge, aligned_counts_nocharge
        ),
        title="日次遷移差分（充電あり日 − 充電なし日）",
        x_label="遷移先クラスタ",
        y_label="遷移元クラスタ",
        colorbar_label="差分 [ポイント]",
        path=plots_dir / "transition_matrix_diff.png",
        cmap="RdBu_r",
        center=0.0,
    )

    # クラスタごとの Jensen–Shannon / TV 距離を算出。
    metrics = []
    for origin_cluster in diff_matrix.index:
        p = aligned_charge.loc[origin_cluster].values
        q = aligned_nocharge.loc[origin_cluster].values
        if not p.size or not q.size:
            continue
        jsd = float(jensenshannon(p, q, base=2))
        tvd = float(0.5 * np.abs(p - q).sum())
        metrics.append(
            {
                "from_cluster": origin_cluster,
                "js_distance": jsd,
                "tv_distance": tvd,
            }
        )

    if metrics:
        pd.DataFrame(metrics).to_csv(
            tables_dir / "transition_diff_metrics.csv", index=False
        )


def run_pipeline(
    csv_path: Path,
    output_root: Path,
    per_vehicle_top_n: int = 10,
    focus_clusters: int = 5,
    freq_minutes: int = DEFAULT_FREQ_MINUTES,
) -> None:
    """EDA の全工程（Step1〜Step3）を順番に実行する。"""

    df = load_sessions(csv_path)
    df = prepare_sessions(df)
    plots_dir, tables_dir = ensure_dirs(output_root)

    focus_cluster_ids = step1_long_park_distribution(
        df,
        plots_dir=plots_dir,
        tables_dir=tables_dir,
        per_vehicle_top_n=per_vehicle_top_n,
        global_top_k=focus_clusters,
    )

    step2_heatmaps_and_charge_effects(
        df,
        focus_clusters=focus_cluster_ids,
        plots_dir=plots_dir,
        tables_dir=tables_dir,
        freq_minutes=freq_minutes,
    )

    step3_transition_analysis(df, plots_dir=plots_dir, tables_dir=tables_dir)


def parse_args() -> argparse.Namespace:
    """CLI 用の引数定義。"""
    parser = argparse.ArgumentParser(
        description="EV バッテリ放置挙動の EDA パイプラインを実行します。",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        required=True,
        help="セッションログ CSV のパス",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="成果物（PNG, CSV）の出力先ディレクトリ",
    )
    parser.add_argument(
        "--per-vehicle-top-n",
        type=int,
        default=10,
        help="Step1 で車両ごとに描画する上位クラスタ数",
    )
    parser.add_argument(
        "--focus-clusters",
        type=int,
        default=5,
        help="Step2 の詳細分析対象となるクラスタ数（全体上位）",
    )
    parser.add_argument(
        "--freq-minutes",
        type=int,
        default=DEFAULT_FREQ_MINUTES,
        help="Step2 のタイムスロット展開に用いる間隔（分）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(
        csv_path=args.csv_path,
        output_root=args.output_dir,
        per_vehicle_top_n=args.per_vehicle_top_n,
        focus_clusters=args.focus_clusters,
        freq_minutes=args.freq_minutes,
    )


if __name__ == "__main__":
    main()
