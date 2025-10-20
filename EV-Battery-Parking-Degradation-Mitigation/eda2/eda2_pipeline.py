"""
EDA2 要件に沿って EV の放置セッションを曜日 × 時間帯 × クラスタ軸で集計し、
充電あり／なしの日別ヒートマップや類似度指標を出力するユーティリティ。

Jupyter Notebook から利用することを想定し、関数をインポートして実行できるように構成する。
コメントとドキュメントは若手エンジニアが背景と意図を理解しやすいよう日本語で記述する。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import math

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


Hour = int
ClusterLabel = str
DayType = str


@dataclass
class Eda2Config:
    """EDA2 の出力仕様をパラメータ化した設定クラス。"""

    top_n_clusters: int = 15
    metric: str = "duration_ratio"
    clip_percentile: float = 0.95
    other_position: str = "top"
    share_color_scale: bool = True
    annotate_cells: bool = True
    min_coverage_weeks: int = 1


class RequirementError(ValueError):
    """要件違反が発生したときに明示的に通知するための例外。"""


def run_weekday_parking_report(
    sessions: pd.DataFrame,
    output_root: Path,
    config: Eda2Config | None = None,
) -> None:
    """
    EV inactive セッションを基に曜日別ヒートマップ・CSV・類似度指標を出力する。

    Parameters
    ----------
    sessions : pd.DataFrame
        要件に定義された列を持つセッションデータ。Jupyter 上で事前に読み込む想定。
    output_root : Path
        出力先のルートディレクトリ（例: Path("result")）。
    config : Eda2Config | None
        振る舞いを調整する設定。None の場合は既定値を使用する。

    Raises
    ------
    RequirementError
        必須列が欠けている、日別区分が判定できない等、要件を満たせない場合。
    """

    if config is None:
        config = Eda2Config()

    # ratio の計算単位は duration（滞在時間）か count（イベント件数）の二択なので、想定外を弾く。
    if config.metric not in {"duration_ratio", "count_ratio"}:
        raise RequirementError("metric は 'duration_ratio' または 'count_ratio' のみサポートしています。")

    # 必須となるカラムが揃っているかチェックする。欠落していると後工程で情報が引けない。
    _require_columns(
        sessions,
        required={
            "hashvin",
            "session_cluster",
            "session_type",
            "start_time",
            "end_time",
        },
    )

    # 入力 DataFrame を解析に扱いやすい形に正規化する（型変換・無効行除去）。
    normalized = _prepare_sessions_dataframe(sessions)
    if normalized.empty:
        raise RequirementError("inactive セッションが存在しないため、レポートを生成できません。")

    # 日単位で充電あり/なしを判定するテーブルと観測週数を算出する。
    day_type_table = _build_day_type_table(normalized)
    coverage_weeks = _compute_coverage_weeks(day_type_table)
    _validate_coverage(coverage_weeks, config.min_coverage_weeks)
    # 描画時に分母情報へ素早くアクセスできるよう辞書化しておく。
    coverage_lookup = {
        (row.hashvin, row.day_type): row.denominator_hours
        for row in coverage_weeks.itertuples(index=False)
    }

    # ヒートマップ対象は inactive セッションのみなので事前に抽出する。
    inactive_sessions = normalized.query("session_type == 'inactive'").copy()
    hourly_records = _accumulate_hourly_numerators(
        inactive_df=inactive_sessions,
        day_table=day_type_table,
        metric=config.metric,
    )
    if hourly_records.empty:
        raise RequirementError("inactive セッションに集計対象の滞在時間がありません。")

    # 1 時間ビンに展開した分子をクラスタ単位に再集約する。
    aggregated = _aggregate_numerators(hourly_records)
    # 充電あり/なし双方で有効な上位クラスタを決定し、それ以外を OTHER へ畳み込む。
    top_cluster_map = _select_top_clusters(aggregated, config.top_n_clusters)
    aggregated = _collapse_other_clusters(aggregated, top_cluster_map)

    # 観測週数を分母としてセル割合を算出する。
    ratios = _compute_cell_ratios(aggregated, coverage_weeks)
    # ヒートマップ表示順を揃えるため、クラスタの並びを曜日ごとに決定する。
    cluster_order_map = _build_cluster_orders(ratios, config.other_position)

    # 出力ディレクトリを生成する。既存の場合は何もしない。
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for hashvin, hashvin_df in ratios.groupby("hashvin"):
        # 車両ごとにサブディレクトリを作成し、その配下に画像・CSV を格納する。
        hashvin_dir = output_root / hashvin
        hashvin_dir.mkdir(parents=True, exist_ok=True)

        # 分母情報は全曜日共通なので 1 ファイルにまとめる。
        denom_df = _build_denominator_table(hashvin, coverage_weeks)
        denom_df.to_csv(hashvin_dir / "denominator_weeks.csv", index=False, encoding="utf-8-sig")

        similarity_rows = []

        for weekday in range(7):
            # 曜日ごとに集計結果をフィルタし、クラスタ順序を取得する。
            weekday_df = hashvin_df.query("weekday == @weekday").copy()
            cluster_order = cluster_order_map[(hashvin, weekday)]

            # ヒートマップ描画と CSV 出力に使いやすいマトリクスへ整形する。
            matrices = _build_matrices_for_outputs(
                weekday_df=weekday_df,
                cluster_order=cluster_order,
            )

            # 指定フォーマットで割合・分子を CSV 出力する。
            _export_matrices(
                matrices=matrices,
                base_path=hashvin_dir,
                weekday=weekday,
            )

            # 左右比較ヒートマップを描画し、画像ファイルとして保存する。
            rendered = _render_weekday_comparison(
                hashvin=hashvin,
                weekday=weekday,
                matrices=matrices,
                coverage_lookup=coverage_lookup,
                metric=config.metric,
                clip_percentile=config.clip_percentile,
                share_color_scale=config.share_color_scale,
                annotate_cells=config.annotate_cells,
            )
            rendered.savefig(hashvin_dir / f"weekday_{weekday}_comparison.png", dpi=200, bbox_inches="tight")
            plt.close(rendered)

            # 後段の類似度計算は曜日単位なので、その結果を収集する。
            similarity_rows.append(
                _compute_similarity_row(
                    hashvin=hashvin,
                    weekday=weekday,
                    matrices=matrices,
                    coverage_weeks=coverage_weeks,
                )
            )

        # 類似度指標を 1 ファイルにまとめて出力する。
        similarity_df = pd.DataFrame(similarity_rows)
        similarity_df.to_csv(hashvin_dir / "similarity_scores.csv", index=False, encoding="utf-8-sig")


def _require_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    """必須列が揃っているかチェックする。"""

    # DataFrame の列セットと required の差分を取る。
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise RequirementError(f"必須列が不足しています: {', '.join(missing)}")


def _prepare_sessions_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    文字列や型を正規化し、解析で扱いやすい DataFrame を返す。

    - start_time / end_time を datetime64[ns] に変換する。
    - 無効な行（end <= start）を除外する。
    - 重要列のみを残してコピーを返す。
    """

    work = df.copy()
    # 文字列で渡されることを想定し、datetime64[ns] へ変換（不正値は NaT）。
    work["start_time"] = pd.to_datetime(work["start_time"], utc=False, errors="coerce")
    work["end_time"] = pd.to_datetime(work["end_time"], utc=False, errors="coerce")
    # タイムゾーン付きのデータが渡された場合は UTC に揃えてから tz 情報を外し、
    # 後続の計算（タイムゾーンなしの datetime）と比較できるようにする。
    if getattr(work["start_time"].dt, "tz", None) is not None:
        work["start_time"] = work["start_time"].dt.tz_convert("UTC").dt.tz_localize(None)
    if getattr(work["end_time"].dt, "tz", None) is not None:
        work["end_time"] = work["end_time"].dt.tz_convert("UTC").dt.tz_localize(None)

    # start/end が欠損していない行だけを保持する。
    valid_mask = work["start_time"].notna() & work["end_time"].notna()
    work = work.loc[valid_mask].copy()
    # end_time <= start_time のような逆転データは解析に使えないので除外する。
    work = work.loc[work["end_time"] > work["start_time"]]

    # 必須列と、存在すれば duration_minutes を残して解析用の最小構成へ絞り込む。
    keep_cols = [
        "hashvin",
        "session_cluster",
        "session_type",
        "start_time",
        "end_time",
    ]
    if "duration_minutes" in work.columns:
        keep_cols.append("duration_minutes")

    work = work[keep_cols].copy()
    # クラスタは int/float の混在を避けるため文字列化しておく。欠損は UNKNOWN 扱いとする。
    work["session_cluster"] = work["session_cluster"].map(lambda v: str(v) if pd.notna(v) else "UNKNOWN")

    # 夜間を跨ぐ処理を行うために、開始日の情報を保持する。
    work["start_date"] = work["start_time"].dt.date
    work["weekday"] = work["start_time"].dt.weekday

    # 後続処理が時間順に進むように、主要キーでソートしてリセットする。
    work.sort_values(["hashvin", "start_time"], inplace=True)
    work.reset_index(drop=True, inplace=True)
    return work


def _build_day_type_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    日単位で「充電あり／なし」を判定したテーブルを作成する。

    inactive 行だけでは日別区分が決められないため、元の DataFrame から charging の有無を参照する。
    """

    # 充電セッションが存在する日を抽出し day_type=with を付与する。
    charging_days = (
        df.loc[df["session_type"] == "charging", ["hashvin", "start_date"]]
        .drop_duplicates()
        .assign(day_type="with")
    )

    # inactive セッションが存在する日を抽出し、weekday を保持する。
    inactive_days = (
        df.loc[df["session_type"] == "inactive", ["hashvin", "start_date", "weekday"]]
        .drop_duplicates()
    )

    if inactive_days.empty:
        raise RequirementError("inactive セッションの日付が存在しません。")

    # inactive 日付と charging 日付を結合し、充電あり/なしを判定する。
    day_table = inactive_days.merge(
        charging_days,
        on=["hashvin", "start_date"],
        how="left",
    )
    day_table["day_type"] = day_table["day_type"].fillna("without")

    # カレンダー週を ISO 規則で求め、分母計算に備える。
    iso_calendar = pd.to_datetime(day_table["start_date"]).dt.isocalendar()
    day_table["iso_year"] = iso_calendar["year"]
    day_table["iso_week"] = iso_calendar["week"]
    day_table["iso_yearweek"] = day_table["iso_year"].astype(str) + "-W" + day_table["iso_week"].astype(str).str.zfill(2)

    return day_table


def _compute_coverage_weeks(day_table: pd.DataFrame) -> pd.DataFrame:
    """充電あり／なし別にユニークなカレンダー週数を算出する。"""

    # hashvin × day_type ごとに登場した ISO 週の種類数を数え、分母用の時間数に変換する。
    coverage = (
        day_table.groupby(["hashvin", "day_type"])["iso_yearweek"]
        .nunique()
        .reset_index(name="weeks")
    )
    coverage["denominator_hours"] = coverage["weeks"] * 1.0  # 1 時間 × 週数
    return coverage


def _validate_coverage(coverage: pd.DataFrame, min_weeks: int) -> None:
    """
    分子が存在する日種別について、観測週数が一定以上あるかをチェックする。

    週数が 0 の場合は分母計算が破綻するため即座に例外を送出する。
    """

    # 週数がゼロだと分母がゼロになるので即座にエラーにする。
    if (coverage["weeks"] == 0).any():
        raise RequirementError("観測週数が 0 の日種別があるため、割合を計算できません。")

    # ユーザーが指定した最小週数しきい値を満たしているかを確認する。
    insufficient = coverage.loc[coverage["weeks"] < min_weeks]
    if not insufficient.empty:
        detail = ", ".join(
            f"{row.hashvin}:{row.day_type}={row.weeks}週"
            for row in insufficient.itertuples(index=False)
        )
        raise RequirementError(f"観測週数が min_coverage_weeks を下回っています: {detail}")


def _accumulate_hourly_numerators(
    inactive_df: pd.DataFrame,
    day_table: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    """
    inactive セッションを 1 時間ビンへ分解して分子（滞在時間[h]）を積算する。

    - 日単位の計算窓を「当日 6:00 〜 翌日 6:00」で固定する。
    - 翌日 6:00 を越える滞在は切り捨てる（ユーザー指定仕様）。
    """

    day_type_map = {
        (row.hashvin, row.start_date): row.day_type
        for row in day_table.itertuples(index=False)
    }

    records: List[Dict[str, object]] = []

    for row in inactive_df.itertuples(index=False):
        # inactive 行が day_table に存在しない場合は異常データなので無視する（念のため防御）。
        key = (row.hashvin, row.start_date)
        day_type = day_type_map.get(key)
        if day_type is None:
            # inactive だが day_table に存在しない場合は異常データなので無視する。
            continue

        # 当日 6:00 を起点に、翌日 6:00 までを分析対象ウィンドウとする。
        window_start = datetime.combine(row.start_date, time(hour=6))
        window_end = window_start + timedelta(hours=24)

        # ユーザー要望に従い、6:00〜30:00 の範囲へクリップする。
        clipped_start = max(row.start_time, window_start)
        clipped_end = min(row.end_time, window_end)
        if clipped_start >= clipped_end:
            continue

        # 1 時間単位でイベントを切り分けながら、分子へ加算する。
        current = clipped_start
        while current < clipped_end:
            hour_start = current.replace(minute=0, second=0, microsecond=0)
            next_hour = hour_start + timedelta(hours=1)
            bucket_end = min(next_hour, clipped_end)

            overlap = (bucket_end - current).total_seconds() / 3600.0
            if overlap <= 0:
                current = bucket_end
                continue

            # metric に応じて「滞在時間」または「件数」で分子を構築する。
            if metric == "duration_ratio":
                numerator_value = overlap
            else:
                numerator_value = 1.0

            records.append(
                {
                    "hashvin": row.hashvin,
                    "day_type": day_type,
                    "weekday": row.weekday,
                    "hour": hour_start.hour,
                    "session_cluster": row.session_cluster,
                    "numerator": numerator_value,
                }
            )

            current = bucket_end

    return pd.DataFrame.from_records(records)


def _aggregate_numerators(hourly_df: pd.DataFrame) -> pd.DataFrame:
    """分子データをクラスタ・曜日単位で集計する。"""

    if hourly_df.empty:
        return hourly_df

    aggregated = (
        hourly_df.groupby(
            ["hashvin", "day_type", "weekday", "hour", "session_cluster"],
            as_index=False,
        )["numerator"]
        .sum()
    )
    return aggregated


def _select_top_clusters(aggregated: pd.DataFrame, top_n: int) -> Dict[str, List[str]]:
    """
    各車両ごとに滞在時間の多いクラスタを抽出する。

    充電あり／なし両方の分子を合算し、合計滞在時間の降順で上位 N 件を採用する。
    """

    cluster_totals = (
        aggregated.groupby(["hashvin", "session_cluster"])["numerator"]
        .sum()
        .reset_index()
    )

    top_map: Dict[str, List[str]] = {}
    for hashvin, hashvin_df in cluster_totals.groupby("hashvin"):
        sorted_clusters = (
            hashvin_df.sort_values("numerator", ascending=False)["session_cluster"].tolist()
        )
        top_map[hashvin] = sorted_clusters[:top_n]
    return top_map


def _collapse_other_clusters(
    aggregated: pd.DataFrame,
    top_cluster_map: Dict[str, List[str]],
) -> pd.DataFrame:
    """
    上位外クラスタおよびノイズクラスタ(-1など)を OTHER 行へまとめる。
    """

    # hashvin ごとにトップクラスタへ含まれるかを判定し、含まれなければ OTHER へ送る。
    def classify_cluster(row: pd.Series) -> str:
        hashvin = row["hashvin"]
        cluster = row["session_cluster"]
        if cluster in top_cluster_map.get(hashvin, []):
            return cluster
        return "OTHER"

    aggregated = aggregated.copy()
    aggregated["session_cluster"] = aggregated.apply(classify_cluster, axis=1)

    grouped = (
        aggregated.groupby(
            ["hashvin", "day_type", "weekday", "hour", "session_cluster"],
            as_index=False,
        )["numerator"]
        .sum()
    )
    return grouped


def _compute_cell_ratios(
    aggregated: pd.DataFrame,
    coverage_weeks: pd.DataFrame,
) -> pd.DataFrame:
    """
    分子と分母を突き合わせてセル割合を計算する。

    分母は「観測週数 × 1 時間」で、日種別（充電あり／なし）ごとに共通値を持つ。
    """

    denom_lookup = {
        (row.hashvin, row.day_type): row.denominator_hours
        for row in coverage_weeks.itertuples(index=False)
    }

    # hashvin × day_type ごとの分母を参照し、割合を計算する。
    def compute_ratio(row: pd.Series) -> float:
        denom = denom_lookup.get((row["hashvin"], row["day_type"]))
        if denom is None or denom == 0:
            return math.nan
        return row["numerator"] / denom

    aggregated = aggregated.copy()
    aggregated["ratio"] = aggregated.apply(compute_ratio, axis=1)
    return aggregated


def _build_cluster_orders(
    ratio_df: pd.DataFrame,
    other_position: str,
) -> Dict[Tuple[str, int], List[str]]:
    """
    ヒートマップ描画用にクラスタの表示順を決定する。

    - OTHER を最上段に固定する（other_position == "top" の場合）。
    - 合計割合が大きいクラスタほど下段に配置する。
    """

    cluster_orders: Dict[Tuple[str, int], List[str]] = {}

    totals = (
        ratio_df.groupby(["hashvin", "weekday", "session_cluster"])["ratio"]
        .sum()
        .reset_index()
    )

    for (hashvin, weekday), group in totals.groupby(["hashvin", "weekday"]):
        ordered = (
            group.assign(
                # 合計割合が大きいものほど最後（下段）に来るよう昇順に並べる。
                sort_key=group["ratio"].fillna(0.0)
            )
            .sort_values("sort_key", ascending=True)
        )

        clusters = ordered["session_cluster"].tolist()

        if other_position == "top" and "OTHER" in clusters:
            clusters = ["OTHER"] + [c for c in clusters if c != "OTHER"]

        cluster_orders[(hashvin, weekday)] = clusters

    return cluster_orders


def _build_matrices_for_outputs(
    weekday_df: pd.DataFrame,
    cluster_order: List[str],
) -> Dict[str, pd.DataFrame]:
    """
    曜日ごとの出力に必要なマトリクス（ratio, numerator）を生成する。

    戻り値のキー:
        - matrix_with, matrix_without
        - numerator_with, numerator_without
    """

    display_hours = list(range(6, 24)) + list(range(0, 6))
    matrices: Dict[str, pd.DataFrame] = {}

    for day_type in ["without", "with"]:
        subset = weekday_df.query("day_type == @day_type").copy()
        # 滞在量または件数のマトリクス（クラスタ行 × 時間列）を生成する。
        numerator = _pivot_matrix(
            subset,
            value_column="numerator",
            cluster_order=cluster_order,
            display_hours=display_hours,
        )
        # 割合のマトリクスも同じ形で生成する。
        ratio = _pivot_matrix(
            subset,
            value_column="ratio",
            cluster_order=cluster_order,
            display_hours=display_hours,
        )
        matrices[f"numerator_{day_type}"] = numerator
        matrices[f"matrix_{day_type}"] = ratio

    matrices["display_hours"] = pd.Index(display_hours, name="hour")
    matrices["cluster_order"] = cluster_order
    return matrices


def _pivot_matrix(
    source: pd.DataFrame,
    value_column: str,
    cluster_order: List[str],
    display_hours: List[int],
) -> pd.DataFrame:
    """クラスタ行 × 時間列のマトリクスへ整形する。"""

    pivoted = (
        source.pivot_table(
            index="session_cluster",
            columns="hour",
            values=value_column,
            aggfunc="sum",
            fill_value=0.0,
        )
        .reindex(cluster_order)
        .reindex(columns=display_hours)
        .fillna(0.0)
    )
    pivoted.index.name = "session_cluster"
    pivoted.columns.name = "hour"
    return pivoted


def _export_matrices(
    matrices: Dict[str, pd.DataFrame],
    base_path: Path,
    weekday: int,
) -> None:
    """割合と分子の CSV を要件どおりに書き出す。"""

    # day_type/値種別の組み合わせで命名規則どおりにファイルへ保存する。
    for key in ["matrix_without", "matrix_with", "numerator_without", "numerator_with"]:
        df = matrices[key]
        df.to_csv(
            base_path / f"weekday_{weekday}_{key}.csv",
            encoding="utf-8-sig",
        )


def _render_weekday_comparison(
    hashvin: str,
    weekday: int,
    matrices: Dict[str, pd.DataFrame],
    coverage_lookup: Dict[Tuple[str, DayType], float],
    metric: str,
    clip_percentile: float,
    share_color_scale: bool,
    annotate_cells: bool,
) -> plt.Figure:
    """
    曜日別に左右比較ヒートマップを描画する。

    左: 充電なし日, 右: 充電あり日。カラースケールは設定に応じて共有する。
    """

    weekday_label = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][weekday]
    hours = matrices["display_hours"]

    left = matrices["matrix_without"]
    right = matrices["matrix_with"]

    all_values = np.concatenate([left.values.flatten(), right.values.flatten()])
    valid_values = all_values[~np.isnan(all_values)]

    if valid_values.size == 0:
        vmax = 1.0
    else:
        vmax = np.quantile(valid_values, clip_percentile)
        if vmax <= 0:
            vmax = float(valid_values.max() or 1.0)

    vmins = [0.0, 0.0]
    vmaxs = [vmax, vmax] if share_color_scale else [
        _compute_vmax(left.values, clip_percentile),
        _compute_vmax(right.values, clip_percentile),
    ]

    denom_without = coverage_lookup.get((hashvin, "without"))
    denom_with = coverage_lookup.get((hashvin, "with"))
    denom_without = float(denom_without) if denom_without is not None else math.nan
    denom_with = float(denom_with) if denom_with is not None else math.nan

    # セルに注記を描画する場合は、確率と分子・分母を組み合わせたテキストを準備する。
    annotation_without = (
        _build_annotation_matrix(
            ratio_df=left,
            numerator_df=matrices["numerator_without"],
            denominator=denom_without,
            metric=metric,
        )
        if annotate_cells
        else None
    )
    annotation_with = (
        _build_annotation_matrix(
            ratio_df=right,
            numerator_df=matrices["numerator_with"],
            denominator=denom_with,
            metric=metric,
        )
        if annotate_cells
        else None
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 9), sharey=True)
    cmap = "YlGnBu"

    for ax, data, title, vmin, vmax_val, annotations in zip(
        axes,
        [left, right],
        ["充電なし日", "充電あり日"],
        vmins,
        vmaxs,
        [annotation_without, annotation_with],
    ):
        # seaborn の heatmap を利用して 2D 行列を描画する。
        sns.heatmap(
            data,
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax_val,
            cbar=True,
            annot=annotations if annotate_cells else False,
            fmt="",
        )
        ax.set_xlabel("hour")
        ax.set_xticklabels(hours, rotation=45, ha="right")
        ax.set_ylabel("session_cluster")
        ax.set_title(title)

    denom_text = (
        f"denom_without={denom_without:.0f}h | denom_with={denom_with:.0f}h"
        if not math.isnan(denom_without) or not math.isnan(denom_with)
        else "denom: n/a"
    )
    fig.suptitle(
        f"{hashvin} | Weekday: {weekday_label} | metric: {metric} | scale: 0–p{int(clip_percentile*100)} | {denom_text}"
    )

    summary_text = _build_summary_text(
        matrices=matrices,
        denom_without=denom_without,
        denom_with=denom_with,
        metric=metric,
    )
    fig.text(0.5, -0.02, summary_text, ha="center", va="top")
    fig.tight_layout()
    return fig


def _build_annotation_matrix(
    ratio_df: pd.DataFrame,
    numerator_df: pd.DataFrame,
    denominator: float,
    metric: str,
) -> pd.DataFrame:
    """
    ヒートマップへ描画する注記テキスト（確率＋分子/分母）を整形する。

    ratio_df, numerator_df は同じ行列形状を想定し、足りない場合も安全に reindex する。
    """

    ratio_aligned = ratio_df.reindex(index=numerator_df.index, columns=numerator_df.columns)
    numerator_aligned = numerator_df.reindex(index=numerator_df.index, columns=numerator_df.columns)

    denom_value = float(denominator) if denominator is not None else math.nan
    denom_text = (
        "n/a" if math.isnan(denom_value) or denom_value <= 0 else f"{denom_value:.0f}h"
    )

    annotation_values: List[List[str]] = []
    for cluster in numerator_aligned.index:
        row_labels: List[str] = []
        for hour in numerator_aligned.columns:
            numerator_value = numerator_aligned.loc[cluster, hour]
            ratio_value = ratio_aligned.loc[cluster, hour]

            if metric == "duration_ratio":
                numerator_text = f"{numerator_value:.1f}h"
            else:
                numerator_text = f"{numerator_value:.0f}件"

            ratio_text = (
                "確率:--" if pd.isna(ratio_value) else f"確率:{ratio_value:.2f}"
            )
            row_labels.append(
                f"{ratio_text}\n分子:{numerator_text}\n分母:{denom_text}"
            )

        annotation_values.append(row_labels)

    return pd.DataFrame(annotation_values, index=numerator_aligned.index, columns=numerator_aligned.columns)


def _compute_vmax(values: np.ndarray, percentile: float) -> float:
    """単独ヒートマップ用の vmax を計算する。"""

    valid = values[~np.isnan(values)]
    if valid.size == 0:
        return 1.0
    vmax = np.quantile(valid, percentile)
    if vmax <= 0:
        vmax = float(valid.max() or 1.0)
    return vmax


def _build_summary_text(
    matrices: Dict[str, pd.DataFrame],
    denom_without: float,
    denom_with: float,
    metric: str,
) -> str:
    """ヒートマップ下に表示する注記テキストを生成する。"""

    # 分子マトリクスの総和（滞在時間または件数）を集計する。
    total_without = matrices["numerator_without"].values.sum()
    total_with = matrices["numerator_with"].values.sum()
    if metric == "duration_ratio":
        label = "Σ滞在"
        unit = "h"
        fmt_without = f"{total_without:.1f}{unit}"
        fmt_with = f"{total_with:.1f}{unit}"
    else:
        label = "Σ件数"
        unit = "件"
        fmt_without = f"{total_without:.0f}{unit}"
        fmt_with = f"{total_with:.0f}{unit}"

    without_text = (
        f"{denom_without:.0f}h" if not math.isnan(denom_without) else "n/a"
    )
    with_text = (
        f"{denom_with:.0f}h" if not math.isnan(denom_with) else "n/a"
    )
    return (
        f"{label}(充電なし)={fmt_without} / 週数={without_text} | "
        f"{label}(充電あり)={fmt_with} / 週数={with_text}"
    )


def _compute_similarity_row(
    hashvin: str,
    weekday: int,
    matrices: Dict[str, pd.DataFrame],
    coverage_weeks: pd.DataFrame,
) -> Dict[str, object]:
    """
    類似度指標を算出し、CSV 出力用に辞書を返す。

    分母の週数は coverage テーブルから取得する。
    """

    vector_without = matrices["matrix_without"].to_numpy(dtype=float).flatten()
    vector_with = matrices["matrix_with"].to_numpy(dtype=float).flatten()

    pearson = _safe_pearson(vector_without, vector_with)
    cosine = _safe_cosine(vector_without, vector_with)
    js_distance = _safe_js_distance(vector_without, vector_with)

    # coverage_weeks を辞書化し、CSV に記録する観測週数を引き出す。
    coverage_lookup = dict(
        ((row.hashvin, row.day_type), row.weeks) for row in coverage_weeks.itertuples(index=False)
    )

    return {
        "hashvin": hashvin,
        "weekday": weekday,
        "pearson_corr": pearson,
        "cosine_sim": cosine,
        "js_distance": js_distance,
        "covered_weeks_with": coverage_lookup.get((hashvin, "with"), np.nan),
        "covered_weeks_without": coverage_lookup.get((hashvin, "without"), np.nan),
    }


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    """ゼロ分散などを考慮しつつ Pearson 相関を計算する。"""

    if np.allclose(x, 0) and np.allclose(y, 0):
        return math.nan
    if np.std(x) == 0 or np.std(y) == 0:
        return math.nan
    return float(np.corrcoef(x, y)[0, 1])


def _safe_cosine(x: np.ndarray, y: np.ndarray) -> float:
    """コサイン類似度を計算する。"""

    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)
    if norm_x == 0 or norm_y == 0:
        return math.nan
    return float(np.dot(x, y) / (norm_x * norm_y))


def _safe_js_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Jensen-Shannon 距離を計算する。"""

    px = x.copy()
    py = y.copy()
    px = np.maximum(px, 0)
    py = np.maximum(py, 0)

    sum_px = px.sum()
    sum_py = py.sum()
    if sum_px == 0 or sum_py == 0:
        return math.nan

    px /= sum_px
    py /= sum_py
    m = 0.5 * (px + py)

    def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
        mask = (p > 0) & (q > 0)
        return float(np.sum(p[mask] * np.log2(p[mask] / q[mask])))

    js = 0.5 * _kl_divergence(px, m) + 0.5 * _kl_divergence(py, m)
    return float(np.sqrt(js))


def _build_denominator_table(hashvin: str, coverage_weeks: pd.DataFrame) -> pd.DataFrame:
    """denominator_weeks.csv に対応するテーブルを構築する。"""

    subset = coverage_weeks.loc[coverage_weeks["hashvin"] == hashvin].copy()
    subset["hours_per_week_bin"] = 1.0
    subset.rename(columns={"day_type": "category"}, inplace=True)
    return subset[["hashvin", "category", "weeks", "hours_per_week_bin"]]
