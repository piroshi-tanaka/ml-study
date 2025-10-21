# -*- coding: utf-8 -*-
"""EVの放置・充電行動を曜日×時間帯×クラスタで可視化するユーティリティ。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import math

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter

# 型エイリアス（読みやすさ向上のために定義）
Hour = int
ClusterLabel = str
DayType = str
PanelKey = str

# 3 面ヒートマップで扱うパネル（左→右）
PANEL_SEQUENCE: List[PanelKey] = [
    "inactive_without",  # 充電なし日の放置セッション
    "inactive_with",     # 充電あり日の放置セッション
    "charging",          # 充電セッションそのもの
]

# ヒートマップタイトルに表示するラベル
PANEL_TITLES: Dict[PanelKey, str] = {
    "inactive_without": "充電なし日",
    "inactive_with": "充電あり日",
    "charging": "充電セッション",
}

# 6:00〜23:00 と 0:00〜5:00 を 24 ビンで扱う
DISPLAY_HOURS = list(range(6, 24)) + list(range(0, 6))


@dataclass
class ParkingAnalysisConfig:
    """可視化・集計処理の動作を切り替えるための設定クラス。"""

    top_n_clusters: int = 15  # 各パネルで表示するクラスタ数の上限
    metric: str = "duration_ratio"  # "duration_ratio" または "count_ratio" を指定
    clip_percentile: float = 0.95  # カラーバー上限に用いるパーセンタイル
    other_position: str = "top"  # OTHER 行の表示位置（"top" / "bottom"）
    share_color_scale: bool = True  # 3 面でカラースケールを共有するか
    annotate_cells: bool = True  # セル内に注記（割合・件数・時間）を描画するか
    min_coverage_weeks: int = 1  # 分母に含める最低週数（未満なら例外）
    panel_denominator_mode: str = "per_panel"  # "per_panel" or "shared" を選択


class RequirementError(ValueError):
    """設定や入力データが要件を満たさない場合に送出する例外。"""


def run_weekday_parking_analysis(
    sessions: pd.DataFrame,
    output_root: Path,
    config: ParkingAnalysisConfig | None = None,
) -> None:
    """EV放置・充電セッションを分析し、ヒートマップと補助 CSV を出力する。"""

    # 設定が省略されていれば既定値を利用
    if config is None:
        config = ParkingAnalysisConfig()

    # 指定可能なオプションを事前に検証
    if config.metric not in {"duration_ratio", "count_ratio"}:
        raise RequirementError("metric は 'duration_ratio' または 'count_ratio' を指定してください。")
    if config.panel_denominator_mode not in {"per_panel", "shared"}:
        raise RequirementError("panel_denominator_mode は 'per_panel' または 'shared' を指定してください。")

    # 必須列が揃っているかチェック
    _require_columns(
        sessions,
        required={"hashvin", "session_cluster", "session_type", "start_time", "end_time"},
    )

    # 日付型変換・欠損除去などの前処理
    normalized = _prepare_sessions_dataframe(sessions)
    if normalized.empty:
        raise RequirementError("inactive セッションが存在しないため、結果を生成できません。")

    # 放置日区分テーブルと充電セッションを準備
    day_type_table = _build_day_type_table(normalized)
    charging_sessions = normalized.loc[normalized["session_type"] == "charging"].copy()

    # パネルごとの観測週数と週集合を算出
    panel_coverage_df, panel_week_sets = _compute_panel_coverage(day_type_table, charging_sessions)
    denominator_lookup, weeks_lookup = _build_denominator_lookups(
        panel_coverage_df,
        panel_week_sets,
        config.panel_denominator_mode,
    )

    # 観測週数が要件（min_coverage_weeks）を満たしているか確認
    coverage_validation = panel_coverage_df.copy()
    if coverage_validation.empty:
        coverage_validation = pd.DataFrame(columns=["hashvin", "panel", "weeks"])
    _validate_coverage(coverage_validation, config.min_coverage_weeks)

    # 放置・充電のイベントを時間ビン単位で集計
    aggregated = _aggregate_panel_metrics(normalized, day_type_table, charging_sessions, config.metric)
    top_cluster_map = _select_top_clusters(aggregated, config.top_n_clusters)
    aggregated = _collapse_other_clusters(aggregated, top_cluster_map)
    ratios = _compute_cell_ratios(aggregated, denominator_lookup)
    cluster_orders = _build_cluster_orders(ratios, config.other_position)

    # 出力ディレクトリを作成
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    similarity_rows: List[Dict[str, object]] = []

    for hashvin, hashvin_df in ratios.groupby("hashvin"):
        hashvin_dir = output_root / hashvin
        hashvin_dir.mkdir(parents=True, exist_ok=True)

        # 分母情報（週数×1h）を CSV に出力
        denom_df = _build_denominator_table(
            hashvin=hashvin,
            weeks_lookup=weeks_lookup,
            denominator_mode=config.panel_denominator_mode,
        )
        denom_df.to_csv(hashvin_dir / "denominator_weeks.csv", index=False, encoding="utf-8-sig")

        for weekday in range(7):
            weekday_df = hashvin_df.loc[hashvin_df["weekday"] == weekday]
            panel_payload = _build_weekday_panel_payload(
                hashvin=hashvin,
                weekday=weekday,
                weekday_df=weekday_df,
                cluster_orders=cluster_orders,
                panels=PANEL_SEQUENCE,
                metric=config.metric,
            )

            # CSV 出力（割合・滞在時間・件数）
            _export_panel_matrices(panel_payload, hashvin_dir, weekday)

            # 3 面比較ヒートマップを描画
            figure = _render_weekday_comparison(
                hashvin=hashvin,
                weekday=weekday,
                panel_payload=panel_payload,
                weeks_lookup=weeks_lookup,
                metric=config.metric,
                clip_percentile=config.clip_percentile,
                share_color_scale=config.share_color_scale,
                annotate_cells=config.annotate_cells,
            )
            figure.savefig(hashvin_dir / f"weekday_{weekday}_comparison.png", dpi=200, bbox_inches="tight")
            plt.close(figure)

            # 充電あり/なしの類似度を記録
            similarity_rows.append(
                _compute_similarity_row(
                    hashvin=hashvin,
                    weekday=weekday,
                    panel_payload=panel_payload,
                    weeks_lookup=weeks_lookup,
                )
            )

    if similarity_rows:
        similarity_df = pd.DataFrame(similarity_rows)
        for hashvin, sub in similarity_df.groupby("hashvin"):
            hashvin_dir = output_root / hashvin
            hashvin_dir.mkdir(parents=True, exist_ok=True)
            sub.to_csv(hashvin_dir / "similarity_scores.csv", index=False, encoding="utf-8-sig")


def _require_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    """必要な列が揃っているか確認し、不足があれば例外を送出する。"""

    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise RequirementError(f"必須列が不足しています: {', '.join(missing)}")


def _prepare_sessions_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """セッションを前処理し、集計に必要な補助列を追加して返す。"""

    work = df.copy()
    work["start_time"] = pd.to_datetime(work["start_time"], utc=False, errors="coerce")
    work["end_time"] = pd.to_datetime(work["end_time"], utc=False, errors="coerce")

    work = work.loc[work["start_time"].notna() & work["end_time"].notna()].copy()
    work = work.loc[work["end_time"] > work["start_time"]]

    keep_cols = ["hashvin", "session_cluster", "session_type", "start_time", "end_time"]
    if "duration_minutes" in work.columns:
        keep_cols.append("duration_minutes")
    work = work[keep_cols].copy()

    work["session_cluster"] = work["session_cluster"].where(work["session_cluster"].notna(), "UNKNOWN")
    work["session_cluster"] = work["session_cluster"].astype(str)

    work["start_date"] = work["start_time"].dt.date
    work["weekday"] = work["start_time"].dt.weekday

    work.sort_values(["hashvin", "start_time"], inplace=True)
    work.reset_index(drop=True, inplace=True)
    return work


def _build_day_type_table(df: pd.DataFrame) -> pd.DataFrame:
    """各日が充電あり日か充電なし日かを判定したテーブルを作成する。"""

    base_days = (
        df[["hashvin", "start_date", "weekday"]]
        .drop_duplicates()
        .copy()
    )

    charging_days = (
        df.loc[df["session_type"] == "charging", ["hashvin", "start_date"]]
        .drop_duplicates()
        .assign(day_type="with")
    )

    day_table = base_days.merge(charging_days, on=["hashvin", "start_date"], how="left")
    day_table["day_type"] = day_table["day_type"].fillna("without")

    iso = pd.to_datetime(day_table["start_date"]).dt.isocalendar()
    day_table["iso_year"] = iso["year"]
    day_table["iso_week"] = iso["week"]
    day_table["iso_yearweek"] = (
        day_table["iso_year"].astype(str)
        + "-W"
        + day_table["iso_week"].astype(str).str.zfill(2)
    )

    return day_table


def _compute_panel_coverage(
    day_table: pd.DataFrame,
    charging_sessions: pd.DataFrame,
) -> tuple[pd.DataFrame, Dict[Tuple[str, PanelKey], set[str]]]:
    """パネルごとの観測週集合と週数テーブルを作成する。"""

    panel_week_sets: Dict[Tuple[str, PanelKey], set[str]] = {}

    for row in day_table.itertuples(index=False):
        panel = f"inactive_{row.day_type}"
        panel_week_sets.setdefault((row.hashvin, panel), set()).add(row.iso_yearweek)

    if not charging_sessions.empty:
        iso = charging_sessions["start_time"].dt.isocalendar()
        charging_sessions = charging_sessions.assign(
            iso_yearweek=(
                iso["year"].astype(str)
                + "-W"
                + iso["week"].astype(str).str.zfill(2)
            )
        )
        for row in charging_sessions.itertuples(index=False):
            panel_week_sets.setdefault((row.hashvin, "charging"), set()).add(row.iso_yearweek)

    coverage_records = [
        {"hashvin": hv, "panel": panel, "weeks": len(weeks)}
        for (hv, panel), weeks in panel_week_sets.items()
    ]
    coverage_df = pd.DataFrame(coverage_records)
    return coverage_df, panel_week_sets


def _validate_coverage(coverage: pd.DataFrame, min_weeks: int) -> None:
    """観測週数が要件を満たしているか検証する。"""

    if coverage.empty:
        raise RequirementError("観測週数を算出できなかったため、分母を計算できません。")

    if (coverage["weeks"] <= 0).any():
        raise RequirementError("観測週数が 0 のパネルがあるため、割合を算出できません。")

    insufficient = coverage.loc[coverage["weeks"] < min_weeks]
    if not insufficient.empty:
        detail = ", ".join(
            f"{row.hashvin}:{row.panel}={row.weeks}週"
            for row in insufficient.itertuples(index=False)
        )
        raise RequirementError(f"観測週数が min_coverage_weeks を下回っています: {detail}")


def _build_denominator_lookups(
    coverage: pd.DataFrame,
    week_sets: Dict[Tuple[str, PanelKey], set[str]],
    mode: str,
) -> tuple[Dict[Tuple[str, PanelKey], float], Dict[Tuple[str, PanelKey], float]]:
    """分母となる週数×1h を求め、辞書として返す。"""

    denominator_hours: Dict[Tuple[str, PanelKey], float] = {}
    weeks_lookup: Dict[Tuple[str, PanelKey], float] = {}

    if mode == "per_panel":
        for row in coverage.itertuples(index=False):
            key = (row.hashvin, row.panel)
            weeks = float(row.weeks)
            denominator_hours[key] = weeks
            weeks_lookup[key] = weeks
        return denominator_hours, weeks_lookup

    for hashvin, subset in coverage.groupby("hashvin"):
        panels = subset["panel"].tolist()
        week_sets_for_vehicle = [week_sets.get((hashvin, panel), set()) for panel in panels]
        shared_weeks = set.intersection(*week_sets_for_vehicle) if week_sets_for_vehicle and all(week_sets_for_vehicle) else set()

        if shared_weeks:
            shared_len = float(len(shared_weeks))
            for panel in panels:
                key = (hashvin, panel)
                denominator_hours[key] = shared_len
                weeks_lookup[key] = shared_len
        else:
            for row in subset.itertuples(index=False):
                key = (hashvin, row.panel)
                weeks = float(row.weeks)
                denominator_hours[key] = weeks
                weeks_lookup[key] = weeks

    return denominator_hours, weeks_lookup


def _collect_hourly_records(
    df: pd.DataFrame,
    panel_resolver,
) -> pd.DataFrame:
    """1 時間ビン単位にイベントを分解し、指定パネルに割り当てる。"""

    records: List[Dict[str, object]] = []

    for row in df.itertuples(index=False):
        panel = panel_resolver(row)
        if panel is None:
            continue

        window_start = datetime.combine(row.start_date, time(hour=6))
        window_end = window_start + timedelta(hours=24)

        clipped_start = max(row.start_time, window_start)
        clipped_end = min(row.end_time, window_end)
        if clipped_start >= clipped_end:
            continue

        total_hours = (clipped_end - clipped_start).total_seconds() / 3600.0
        if total_hours <= 0:
            continue

        current = clipped_start
        while current < clipped_end:
            hour_start = current.replace(minute=0, second=0, microsecond=0)
            next_hour = hour_start + timedelta(hours=1)
            bucket_end = min(next_hour, clipped_end)
            overlap = (bucket_end - current).total_seconds() / 3600.0
            if overlap > 0:
                records.append(
                    {
                        "hashvin": row.hashvin,
                        "panel": panel,
                        "weekday": row.weekday,
                        "hour": hour_start.hour,
                        "session_cluster": row.session_cluster,
                        "duration_hours": overlap,
                        "event_share": overlap / total_hours,
                    }
                )
            current = bucket_end

    return pd.DataFrame.from_records(records)


def _aggregate_panel_metrics(
    normalized: pd.DataFrame,
    day_type_table: pd.DataFrame,
    charging_sessions: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    """放置・充電イベントを時間ビンごとに集計し、割合計算に必要な値を準備する。"""

    day_type_map = {
        (row.hashvin, row.start_date): row.day_type
        for row in day_type_table.itertuples(index=False)
    }

    inactive_df = normalized.loc[normalized["session_type"] == "inactive"]
    charging_df = charging_sessions

    def inactive_panel(row) -> PanelKey:
        day_type = day_type_map.get((row.hashvin, row.start_date), "without")
        return f"inactive_{day_type}"

    def charging_panel(_row) -> PanelKey:
        return "charging"

    inactive_records = _collect_hourly_records(inactive_df, inactive_panel)
    charging_records = _collect_hourly_records(charging_df, charging_panel)

    frames = [df for df in [inactive_records, charging_records] if not df.empty]
    if not frames:
        return pd.DataFrame(
            columns=[
                "hashvin",
                "panel",
                "weekday",
                "hour",
                "session_cluster",
                "duration_hours",
                "event_count",
                "metric_value",
            ]
        )

    combined = pd.concat(frames, ignore_index=True)
    aggregated = (
        combined.groupby(
            ["hashvin", "panel", "weekday", "hour", "session_cluster"],
            as_index=False,
        )
        .agg(
            duration_hours=("duration_hours", "sum"),
            event_count=("event_share", "sum"),
        )
        .sort_values(["hashvin", "panel", "weekday", "hour", "session_cluster"])
        .reset_index(drop=True)
    )
    if aggregated.empty:
        return aggregated

    if metric == "duration_ratio":
        aggregated["metric_value"] = aggregated["duration_hours"]
    else:
        aggregated["metric_value"] = aggregated["event_count"]

    return aggregated


def _select_top_clusters(aggregated: pd.DataFrame, top_n: int) -> Dict[str, set[str]]:
    """各車両で表示対象とする上位クラスタを選び出す。"""

    if aggregated.empty or top_n <= 0:
        return {}

    selection: Dict[str, set[str]] = {}
    inactive_panels = {"inactive_with", "inactive_without"}

    for hashvin, hv_df in aggregated.groupby("hashvin"):
        ranking_source = hv_df.loc[hv_df["panel"].isin(inactive_panels)]
        if ranking_source.empty:
            ranking_source = hv_df

        order = (
            ranking_source.groupby("session_cluster")["duration_hours"]
            .sum()
            .sort_values(ascending=False)
        )
        top_clusters = order.head(top_n).index.tolist()
        selection[hashvin] = set(map(str, top_clusters))

    return selection


def _collapse_other_clusters(
    aggregated: pd.DataFrame,
    top_cluster_map: Dict[str, set[str]],
) -> pd.DataFrame:
    """上位以外のクラスタを OTHER にまとめて再集計する。"""

    if aggregated.empty:
        return aggregated

    work = aggregated.copy()

    def normalize_cluster(row) -> str:
        allowed = top_cluster_map.get(row.hashvin)
        label = str(row.session_cluster)
        if not allowed:
            return label
        return label if label in allowed else "OTHER"

    work["session_cluster"] = work.apply(normalize_cluster, axis=1)

    collapsed = (
        work.groupby(["hashvin", "panel", "weekday", "hour", "session_cluster"], as_index=False)
        .agg(
            duration_hours=("duration_hours", "sum"),
            event_count=("event_count", "sum"),
            metric_value=("metric_value", "sum"),
        )
        .sort_values(["hashvin", "panel", "weekday", "hour", "session_cluster"])
        .reset_index(drop=True)
    )
    return collapsed


def _compute_cell_ratios(
    aggregated: pd.DataFrame,
    denominator_lookup: Dict[tuple[str, PanelKey], float],
) -> pd.DataFrame:
    """集計値を分母（週数）で割り、ヒートマップ用のセル値を算出する。"""

    if aggregated.empty:
        return aggregated.assign(
            denominator_weeks=pd.Series(dtype=float),
            cell_ratio=pd.Series(dtype=float),
        )

    work = aggregated.copy()

    def resolve_weeks(row) -> float:
        key = (row.hashvin, row.panel)
        if key not in denominator_lookup:
            raise RequirementError(f"分母が定義されていないパネルです: {key}")
        return float(denominator_lookup[key])

    work["denominator_weeks"] = work.apply(resolve_weeks, axis=1)
    work["cell_ratio"] = np.where(
        work["denominator_weeks"] > 0,
        work["metric_value"] / work["denominator_weeks"],
        0.0,
    )
    work["cell_ratio"] = work["cell_ratio"].fillna(0.0)
    return work


def _build_cluster_orders(ratios: pd.DataFrame, other_position: str) -> Dict[str, Dict[str, List[str]]]:
    """クラスタ表示順をパネル別に決定して返す。"""

    orders: Dict[str, Dict[str, List[str]]] = {}

    if ratios.empty:
        return orders

    for hashvin, hv_df in ratios.groupby("hashvin"):
        hv_orders: Dict[str, List[str]] = {}
        for panel, panel_df in hv_df.groupby("panel"):
            ranking = (
                panel_df.loc[panel_df["session_cluster"] != "OTHER"]
                .groupby("session_cluster")["duration_hours"]
                .sum()
                .sort_values(ascending=False)
            )
            clusters = ranking.index.tolist()

            if (panel_df["session_cluster"] == "OTHER").any():
                if other_position == "top":
                    clusters = ["OTHER"] + clusters
                else:
                    clusters.append("OTHER")

            if not clusters:
                clusters = ["OTHER"]

            hv_orders[panel] = [str(c) for c in clusters]

        # inactive パネル同士は同一順序になるよう統一
        inactive_orders = [
            hv_orders.get("inactive_without", []),
            hv_orders.get("inactive_with", []),
        ]
        merged: List[str] = []
        for order in inactive_orders:
            for cluster in order:
                if cluster not in merged:
                    merged.append(cluster)
        if merged:
            hv_orders["inactive_without"] = merged.copy()
            hv_orders["inactive_with"] = merged.copy()

        orders[hashvin] = hv_orders

    return orders


def _build_denominator_table(
    hashvin: str,
    weeks_lookup: Dict[tuple[str, PanelKey], float],
    denominator_mode: str,
) -> pd.DataFrame:
    """分母情報を CSV 出力しやすい形にまとめる。"""

    records: List[Dict[str, object]] = []

    for panel in PANEL_SEQUENCE:
        weeks = weeks_lookup.get((hashvin, panel))
        if weeks is None:
            continue
        records.append(
            {
                "hashvin": hashvin,
                "panel": panel,
                "panel_label": PANEL_TITLES.get(panel, panel),
                "weeks": float(weeks),
                "denominator_hours": float(weeks),  # 1 時間ビンあたり 1h × 週数
                "denominator_mode": denominator_mode,
            }
        )

    return pd.DataFrame(records)


def _build_weekday_panel_payload(
    hashvin: str,
    weekday: int,
    weekday_df: pd.DataFrame,
    cluster_orders: Dict[str, Dict[str, List[str]]],
    panels: Iterable[PanelKey],
    metric: str,
) -> Dict[str, object]:
    """描画・CSV 出力で使う曜日別データを整形する。"""

    hours = DISPLAY_HOURS
    payload_panels: Dict[PanelKey, Dict[str, object]] = {}
    panel_order_map: Dict[PanelKey, List[str]] = {}
    hv_orders = cluster_orders.get(hashvin, {})

    for panel in panels:
        panel_df = weekday_df.loc[weekday_df["panel"] == panel].copy()
        requested_order = hv_orders.get(panel, [])

        if not requested_order:
            unique_clusters = (
                panel_df["session_cluster"].astype(str).unique().tolist()
                if not panel_df.empty
                else []
            )
            requested_order = sorted(unique_clusters) if unique_clusters else ["OTHER"]

        clusters = [str(c) for c in requested_order]
        panel_order_map[panel] = clusters

        ratio_matrix = pd.DataFrame(0.0, index=clusters, columns=hours)
        duration_matrix = pd.DataFrame(0.0, index=clusters, columns=hours)
        count_matrix = pd.DataFrame(0.0, index=clusters, columns=hours)

        if not panel_df.empty:
            panel_df["session_cluster"] = panel_df["session_cluster"].astype(str)
            ratio_pivot = panel_df.pivot_table(
                index="session_cluster",
                columns="hour",
                values="cell_ratio",
                aggfunc="sum",
                fill_value=0.0,
            )
            duration_pivot = panel_df.pivot_table(
                index="session_cluster",
                columns="hour",
                values="duration_hours",
                aggfunc="sum",
                fill_value=0.0,
            )
            count_pivot = panel_df.pivot_table(
                index="session_cluster",
                columns="hour",
                values="event_count",
                aggfunc="sum",
                fill_value=0.0,
            )

            ratio_matrix = ratio_pivot.reindex(index=clusters, columns=hours, fill_value=0.0)
            duration_matrix = duration_pivot.reindex(index=clusters, columns=hours, fill_value=0.0)
            count_matrix = count_pivot.reindex(index=clusters, columns=hours, fill_value=0.0)

        denominator = float(panel_df["denominator_weeks"].iloc[0]) if not panel_df.empty else math.nan
        numerator_matrix = duration_matrix if metric == "duration_ratio" else count_matrix
        payload_panels[panel] = {
            "ratio": ratio_matrix,
            "duration": duration_matrix,
            "count": count_matrix,
            "numerator": numerator_matrix,
            "denominator_weeks": denominator,
            "panel_label": PANEL_TITLES.get(panel, panel),
            "total_duration_hours": float(duration_matrix.values.sum()),
            "total_events": float(count_matrix.values.sum()),
        }

    return {
        "hashvin": hashvin,
        "weekday": weekday,
        "hours": hours,
        "panels": payload_panels,
        "panel_orders": panel_order_map,
        "metric": metric,
    }


def _export_panel_matrices(
    panel_payload: Dict[str, object],
    output_dir: Path,
    weekday: int,
) -> None:
    """曜日ごとの行列を CSV に書き出す。"""

    for panel, info in panel_payload["panels"].items():
        suffix = panel.replace("inactive_", "")
        ratio_df = info["ratio"].copy()
        numerator_df = info["numerator"].copy()

        ratio_df.index.name = "session_cluster"
        numerator_df.index.name = "session_cluster"

        hour_labels = [f"{hour:02d}" for hour in ratio_df.columns]
        ratio_df.columns = hour_labels
        numerator_df.columns = hour_labels

        ratio_path = output_dir / f"weekday_{weekday}_matrix_{suffix}.csv"
        numerator_path = output_dir / f"weekday_{weekday}_numerator_{suffix}.csv"

        ratio_df.to_csv(ratio_path, encoding="utf-8-sig")
        numerator_df.to_csv(numerator_path, encoding="utf-8-sig")


def _render_weekday_comparison(
    hashvin: str,
    weekday: int,
    panel_payload: Dict[str, object],
    weeks_lookup: Dict[tuple[str, PanelKey], float],
    metric: str,
    clip_percentile: float,
    share_color_scale: bool,
    annotate_cells: bool,
) -> plt.Figure:
    """曜日別の 3 面ヒートマップを描画する。"""

    panels = [panel for panel in PANEL_SEQUENCE if panel in panel_payload["panels"]]
    if not panels:
        panels = PANEL_SEQUENCE

    hours = panel_payload.get("hours", DISPLAY_HOURS)
    panel_orders = panel_payload.get("panel_orders", {})

    reindexed_arrays = []
    for panel in panels:
        info = panel_payload["panels"].get(panel)
        clusters = panel_orders.get(panel) or (info["ratio"].index.tolist() if info else [])
        if not clusters:
            clusters = ["OTHER"]
        ratio_df = (
            info["ratio"].reindex(index=clusters, columns=hours, fill_value=0.0)
            if info is not None
            else pd.DataFrame(0.0, index=clusters, columns=hours)
        )
        reindexed_arrays.append(ratio_df.to_numpy().ravel())

    merged = np.concatenate([arr for arr in reindexed_arrays if arr.size > 0], axis=0) if any(
        arr.size for arr in reindexed_arrays
    ) else np.array([0.0])

    vmax = float(np.percentile(merged, clip_percentile * 100)) if merged.size > 0 else 0.0
    if vmax <= 0:
        vmax = 1.0

    max_rows = max((len(panel_orders.get(panel) or []) for panel in panels), default=1)
    if max_rows <= 0:
        max_rows = 1
    # height = max(2.5, 0.45 * max_rows + 1.6)
    # width = max(4.0 * len(panels), 6.0)

    figure, axes = plt.subplots(
        1,
        len(panels),
        figsize=(25, 12),
        sharey=False,
    )
    if len(panels) == 1:
        axes = [axes]

    sns.set_style("whitegrid")

    for ax, panel in zip(axes, panels):
        info = panel_payload["panels"].get(panel)
        clusters = panel_orders.get(panel) or (info["ratio"].index.tolist() if info else [])
        if not clusters:
            clusters = ["OTHER"]

        ratio_df = (
            info["ratio"].reindex(index=clusters, columns=hours, fill_value=0.0)
            if info is not None
            else pd.DataFrame(0.0, index=clusters, columns=hours)
        )
        duration_df = (
            info["duration"].reindex(index=clusters, columns=hours, fill_value=0.0)
            if info is not None
            else pd.DataFrame(0.0, index=clusters, columns=hours)
        )
        count_df = (
            info["count"].reindex(index=clusters, columns=hours, fill_value=0.0)
            if info is not None
            else pd.DataFrame(0.0, index=clusters, columns=hours)
        )

        panel_vmax = vmax
        if not share_color_scale:
            flat = ratio_df.to_numpy().ravel()
            panel_vmax = float(np.percentile(flat, clip_percentile * 100)) if flat.size else 1.0
            if panel_vmax <= 0:
                panel_vmax = 1.0

        annot_matrix = None
        if annotate_cells:
            percent_matrix = ratio_df.to_numpy() * 100.0
            count_matrix = count_df.to_numpy()
            duration_matrix = duration_df.to_numpy()
            annot_matrix = np.empty(percent_matrix.shape, dtype=object)
            for i in range(percent_matrix.shape[0]):
                for j in range(percent_matrix.shape[1]):
                    pct = percent_matrix[i, j]
                    cnt = count_matrix[i, j]
                    dur = duration_matrix[i, j]
                    cnt_text = f"{cnt:.1f}".rstrip('0').rstrip('.')
                    dur_text = f"{dur:.1f}".rstrip('0').rstrip('.')
                    annot_matrix[i, j] = f"{pct:.1f}%\n{cnt_text}件/{dur_text}h"

        sns.heatmap(
            ratio_df,
            ax=ax,
            cmap="YlGnBu",
            vmin=0,
            vmax=panel_vmax,
            cbar=True,
            annot=annot_matrix if annot_matrix is not None else False,
            fmt="" if annot_matrix is not None else ".2f",
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"format": FuncFormatter(lambda val, _: f"{val:.2f}")},
        )

        ax.set_xticks(np.arange(len(hours)) + 0.5)
        ax.set_yticks(np.arange(len(clusters)) + 0.5)
        ax.set_xlabel("hour")
        ylabel = "cluster"
        if panel == "charging":
            ylabel = "charging cluster"
        ax.set_ylabel(ylabel if ax is axes[0] else (ylabel if panel == "charging" else ""))
        ax.set_xticklabels([f"{h:02d}" for h in hours], rotation=45, ha="right")
        y_labels = [str(c) for c in clusters]
        ax.set_yticklabels(y_labels, rotation=0)

        denom_weeks = info.get("denominator_weeks") if info else math.nan
        if np.isnan(denom_weeks):
            denom_weeks = float(weeks_lookup.get((hashvin, panel), math.nan))
        if np.isnan(denom_weeks) or denom_weeks <= 0:
            denom_text = "weeks: –"
        else:
            denom_text = f"weeks: {denom_weeks:.0f}h"
        ax.set_title(f"{PANEL_TITLES.get(panel, panel)}\n({denom_text})", fontsize=11)

    weekday_label = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][weekday % 7]
    scale_label = f"0–p{int(clip_percentile * 100)}"

    denom_values = {panel: weeks_lookup.get((hashvin, panel)) for panel in panels}
    denom_unique = {val for val in denom_values.values() if val is not None}
    if len(denom_unique) == 1:
        denom_header = f"denom: {next(iter(denom_unique)):.0f}h"
    else:
        denom_header = "denom: per panel"

    figure.suptitle(
        f"{hashvin} | Weekday: {weekday_label} | scale: {scale_label} | {denom_header}",
        fontsize=13,
    )

    summary_parts: List[str] = []
    label_map = {
        "inactive_without": "Σ滞在(充電なし)",
        "inactive_with": "Σ滞在(充電あり)",
        "charging": "Σ充電",
    }
    value_unit = "h" if metric == "duration_ratio" else "件"
    for panel in panels:
        info = panel_payload["panels"].get(panel)
        if info is None:
            continue
        numerator_total = info["numerator"].to_numpy().sum()
        denom_weeks = info.get("denominator_weeks")
        if np.isnan(denom_weeks):
            denom_weeks = float(weeks_lookup.get((hashvin, panel), math.nan))
        if np.isnan(denom_weeks):
            weeks_text = "週数=–"
        else:
            weeks_text = f"週数={denom_weeks:.0f}h"
        summary_parts.append(
            f"{label_map.get(panel, panel)}={numerator_total:.1f}{value_unit} / {weeks_text}"
        )

    if summary_parts:
        figure.text(
            0.5,
            0.02,
            " | ".join(summary_parts),
            ha="center",
            va="bottom",
            fontsize=10,
        )

    figure.tight_layout(rect=(0, 0.05, 1, 0.94))
    return figure


def _compute_similarity_row(
    hashvin: str,
    weekday: int,
    panel_payload: Dict[str, object],
    weeks_lookup: Dict[tuple[str, PanelKey], float],
) -> Dict[str, object]:
    """充電あり・なしの行動類似度指標を算出する。"""

    with_panel = panel_payload["panels"].get("inactive_with")
    without_panel = panel_payload["panels"].get("inactive_without")

    if with_panel is None or without_panel is None:
        return {
            "hashvin": hashvin,
            "weekday": weekday,
            "pearson_corr": math.nan,
            "cosine_sim": math.nan,
            "js_distance": math.nan,
            "covered_weeks_with": weeks_lookup.get((hashvin, "inactive_with"), math.nan),
            "covered_weeks_without": weeks_lookup.get((hashvin, "inactive_without"), math.nan),
        }

    vec_with = with_panel["ratio"].to_numpy().ravel()
    vec_without = without_panel["ratio"].to_numpy().ravel()

    def safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
        if a.size == 0 or b.size == 0:
            return math.nan
        if np.allclose(a, a[0]) and np.allclose(b, b[0]):
            return math.nan
        if np.std(a) == 0 or np.std(b) == 0:
            return math.nan
        return float(np.corrcoef(a, b)[0, 1])

    def safe_cosine(a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return math.nan
        return float(np.dot(a, b) / (norm_a * norm_b))

    def safe_js_distance(a: np.ndarray, b: np.ndarray) -> float:
        a = np.maximum(a, 0)
        b = np.maximum(b, 0)
        if a.sum() == 0 and b.sum() == 0:
            return 0.0
        if a.sum() == 0 or b.sum() == 0:
            return 1.0
        a = a / a.sum()
        b = b / b.sum()
        m = 0.5 * (a + b)

        def _kl_div(p: np.ndarray, q: np.ndarray) -> float:
            mask = (p > 0) & (q > 0)
            if not np.any(mask):
                return 0.0
            return float(np.sum(p[mask] * np.log2(p[mask] / q[mask])))

        js_div = 0.5 * _kl_div(a, m) + 0.5 * _kl_div(b, m)
        js_div = max(js_div, 0.0)
        return float(math.sqrt(min(js_div, 1.0)))

    return {
        "hashvin": hashvin,
        "weekday": weekday,
        "pearson_corr": safe_pearson(vec_without, vec_with),
        "cosine_sim": safe_cosine(vec_without, vec_with),
        "js_distance": safe_js_distance(vec_without, vec_with),
        "covered_weeks_with": weeks_lookup.get((hashvin, "inactive_with"), math.nan),
        "covered_weeks_without": weeks_lookup.get((hashvin, "inactive_without"), math.nan),
    }


__all__ = [
    "ParkingAnalysisConfig",
    "RequirementError",
    "run_weekday_parking_analysis",
]
