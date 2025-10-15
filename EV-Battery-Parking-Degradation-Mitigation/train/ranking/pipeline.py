"""hashvin（車両）ごとのランキング学習と推論をまとめたパイプライン。"""

from __future__ import annotations


import json


from dataclasses import dataclass


from pathlib import Path


from typing import Dict, Iterable, List, Optional, Tuple, Union


import numpy as np


import pandas as pd


from autogluon.tabular import TabularPredictor


from .config import RankingConfig


from .evaluation import evaluate_user_model, summarize_validation_scores


from .features import build_training_table


from .utils import haversine_km


# ---------------------------------------------------------------------------


# 共通前処理


# ---------------------------------------------------------------------------


def load_sessions(csv_path: Path) -> pd.DataFrame:
    """セッションCSVから学習に必要な共通派生列を作成して返す。"""

    df = pd.read_csv(csv_path)
    # 入力直後に開始・終了時刻を日本時間へ統一し、タイムゾーン差異を吸収する

    for col in ["start_time", "end_time"]:
        ts = pd.to_datetime(df[col], errors="coerce")

        if pd.api.types.is_datetime64tz_dtype(ts.dtype):
            df[col] = ts.dt.tz_convert("Asia/Tokyo").dt.tz_localize(None)

        else:
            df[col] = ts.dt.tz_localize(
                "Asia/Tokyo", nonexistent="shift_forward", ambiguous="NaT"
            ).dt.tz_localize(None)

    df = df.sort_values(["hashvin", "start_time"]).reset_index(drop=True)
    # 滞在時間や曜日・開始時刻などモデルで使う派生列をまとめて追加する

    df["duration_minutes"] = pd.to_numeric(df["duration_minutes"], errors="coerce")

    df["weekday"] = df["start_time"].dt.dayofweek

    df["start_hour"] = df["start_time"].dt.hour

    df["date"] = df["start_time"].dt.date

    df["is_long_park"] = (df["session_type"] == "inactive") & (
        df["duration_minutes"] >= 360
    )

    return df


# ---------------------------------------------------------------------------


# ユーザー（hashvin）単位の統計データ


# ---------------------------------------------------------------------------


@dataclass
class UserData:
    """ユーザー単位で学習に必要なテーブルを保持するデータクラス。"""

    hashvin: str

    sessions: pd.DataFrame

    links: pd.DataFrame

    presence: pd.DataFrame

    start_prob: pd.DataFrame

    charge_prior: pd.DataFrame

    hour_prior: pd.DataFrame

    cluster_profile: pd.DataFrame


class UserDataBuilder:
    """hashvin ごとの統計テーブルを構築するヘルパー。"""

    def __init__(self, sessions: pd.DataFrame, config: RankingConfig) -> None:
        self.sessions = sessions.copy()

        self.config = config

        self.ref_time = self.sessions["start_time"].max()

        self._annotate_time_weight()

    # ---- 内部ユーティリティ --------------------------------------------------

    def _annotate_time_weight(self) -> None:
        """イベントの経過日数と時間減衰重みを計算する。"""

        # 参照時刻からの経過日数を日単位で計算し、減衰重みの基礎とする
        age_days = (
            self.ref_time - self.sessions["start_time"]
        ).dt.total_seconds() / 86400

        self.sessions["age_days"] = age_days

        if self.config.use_decay_weight and self.config.halflife_days > 0:
            # 半減期パラメータに従って、古いイベントほど指数的に重みを下げる
            weight = np.exp(-np.log(2) * age_days / self.config.halflife_days)

        else:
            weight = np.ones(len(self.sessions))

        if self.config.window_days > 0:
            # 指定期間より前のイベントは学習対象から除外するため重み0にする
            weight = np.where(age_days <= self.config.window_days, weight, 0.0)

        self.sessions["time_weight"] = weight

    # ---- テーブル構築 --------------------------------------------------

    def build_links(self) -> pd.DataFrame:
        """充電イベントに続く最初の長時間放置をリンクしたテーブルを作成する。"""

        charges = self.sessions[self.sessions["session_type"] == "charging"].copy()

        long_parks = self.sessions[self.sessions["is_long_park"]].copy()

        rows: List[Dict[str, object]] = []

        prev_charge_cluster = None

        prev_charge_end = None

        prev_end_soc = None

        event_id = 0

        for _, charge in charges.iterrows():
            event_id += 1

            start = charge["start_time"]

            end = charge["end_time"]

            subsequent_charges = charges[charges["start_time"] > end]

            next_charge_start = (
                subsequent_charges["start_time"].min()
                if not subsequent_charges.empty
                else None
            )

            if next_charge_start is not None:
                candidate_long = long_parks[
                    (long_parks["start_time"] >= end)
                    & (long_parks["start_time"] < next_charge_start)
                ]

            else:
                candidate_long = long_parks[long_parks["start_time"] >= end]

            candidate_long = candidate_long.sort_values("start_time")

            first_long = candidate_long.head(1)

            park_cluster = (
                first_long["session_cluster"].iloc[0]
                if not first_long.empty
                else np.nan
            )

            park_start = (
                first_long["start_time"].iloc[0] if not first_long.empty else pd.NaT
            )

            park_end = (
                first_long["end_time"].iloc[0] if not first_long.empty else pd.NaT
            )

            park_duration = (
                first_long["duration_minutes"].iloc[0]
                if not first_long.empty
                else np.nan
            )

            park_lat = (
                first_long["start_lat"].iloc[0] if not first_long.empty else np.nan
            )

            park_lon = (
                first_long["start_lon"].iloc[0] if not first_long.empty else np.nan
            )

            gap_minutes = (
                (park_start - end).total_seconds() / 60
                if pd.notnull(park_start)
                else np.nan
            )

            dist = haversine_km(
                charge.get("end_lat", np.nan),
                charge.get("end_lon", np.nan),
                park_lat,
                park_lon,
            )

            age_days = (self.ref_time - start).total_seconds() / 86400

            weight = (
                np.exp(-np.log(2) * age_days / self.config.halflife_days)
                if (self.config.use_decay_weight and self.config.halflife_days > 0)
                else 1.0
            )

            if self.config.window_days > 0 and age_days > self.config.window_days:
                weight = 0.0

            time_since_last = (
                (start - prev_charge_end).total_seconds() / 60
                if prev_charge_end is not None
                else np.nan
            )

            soc_drop = (
                (prev_end_soc - charge.get("start_soc", np.nan))
                if prev_end_soc is not None
                else np.nan
            )

            rows.append(
                {
                    "hashvin": charge["hashvin"],
                    "event_id": event_id,
                    "weekday": charge["weekday"],
                    "charge_cluster": str(charge["session_cluster"]),
                    "charge_start_time": start,
                    "charge_start_hour": charge["start_hour"],
                    "charge_end_time": end,
                    "charge_end_lat": charge.get("end_lat", np.nan),
                    "charge_end_lon": charge.get("end_lon", np.nan),
                    "park_cluster": park_cluster
                    if pd.isna(park_cluster)
                    else str(park_cluster),
                    "park_start_time": park_start,
                    "park_start_hour": park_start.hour
                    if pd.notnull(park_start)
                    else np.nan,
                    "park_duration_minutes": park_duration,
                    "park_start_lat": park_lat,
                    "park_start_lon": park_lon,
                    "gap_minutes": gap_minutes,
                    "dist_charge_to_park_km": dist,
                    "age_days": age_days,
                    "weight_time": weight,
                    "start_soc": charge.get("start_soc", np.nan),
                    "end_soc": charge.get("end_soc", np.nan),
                    "time_since_last_charge_min": time_since_last,
                    "soc_drop_since_prev": soc_drop,
                    "prev_charge_cluster": prev_charge_cluster
                    if prev_charge_cluster is None
                    else str(prev_charge_cluster),
                }
            )

            prev_charge_cluster = charge["session_cluster"]

            prev_charge_end = end

            prev_end_soc = charge.get("end_soc", np.nan)

        return pd.DataFrame(rows)

    def build_presence(self) -> pd.DataFrame:
        """曜日×時間帯ごとの存在確率テーブルを作成する。"""

        records: List[Dict[str, object]] = []
        # 6時間以上の放置で減衰重みが正の行だけを取り出す
        long_df = self.sessions[
            self.sessions["is_long_park"] & (self.sessions["time_weight"] > 0)
        ]

        for _, row in long_df.iterrows():
            start = row["start_time"]

            end = row["end_time"]

            weight = row["time_weight"]

            current = start.floor("H")

            if current > start:
                current -= pd.Timedelta(hours=1)

            while current < end:
                nxt = current + pd.Timedelta(hours=1)
                # 1時間ごとの重なりを分計算で取得し、滞在時間として積算する
                overlap = min(end, nxt) - max(start, current)

                minutes = overlap.total_seconds() / 60

                if minutes > 0:
                    weekday = (max(start, current)).dayofweek

                    records.append(
                        {
                            "weekday": weekday,
                            "hour": current.hour,
                            "cluster": str(row["session_cluster"]),
                            "weight": weight * (minutes / 60),
                        }
                    )

                current = nxt

        if not records:
            return pd.DataFrame(
                columns=[
                    "weekday",
                    "hour",
                    "cluster",
                    "presence_weight",
                    "presence_prob",
                    "long_park_ratio",
                ]
            )

        df = pd.DataFrame(records)

        grouped = (
            df.groupby(["weekday", "hour", "cluster"], as_index=False)["weight"]
            .sum()
            .rename(columns={"weight": "presence_weight"})
        )

        totals = (
            grouped.groupby(["weekday", "hour"], as_index=False)["presence_weight"]
            .sum()
            .rename(columns={"presence_weight": "total_weight"})
        )

        merged = grouped.merge(totals, on=["weekday", "hour"], how="left")

        alpha = self.config.alpha_smooth

        clusters = grouped["cluster"].nunique()

        # 平滑化しながら「曜日×時間帯にそのクラスタへ滞在する確率」をpresence_probとして持つ
        merged["presence_prob"] = (merged["presence_weight"] + alpha) / (
            merged["total_weight"] + alpha * clusters
        )

        cluster_total = (
            merged.groupby("cluster")["presence_weight"]
            .sum()
            .rename("cluster_total")
            .reset_index()
        )

        merged = merged.merge(cluster_total, on="cluster", how="left")

        total_sum = cluster_total["cluster_total"].sum()

        # クラスタ全体における滞在比率（どの放置先がホーム的か）も保持する
        merged["long_park_ratio"] = (
            merged["cluster_total"] / total_sum if total_sum > 0 else 0.0
        )

        return merged.drop(columns=["total_weight", "cluster_total"], errors="ignore")

    def build_start_prob(self) -> pd.DataFrame:
        """曜日×開始時刻ごとの長時間放置開始確率を計算する。"""

        df = self.sessions[
            self.sessions["is_long_park"] & (self.sessions["time_weight"] > 0)
        ]

        if df.empty:
            return pd.DataFrame(
                columns=["weekday", "hour", "cluster", "start_prob", "start_weight"]
            )

        grouped = df.groupby(
            ["weekday", "start_hour", "session_cluster"], as_index=False
        )["time_weight"].sum()

        grouped = grouped.rename(
            columns={
                "start_hour": "hour",
                "session_cluster": "cluster",
                "time_weight": "start_weight",
            }
        )

        grouped["cluster"] = grouped["cluster"].astype(str)

        totals = (
            grouped.groupby(["weekday", "hour"], as_index=False)["start_weight"]
            .sum()
            .rename(columns={"start_weight": "total_weight"})
        )

        merged = grouped.merge(totals, on=["weekday", "hour"], how="left")

        alpha = self.config.alpha_smooth

        clusters = grouped["cluster"].nunique()

        # 曜日×開始時刻ごとに「そのクラスタから放置を始めた重み付き件数」を確率化する
        merged["start_prob"] = (merged["start_weight"] + alpha) / (
            merged["total_weight"] + alpha * clusters
        )

        return merged.drop(columns=["total_weight"], errors="ignore")

    def build_charge_prior(self, links: pd.DataFrame) -> pd.DataFrame:
        """充電クラスタ×時間帯で条件付けした放置先確率を作成する。"""

        df = links.dropna(subset=["park_cluster"]).copy()

        if df.empty:
            return pd.DataFrame(
                columns=[
                    "charge_cluster",
                    "weekday",
                    "hour",
                    "park_cluster",
                    "prob",
                    "weight",
                ]
            )

        df["charge_cluster"] = df["charge_cluster"].astype(str)

        df["park_cluster"] = df["park_cluster"].astype(str)

        grouped = df.groupby(
            ["charge_cluster", "weekday", "charge_start_hour", "park_cluster"],
            as_index=False,
        )["weight_time"].sum()

        grouped = grouped.rename(
            columns={"charge_start_hour": "hour", "weight_time": "weight"}
        )

        totals = (
            grouped.groupby(["charge_cluster", "weekday", "hour"], as_index=False)[
                "weight"
            ]
            .sum()
            .rename(columns={"weight": "total_weight"})
        )

        merged = grouped.merge(
            totals, on=["charge_cluster", "weekday", "hour"], how="left"
        )

        alpha = self.config.alpha_smooth

        count = (
            grouped.groupby(["charge_cluster", "weekday", "hour"])["park_cluster"]
            .nunique()
            .rename("cluster_count")
            .reset_index()
        )

        merged = merged.merge(
            count, on=["charge_cluster", "weekday", "hour"], how="left"
        )

        # 特定の充電クラスタからどの放置クラスタへ流れやすいかを条件付き確率で保持する
        merged["prob"] = (merged["weight"] + alpha) / (
            merged["total_weight"] + alpha * merged["cluster_count"]
        )

        return merged.drop(columns=["total_weight"], errors="ignore")

    def build_hour_prior(self, links: pd.DataFrame) -> pd.DataFrame:
        """時間帯のみで条件付けした放置先確率を作成する。"""

        df = links.dropna(subset=["park_cluster"]).copy()

        if df.empty:
            return pd.DataFrame(
                columns=["weekday", "hour", "park_cluster", "prob", "weight"]
            )

        df["park_cluster"] = df["park_cluster"].astype(str)

        grouped = df.groupby(
            ["weekday", "charge_start_hour", "park_cluster"], as_index=False
        )["weight_time"].sum()

        grouped = grouped.rename(
            columns={"charge_start_hour": "hour", "weight_time": "weight"}
        )

        totals = (
            grouped.groupby(["weekday", "hour"], as_index=False)["weight"]
            .sum()
            .rename(columns={"weight": "total_weight"})
        )

        merged = grouped.merge(totals, on=["weekday", "hour"], how="left")

        alpha = self.config.alpha_smooth

        clusters = grouped["park_cluster"].nunique()

        merged["prob"] = (merged["weight"] + alpha) / (
            merged["total_weight"] + alpha * clusters
        )

        return merged.drop(columns=["total_weight"], errors="ignore")

    def build_cluster_profile(self) -> pd.DataFrame:
        """クラスタごとの代表座標・代表時刻を算出する。"""

        df = self.sessions[self.sessions["is_long_park"]].copy()

        if df.empty:
            return pd.DataFrame(
                columns=[
                    "cluster",
                    "mean_lat",
                    "mean_lon",
                    "peak_hour",
                    "peak_hour_std",
                ]
            )

        profile = (
            df.groupby("session_cluster")
            .agg(
                mean_lat=("start_lat", "mean"),
                mean_lon=("start_lon", "mean"),
                peak_hour=("start_hour", lambda s: s.value_counts().idxmax()),
                peak_hour_std=("start_hour", "std"),
            )
            .reset_index()
            .rename(columns={"session_cluster": "cluster"})
        )

        profile["cluster"] = profile["cluster"].astype(str)

        return profile

    def build(self) -> UserData:
        """各種テーブルをまとめた UserData を返す。"""

        links = self.build_links()

        presence = self.build_presence()

        start_prob = self.build_start_prob()

        charge_prior = self.build_charge_prior(links)

        hour_prior = self.build_hour_prior(links)

        cluster_profile = self.build_cluster_profile()

        return UserData(
            hashvin=self.sessions["hashvin"].iloc[0],
            sessions=self.sessions,
            links=links,
            presence=presence,
            start_prob=start_prob,
            charge_prior=charge_prior,
            hour_prior=hour_prior,
            cluster_profile=cluster_profile,
        )


# ---------------------------------------------------------------------------


# 候補生成と特徴量化


# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------


# 学習・評価・推論


# ---------------------------------------------------------------------------


@dataclass
class TrainedUserModel:
    """学習済みモデルと関連データをまとめたコンテナ。"""

    hashvin: str

    predictor: TabularPredictor

    features: List[str]

    train_data: pd.DataFrame

    validation_data: pd.DataFrame

    training_table: pd.DataFrame

    user_data: UserData


def train_user_model(
    training_table: pd.DataFrame,
    user_data: UserData,
    config: RankingConfig,
    model_root: Optional[Path] = None,
) -> Optional[TrainedUserModel]:
    """1ユーザー分の学習データから AutoGluon モデルを学習する。"""

    if training_table.empty or training_table["label"].sum() == 0:
        return None

    if training_table["label"].nunique() < 2:
        # 正例のみ（または負例のみ）では分類器が学習できないためスキップ

        return None

    dataset = training_table.sort_values("age_days").reset_index(drop=True)

    split_idx = max(1, int(len(dataset) * 0.8))

    train_df = dataset.iloc[:split_idx]

    val_df = dataset.iloc[split_idx:]

    if val_df.empty:
        val_df = train_df.copy()

    features = [c for c in dataset.columns if c not in {"label", "event_id", "hashvin"}]

    model_path: Optional[Path] = None

    if model_root is not None:
        model_path = model_root / f"autogluon_{user_data.hashvin}"

        model_path.mkdir(parents=True, exist_ok=True)

    predictor = TabularPredictor(
        label="label",
        problem_type="binary",
        path=str(model_path) if model_path else None,
        eval_metric="roc_auc",
    )

    try:
        predictor.fit(
            train_data=train_df[[*features, "label"]],
            tuning_data=val_df[[*features, "label"]] if not val_df.empty else None,
            time_limit=config.time_limit,
            presets=config.ag_presets,
        )

    except Exception:
        # モデルが学習できない場合は None を返して呼び出し元でスキップ

        return None

    return TrainedUserModel(
        hashvin=user_data.hashvin,
        predictor=predictor,
        features=features,
        train_data=train_df[[*features, "label"]].copy(),
        validation_data=val_df[["event_id", *features, "label"]].copy(),
        training_table=dataset,
        user_data=user_data,
    )




def predict_topk_for_user(model: TrainedUserModel, top_k: int) -> pd.DataFrame:
    """学習済みモデルを用いて Top-k 候補を算出する。"""

    scoring_df = model.training_table.copy()

    proba = model.predictor.predict_proba(scoring_df[model.features])

    if isinstance(proba, pd.DataFrame):
        if 1 in proba.columns:
            scoring_df["score"] = proba[1].to_numpy()

        else:
            scoring_df["score"] = proba.iloc[:, -1].to_numpy()

    else:
        scoring_df["score"] = np.asarray(proba)

    records: List[Dict[str, object]] = []

    for event_id, group in scoring_df.groupby("event_id"):
        top_rows = group.sort_values("score", ascending=False).head(top_k)

        records.append(
            {
                "hashvin": model.hashvin,
                "event_id": event_id,
                "candidate_clusters": ",".join(
                    top_rows["candidate_cluster"].astype(str).tolist()
                ),
                "scores": ",".join(f"{s:.6f}" for s in top_rows["score"].tolist()),
            }
        )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------


# パイプライン統合（Notebook からの利用想定）


# ---------------------------------------------------------------------------


class RankingPipeline:
    """ノートブックから学習・推論をまとめて呼び出すためのラッパークラス。"""

    def __init__(
        self,
        sessions: pd.DataFrame,
        config: Optional[RankingConfig] = None,
        model_root: Optional[Path] = None,
    ) -> None:
        self.sessions = sessions

        self.config = config or RankingConfig()

        self.model_root = model_root

        self.result_root = (
            Path(self.config.result_root)
            if self.config.result_root is not None
            else None
        )

        self.user_models: Dict[str, TrainedUserModel] = {}

        self.metrics: Dict[str, Dict[str, float]] = {}

    def fit_all(self) -> Dict[str, Dict[str, float]]:
        """hashvinごとにモデルを学習・評価し、指標を集計して返す。"""

        self.user_models.clear()

        self.metrics.clear()

        for hashvin, df_user in self.sessions.groupby("hashvin"):
            builder = UserDataBuilder(df_user, self.config)

            user_data = builder.build()

            # features.build_training_table で候補生成と特徴量作成をまとめて実行
            training_table = build_training_table(user_data, self.config)

            if training_table.empty:
                continue

            trained = train_user_model(
                training_table, user_data, self.config, self.model_root
            )

            if trained is None:
                continue

            metrics, val_scored = evaluate_user_model(
                trained, self.config.topk_eval, return_scored=True
            )

            self.user_models[hashvin] = trained

            self.metrics[hashvin] = metrics

            # 評価指標と検証スコアを保存し、後から分析できるようにする

            self._save_user_results(trained, metrics, val_scored)

        return self.metrics

    def predict_all(self, top_k: int = 3) -> pd.DataFrame:
        """学習済みモデルがあるユーザーについて Top-k 候補をまとめて返す。"""

        if not self.user_models:
            raise RuntimeError("先に fit_all() を実行してください。")

        frames = [
            predict_topk_for_user(model, top_k) for model in self.user_models.values()
        ]

        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def get_user_model(self, hashvin: str) -> Optional[TrainedUserModel]:
        """特定ユーザーの学習済みモデルを取得する。"""

        return self.user_models.get(hashvin)

    def _save_user_results(
        self,
        model: TrainedUserModel,
        metrics: Dict[str, float],
        scored_validation: pd.DataFrame,
    ) -> None:
        """評価指標・検証スコア・集計結果をユーザー別に保存する。"""

        if self.result_root is None:
            return

        user_dir = self.result_root / model.hashvin

        user_dir.mkdir(parents=True, exist_ok=True)

        metrics_path = user_dir / "metrics.json"

        with metrics_path.open("w", encoding="utf-8") as fh:
            json.dump(metrics, fh, ensure_ascii=False, indent=2)

        scored_path = user_dir / "validation_scores.csv"

        scored_validation.to_csv(scored_path, index=False)

        event_summary, cluster_summary = summarize_validation_scores(
            scored_validation, self.config.topk_eval
        )

        event_path = user_dir / "event_summary.csv"

        event_summary.to_csv(event_path, index=False)

        cluster_path = user_dir / "cluster_summary.csv"

        cluster_summary.to_csv(cluster_path, index=False)

        try:
            feature_importance = model.predictor.feature_importance(model.train_data)

        except Exception:
            feature_importance = None

        if feature_importance is not None:
            fi_df = (
                feature_importance
                if isinstance(feature_importance, pd.DataFrame)
                else pd.DataFrame(feature_importance)
            )

            if not fi_df.empty:
                fi_path = user_dir / "feature_importance.csv"

                fi_df.to_csv(fi_path, index=True)
