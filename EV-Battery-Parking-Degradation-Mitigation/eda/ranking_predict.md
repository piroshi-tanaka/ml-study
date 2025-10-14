# 充電→長時間放置 ランキング予測（最終仕様）

本仕様は、充電イベント直後に最初に発生する「長時間放置（inactive, duration_minutes ≥ 360）」の場所クラスタを、AutoGluon によるランキングで予測する最終版の要件です。過去の方法は不要で、本仕様のコードのみを使用します。

---

## 1. 入力データ（セッション単位）
- 必須列: `hashvin, session_cluster, session_type (inactive/charging), start_time, end_time, duration_minutes, start_lat, start_lon, end_lat, end_lon`
- タイムゾーン: Asia/Tokyo に正規化（tz-aware は変換、naive はローカライズ）

派生列（前処理で付与）
- `weekday = start_time.dayofweek`（0=Mon … 6=Sun）
- `start_hour = start_time.hour`
- `date = start_time.date`
- `is_long_park = (session_type=='inactive') & (duration_minutes>=360)`
- `prev_* / next_*`（同一 hashvin 内でシフトして直前/直後の種別・クラスタなどを参照）

実装: `train/ranking/dataset.py` の `load_sessions`, `prepare_sessions`

---

## 2. 充電→最初の長時間放置 リンク規則（ラベル）
- 充電イベント c の終了から「次の充電開始まで」の区間に存在する「最初の長時間放置 p」を c にリンク（1件のみ）。
- 途中で別の充電が挟まる場合はリンクしない（＝該当なし）。
- 出力列（例）:
  - 充電側: `hashvin, weekday, charge_cluster, charge_start_time, charge_start_hour, charge_end_time`
  - 放置側: `park_cluster(NA可), park_start_time(NA), park_start_hour(NA), park_duration_minutes(NA)`
  - 充電→放置の距離: `dist_charge_to_park_km`（充電終了地点→放置開始地点）

実装: `train/ranking/dataset.py` の `build_charge_to_next_long_table`

---

## 3. 候補生成と特徴量（学習に使用）

候補生成（ランキング対象）
- 車両別の長時間放置クラスタ出現頻度 Top-N を候補集合に採用。
- 履歴が乏しい場合は全体 Top-N を候補として補完。
- 実装: `build_candidate_pool_per_vehicle(top_n_per_vehicle, global_top_n)`

特徴量（抜粋と計算方法）
- 充電側文脈
  - `weekday`, `charge_start_hour`
  - 循環埋め込み: `charge_hour_sin = sin(2π*hour/24)`, `charge_hour_cos = cos(2π*hour/24)`
  - `charge_cluster`
- 候補クラスタの履歴傾向
  - 人気度（相対頻度）:
    - `cand_pop_vehicle`（その車両内での長時間放置における候補クラスタの相対頻度）
    - `cand_pop_global`（全車両合算での相対頻度）
  - 時間帯傾向:
    - `cand_mean_start_hour`（候補クラスタの「長時間放置開始時刻」の平均）
    - `cand_hour_diff`（充電開始時刻と `cand_mean_start_hour` の循環距離）
  - 空間距離:
    - `dist_charge_to_cand_km`（充電終了地点→候補クラスタ重心のハーサイン距離）
    - `cand_hist_count`（車両×候補クラスタでの長時間放置件数）
  - `same_as_charge_cluster`（充電クラスタと候補クラスタが同一か）
- 遷移履歴（新機能）
  - 充電開始時刻を `HOUR_BIN_SIZE` 時間幅で離散化し `h_c_bin` として使用。
  - 条件付き遷移の回数/確率を、グローバル/車両別で作成：
    - `trans_cnt_global_cchc`, `trans_prob_global_cchc` … 条件 (charge_cluster=c_c, h_c_bin=h) 下で park_cluster=c_p の件数/確率。
    - `trans_cnt_vehicle_cchc`, `trans_prob_vehicle_cchc` … 同（車両別）。
    - `trans_cnt_global_hc`, `trans_prob_global_hc` … 条件 (h_c_bin=h) のみ（疎セル補完用）。
  - 確率はラプラス平滑化: `(cnt + α) / (denom + α*K)`（K=候補クラスタ種類数）。
- 直前行動（新機能）
  - `prev_same_as_candidate` … 充電直前のセッションのクラスタが候補と一致するか（1/0）
  - `prev_is_long_park` … 直前が長時間放置か（1/0）
  - `prev_to_cand_dist_km` … 直前セッション終了地点→候補重心の距離

実装: `train/ranking/dataset.py` の `build_ranking_training_data`（内部で遷移テーブルを構築して特徴へ展開）

---

## 4. 学習・推論方針
- 学習: （充電, 候補クラスタ）のペアを 1 行とする二値分類（正例=真の放置クラスタ）。
- 推論: 陽性確率をスコアとし、候補を降順に並べてランキング。
- 分割: group（充電イベント＝`group_id`）単位で学習/検証を分ける。
- 実装: `train/ranking/train_rank.py`, `train/ranking/predict_rank.py`

---

## 5. 評価方法・指標（読み方含む）
- 評価は group（充電イベント）ごとに候補をスコア降順へ並べ、以下を算出して平均。
- Top-k Accuracy: 上位 k に正解が含まれれば 1、なければ 0 の平均。運用で上位候補提示が有効かを判断。
- MRR（Mean Reciprocal Rank）: 最初の正解の順位の逆数（1/rank）の平均。1 に近いほど上位に正解が来る。
- MAP（Mean Average Precision）: 正解位置 i での precision@i を平均した AP を group 平均。複数正解がある場合に有効。
- NDCG@k: 上位 k の DCG を理想 DCG で割った正規化値。上位に正解が集中するほど高い。

読み方の例:
- Top1 と MRR が高い → 単一候補提示でも当てやすい。
- Top1 は中程度だが Top3 が高い → 上位候補提示と人の判断の組み合わせに適する。
- MAP/NDCG が高い → 複数の正解候補があるケースで全体的な並べ方が良好。

---

## 6. パラメータ（Notebook 冒頭で調整）
`train/ranking_predict.ipynb` の先頭付近に「パラメータ」セルを用意済み。用途に応じて変更してください。

- `LONG_PARK_THRESHOLD_MIN` … 長時間放置の閾値（分）
- `CAND_TOP_N_PER_VEHICLE`, `CAND_GLOBAL_TOP_N` … 候補 Top-N（車両別/全体）
- `NEG_SAMPLE_K` … 学習時に各 group でサンプリングする負例数（0/負ならサンプリングなし）
- `HOUR_BIN_SIZE` … 充電開始時刻のビン幅（時間）。3 を推奨
- `ALPHA_SMOOTH` … 遷移確率計算のラプラス平滑化パラメータ α
- `AG_PRESETS`, `TIME_LIMIT` … AutoGluon の学習設定

---

## 7. 実行方法（CLI）
学習（リポジトリ直下）
```
python -m train.ranking.train_rank \
  --sessions_csv EV-Battery-Parking-Degradation-Mitigation/eda/ev_sessions_test.csv \
  --outdir EV-Battery-Parking-Degradation-Mitigation/train/outputs/ranking_model \
  --time_limit 600
```

推論（Top-3）
```
python -m train.ranking.predict_rank \
  --sessions_csv EV-Battery-Parking-Degradation-Mitigation/eda/ev_sessions_test.csv \
  --model_dir EV-Battery-Parking-Degradation-Mitigation/train/outputs/ranking_model/autogluon \
  --top_k 3
```

---

## 8. 実装の所在
- データセット/特徴量: `train/ranking/dataset.py`
- 指標: `train/ranking/metrics.py`
- 学習 CLI: `train/ranking/train_rank.py`
- 推論 CLI: `train/ranking/predict_rank.py`
- ノートブック: `train/ranking_predict.ipynb`

