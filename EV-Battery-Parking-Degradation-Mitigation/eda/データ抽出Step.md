# EV バッテリ劣化抑制\_放置予測 — CodeX 要件書（改訂版）

## 0. 概要

- **目的**:  
  EV ユーザーが充電ステーションを利用した後、どこで長時間（≥6h）放置する傾向があるかを可視化し、  
  「充電時点での行動予測モデル」構築に向けた EDA を行う。  
  特に、充電前後の行動パターンや時間帯・曜日・充電開始時刻の影響を明確化する。

- **対象フェーズ**: PoC（Jupyter ベース EDA）
- **分析単位**: `hashvin`（車両）ごと

---

## 1. あなた（ChatGPT）のロール

- 機械学習・データ分析・EV 領域の専門家として、  
  CodeX での EDA スクリプト設計・自動化テンプレートの生成を支援する。
- Notebook レベルで再利用可能な EDA 可視化コードを出力する。

---

## 2. データ仕様（入力）

**1 行＝ 1 セッションイベント（充電または放置）**

| 列名                                           | 説明                           |
| ---------------------------------------------- | ------------------------------ |
| `session_cluster`                              | 滞在クラスタ ID（DBSCAN 結果） |
| `session_type`                                 | {`inactive`, `charging`}       |
| `start_time`                                   | 滞在/充電開始時刻              |
| `end_time`                                     | 滞在/充電終了時刻              |
| `duration_minutes`                             | 滞在/充電継続時間              |
| `start_soc`, `end_soc`, `change_soc`           | SOC 関連値                     |
| `start_lat`, `start_lon`, `end_lat`, `end_lon` | 位置座標                       |

**除外条件**:

- 移動中セッション、および「充電してその場に留まる」イベントは除外済み。

---

## 3. 派生列（EDA 処理で追加）

| 列名                                | 内容                                                 |
| ----------------------------------- | ---------------------------------------------------- |
| `weekday`                           | `start_time.dayofweek`（0=Mon,…,6=Sun）              |
| `start_hour`                        | `start_time.hour`                                    |
| `is_long_park`                      | `session_type=='inactive'` & `duration_minutes>=360` |
| `prev_session_type`, `prev_cluster` | 直前セッション情報（shift(+1))                       |
| `next_session_type`, `next_cluster` | 直後セッション情報（shift(-1))                       |
| `after_charge`                      | `inactive`かつ`prev_session_type=='charging'`        |
| `charge_start_hour`                 | `start_hour`（充電セッションのみ）                   |
| `date`                              | `start_time.dt.date`（充電有無日比較に使用）         |

---

## 4. 分析構成（再設計版）

| Step | テーマ                 | 目的                                                        | 出力可視化                       |
| ---- | ---------------------- | ----------------------------------------------------------- | -------------------------------- |
| 1    | 放置場所分布           | 各車両の長時間放置クラスタ傾向を把握                        | 棒グラフ                         |
| 2    | 滞在傾向＋充電時刻影響 | 曜日 × 時間帯の滞在傾向＋充電後偏り＋充電開始時刻による影響 | ヒートマップ＋条件付き棒グラフ   |
| 3    | 充電前後遷移比較       | 充電前後・充電有無日別の行動遷移傾向を把握                  | ネットワーク図＋ヒートマップ差分 |

---

## 5. Step 詳細と可視化仕様

### 🟦 Step 1：放置場所の全体傾向（Bar Plot）

**目的**: 各車両がどのクラスタで長時間放置しているかの全体像を把握。  
**対象**: `session_type=='inactive'` & `duration_minutes>=360`

- 集計単位：`hashvin × session_cluster`
- 指標：総滞在時間[h]
- 出力：棒グラフ（降順、上位 10 クラスタ）
- 次工程：上位 5 クラスタを Step2 の可視化対象とする

---

### 🟩 Step 2：クラスタ別・曜日 × 時間帯ヒートマップ（充電後偏り＋充電時刻影響）

**目的**

- 「どの時間帯・曜日にどのクラスタに長時間滞在するか」
- 「充電後に偏る時間帯や曜日」
- 「充電開始時刻が次放置先に与える影響」

**データ**

- 長時間放置イベントのみ（`inactive & duration_minutes>=360`）
- `after_charge`フラグを使用（充電後放置のみ抽出）

**処理**

1. 放置イベントを 15 分スロット展開し、`weekday×hour`で滞在集計
2. 上位 5 クラスタについて、以下の 3 種ヒートマップを作成
   - 全体滞在（All long inactive）
   - 充電後滞在（After-charge）
   - 差分（After-charge − All）
3. 充電開始時刻の影響可視化
   - 条件付き棒グラフ：  
     x=`charge_start_hour`、y=`P(next_long_park_cluster=c)`  
     facet：`charge_cluster` or `weekday`

---

### 🟨 Step 3：充電前後遷移と充電有無日比較

#### 3-A. 充電前の遷移（どこから充電に来るか）

- 対象：`session_type=='charging'`
- 直前セッション：`prev_cluster`
- 集計：`P(charge_cluster | prev_cluster, charge_start_hour, weekday)`
- 可視化：
  - ネットワーク図（prev_cluster → charge_cluster）
  - ヒートマップ（x=hour, y=prev_cluster, color=頻度）

#### 3-B. 充電後の遷移（どこに放置するか）

- 対象：`charging` → `inactive & is_long_park==1`
- 集計：`P(next_cluster | charge_cluster, charge_start_hour, weekday)`
- 可視化：
  - ネットワーク図（charge_cluster → next_long_park_cluster）
  - 確率棒グラフ（x=hour, y=prob, hue=next_cluster）

#### 3-C. 充電あり日 vs なし日 の日次遷移比較

- 定義：
  - 充電あり日：同一`date`内に`charging`が存在
  - 充電なし日：存在しない日
- 抽出：
  - 日ごとの`inactive`遷移（`cluster_i → cluster_j`）を集計
- 出力：
  1. `T_charge` / `T_nocharge` の遷移確率行列ヒートマップ
  2. 差分ヒートマップ（`T_charge - T_nocharge`）
  3. 距離指標（Jensen–Shannon 距離 or TV 距離）

---

## 6. 可視化サマリ表

| 可視化種別               | 対象                              | 目的                         |
| ------------------------ | --------------------------------- | ---------------------------- |
| 棒グラフ                 | 長時間放置クラスタ分布            | 放置拠点の把握               |
| ヒートマップ ×3          | 上位 5 クラスタの全体/充電後/差分 | 時間・曜日傾向＋充電後偏り   |
| 棒グラフ（条件付き）     | 充電開始時刻 × 放置確率           | 充電時刻の影響確認           |
| ネットワーク図（充電前） | prev→charge                       | 充電行動の文脈を把握         |
| ネットワーク図（充電後） | charge→long park                  | 充電後放置遷移の定型化を確認 |
| 遷移行列ヒートマップ     | 充電あり/なし日比較               | 日常行動への影響を確認       |
| 差分ヒートマップ         | T_charge−T_nocharge               | 行動パターン変化を定量化     |

---

## 7. 実装要点（共通）

- `groupby('hashvin')`で逐次シフトし、`prev_*`/`next_*`を生成
- 15 分スロット展開：`pd.date_range(start, end, freq='15T')`
- ネットワーク描画：`networkx`
- 可視化：`matplotlib`, `seaborn`
- サマリ指標計算：`scipy.spatial.distance.jensenshannon`

---

## 8. 出力成果物

- 図表：
  - `bar_cluster_distribution.png`
  - `heatmap_cluster_[1-5]_all.png`
  - `heatmap_cluster_[1-5]_aftercharge.png`
  - `heatmap_cluster_[1-5]_diff.png`
  - `network_before_charge.png`
  - `network_after_charge.png`
  - `transition_matrix_diff.png`
- テーブル：
  - `transition_prob_before.csv`
  - `transition_prob_after.csv`
  - `transition_diff_metrics.csv`

---

## 9. 評価・モデル接続想定

EDA で確認された構造をもとに、以下特徴量を抽出して  
放置先クラスタ予測モデルに活用。

| 特徴量名                                  | 説明                 |
| ----------------------------------------- | -------------------- |
| `weekday`, `hour`                         | 曜日・時間帯         |
| `charge_cluster`, `charge_start_hour`     | 充電条件             |
| `stay_prob_all`, `stay_prob_after_charge` | 通常／充電後滞在確率 |
| `transition_prob_before/after`            | 前後遷移確率         |
| `long_park_ratio`                         | クラスタ長時間滞在率 |
| `has_charge_day`                          | その日の充電有無     |
| `dist(prev, charge)`                      | 移動距離特徴（任意） |

---

## 10. 受け入れ基準

- [ ] Step1：クラスタごとの棒グラフが生成される（上位クラスタ選出可能）
- [ ] Step2：上位 5 クラスタの全体/充電後/差分ヒートマップ＋充電時刻影響グラフが描画される
- [ ] Step3：充電前後のネットワーク図＋充電有無日比較の行列ヒートマップが出力される
- [ ] 差分や距離指標が計算され、出力ファイルとして保存される
- [ ] Notebook は`hashvin`単位で切替可能

---

## 11. 非スコープ

- 学習モデル構築（AutoGluon など）は次フェーズ
- リアルタイム推論、API 化
- 交通・天候・地図連携

---

## 12. 成果物

- Notebook: `eda_ev_parking_behavior.ipynb`
- 可視化画像：`/outputs/plots/`
- 集計テーブル：`/outputs/tables/`
- Markdown レポート：`eda_summary.md`

---
