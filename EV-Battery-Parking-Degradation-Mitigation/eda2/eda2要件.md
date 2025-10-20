# 🚗 CodeX 指示書：EV行動ヒートマップ可視化（充電あり/なし × 曜日別）

## 🎯 目的
EV利用者の放置行動を「曜日 × 時間帯 × 場所クラスタ」で可視化し、  
充電の有無による行動パターンの差異を把握する。  
また、分子・分母に基づく割合を定量的に算出し、週次スケールで比較可能な形式で出力する。

---

## 🧩 入力データ仕様
入力は以下の列を持つCSVまたはDataFrameとする。

|列名|型|説明|
|----|--|----|
|`hashvin`|str|車両ID（ユーザー単位で分析）|
|`session_cluster`|int or str|滞在場所クラスタ（DBSCAN結果など）|
|`session_type`|str|`inactive`または`charging`|
|`start_time`|datetime|イベント開始時刻|
|`end_time`|datetime|イベント終了時刻|
|`duration_minutes`|float|滞在時間（分）|
|`start_lat`|float|開始緯度|
|`start_lon`|float|開始経度|
|`end_lat`|float|終了緯度|
|`end_lon`|float|終了経度|
|`start_soc`, `end_soc`, `change_soc`|float|SOC情報（任意）|

---

## 🧮 集計ロジック

### 1️⃣ データ分割
- `session_type == "inactive"` のみ対象とする。
- 各イベントの `start_time.date()` を所属日とする。  
  → 夜間を跨いでも開始日に帰属（例：20:30–07:30 → 20:30の属する日）。
- 同一日内に `session_type == "charging"` が1件以上あれば「充電あり日」、無ければ「充電なし日」と分類。
- 曜日を `weekday = start_time.weekday()` で算出（0=月曜, 6=日曜）。

### 2️⃣ 時間軸のビニング
- 1時間ビンで集計する。
- X軸は **6時〜30時**（=6〜23h＋翌日0〜6h）をロールして表示。
- 各 inactive イベントの `[start_time, end_time)` とビンの重なり時間を積算し、
  各 `(weekday, hour, cluster)` に滞在時間[h]を加算。

### 3️⃣ 分子・分母・セル値
- **分子 (cell_numerator)**：該当(weekday × hour × cluster)の滞在時間合計[h]  
- **分母 (cell_denominator)**：  
  「観測期間に含まれるカレンダー週数 × 1時間」  
  （理想値固定。例：1年＝52週）  
- **セル値 (cell_value)**：`cell_numerator / cell_denominator`（割合 0〜1）

### 4️⃣ クラスタ選定と並び
- 上位クラスタを **(充電あり＋なし)の分子合計降順**でソート。
- 上位 `N=15`（引数指定可）を描画。
- 上位外のクラスタ＋DBSCANノイズ(-1など)を **OTHER** に合算（既定で最上段表示）。
- クラスタは充電あり／なし両方で同じ順序を固定。
- 割合が大きいクラスタほど**下側に配置**。

### 5️⃣ 分子・分母表示
- 各ヒートマップ下部に以下を注記：
```
Σ滞在(充電なし)=XXX.Xh / 週数=YYh | Σ滞在(充電あり)=ZZZ.Zh / 週数=YYh
```

- 各セルごとの分子/分母はCSVに出力し、画像には表示しない（視認性優先）。

---

## 📊 出力仕様

### 📁 ディレクトリ構造
```
result/
└─ {hashvin}/
├─ weekday_0_comparison.png
├─ weekday_1_comparison.png
├─ ...
├─ weekday_6_comparison.png
├─ weekday_0_matrix_with.csv
├─ weekday_0_matrix_without.csv
├─ weekday_0_numerator_with.csv
├─ weekday_0_numerator_without.csv
├─ ...
└─ denominator_weeks.csv
```


### 🖼️ 画像仕様
- 1枚につき **左右2面構成**：  
  左＝充電なし日、右＝充電あり日。
- タイトル例：
```
hv_0001_demo | Weekday: Mon | scale: 0–p95 | denom: 52h
```
- カラーマップ：`YlGnBu`（共有スケール、0〜p95でクリップ）
- 注釈：ヒートマップ下部に分子/分母合計を記載。

### 📈 CSV出力
|ファイル|内容|
|--------|----|
|`weekday_{w}_matrix_with.csv`|割合（cell_value）|
|`weekday_{w}_matrix_without.csv`|割合（cell_value）|
|`weekday_{w}_numerator_with.csv`|滞在時間[h]|
|`weekday_{w}_numerator_without.csv`|滞在時間[h]|
|`denominator_weeks.csv`|週数情報（全曜日共通）|

---

## 📏 定量評価：充電あり/なし類似度
- 比較単位：各曜日ごと。
- 対象：上位Nクラスタ × 時間（6〜30）のセル値ベクトル。
- 指標：

|指標|意味|スケール|
|----|----|----|
|`pearson_corr`|構造的相関|[-1,1]|
|`cosine_sim`|全体方向一致度|[0,1]|
|`js_distance`|確率分布乖離|[0,1]（小さいほど類似）|

- 出力：`result/{hashvin}/similarity_scores.csv`
```csv
hashvin,weekday,pearson_corr,cosine_sim,js_distance,covered_weeks_with,covered_weeks_without
hv_0001_demo,0,0.83,0.91,0.07,52,52
```


### パラメータ
| 項目        | 変数名                  | 既定値              | 説明                  |
| --------- | -------------------- | ---------------- | ------------------- |
| 上位描画クラスタ数 | `top_n_clusters`     | 15               | 描画対象クラスタ上限          |
| 集計指標      | `metric`             | "duration_ratio" | `"count_ratio"`に切替可 |
| カラースケール上限 | `clip_percentile`    | 0.95             | 上位5%でクリップ           |
| OTHER行位置  | `other_position`     | "top"            | 上部固定                |
| カラースケール共有 | `share_color_scale`  | True             | 左右で同一スケール           |
| セル注釈      | `annotate_cells`     | False            | セルに数値注記するか          |
| 最小観測週数    | `min_coverage_weeks` | 1                | 分母に含める最小週数          |

### 💡 エッジケース処理

- 部分週：理想値（週数×1h）を維持、欠測は無視。
- クラスタ乱立：上位N基準は「充電あり＋なし合計の分子値」。
- OTHER扱い：上位外＋ノイズ（-1）を合算し最上段固定。

### ✅ 出力概要（1hashvinあたり）

- ヒートマップ画像：7枚（曜日別）
- CSV（割合・分子）：各7×2×2ファイル
- 類似度スコアCSV：1ファイル
- 総出力点数：15ファイル＋画像7枚程度／車両