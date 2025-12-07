# ONNX変換前後の精度影響と内部動作の整理

このドキュメントでは、以下の論点を整理します。

1. ONNX変換前と変換後で精度影響が発生する理由  
2. ONNX変換時に内部構造に影響があるのか  
3. TreeEnsembleRegressor・ColumnTransformer・initial_types などの役割整理  
4. 参考にした公式ドキュメント／情報ソース

---

# 1. ONNX変換前後で精度差が出る理由

## 1.1 入力数値型（double → float）による違い

- 変換前（scikit-learn 側）
  - `HistGradientBoostingRegressor` や `ColumnTransformer` は、デフォルトで `numpy.float64`（double）を使って計算する。
  - 学習時・推論時ともに Python / NumPy / scikit-learn の世界で完結。

- 変換後（ONNX 側）
  - `skl2onnx` で変換する際、多くのケースで **ONNX モデルの入出力型は `float`（float32）** に落とされる。
  - 特に `initial_types` に `FloatTensorType` を指定すると、ONNX のモデル入力は `tensor(float)` になる。

- この違いによる影響
  - 学習済みパラメータ（木の構造・閾値・葉の値）は基本的に同じだが、  
    演算精度が `float64 → float32` に縮むことで、しきい値近傍などで分岐がわずかに変わる可能性がある。
  - その結果、**scikit-learn と ONNX の予測値に微小な差**が出ることがある。

- 実務的な評価のポイント
  - 木モデル（GBDT / HistGBDT）は、そもそもデータノイズや汎化誤差が支配的。
  - `float64 → float32` による差は通常「非常に小さい」。
  - SOC予測のようなタスクでは、業務上許容される誤差（例：SOC数％）と比較すると、ほぼ無視できることが多い。
  - とはいえ、**sklearn と ONNX の予測を比較して MAE / 最大絶対誤差を確認**しておくのがベスト。

---

## 1.2 「ONNX を double で実行しようとしてエラー」について

### 1.2.1 試したことのイメージ

- 試行内容
  - `ColumnTransformer` 内で double（`np.float64`）想定で前処理を構成。
  - `skl2onnx.convert_sklearn` の `initial_types` に `DoubleTensorType` を指定。
  - ONNX Runtime で `InferenceSession` を作成し、double 入力を前提に推論しようとした。

- 発生したエラー（要約）
  - `TypeError: Type(tensor(double)) of output arg (...) of node (TreeEnsembleRegressor) does not match expected type (tensor(float))`
  - → TreeEnsembleRegressor ノードの出力が `tensor(double)` になっており、仕様上の `tensor(float)` と矛盾している。

### 1.2.2 エラーの直接原因はどこか？

- TreeEnsembleRegressor の公式仕様（簡略版）
  - 入力 `X`:
    - 型制約 `T in (tensor(double), tensor(float), tensor(int32), tensor(int64))`
    - → 入力は複数の数値型を許容する。  
      （`ai.onnx.ml - TreeEnsembleRegressor` の仕様より）
  - 出力 `Y`:
    - 型: `tensor(float)`（float32 固定）
    - → 出力は double 非対応。  
      （同じく TreeEnsembleRegressor の仕様に明記）

- エラーの本質
  - ONNX モデル内の TreeEnsembleRegressor ノードの **出力** が `tensor(double)` になっていた。
  - しかし、公式仕様は「出力は `tensor(float)` 固定」のため、  
    ONNX Runtime が「仕様違反」と判断し、モデルロード時に Type Error を出している。

- 直接原因の整理
  - ColumnTransformer 自体は sklearn 内の前処理であり、ONNX 型仕様の直接の原因ではない。
  - `initial_types` を `DoubleTensorType` にしたことなどがトリガーとなり、
    - skl2onnx が「モデル入力は double」と解釈し、
    - その型情報がグラフ全体に伝播、
    - TreeEnsembleRegressor の出力まで `tensor(double)` として保存されてしまった。
  - これが TreeEnsembleRegressor の仕様（Y: tensor(float)）と矛盾し、Runtime ロードエラーとなった可能性が高い。

### 1.2.3 initial_types とは何か？何のための指定か？

- 役割
  - **ONNX モデルの「外部入力」の名前と型を定義するための情報**。
  - 例: `('input', FloatTensorType([None, n_features]))`
  - これを基点として、コンバータ（skl2onnx）が
    - 前処理ノード（ColumnTransformer 等）
    - 学習器ノード（TreeEnsembleRegressor 等）
    までの **テンソル型（float/double）を推論・決定**する。  
    （`sklearn-onnx API Summary` / `Convert a pipeline` ドキュメント参照）

- `DoubleTensorType` を指定した場合の挙動
  - 「外部から double が入ってくる」と認識される。
  - skl2onnx はその型を伝播させ、最終ノードの出力まで double と解釈するケースがある。
  - しかし TreeEnsembleRegressor の公式仕様は「出力は float32 固定」なので、
    - 出力が double だと仕様と矛盾し、
    - Runtime ロード時に Type Error になる。

- 推奨
  - TreeEnsembleRegressor を使う限り、
    - `initial_types` は **FloatTensorType** を使い、
    - ONNX モデル全体を float32 ベースで構成するのが安全。

### 1.2.4 ColumnTransformer の役割と「型指定」の意味

- ColumnTransformer の役割（sklearn 内）
  - 複数のカラムに対して異なる前処理（標準化、OneHot、sin/cos 生成など）を適用し、横方向に結合した特徴量行列を作る。
  - sklearn のパイプライン上で、入力 DataFrame のカラムを「数値系」「カテゴリ系」などに切り分けて処理するためのクラス。

- 型指定の意味
  - sklearn 内部での dtype 管理（例: 数値カラムを float64 で統一）に用いる。
  - これはあくまで **Python / NumPy / scikit-learn の世界**の話であり、  
    ONNX の型（FloatTensorType / DoubleTensorType）とは別物。

- ONNX 変換時
  - skl2onnx は ColumnTransformer の構造を読み取り、
    - `StandardScaler`, `OneHotEncoder`, `Concat`, `Cast` などの ONNX ノード列に展開する。  
      （`Convert a pipeline` ドキュメントの記述）
  - ここで ONNX の型がどうなるかは、
    - `initial_types` で指定した外部入力型、
    - 各変換ステップのコンバータ実装
    によって決まる。

---

## 1.2.5 TreeEnsembleRegressor とは何か？

### TreeEnsembleRegressor とは？

- ONNX-ML ドメインに定義されている「**木ベースの回帰モデル用オペレータ（Operator）**」。  
  （`ai.onnx.ml - TreeEnsembleRegressor` 仕様）
- 表現対象の例
  - ランダムフォレスト回帰
  - Gradient Boosting / HistGradientBoosting 回帰
  - Extra Trees 回帰
- 役割
  - 多数の決定木からなる回帰モデル（アンサンブル）を、  
    ONNX グラフ上では **1 ノード** で表現するための標準化された演算子。

- ノード内部に持つ情報（属性の例）
  - 各木の構造（ノード ID、親子関係）
  - 分割に使う特徴量インデックス
  - 分割閾値
  - 葉ノードの出力値
  - 各木の重み など

### TreeEnsembleRegressor はどの段階で指定されるのか？

- sklearn → ONNX の変換時（skl2onnx）に、
  - `HistGradientBoostingRegressor` などを解析し、
  - 「これは木ベースの回帰モデルだから、ONNX 側では TreeEnsembleRegressor を使おう」
  - と **コンバータ側が自動的に選択**する。
- ユーザー自身が Python コード内で TreeEnsembleRegressor と直接書くことは通常なく、  
  あくまで「ONNX の中身に出てくるノード名」として存在する。

### TreeEnsembleRegressor の型仕様のポイント

- 入力 `X`:
  - `tensor(double)`, `tensor(float)`, `tensor(int32)`, `tensor(int64)` を許容。  
    （公式仕様に型制約として明記）
- 出力 `Y`:
  - `tensor(float)`（float32 固定）。

このため、

- **ONNX モデル内で TreeEnsembleRegressor ノードの出力を double にしてしまうと、必ず仕様違反になりエラーとなる**。
- 今回のエラーはまさにこのパターン。

---

## 1.2.6 「X に double を入れるとエラーになる」の整理

- 仕様上は、
  - 入力 `X` に `tensor(double)` を指定しても TreeEnsembleRegressor は受け付け可能。
  - ただし出力 `Y` は `tensor(float)` のまま。
- 本来は、
  - 「外部入力 double → （必要なら Cast） → TreeEnsembleRegressor（float入力・float出力）」  
    という構造であれば矛盾しない。

- 今回のケースでは、
  - `initial_types` 等の指定が原因となり、
  - skl2onnx が「TreeEnsembleRegressor の**出力まで** double」とした ONNX を生成してしまった。
  - その結果、公式仕様と矛盾し、ONNX Runtime がロードを拒否した。

---

## 1.3 結論（1章のまとめ）

- **木構造の TreeEnsembleRegressor では「出力を double にする」ことは仕様上できない。**
  - 出力は `tensor(float)`（float32）に固定。
- 無理に double を通そうとして `initial_types` 等を double にすると、
  - コンバータの型推論の結果、ノード出力まで double と解釈され、
  - 公式仕様に反して Runtime ロードエラーを引き起こす可能性が高い。
- 実務的には、
  - **ONNX 側は float32 に統一**するのが最も安全・現実的。
  - sklearn（float64）と ONNX（float32）の予測を比較し、
    - MAE / 最大絶対誤差などで影響の有無を確認。
  - 多くのケースで、float32 でも SOC 予測タスクにおいて実用上問題ない精度が得られる。

---

# 2. ONNX変換時に内部構造に影響があるのか？

## 2.1 ONNX変換で変わるもの・変わらないもの

- 変わるもの
  - 数値型
    - 多くのケースで `float64 → float32` に縮小される。
  - 演算のレイヤー
    - scikit-learn の高レベル API／Python 実装  
      → ONNX の低レベルオペレータ列  
      → ONNX Runtime の C++ 実装。
  - 一部の演算の表現形式
    - 例: ColumnTransformer が `Scaler`, `OneHotEncoder`, `Concat`, `Cast` などの ONNX ノード列に展開される。  
      （`Convert a pipeline` ドキュメントより）

- 原則として変わらないもの
  - モデルのロジック（構造）
    - 木の分岐構造（どの特徴量・どの閾値で分割するか）
    - 葉の値
    - 各木の重み
  - ただし、それらの値が float32 に落とされることで若干の丸めは入る。

- まとめ
  - **ONNX変換によって「木の構造そのもの」が変わるわけではない。**
  - 変わるのは「数値表現の精度」と「実行エンジンの層」であり、  
    精度影響は主に「丸め誤差／境界の判定がわずかに変わる程度」。

---

## 2.2 ONNX Runtime の providers と精度・速度

### 2.2.1 providers とは？

- `providers` は、「どの実行バックエンドを使うか」を指定するもの。
  - 例:
    - `"CPUExecutionProvider"`
    - `"CUDAExecutionProvider"`（GPU）
    - `"TensorRTExecutionProvider"`, `"OpenVINOExecutionProvider"` など。  
      （`ONNX Runtime Execution Providers` ドキュメント）

- 主に変わるもの
  - **性能（速度・スループット）**
    - CPU: 汎用用途。小さなモデルや低頻度推論に向く。
    - GPU / TensorRT 等: 大量バッチ推論や大型 NN モデルに向く。

### 2.2.2 精度への影響

- 目標としては「同じ ONNX グラフなら、どの provider でも同じ出力」を目指している。  
  （ただし実装上の都合によるごく微小な差はあり得る）
- 浮動小数点演算の順序や最適化（FMA, 並列化など）の違いにより、
  - 小数点以下のごくわずかな差（丸め誤差単位）は発生し得る。

- TreeEnsembleRegressor に関しては、
  - 計算は主に「比較 + 分岐 + 加算」であり、
  - 大規模行列積のような誤差蓄積は相対的に小さい。
  - providers の違いで精度差が出るとしても、ごくわずかなレベルと考えられる。

- 実務での扱い
  - providers の違いは主に「速度・リソース効率の観点」で評価する。
  - 精度影響が気になる場合は、
    - CPU と GPU で同一入力に対する出力を比較し、
    - 誤差が業務上許容範囲かどうかを確認すればよい。

---

# 3. 参考文献・情報ソース

以下は、上記整理の根拠となる主な公式ドキュメント／情報ソースです。

1. **TreeEnsembleRegressor 仕様（型定義・入出力）**  
   - ONNX 公式 Operators ドキュメント（ai.onnx.ml - TreeEnsembleRegressor）  
     - https://onnx.ai/onnx/operators/onnx_aionnxml_TreeEnsembleRegressor.html  
   - 仕様サマリ・バージョン情報  
     - https://natke.github.io/onnx/operators/onnx_aionnxml_TreeEnsembleRegressor.html  

2. **ONNX のコンセプト・強い型付け・ドメイン**  
   - ONNX Concepts  
     - https://onnx.ai/onnx/intro/concepts.html  

3. **Cast オペレータ（明示的な型変換）**  
   - Cast オペレータ仕様  
     - https://onnx.ai/onnx/operators/onnx__Cast.html  
   - 「ONNX は strongly typed なので、型が違えば Cast が必要」とする議論（GitHub issue）  
     - https://github.com/onnx/onnx/issues/4443  

4. **skl2onnx / sklearn-onnx の API と initial_types**  
   - sklearn-onnx API Summary（`convert_sklearn`, `initial_types` の説明）  
     - https://onnx.ai/sklearn-onnx/api_summary.html  
   - パイプライン変換（ColumnTransformer などを ONNX グラフに展開する流れ）  
     - https://onnx.ai/sklearn-onnx/pipeline.html  
   - `FloatTensorType` / `DoubleTensorType` 使用例（コード中で両者を指定するケース）  
     - https://github.com/onnx/sklearn-onnx/blob/master/docs/examples/plot_gpr.py  

5. **ONNX Runtime の Python API / InferenceSession**  
   - Python API Summary（InferenceSession の説明）  
     - https://onnxruntime.ai/docs/api/python/api_summary.html  
   - ONNX Runtime + scikit-learn パイプラインのチュートリアル  
     - https://onnxruntime.ai/docs/api/python/tutorial.html  

6. **ONNX Runtime Execution Providers（CPU/GPU/TensorRT など）**  
   - Execution Providers 一覧と概要  
     - https://onnxruntime.ai/docs/execution-providers/  
   - CUDA Execution Provider  
     - https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html  
   - TensorRT Execution Provider  
     - https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html  

7. **ONNX 全体のオペレータ一覧**  
   - ONNX Operators 一覧（ai.onnx / ai.onnx.ml 含む）  
     - https://onnx.ai/onnx/operators/  

これらのドキュメントをベースに、「TreeEnsembleRegressor の出力型は float に固定」「ONNX は暗黙キャストを持たず Cast オペレータで型変換する」「skl2onnx の initial_types が ONNX グラフ全体の型推論の起点になる」といった前提を組み立てています。
