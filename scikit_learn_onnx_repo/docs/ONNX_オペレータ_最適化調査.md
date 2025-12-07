# ONNX・オペレータ・最適化の整理メモ

## 1. ONNXとは？

### 1-1. 位置づけ・役割

- ONNX は、**機械学習モデルを保存・やり取りするための共通フォーマット（モデルの設計図ファイル）**。
- PyTorch・TensorFlow・scikit-learn など、各フレームワークごとにバラバラなモデル表現を、
  - 「入力テンソル → 演算ノード → 出力テンソル」という **計算グラフ**
  - そのグラフで使う **重み（initializer / tensor）**
 で共通に表現する。

ONNX モデルの本体は、`ModelProto` という protobuf 構造体で定義されており、`graph`（ノード・エッジ・initializer）、`opset_import`（使用するオペレータのバージョン）などを含む。

参考：
- ONNX 公式仕様書（ModelProto / GraphProto / NodeProto など）  
  https://github.com/onnx/onnx/blob/main/docs/IR.md

---

## 2. ONNXの「オペレータ」とは？

### 2-1. オペレータの定義

- **オペレータ = ONNX の世界で使える「計算の部品（関数）」の種類**。
- 例：
  - 基本演算: `Add`, `Mul`, `MatMul`, `Relu`, `Sigmoid` など  
  - 畳み込み系: `Conv`, `BatchNormalization` など  
  - ツリーモデル系: `TreeEnsembleRegressor`, `TreeEnsembleClassifier` など
- 各オペレータは、
  - 入力テンソルの型・次元（型制約）
  - 挙動を決める属性（attributes）
  - 出力テンソルの型
  をスキーマとして持つ。

参考（オペレータ仕様の例）：
- Conv オペレータのスキーマ  
  https://github.com/onnx/onnx/blob/main/docs/Operators.md#Conv
- TreeEnsembleRegressor オペレータのスキーマ  
  https://github.com/onnx/onnx/blob/main/docs/Operators-ml.md#TreeEnsembleRegressor

### 2-2. モデル → オペレータのマッピング

- scikit-learn の `HistGradientBoostingRegressor` や `StandardScaler`, `OneHotEncoder` は、
  - ONNX に変換されると、対応する ONNX オペレータの組み合わせで表現される。
- 例：  
  - `HistGradientBoostingRegressor` → `TreeEnsembleRegressor`  
  - `StandardScaler` → `Scaler`  
  - `OneHotEncoder` → `OneHotEncoder`
- これにより、「元は scikit-learn か PyTorch か」を隠蔽し、ONNX Runtime など任意のランタイムが同じルールで実行できる。

参考：
- sklearn-onnx API 概要  
  “Both functions convert a scikit-learn model into ONNX. … An ONNX model (type: ModelProto) which is equivalent to the input scikit-learn model.”  
  https://onnx.ai/sklearn-onnx/api_summary.html

---

## 3. ONNXの「最適化」とは？

### 3-1. グラフ最適化（構造を変えずに速く・小さくする）

**目的：**  
入出力の意味（予測結果）はそのままに、計算量やメモリを減らし、推論を高速化・省メモリ化すること。

ONNX Runtime のドキュメントでは、グラフ最適化を以下のように説明している：

> “Graph optimizations are applied to the model to improve performance. They include node eliminations, node fusions, and layout optimizations while preserving the model’s semantics.”  
> （グラフ最適化は性能向上のためにノード削除・ノード融合・レイアウト最適化などを行うが、モデルの意味は保持する）

出典：  
https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html

代表的な最適化内容：

- **意味のないノードの削除**
  - 例: identity ノードや、どこからも参照されていない出力を消す（dead-end elimination）。
- **ノードの融合（fusion）**
  - 例: `MatMul` + `Add` → `Gemm` にまとめる  
  - 例: Conv + Bias Add → 1つの Conv にまとめる
- **定数計算の前取り（constant folding）**
  - 推論時に変わらない計算をあらかじめ実行し、その結果だけを保存しておく。
- **レイアウト・メモリアクセスの改善**
  - ハードウェアに適したデータレイアウトに変更することでキャッシュ効率を上げる。

ONNX Optimizer の解説記事では、最適化パスを次のように分類している：

> 「サポートしている最適化は、get_available_passesで取得できます。大きく3つに分類できます。
> 1. 意味のないOpの削除（eliminate_deadend 等）
> 2. 2つのOpの fusion（fuse_matmul_add_bias_into_gemm 等）
> 3. Convへのfusion（fuse_add_bias_into_conv 等）」  
> https://natsutan.hatenablog.com/entry/2019/10/02/095939

### 3-2. 量子化など（精度とのトレードオフ）

**量子化（Quantization）** は、float32 などの浮動小数点を int8 / float16 に落とし、モデルサイズ削減・推論高速化を狙う手法。

ONNX Runtime の量子化ドキュメントより：

> “ONNX Runtime provides python APIs for converting 32-bit floating point model to an 8-bit integer model, a.k.a. quantization. These APIs include dynamic quantization, static quantization, and QAT.”  
> https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html

- メリット：
  - モデルサイズが小さくなる
  - メモリ帯域の削減により推論が速くなる場合が多い
- デメリット：
  - 数値精度がわずかに劣化する可能性があるため、元モデルとの誤差比較が必要

---

## 4. どの段階でどのような最適化手法をとれるか？

### 段階A：学習・設計（Python / scikit-learn 等）

**やること：**

- モデル構造そのものの軽量化
  - 木の深さ・本数の調整
  - 特徴量数の削減
  - 履歴長・時系列ウィンドウの調整

**狙い：**

- そもそも軽量なモデルにしておくことで、ONNX 変換後も推論が軽くなる。

※ここは ONNX というより、モデリング・ハイパーパラメータ設計の領域。

---

### 段階B：Python → ONNX 変換（コンバータ側）

代表例：`skl2onnx`（scikit-learn → ONNX）

**やること：**

- scikit-learn のモデル・パイプラインを対応する ONNX オペレータ群に変換。
- オプションにより、簡易最適化（Identity ノード削除など）を有効化。

sklearn-onnx API 概要より：

> “Both functions convert a scikit-learn model into ONNX. The main difference is … The result is an ONNX model (ModelProto) which is equivalent to the input scikit-learn model.”  
> https://onnx.ai/sklearn-onnx/api_summary.html

**狙い：**

- 「ONNX として妥当で、無駄の少ないグラフ」を最初から出力する。

**代表的なメソッド・パラメータ例：**

```python
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# scikit-learn モデル → ONNX
onnx_model = convert_sklearn(
    skl_model,
    initial_types=[('input', FloatTensorType([None, n_features]))],
    options={
        "model_optim": True,  # 変換直後の簡易最適化（Identity ノード削除等）
    },
)
```

※ `options` の内容・キーはライブラリのバージョンに依存するため、実際には公式ドキュメント・ソースコードを要確認。

skl2onnx の公開パッケージ情報：  
https://pypi.org/project/skl2onnx/

---

### 段階C：ONNX グラフのオフライン最適化（onnx.optimizer）

**やること：**

- `onnx.optimizer` を使い、ONNX モデルを一度読み込んで、指定パスで最適化済みモデルを書き出す。

onnx.optimizer の利用例（公式サンプル・ブログ）：

```python
import onnx
from onnx import optimizer

model = onnx.load("model.onnx")

# 利用可能なパス一覧の取得
all_passes = optimizer.get_available_passes()
print(all_passes)

# 適用したいパスを選択
passes = [
    "eliminate_identity",
    "eliminate_deadend",
    "fuse_matmul_add_bias_into_gemm",
]
optimized_model = optimizer.optimize(model, passes)

onnx.save(optimized_model, "model_optimized.onnx")
```

参考：  
- ONNX Optimizer の概説とパス一覧  
  https://natsutan.hatenablog.com/entry/2019/10/02/095939  
- onnx/onnx の optimizer 実装  
  https://github.com/onnx/onnx/tree/main/onnx/optimizer

**狙い：**

- 端末に配布する前にグラフを整理し、不要ノード削除や fusion を済ませておくことで、
  - モデルサイズ削減
  - 推論性能向上
  を狙う。

---

### 段階D：ONNX Runtime によるグラフ最適化

**やること：**

- ONNX Runtime（ORT）の `SessionOptions` でグラフ最適化レベルを指定し、モデル読み込み時に最適化を適用する。

ONNX Runtime のドキュメントより：

> “Graph optimizations are divided into levels … ORT_DISABLE_ALL, ORT_ENABLE_BASIC, ORT_ENABLE_EXTENDED, ORT_ENABLE_ALL … To enable model serialization after graph optimization set SessionOptions.optimized_model_filepath.”  
> https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html

GraphOptimizationLevel の列挙型定義（C#/C API ドキュメント）：
https://onnxruntime.ai/docs/api/csharp/api/Microsoft.ML.OnnxRuntime.GraphOptimizationLevel.html

**代表的な最適化レベル：**

- `ORT_DISABLE_ALL`  
  → 最適化なし
- `ORT_ENABLE_BASIC`  
  → 基本的で安全な最適化（ノード削除など）
- `ORT_ENABLE_EXTENDED`  
  → BASIC に加え、より積極的な fusion など
- `ORT_ENABLE_ALL` / `ORT_ENABLE_LAYOUT`  
  → レイアウト変換などすべての最適化（GPU 等向け）

**コード例（Python）：**

```python
import onnxruntime as ort

so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

# 最適化後モデルをファイルに保存したい場合
so.optimized_model_filepath = "model_ort_optimized.onnx"

sess = ort.InferenceSession(
    "model.onnx",
    sess_options=so,
    providers=["CPUExecutionProvider"],  # 例: CPU 実行
)
```

**狙い：**

- 実行環境（CPU / GPU / 特定 SoC）に合わせて、Runtime 側で最適化を自動適用し、推論を高速化する。

---

### 段階E：量子化（Quantization）

**やること：**

- ONNX Runtime の量子化ツールで、float32 モデルを int8 / float16 などに変換。

ONNX Runtime の量子化ドキュメント：

> “ONNX Runtime provides python APIs for converting 32-bit floating point model to an 8-bit integer model, a.k.a. quantization. These APIs include dynamic quantization, static quantization, and QAT.”  
> https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html

**代表的な API（Python）：**

- 動的量子化（Dynamic Quantization）

```python
from onnxruntime.quantization import quantize_dynamic, QuantType

model_fp32 = "model.onnx"
model_quant = "model.quant.onnx"

quantize_dynamic(
    model_fp32,
    model_quant,
    weight_type=QuantType.QUInt8,  # or QuantType.QInt8
    # op_types_to_quantize=["MatMul", "Gemm", ...] なども指定可能
)
```

- 静的量子化: `quantize_static`
- 学習時量子化: `quantize_qat`

日本語の解説（API 呼び出しの概略）：

> 「動的量子化のPython APIは、モジュール `onnxruntime.quantization.quantize` の関数 `quantize_dynamic()` にあります。静的量子化では `quantize_static()` を使用します。」  
> https://zenn.dev/nnn112358/scraps/8222cfbfb067c5

**狙い：**

- モデルサイズの削減・推論高速化（特にメモリ・帯域が厳しいエッジ環境向け）。

**注意点：**

- 精度がわずかに低下することがあるため、
  - 元の float32 モデルと
  - 量子化モデル
  を同じ検証データで比べ、MAE / RMSE / 最大誤差などで差を確認する。

---

## 5. まとめ（設計・実装の観点）

1. **ONNXとは？**  
   - フレームワーク依存のモデル表現を、共通の計算グラフ＋重み形式（ModelProto）に変換する「モデルの設計図フォーマット」。

2. **ONNXのオペレータとは？**  
   - ONNX の世界で使える「計算の部品」の種類（`Add`, `Conv`, `TreeEnsembleRegressor` など）。  
   - 各オペレータは、入力/出力の型や属性をスキーマとして持つ。

3. **ONNXの最適化とは？**  
   - グラフの構造を賢く書き換え、
     - 不要ノード削除
     - ノード fusion
     - 定数計算の前取り
     などで、**同じ入出力を保ちつつ速度・メモリを改善**すること。  
   - 量子化は、精度とトレードオフしながらさらにサイズ・速度を追求するオプション。

4. **どの段階でどの手法がとれる？（狙いと効果）**
   - A: 学習・設計段階  
     → モデル構造をそもそも軽量にする（最も効果大）
   - B: 変換段階（skl2onnx など）  
     → `options={"model_optim": True}` などで簡易なグラフ整理
   - C: ONNX Optimizer  
     → `optimizer.optimize(model, passes)` で不要ノード削除・fusion を事前適用
   - D: ONNX Runtime  
     → `SessionOptions.graph_optimization_level` で実行環境に応じた最適化を自動適用
   - E: 量子化  
     → `quantize_dynamic` / `quantize_static` / `quantize_qat` で int8/float16 などに落とし、サイズ・速度をさらに改善（精度とのトレードオフ）

5. **EV 向けエッジ端末（ECU）での実務的なすすめ方（例）**
   - まずは B + D（変換時の軽量最適化＋ORT の EXTENDED 最適化）まででベンチマーク。
   - 性能が足りなければ、
     - A（モデル構造見直し）を行い、
     - それでも厳しければ E（量子化）を検討し、元モデルとの誤差を評価する。

以上をベースに、実際の SOC 予測モデルについても「どの段階まで最適化するか」「ECU のリソース前提でどこまで攻めるか」を設計していけばよい。
