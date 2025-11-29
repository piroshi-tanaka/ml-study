# ONNX変換と精度検証メモ

車載ECU向けのデプロイを想定して、scikit-learnの時系列モデル（HistGradientBoostingRegressor）をONNXへ変換し、前後での精度差を確認するための実験環境とメモをまとめました。

- 実装: `scikit_learn_onnx_repo/onnx_conversion_accuracy_evaluation/evaluate_hgb_time_series.py`
- 実験出力: `scikit_learn_onnx_repo/onnx_conversion_accuracy_evaluation/outputs/`（ONNXモデルとメトリクスを保存）

## ONNXモデルの仕組み（ざっくり）
- ONNXは計算グラフ（GraphProto）としてモデルを表現し、`Graph = {inputs, outputs, nodes, initializers, opset}`で構成される。
- ノードは`op_type`（演算種別）、`inputs/outputs`（テンソル名）、`attributes`（演算パラメータ）を持つ。例えば木モデルは`TreeEnsembleRegressor`ノード1つに木のパラメータが全て埋め込まれる。
- 型・形状は`ValueInfoProto`に定義され、デフォルトでは浮動小数は`float32`が多い（変換時の指定に依存）。
- `opset`は演算仕様のバージョン。変換時の`target_opset`を固定すると再現性が上がる（本実験では17）。
- 推論はランタイム（ここではONNX Runtime, ORT）が実装する。CPU/GPU/カスタムビルドで挙動や精度・速度が変わる可能性がある。

## 変換前後で精度が変わり得る主な要因
- **型の違い（float64→float32）**: scikit-learnはfloat64学習・推論が基本。ONNX変換では初期値がfloat32になりやすく、入力もfloat32にキャストされるため丸め誤差が発生。  
  - 対策: ① scikit-learn側も評価時にfloat32へキャストし、丸めだけの差を切り分ける。② `DoubleTensorType`で変換してfloat64を維持（ただしターゲット環境が対応しているか要確認）。③ 前処理段階で明示的にfloat32へ統一して学習・推論する。
- **演算仕様/近似の違い**: ランタイムごとに乱数生成・数学ライブラリ・並列計算の順序が異なり、累積誤差が変わる。木モデルでは分岐の閾値比較の丸め方でも差が出る。
- **未対応/代替実装**: 一部のスキーマ外の処理（カスタム前処理、Estimatorの未サポート機能）が外れたり、近似演算に置き換わると精度が変わる。変換警告を必ず確認する。
- **前処理・後処理の省略**: 標準化、欠損補完、特徴スケーリング、後段の活性化（例: ロジスティック変換）の取り扱いがモデル本体と分離されていると抜け漏れや設定違いが起こる。
- **乱数シードの不一致**: 木モデル/ブースティングではシード差で学習結果が変わる。`random_state`を固定する。
- **量子化・最適化パス**: デプロイ時に量子化（int8など）やグラフ最適化をかけると、更に丸め誤差や閾値のズレが増える。量子化後は再度精度を検証する。

## 実験スクリプトの概要
- 時系列（単変量）の系列をラグ特徴量に展開し、`HistGradientBoostingRegressor`で回帰。
- 学習済みモデルをONNXへ変換（`target_opset=17`、入力は`float32`）し、ORTで推論。
- 以下の3通りの予測を比較して差分を数値化。
  - scikit-learn (float64入力)
  - scikit-learn (float32入力) — 丸めだけの影響を見るためのベースライン
  - ONNX Runtime (float32入力)
- メトリクス: RMSE/MAE、およびscikit-learnとの予測差分（平均・最大絶対値）。

### 使い方（Kaggle等のCSVでもOK）
```bash
# 1) 合成データで動作確認
python scikit_learn_onnx_repo/onnx_conversion_accuracy_evaluation/evaluate_hgb_time_series.py \
  --output-dir scikit_learn_onnx_repo/onnx_conversion_accuracy_evaluation/outputs

# 2) CSV（例: Kaggleの時系列データ）を使う場合
#   - time_col: 時間順ソートに使う列（任意）
#   - value_col: 予測対象となる系列の列名（数値）
python scikit_learn_onnx_repo/onnx_conversion_accuracy_evaluation/evaluate_hgb_time_series.py \
  --csv-path your_dataset.csv \
  --time-col timestamp_column \
  --value-col target_value_column \
  --lags 48 \
  --test-size 0.2
```
- `--lags`は作成するラグ特徴量の数（系列を何ステップ遡るか）。
- 出力: `outputs/hist_gradient_boosting.onnx`, `outputs/metrics.json` に保存。

### 実行結果サンプル（合成データ）
- RMSE/MAE（scikit-learn float64）: 0.1035 / 0.0786
- RMSE/MAE（scikit-learn float32）: 0.1035 / 0.0786
- RMSE/MAE（ONNX Runtime）: 0.1035 / 0.0786
- 予測差分: 平均絶対差 ≈ 1.9e-7, 最大絶対差 ≈ 8.0e-7  
→ 丸め（float64→float32）起因の差が支配的で、木モデルの論理は一致していることを確認。

## 精度差を潰すためのチェックリスト
- **入力dtypeを固定する**: デプロイ経路がfloat32なら学習・評価もfloat32に統一し、scikit-learnとの比較もfloat32で行う。
- **変換時のdtypeを意識**: どうしてもfloat64を保持したい場合は`DoubleTensorType`で変換し、ターゲット環境でdouble演算が許容されるか確認する。
- **前処理のONNX化**: 正規化やラグ生成など、前処理をPython側で行っている場合は同じ手順で推論時にも適用するか、ONNXパイプラインに含める。
- **opsetとサポート状況を確認**: 変換ログにWARN/ERRORがないか見る。未対応オプション（例: 特殊なロスや後処理）がある場合は近似手段を検討する。
- **ベースラインを揃えて差分を切り分ける**: 「sklearn float64」「sklearn float32」「ONNX float32」「量子化後」など段階を分けて比較し、どこで差が拡大するか特定する。
- **閾値のズレを監視**: 木モデルは浮動小数の比較で分岐が変わる可能性がある。入力スケーリングや丸めを一定にして、閾値近傍のサンプルで差が出やすいか確認する。

## 追加でやれること（必要に応じて）
- ONNXグラフの中身確認: `onnx.helper.printable_graph(onnx_model.graph)` でノードやパラメータを確認。
- ORTの最適化レベル変更: `SessionOptions().graph_optimization_level` を切り替えて差分を観察。
- 量子化実験: `onnxruntime.quantization.quantize_dynamic` などでint8化し、再度差分を計測。
