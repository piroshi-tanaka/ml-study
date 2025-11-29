# 01. PythonからONNX形式を作成する手順

## 📋 目次

1. [概要](#概要)
2. [前提条件](#前提条件)
3. [環境構築](#環境構築)
4. [Jupyter Notebookの実行](#jupyter-notebookの実行)
5. [ONNX変換の確認](#onnx変換の確認)
6. [トラブルシューティング](#トラブルシューティング)

---

## 概要

### 目的

- scikit-learnで時系列予測モデルをトレーニング
- トレーニング済みモデルをONNX形式に変換
- C++推論用のテストデータをエクスポート

### 成果物

| ファイル名 | 説明 | 場所 |
|-----------|------|------|
| `time_series_model.onnx` | ONNX形式のモデル | `cpp_inference/` |
| `test_data.csv` | テストデータ（特徴量） | `cpp_inference/` |
| `test_labels.csv` | 正解ラベル+Python予測結果 | `cpp_inference/` |

### 所要時間

- 初回: 約20-30分（パッケージインストール含む）
- 2回目以降: 約5-10分

---

## 前提条件

### システム要件

- **OS**: Windows 10/11
- **Python**: 3.12以上
- **ディスク容量**: 約2GB以上の空き容量

### 必要なツール

- ✅ **uv** (Pythonパッケージマネージャー)
- ✅ **Jupyter Notebook**
- ✅ **VS Code** (推奨)

### プロジェクト構造

```
ml-study/
├── pyproject.toml
└── scikit_learn_onnx_repo/
    └── time_series_onnx_demo.ipynb  ← このNotebookを使用
```

---

## 環境構築

### ステップ1: プロジェクトディレクトリに移動

**PowerShell または コマンドプロンプト** を開く：

```powershell
cd C:\workspace\src\ml-study
```

### ステップ2: 依存関係の確認

`pyproject.toml` に以下の依存関係が含まれているか確認：

```toml
[project]
name = "ml-study"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "scikit-learn>=1.5.0",
    "skl2onnx>=1.17.0",
    "onnxruntime>=1.18.0",
    "onnx>=1.16.0",
    "pandas>=2.2.0",
    "numpy>=1.26.0",
    "matplotlib>=3.8.0",
    "seaborn>=0.13.0",
    "jupyter>=1.0.0",
    "statsmodels>=0.14.0",
]
```

### ステップ3: 依存関係のインストール

```powershell
# パッケージのインストールと同期
uv sync
```

**期待される出力**:
```
Resolved XX packages in XXXms
...
✓ すべてのパッケージがインストールされました
```

**所要時間**: 初回 約5-10分

---

## Jupyter Notebookの実行

### ステップ1: Jupyter Notebookの起動

```powershell
# プロジェクトルートで実行
uv run jupyter notebook
```

**ブラウザが自動的に開きます**:
```
http://localhost:8888/tree
```

### ステップ2: Notebookを開く

1. ブラウザでファイル一覧から `scikit_learn_onnx_repo` フォルダをクリック
2. `time_series_onnx_demo.ipynb` をクリックして開く

### ステップ3: セルを順番に実行

**方法A: すべてのセルを一括実行（推奨）**

メニューから：
```
Kernel → Restart & Run All
```

**方法B: セルを1つずつ実行**

各セルを選択して：
- `Shift + Enter`: 実行して次のセルへ
- `Ctrl + Enter`: 実行してそのまま

### ステップ4: 実行の進捗確認

#### セル0-4: ライブラリのインポート

**期待される出力**:
```
✓ ライブラリのインポート完了
```

#### セル5-6: データの読み込み

**期待される出力**:
```
データ期間: 1949年1月 〜 1960年12月
総サンプル数: 144 ヶ月
値の範囲: 104 〜 622 (千人)

データ準備完了:
  特徴量の形状: (132, 12)
  ターゲットの形状: (132,)
  特徴量数: 12 (ラグ6 + 統計量5 + 月1)
```

✅ サンプル数が132になっていればOK

#### セル7-8: データの可視化

航空旅客数のグラフが表示されます：
- 上段: 全期間の推移（トレンドが右肩上がり）
- 下段: 年別の季節性パターン（夏に高く、冬に低い）

#### セル9-10: モデルのトレーニング

**期待される出力**:
```
データ分割:
  トレーニング: 79 サンプル (60.0%)
  検証: 26 サンプル (20.0%)
  テスト: 27 サンプル (20.0%)

モデルをトレーニング中...
✓ モデルのトレーニング完了

【トレーニングセット】
  RMSE: X.XXXX
  MAE: X.XXXX
  R²: X.XXXX

【検証セット】
  RMSE: XX.XXXX
  MAE: XX.XXXX
  R²: X.XXXX
```

✅ R²が0.5以上あれば合格

#### セル11-12: ONNX変換 ⭐ 重要

**期待される出力**:
```
ONNXへの変換中...
✓ ONNXモデルを保存: time_series_model.onnx
✓ scikit-learnモデルを保存: time_series_model.pkl
```

✅ エラーなく完了すればOK

#### セル13-16: ONNX推論と精度検証

**期待される出力**:
```
ONNX推論の実行中...
  入力名: float_input
  出力名: variable

✓ 推論完了

============================================================
【テストセット】精度比較
============================================================

【scikit-learn】
  RMSE: XX.XXXXXX
  MAE:  XX.XXXXXX
  R²:   X.XXXXXX

【ONNX】
  RMSE: XX.XXXXXX
  MAE:  XX.XXXXXX
  R²:   X.XXXXXX

【scikit-learn vs ONNX】予測値の差分
  最大差分: X.XXXXXXe-XX
  平均差分: X.XXXXXXe-XX
  → ONNXとscikit-learnの予測はほぼ一致！
============================================================
```

✅ **「ほぼ一致！」と表示されればOK**（差分が1e-5以下が理想）

#### セル17-20: 結果の可視化

3つのグラフが表示されます：

1. **prediction_comparison.png**
   - 実績値とPython予測の比較
   - sklearn vs ONNX の差分プロット

2. **residual_analysis.png**
   - 残差の時系列プロット
   - 残差の分布
   - 実績値 vs 予測値
   - sklearn vs ONNX の比較

3. **error_analysis.png**
   - 絶対誤差の推移
   - sklearn と ONNX の差分

#### セル21-22: C++推論用データのエクスポート ⭐ 重要

**期待される出力**:
```
✓ C++推論用データのエクスポート完了:
  📁 ディレクトリ: cpp_inference/
  📊 テストデータ: cpp_inference/test_data.csv (27 サンプル, 12 特徴量)
  🎯 正解ラベル: cpp_inference/test_labels.csv
  🔮 ONNXモデル: cpp_inference/time_series_model.onnx

テストデータの最初の3行:
      lag_1     lag_2     lag_3  ...
0  148.0000  148.0000  136.0000  ...
1  148.0000  136.0000  119.0000  ...
2  136.0000  119.0000  104.0000  ...
```

✅ `cpp_inference/` ディレクトリが作成され、3つのファイルが保存されればOK

#### セル23-24: 完了メッセージ

**期待される出力**:
```
============================================================
✓ すべての処理が完了しました！
============================================================

生成されたファイル:
  Python推論用:
    - time_series_model.onnx
    - time_series_model.pkl
  可視化:
    - airline_data_overview.png
    - prediction_comparison.png
    - residual_analysis.png
    - error_analysis.png
  C++推論用:
    - cpp_inference/time_series_model.onnx
    - cpp_inference/test_data.csv
    - cpp_inference/test_labels.csv
```

---

## ONNX変換の確認

### ステップ1: ファイルの存在確認

エクスプローラーまたはVS Codeで以下のファイルが作成されたか確認：

```
scikit_learn_onnx_repo/
├── time_series_model.onnx     ✓
├── time_series_model.pkl      ✓
├── *.png (4つ)                ✓
└── cpp_inference/
    ├── time_series_model.onnx ✓ ← C++推論用
    ├── test_data.csv          ✓ ← C++推論用
    └── test_labels.csv        ✓ ← C++推論用
```

### ステップ2: ONNXモデルのサイズ確認

**PowerShellで確認**:
```powershell
cd C:\workspace\src\ml-study\scikit_learn_onnx_repo\cpp_inference
dir time_series_model.onnx
```

**期待されるサイズ**: 約1-5MB

### ステップ3: テストデータの確認

**CSVファイルを開いて確認**:

`test_data.csv` の内容：
- ヘッダー行: `lag_1, lag_2, ..., lag_6, mean, std, min, max, trend, month`
- データ行: 27行（テストサンプル数）

`test_labels.csv` の内容：
- ヘッダー行: `true_value, sklearn_pred, onnx_pred`
- データ行: 27行

✅ すべて確認できれば完了！

---

## トラブルシューティング

### Q1: パッケージのインストールに失敗する

**エラー例**:
```
Failed to download package...
```

**解決策**:
```powershell
# インターネット接続を確認
# プロキシ設定が必要な場合は設定

# 再試行
uv sync

# それでもダメな場合は個別インストール
uv add scikit-learn skl2onnx onnxruntime onnx pandas numpy matplotlib seaborn jupyter statsmodels
```

### Q2: Jupyter Notebookが起動しない

**エラー例**:
```
Jupyter command not found
```

**解決策**:
```powershell
# 明示的にインストール
uv add jupyter

# 再起動
uv run jupyter notebook
```

### Q3: ONNX変換でエラーが発生

**エラー例**:
```
ValueError: Model input type not supported
```

**解決策**:
1. scikit-learnとskl2onnxのバージョンを確認
2. Notebookを最初から再実行（Kernel → Restart & Run All）
3. エラーメッセージをよく読んで対処

### Q4: カーネルがビジー状態のまま

**現象**: セルの実行が終わらない

**解決策**:
```
Kernel → Interrupt
```

その後、問題のセルを再実行

### Q5: グラフが表示されない

**解決策**:
```python
# Notebookの最初のセルで以下を実行
%matplotlib inline
```

その後、グラフを表示するセルを再実行

---

## ✅ チェックリスト

完了したら、以下を確認してください：

- [ ] `pyproject.toml` に必要な依存関係が含まれている
- [ ] `uv sync` が正常に完了した
- [ ] Jupyter Notebookがブラウザで開いた
- [ ] すべてのセルがエラーなく実行された
- [ ] 「ONNXとscikit-learnの予測はほぼ一致！」と表示された
- [ ] `cpp_inference/` ディレクトリが作成された
- [ ] `time_series_model.onnx` が作成された（約1-5MB）
- [ ] `test_data.csv` が作成された（27行）
- [ ] `test_labels.csv` が作成された（27行）
- [ ] 4つの可視化画像が作成された

---

## 📚 次のステップ

✅ **このステップが完了したら、次のドキュメントへ進んでください：**

👉 [**02_WSL_Ubuntu_環境構築手順.md**](./02_WSL_Ubuntu_環境構築手順.md)

WSL Ubuntu環境でC++推論を実行するための環境を構築します。

---

## 📊 補足情報

### 使用しているデータセット

- **名前**: 国際航空旅客数データ（AirPassengers）
- **期間**: 1949年1月 〜 1960年12月
- **サンプル数**: 144ヶ月
- **出典**: R統計ソフトウェアの標準データセット
- **特徴**: 
  - 明確なトレンド（時間とともに増加）
  - 明確な季節性（夏季に増加、冬季に減少）
  - 実際のビジネスデータとして解釈しやすい

### モデルの詳細

- **アルゴリズム**: RandomForestRegressor
- **ハイパーパラメータ**:
  - n_estimators: 50
  - max_depth: 5
  - min_samples_split: 5
  - min_samples_leaf: 2
- **特徴量**: 12個
  - ラグ特徴量（過去6ヶ月）
  - 統計的特徴量（mean, std, min, max, trend）
  - 月情報

### ONNX変換の意義

1. **プラットフォーム独立性**
   - Pythonだけでなく、C++、Java、C#などで推論可能

2. **パフォーマンス最適化**
   - 推論速度の向上
   - メモリ使用量の削減

3. **組み込みシステムへの展開**
   - AUTOSAR環境での実行可能性
   - エッジデバイスでの推論

4. **精度の保証**
   - Python推論と同等の精度
   - 数値誤差が非常に小さい（1e-5以下）

---

**作成日**: 2025-11-02  
**バージョン**: 1.0  
**対象**: Python初心者〜中級者

