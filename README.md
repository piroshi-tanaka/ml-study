# ML Study Project

機械学習の学習・研究用プロジェクトです。

## プロジェクト構成

```
ml-study/
├── venv/                    # 仮想環境（.gitignoreで除外）
├── requirements.txt         # 依存関係
├── .gitignore              # Git除外設定
├── README.md               # このファイル
├── EV-Battery-Parking-Degradation-Mitigation/
│   └── main.ipynb          # EVバッテリー劣化軽減プロジェクト
└── nyc-taxi-trip-duration/
    ├── data/               # データファイル（.gitignoreで除外）
    ├── main.ipynb          # NYCタクシー旅行時間予測プロジェクト
    └── AutogluonModels/    # AutoMLモデル（.gitignoreで除外）
```

## セットアップ手順

### 1. 仮想環境の作成とアクティベート

```bash
# 仮想環境の作成
python -m venv venv

# 仮想環境のアクティベート
# Windows PowerShell:
.\venv\Scripts\Activate.ps1

# Windows Command Prompt:
.\venv\Scripts\activate.bat

# macOS/Linux:
source venv/bin/activate
```

### 2. 依存関係のインストール

```bash
pip install -r requirements.txt
```

### 3. Jupyter Notebookの起動

```bash
jupyter notebook
# または
jupyter lab
```

## インストール済みライブラリ

### 基本ライブラリ
- **pandas** - データ分析・操作
- **numpy** - 数値計算
- **matplotlib** - グラフ描画
- **seaborn** - 統計的データ可視化
- **scikit-learn** - 機械学習アルゴリズム
- **jupyter** - ノートブック環境

### 可視化ライブラリ
- **plotly** - インタラクティブなグラフ
- **bokeh** - インタラクティブな可視化
- **altair** - 宣言的な統計的グラフィックス
- **folium** - 地図可視化
- **wordcloud** - ワードクラウド生成

### 自然言語処理・テキスト分析
- **nltk** - 自然言語処理ツールキット
- **spacy** - 高度な自然言語処理
- **gensim** - トピックモデリング・文書類似度
- **transformers** - Hugging Face Transformers

### ディープラーニング
- **torch** - PyTorch（ディープラーニングフレームワーク）
- **torchvision** - コンピュータビジョン用PyTorch
- **torchaudio** - 音声処理用PyTorch

## プロジェクト

### 1. EV-Battery-Parking-Degradation-Mitigation
電気自動車のバッテリー劣化軽減に関するプロジェクト

### 2. nyc-taxi-trip-duration
NYCタクシーの旅行時間予測プロジェクト

## 注意事項

- 大きなデータファイル（.csv, .json等）は`.gitignore`で除外されています
- モデルファイルやチェックポイントも除外されています
- 仮想環境（venv/）も除外されています

## データの管理

データファイルは以下の方法で管理することを推奨します：

1. **小さいサンプルデータ**: リポジトリに含める
2. **大きなデータセット**: 外部ストレージ（Google Drive, AWS S3等）に保存
3. **データの取得方法**: READMEに記載する

## ライセンス

このプロジェクトは学習目的で作成されています。
