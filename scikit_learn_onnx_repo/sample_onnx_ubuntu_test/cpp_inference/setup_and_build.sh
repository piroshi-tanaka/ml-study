#!/bin/bash

# ONNX C++推論環境のセットアップとビルドスクリプト
# WSL Ubuntu環境用

set -e  # エラーが発生したら停止

echo "========================================"
echo "  ONNX Runtime C++ セットアップ"
echo "========================================"

# ONNX Runtimeのバージョン
ONNX_VERSION="1.18.0"
ONNX_DIR="onnxruntime-linux-x64-${ONNX_VERSION}"
ONNX_ARCHIVE="${ONNX_DIR}.tgz"
DOWNLOAD_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/${ONNX_ARCHIVE}"

# ステップ1: 必要なパッケージのインストール
echo ""
echo "📦 ステップ1: 必要なパッケージのインストール"
sudo apt update
sudo apt install -y build-essential cmake wget

# ステップ2: ONNX Runtimeのダウンロードとインストール
echo ""
echo "📥 ステップ2: ONNX Runtimeのダウンロード"

# 既にダウンロード済みかチェック
if [ -f "/usr/local/lib/libonnxruntime.so" ]; then
    echo "✓ ONNX Runtimeは既にインストールされています"
    ldconfig -p | grep onnxruntime || true
else
    echo "ONNX Runtime v${ONNX_VERSION}をダウンロード中..."
    
    cd ~
    
    # 既にダウンロード済みのアーカイブがあるかチェック
    if [ ! -f "${ONNX_ARCHIVE}" ]; then
        wget "${DOWNLOAD_URL}"
    else
        echo "✓ アーカイブは既にダウンロード済み"
    fi
    
    # 解凍
    if [ ! -d "${ONNX_DIR}" ]; then
        echo "解凍中..."
        tar -xzf "${ONNX_ARCHIVE}"
    else
        echo "✓ 既に解凍済み"
    fi
    
    # システムディレクトリにコピー
    echo "システムディレクトリにインストール中..."
    sudo cp -r "${ONNX_DIR}/include/"* /usr/local/include/
    sudo cp -r "${ONNX_DIR}/lib/"* /usr/local/lib/
    
    # ライブラリパスの更新
    sudo ldconfig
    
    echo "✓ ONNX Runtimeのインストール完了"
fi

# インストールの確認
echo ""
echo "🔍 インストールの確認:"
if [ -d "/usr/local/include/onnxruntime" ]; then
    echo "  ✓ ヘッダーファイル: /usr/local/include/onnxruntime/"
    ls /usr/local/include/onnxruntime/ | head -3
else
    echo "  ❌ ヘッダーファイルが見つかりません"
    exit 1
fi

if [ -f "/usr/local/lib/libonnxruntime.so" ]; then
    echo "  ✓ ライブラリ: /usr/local/lib/libonnxruntime.so"
    ls -lh /usr/local/lib/libonnxruntime.so*
else
    echo "  ❌ ライブラリが見つかりません"
    exit 1
fi

# ステップ3: プロジェクトのビルド
echo ""
echo "🔨 ステップ3: プロジェクトのビルド"

# スクリプトのディレクトリに戻る
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}"

# ビルドディレクトリの作成
if [ -d "build" ]; then
    echo "既存のbuildディレクトリをクリーンアップ..."
    rm -rf build
fi

mkdir build
cd build

# CMake実行
echo "CMake実行中..."
cmake ..

# ビルド実行
echo "ビルド実行中..."
make

# ビルド結果の確認
if [ -f "onnx_inference" ]; then
    echo ""
    echo "========================================"
    echo "  ✓ ビルド成功！"
    echo "========================================"
    echo ""
    echo "実行方法:"
    echo "  cd ${SCRIPT_DIR}/build"
    echo "  ./onnx_inference ../time_series_model.onnx ../test_data.csv ../test_labels.csv"
    echo ""
    echo "または:"
    echo "  cd ${SCRIPT_DIR}"
    echo "  ./build/onnx_inference"
    echo ""
else
    echo "❌ ビルドに失敗しました"
    exit 1
fi

