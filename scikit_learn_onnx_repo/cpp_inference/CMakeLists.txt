cmake_minimum_required(VERSION 3.15)
project(ONNXTimeSeriesInference)

# C++14を使用
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ONNX Runtimeのパスを指定（環境に応じて変更）
# Ubuntu/Linux環境でのデフォルトパス
set(ONNXRUNTIME_ROOTDIR "/usr/local" CACHE PATH "ONNX Runtime root directory")
set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime" CACHE PATH "ONNX Runtime include directory")
set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_ROOTDIR}/lib" CACHE PATH "ONNX Runtime library directory")

# インクルードディレクトリの追加
include_directories(${ONNXRUNTIME_INCLUDE_DIR})

# ライブラリディレクトリの追加
link_directories(${ONNXRUNTIME_LIB_DIR})

# 実行ファイルの作成
add_executable(onnx_inference onnx_inference.cpp)

# ONNX Runtimeライブラリのリンク
target_link_libraries(onnx_inference onnxruntime)

# デバッグ情報の出力
message(STATUS "ONNX Runtime Include Dir: ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "ONNX Runtime Lib Dir: ${ONNXRUNTIME_LIB_DIR}")

# インストール設定
install(TARGETS onnx_inference DESTINATION bin)

