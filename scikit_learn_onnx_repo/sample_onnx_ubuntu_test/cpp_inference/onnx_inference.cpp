#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip>
#include <onnxruntime_cxx_api.h>

// CSVファイル読み込み関数
std::vector<std::vector<float>> loadCSV(const std::string& filename, bool skipHeader = true) {
    std::vector<std::vector<float>> data;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "❌ ファイルを開けません: " << filename << std::endl;
        return data;
    }
    
    std::string line;
    bool firstLine = true;
    
    while (std::getline(file, line)) {
        if (skipHeader && firstLine) {
            firstLine = false;
            continue;
        }
        
        std::vector<float> row;
        std::stringstream ss(line);
        std::string value;
        
        while (std::getline(ss, value, ',')) {
            row.push_back(std::stof(value));
        }
        
        if (!row.empty()) {
            data.push_back(row);
        }
    }
    
    file.close();
    return data;
}

// 精度評価関数
struct Metrics {
    float rmse;
    float mae;
    float r2;
};

Metrics calculateMetrics(const std::vector<float>& true_values, const std::vector<float>& predictions) {
    Metrics metrics = {0.0f, 0.0f, 0.0f};
    
    if (true_values.size() != predictions.size() || true_values.empty()) {
        return metrics;
    }
    
    size_t n = true_values.size();
    
    // 平均の計算
    float mean = 0.0f;
    for (float val : true_values) {
        mean += val;
    }
    mean /= n;
    
    // RMSE, MAE, R²の計算
    float sse = 0.0f;  // Sum of Squared Errors
    float sst = 0.0f;  // Total Sum of Squares
    float mae_sum = 0.0f;
    
    for (size_t i = 0; i < n; ++i) {
        float error = true_values[i] - predictions[i];
        sse += error * error;
        mae_sum += std::abs(error);
        
        float deviation = true_values[i] - mean;
        sst += deviation * deviation;
    }
    
    metrics.rmse = std::sqrt(sse / n);
    metrics.mae = mae_sum / n;
    metrics.r2 = 1.0f - (sse / sst);
    
    return metrics;
}

int main(int argc, char* argv[]) {
    std::cout << "========================================" << std::endl;
    std::cout << "  ONNX時系列予測推論（C++版）" << std::endl;
    std::cout << "========================================" << std::endl;
    
    try {
        // ファイルパスの設定
        std::string model_path = "time_series_model.onnx";
        std::string test_data_path = "test_data.csv";
        std::string test_labels_path = "test_labels.csv";
        
        // コマンドライン引数からファイルパスを取得（オプション）
        if (argc >= 2) model_path = argv[1];
        if (argc >= 3) test_data_path = argv[2];
        if (argc >= 4) test_labels_path = argv[3];
        
        std::cout << "\n📂 ファイル読み込み中..." << std::endl;
        std::cout << "  ONNXモデル: " << model_path << std::endl;
        std::cout << "  テストデータ: " << test_data_path << std::endl;
        std::cout << "  正解ラベル: " << test_labels_path << std::endl;
        
        // テストデータの読み込み
        auto test_data = loadCSV(test_data_path, true);
        if (test_data.empty()) {
            std::cerr << "❌ テストデータの読み込みに失敗" << std::endl;
            return 1;
        }
        
        size_t num_samples = test_data.size();
        size_t num_features = test_data[0].size();
        
        std::cout << "\n✓ データ読み込み完了" << std::endl;
        std::cout << "  サンプル数: " << num_samples << std::endl;
        std::cout << "  特徴量数: " << num_features << std::endl;
        
        // 正解ラベルの読み込み
        auto labels_data = loadCSV(test_labels_path, true);
        std::vector<float> true_values, sklearn_predictions, onnx_python_predictions;
        
        for (const auto& row : labels_data) {
            if (row.size() >= 3) {
                true_values.push_back(row[0]);
                sklearn_predictions.push_back(row[1]);
                onnx_python_predictions.push_back(row[2]);
            }
        }
        
        // ONNX Runtimeの初期化
        std::cout << "\n🔧 ONNX Runtimeの初期化中..." << std::endl;
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNXInference");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        
        // セッションの作成
        Ort::Session session(env, model_path.c_str(), session_options);
        
        // 入力情報の取得
        Ort::AllocatorWithDefaultOptions allocator;
        size_t num_input_nodes = session.GetInputCount();
        size_t num_output_nodes = session.GetOutputCount();
        
        std::cout << "  入力ノード数: " << num_input_nodes << std::endl;
        std::cout << "  出力ノード数: " << num_output_nodes << std::endl;
        
        // 入力名の取得
        auto input_name_alloced = session.GetInputNameAllocated(0, allocator);
        const char* input_name = input_name_alloced.get();
        std::cout << "  入力名: " << input_name << std::endl;
        
        // 出力名の取得
        auto output_name_alloced = session.GetOutputNameAllocated(0, allocator);
        const char* output_name = output_name_alloced.get();
        std::cout << "  出力名: " << output_name << std::endl;
        
        // 推論実行
        std::cout << "\n🚀 推論実行中..." << std::endl;
        std::vector<float> cpp_predictions;
        
        // 入力・出力のテンソル形状
        std::vector<int64_t> input_shape = {1, static_cast<int64_t>(num_features)};
        std::vector<const char*> input_names = {input_name};
        std::vector<const char*> output_names = {output_name};
        
        // 各サンプルに対して推論
        for (size_t i = 0; i < num_samples; ++i) {
            // 入力テンソルの作成
            auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            std::vector<float> input_data = test_data[i];
            
            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                memory_info, 
                input_data.data(), 
                input_data.size(),
                input_shape.data(), 
                input_shape.size()
            );
            
            // 推論実行
            auto output_tensors = session.Run(
                Ort::RunOptions{nullptr},
                input_names.data(),
                &input_tensor,
                1,
                output_names.data(),
                1
            );
            
            // 結果の取得
            float* output_data = output_tensors[0].GetTensorMutableData<float>();
            cpp_predictions.push_back(output_data[0]);
            
            // 進捗表示（10件ごと）
            if ((i + 1) % 10 == 0 || i == num_samples - 1) {
                std::cout << "  進捗: " << (i + 1) << "/" << num_samples << " サンプル" << std::endl;
            }
        }
        
        std::cout << "\n✓ 推論完了" << std::endl;
        
        // 精度評価
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "【精度評価】" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        
        auto metrics_cpp = calculateMetrics(true_values, cpp_predictions);
        
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "\n【C++ ONNX推論】" << std::endl;
        std::cout << "  RMSE: " << metrics_cpp.rmse << std::endl;
        std::cout << "  MAE:  " << metrics_cpp.mae << std::endl;
        std::cout << "  R²:   " << metrics_cpp.r2 << std::endl;
        
        // Python ONNX推論との比較
        std::vector<float> diff_python_cpp;
        float max_diff = 0.0f;
        float sum_diff = 0.0f;
        
        for (size_t i = 0; i < num_samples; ++i) {
            float diff = std::abs(onnx_python_predictions[i] - cpp_predictions[i]);
            diff_python_cpp.push_back(diff);
            max_diff = std::max(max_diff, diff);
            sum_diff += diff;
        }
        
        float mean_diff = sum_diff / num_samples;
        
        std::cout << "\n【Python ONNX vs C++ ONNX】" << std::endl;
        std::cout << "  最大差分: " << std::scientific << max_diff << std::endl;
        std::cout << "  平均差分: " << mean_diff << std::endl;
        
        if (mean_diff < 1e-5) {
            std::cout << "  ✓ Python ONNXとC++ ONNXの予測はほぼ一致！" << std::endl;
        } else {
            std::cout << "  ⚠ 若干の差異があります" << std::endl;
        }
        
        // 最初の5件の予測結果を表示
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "【予測結果サンプル（最初の5件）】" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "No.  実績値    Python    C++予測   誤差" << std::endl;
        std::cout << std::string(60, '-') << std::endl;
        
        for (size_t i = 0; i < std::min(size_t(5), num_samples); ++i) {
            float error = std::abs(true_values[i] - cpp_predictions[i]);
            std::cout << std::setw(3) << (i+1) << "  "
                      << std::setw(8) << true_values[i] << "  "
                      << std::setw(8) << onnx_python_predictions[i] << "  "
                      << std::setw(8) << cpp_predictions[i] << "  "
                      << std::setw(6) << error << std::endl;
        }
        
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "✓ すべての処理が完了しました！" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        
    } catch (const Ort::Exception& e) {
        std::cerr << "❌ ONNX Runtime エラー: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "❌ エラー: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

