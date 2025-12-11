#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <fstream>
#include <iomanip> // 用于 std::fixed 和 std::setprecision

#include <opencv2/opencv.hpp>
#include "super_resolution.h" // 包含我们刚重构的头文件

// ==================== 辅助函数：保存矩阵到 JSON ====================
void saveMatToJson(const cv::Mat& mat, const std::string& filename) {
    if (mat.empty() || mat.type() != CV_32F) {
        std::cerr << "错误：矩阵为空或类型不是 CV_32F。" << std::endl;
        return;
    }

    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "错误：无法打开文件进行写入: " << filename << std::endl;
        return;
    }

    file << "{\n";
    file << "    \"rows\": " << mat.rows << ",\n";
    file << "    \"cols\": " << mat.cols << ",\n";
    file << "    \"data\": [\n";

    for (int i = 0; i < mat.rows; ++i) {
        file << "        [";
        for (int j = 0; j < mat.cols; ++j) {
            file << std::fixed << std::setprecision(8) << mat.at<float>(i, j);
            if (j < mat.cols - 1) {
                file << ", ";
            }
        }
        file << "]";
        if (i < mat.rows - 1) {
            file << ",\n";
        } else {
            file << "\n";
        }
    }

    file << "    ]\n";
    file << "}\n";
    file.close();
    std::cout << "成功将矩阵保存到 " << filename << std::endl;
}
// 保存 vector<Mat> (仿射矩阵数组) 到 JSON
void saveAffineMatricesToJson(const std::vector<cv::Mat>& mats, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) return;

    file << "[\n"; // 数组开始
    for (size_t k = 0; k < mats.size(); ++k) {
        const cv::Mat& mat = mats[k];
        file << "  {\n";
        file << "    \"rows\": " << mat.rows << ",\n";
        file << "    \"cols\": " << mat.cols << ",\n";
        file << "    \"data\": [\n";
        for (int i = 0; i < mat.rows; ++i) {
            file << "      [";
            for (int j = 0; j < mat.cols; ++j) {
                // 使用高精度保存 double
                file << std::fixed << std::setprecision(12) << mat.at<double>(i, j); 
                if (j < mat.cols - 1) file << ", ";
            }
            file << "]";
            if (i < mat.rows - 1) file << ",\n";
            else file << "\n";
        }
        file << "    ]\n";
        file << "  }";
        if (k < mats.size() - 1) file << ",\n";
        else file << "\n";
    }
    file << "]\n";
    file.close();
    std::cout << "Saved Affine Matrices to " << filename << std::endl;
}

// ==================== 主函数 ====================
int main() {
    // --- 1. 配置参数 ---
    const int WIDTH = 255;
    const int HEIGHT = 255;
    const int RES_FACTOR = 4;
    
    // 请根据你的本地路径修改这里
    // const std::string IMAGE_PATH = "D:/chaofen2/image/1341/";
    const std::string IMAGE_PATH = "D:/image/test16/";
    const std::string IMAGE_EXT = ".bmp";

    // 定义要处理的图像索引 (与你之前的代码保持一致)
    // const std::vector<int> IMAGE_INDICES = {
    //     0, 1, 2, 3, 4, 5, 6, 7,
    //     8, 9, 10, 11, 12, 13,
    //     14, 15, 16, 17, 18, 19,
    //     20, 21, 22, 23, 24, 25, 
    //     26, 27, 28, 29, 30, 31, 
    //     32, 33, 34, 35
    // };
   const std::vector<int> IMAGE_INDICES = {
        5, 6, 7, 9,10,11,12,13,14,
    };

    const int NUM_FRAMES = IMAGE_INDICES.size();
    std::cout << "配置信息: " << WIDTH << "x" << HEIGHT << " | " 
              << NUM_FRAMES << " 帧 | " << RES_FACTOR << "x 超分倍率" << std::endl;

    // --- 2. 加载图像 ---
    std::cout << "\n正在加载 " << NUM_FRAMES << " 张指定的低分辨率图像..." << std::endl;
    std::vector<cv::Mat> lr_frames;

    for (int index : IMAGE_INDICES) {
        // 注意：这里假设文件名是 1.png 对应 index 0 (即 index + 1)
        std::string filename = IMAGE_PATH + std::to_string(index + 1) + IMAGE_EXT;
        cv::Mat img = cv::imread(filename, cv::IMREAD_GRAYSCALE);

        if (img.empty()) {
            std::cerr << "错误：无法加载图像: " << filename << std::endl;
            return -1;
        }
        if (img.cols != WIDTH || img.rows != HEIGHT) {
            std::cerr << "错误：图像 " << filename << " 尺寸不正确。" << std::endl;
            return -1;
        }
        lr_frames.push_back(img);
        // std::cout << "  已加载: " << filename << std::endl; // 减少刷屏
    }
    std::cout << "所有图像加载完成。" << std::endl;

    // --- 3. 初始化 SuperResolution 模块 ---
    // 这里会自动调用 initCudaMemory
    std::cout << "\n正在初始化 SuperResolution 模块..." << std::endl;
    SuperResolution sr_module(WIDTH, HEIGHT, NUM_FRAMES, RES_FACTOR);

    // --- 4. 阶段一：运动估计 (Motion Estimation) ---
    std::cout << "\n--- 阶段一：执行运动估计 ---" << std::endl;
    cv::Mat Tvec, Rvec;
    std::vector<cv::Mat> registered_frames_vis; // 仅用于调试查看，process 内部会重新处理配准

    auto start_motion = std::chrono::high_resolution_clock::now();
    
    // 调用封装好的运动估计
    sr_module.estimateMotion(lr_frames, registered_frames_vis, Tvec, Rvec);
    
    auto end_motion = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> motion_duration = end_motion - start_motion;
    // [新增] 保存完整矩阵
    std::vector<cv::Mat> aff_mats = sr_module.getAffineMatrices();
    saveAffineMatricesToJson(aff_mats, "fixed_Affine.json");
    if (Tvec.empty() || Rvec.empty()) {
        std::cerr << "错误：运动估计失败。" << std::endl;
        return -1;
    }
    std::cout << "运动估计完成，耗时 " << motion_duration.count() << " ms." << std::endl;

    // 保存参数便于分析
    saveMatToJson(Tvec, "fixed_Tvec.json");
    saveMatToJson(Rvec, "fixed_Rvec.json");

    // --- 5. 阶段二：超分辨率处理 (SR Process) ---
    std::cout << "\n--- 阶段二：执行超分辨率处理 (新算法) ---" << std::endl;
    std::cout << "提示：process 函数内部将自动执行以下步骤：" << std::endl;
    std::cout << "  1. 根据 Tvec/Rvec 对图像进行配准" << std::endl;
    std::cout << "  2. 放大参考帧作为底图 (Base Map)" << std::endl;
    std::cout << "  3. 调用 CUDA 核函数进行亚像素填充" << std::endl;

    auto start_process = std::chrono::high_resolution_clock::now();
    
    // === 核心调用 ===
    // 我们传入原始序列 lr_frames 和计算好的 Tvec/Rvec
    cv::Mat sr_image = sr_module.process(lr_frames, Tvec, Rvec);
    
    auto end_process = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> process_duration = end_process - start_process;

    if (sr_image.empty()) {
        std::cerr << "错误：超分辨率处理返回空图像。" << std::endl;
        return -1;
    }

    std::cout << "超分辨率处理完成，耗时 " << process_duration.count() << " ms." << std::endl;
    std::cout << "结果尺寸: " << sr_image.cols << "x" << sr_image.rows << std::endl;

    // --- 6. 保存并显示结果 ---
    std::string out_filename = "super_resolution_result_new.png";
    std::cout << "\n正在保存结果到 '" << out_filename << "'..." << std::endl;
    cv::imwrite(out_filename, sr_image);
    
    std::cout << "显示结果 (按任意键退出)..." << std::endl;
    cv::imshow("Super-Resolution Result (New Algorithm)", sr_image);
    
    // 可选：显示底图（如果你在 estimateMotion 里保存了或者在这里手动放大一张看对比）
    // cv::Mat base_preview;
    // cv::resize(lr_frames[0], base_preview, cv::Size(), RES_FACTOR, RES_FACTOR, cv::INTER_NEAREST);
    // cv::imshow("Reference Base (Nearest Neighbor)", base_preview);

    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}