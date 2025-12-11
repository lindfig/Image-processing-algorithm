
#include "super_resolution.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <cmath>

using namespace cv;
using namespace std;
using namespace std::chrono;

extern "C" {
    void initCudaMemory(int num, int width, int height, int resFactor, int kernel_len);
    void freeCudaMemory();
    void cudaFastRobustSR(const unsigned char* img_seq, float* HR, float* Z, float* HR_A,
        const float* Tvec, const float* Rvec, int num, int width, int height,
        int resFactor, const float* kernel, int kernel_size);
}

SuperResolution::SuperResolution(int width, int height, int num_frames, int res_factor)
    : m_width(width), m_height(height), m_num_frames(num_frames), m_res_factor(res_factor), m_initialized(false)
{
    if (m_width > 0 && m_height > 0 && m_num_frames > 0 && m_res_factor > 0) {
        initCudaMemory(m_num_frames, m_width, m_height, m_res_factor, 9);
        m_initialized = true;
        // 预分配矩阵缓存
        m_cached_affine_mats.resize(m_num_frames);
        std::cout << "CUDA 内存初始化完成。" << std::endl;
    }
    else {
        std::cerr << "Error: Invalid parameters." << std::endl;
    }
}

SuperResolution::~SuperResolution() {
    if (m_initialized) {
        freeCudaMemory();
        m_initialized = false;
    }
}

// ==================== 运动估计 (保存完整矩阵) ====================
void SuperResolution::estimateMotion(const std::vector<Mat>& in_seq,
    std::vector<Mat>& out_seq,
    Mat& Tvec, Mat& Rvec) {
    
    int num = in_seq.size();
    Mat img0 = in_seq[0];
    Tvec = Mat::zeros(num, 2, CV_32F);
    Rvec = Mat::zeros(num, 1, CV_32F);
    out_seq.clear();
    out_seq.reserve(num);
    
    // 清空并调整缓存大小
    m_cached_affine_mats.clear();
    m_cached_affine_mats.resize(num);

    for (int i = 0; i < num; i++) {
        // 第一帧是参考帧，矩阵是单位矩阵
        if (i == 0) {
            out_seq.push_back(img0.clone());
            m_cached_affine_mats[i] = Mat::eye(2, 3, CV_64F); // 保存单位矩阵
            continue;
        }

        Mat imgi = in_seq[i];
        
        // --- 1. 特征提取 ---
        std::vector<Point2f> p1;
        goodFeaturesToTrack(img0, p1, 1000, 0.01, 20.0, Mat(), 5, true, 0.04);

        // 如果特征点不足，使用单位矩阵 (不变换)
        if (p1.size() < 8) {
            out_seq.push_back(imgi.clone());
            m_cached_affine_mats[i] = Mat::eye(2, 3, CV_64F); 
            continue;
        }

        // --- 2. 光流 ---
        std::vector<Point2f> p2;
        std::vector<uchar> status;
        std::vector<float> err;
        TermCriteria termcrit(TermCriteria::MAX_ITER | TermCriteria::EPS, 100, 0.01);
        calcOpticalFlowPyrLK(img0, imgi, p1, p2, status, err, Size(23, 23), 7, termcrit, 0, 0.001);

        std::vector<Point2f> p1_filtered, p2_filtered;
        for (size_t k = 0; k < p1.size(); k++) {
            if (status[k] && err[k] < 50.0) {
                p1_filtered.push_back(p1[k]);
                p2_filtered.push_back(p2[k]);
            }
        }

        if (p1_filtered.size() < 8) {
            out_seq.push_back(imgi.clone());
            m_cached_affine_mats[i] = Mat::eye(2, 3, CV_64F);
            continue;
        }

        // --- 3. 估计变换 (包含 Scale) ---
        std::vector<uchar> inliers;
        Mat affine_mat = estimateAffinePartial2D(p2_filtered, p1_filtered, inliers,
            RANSAC, 3.0, 2000, 0.99, 10);

        if (affine_mat.empty()) {
            out_seq.push_back(imgi.clone());
            m_cached_affine_mats[i] = Mat::eye(2, 3, CV_64F);
            continue;
        }

        // [重点] 保存原始的高精度矩阵到成员变量
        m_cached_affine_mats[i] = affine_mat.clone();

        // 提取参数用于 CUDA (仅用于位移计算，不用于图像配准)
        double cos_theta = affine_mat.at<double>(0, 0);
        double sin_theta = affine_mat.at<double>(1, 0);
        double tx = affine_mat.at<double>(0, 2);
        double ty = affine_mat.at<double>(1, 2);
        
        float angle = static_cast<float>(atan2(sin_theta, cos_theta));

        Tvec.at<float>(i, 0) = static_cast<float>(tx);
        Tvec.at<float>(i, 1) = static_cast<float>(ty);
        Rvec.at<float>(i, 0) = angle;

        // 生成预览图
        Mat dst;
        warpAffine(imgi, dst, affine_mat, imgi.size(), INTER_LINEAR, BORDER_REPLICATE);
        out_seq.push_back(dst);
    }
}

// ==================== 处理函数 (优先使用缓存矩阵) ====================
cv::Mat SuperResolution::process(const std::vector<Mat>& lr_frames,
    const Mat& precomputed_Tvec,
    const Mat& precomputed_Rvec)
{
    if (!m_initialized) { return Mat(); }
    if (lr_frames.size() != m_num_frames) { return Mat(); }

    std::vector<unsigned char> img_seq_flat(m_num_frames * m_width * m_height);
    
    // 第 0 帧处理
    if (lr_frames[0].isContinuous()) {
        memcpy(&img_seq_flat[0], lr_frames[0].data, m_width * m_height);
    } else {
        Mat temp = lr_frames[0].clone();
        memcpy(&img_seq_flat[0], temp.data, m_width * m_height);
    }

    // --- 关键修改：检查是否有缓存的矩阵 ---
    bool use_cached_mats = (m_cached_affine_mats.size() == m_num_frames);
    if (use_cached_mats) {
        // std::cout << "Debug: 使用缓存的高精度矩阵进行配准 (保留Scale)" << std::endl;
    } else {
        // std::cout << "Warning: 未找到缓存矩阵，将从 Tvec/Rvec 重建 (可能会丢失 Scale 导致模糊)" << std::endl;
    }

    for (int i = 1; i < m_num_frames; i++) {
        Mat affine_mat;

        if (use_cached_mats && !m_cached_affine_mats[i].empty()) {
            // [方案A] 直接使用原始计算出的矩阵，包含 Scale，绝对精确
            affine_mat = m_cached_affine_mats[i];
        } 
        else {
            // [方案B] 如果没有缓存 (比如从文件读取 Tvec)，则必须重建 (会丢失 Scale)
            double tx = static_cast<double>(precomputed_Tvec.at<float>(i, 0));
            double ty = static_cast<double>(precomputed_Tvec.at<float>(i, 1));
            double angle = static_cast<double>(precomputed_Rvec.at<float>(i, 0));

            affine_mat = Mat::zeros(2, 3, CV_64F);
            double cosA = cos(angle);
            double sinA = sin(angle);

            affine_mat.at<double>(0, 0) = cosA;
            affine_mat.at<double>(0, 1) = -sinA;
            affine_mat.at<double>(1, 0) = sinA;
            affine_mat.at<double>(1, 1) = cosA;
            affine_mat.at<double>(0, 2) = tx;
            affine_mat.at<double>(1, 2) = ty;
        }

        // 配准图像
        Mat aligned_img;
        warpAffine(lr_frames[i], aligned_img, affine_mat, Size(m_width, m_height), 
                   INTER_LINEAR, BORDER_REPLICATE);

        if (aligned_img.isContinuous()) {
            memcpy(&img_seq_flat[i * m_width * m_height], aligned_img.data, m_width * m_height);
        } else {
             Mat continuous_img = aligned_img.clone();
             memcpy(&img_seq_flat[i * m_width * m_height], continuous_img.data, m_width * m_height);
        }
    }

    // --- 后续逻辑不变 ---
    Mat reference_upsampled;
    resize(lr_frames[0], reference_upsampled, 
           Size(m_width * m_res_factor, m_height * m_res_factor), 
           0, 0, INTER_NEAREST);
    
    std::vector<float> HR(m_width * m_res_factor * m_height * m_res_factor);
    Mat reference_float;
    reference_upsampled.convertTo(reference_float, CV_32F);
    
    if (reference_float.isContinuous()) {
        memcpy(HR.data(), reference_float.data, HR.size() * sizeof(float));
    } else {
        for(int r=0; r<reference_float.rows; ++r) {
             memcpy(HR.data() + r * reference_float.cols, 
                    reference_float.ptr<float>(r), 
                    reference_float.cols * sizeof(float));
        }
    }

    std::vector<float> Tvec_flat(m_num_frames * 2);
    std::vector<float> Rvec_flat(m_num_frames);
    memcpy(Tvec_flat.data(), precomputed_Tvec.ptr<float>(0), m_num_frames * 2 * sizeof(float));
    memcpy(Rvec_flat.data(), precomputed_Rvec.ptr<float>(0), m_num_frames * sizeof(float));

    std::vector<float> kernel_flat(9, 0.0f);
    std::vector<float> Z(m_width * m_res_factor * m_height * m_res_factor);
    std::vector<float> HR_A(m_width * m_res_factor * m_height * m_res_factor);

    cudaFastRobustSR(img_seq_flat.data(), HR.data(), Z.data(), HR_A.data(),
        Tvec_flat.data(), Rvec_flat.data(), m_num_frames, m_width, m_height,
        m_res_factor, kernel_flat.data(), 9);

    Mat HR_mat(m_height * m_res_factor, m_width * m_res_factor, CV_32F, HR.data());
    Mat HR_vis;
    normalize(HR_mat, HR_vis, 0, 255, NORM_MINMAX);
    HR_vis.convertTo(HR_vis, CV_8U);

    return HR_vis;
}



std::vector<cv::Mat> SuperResolution::getAffineMatrices() const {
    return m_cached_affine_mats;
}

void SuperResolution::setAffineMatrices(const std::vector<cv::Mat>& mats) {
    if (mats.size() == m_num_frames) {
        m_cached_affine_mats.clear();
        for (const auto& m : mats) {
            // 确保深拷贝，防止外部释放
            m_cached_affine_mats.push_back(m.clone()); 
        }
        std::cout << "Debug: 已手动设置缓存仿射矩阵，Process 将使用高精度配准。" << std::endl;
    } else {
        std::cerr << "Error: 设置的矩阵数量与帧数不匹配！" << std::endl;
    }
}