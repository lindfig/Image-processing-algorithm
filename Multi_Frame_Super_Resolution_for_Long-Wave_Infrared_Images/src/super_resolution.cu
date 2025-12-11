#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <stdio.h>

using namespace std;

// 检查 CUDA 错误
void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err));
        exit(-1);
    }
}

// ==================== 全局设备内存 ====================
static unsigned char* d_img_seq = nullptr;
static float* d_HR = nullptr, * d_Z = nullptr, * d_HR_A = nullptr;
static float* d_Tvec = nullptr, * d_Rvec = nullptr, * d_kernel = nullptr;
static int* d_D = nullptr;
static size_t img_seq_size = 0, HR_size = 0, Tvec_size = 0, Rvec_size = 0, D_size = 0, kernel_size = 0;
static cudaStream_t stream;

// ==================== 内存管理 ====================
extern "C" void initCudaMemory(int num, int width, int height, int resFactor, int kernel_len) {
    size_t new_img_seq_size = num * width * height * sizeof(unsigned char);
    size_t new_HR_size = width * resFactor * height * resFactor * sizeof(float);
    size_t new_Tvec_size = num * 2 * sizeof(float);
    size_t new_Rvec_size = num * sizeof(float);
    size_t new_D_size = num * 2 * sizeof(int);
    size_t new_kernel_size = kernel_len * sizeof(float);

    // 重新分配内存（如果需要）
    if (d_img_seq && img_seq_size < new_img_seq_size) { cudaFree(d_img_seq); d_img_seq = nullptr; }
    if (d_HR && HR_size < new_HR_size) { cudaFree(d_HR); cudaFree(d_Z); cudaFree(d_HR_A); d_HR = d_Z = d_HR_A = nullptr; }
    if (d_Tvec && Tvec_size < new_Tvec_size) { cudaFree(d_Tvec); d_Tvec = nullptr; }
    if (d_Rvec && Rvec_size < new_Rvec_size) { cudaFree(d_Rvec); d_Rvec = nullptr; }
    if (d_D && D_size < new_D_size) { cudaFree(d_D); d_D = nullptr; }
    if (d_kernel && kernel_size < new_kernel_size) { cudaFree(d_kernel); d_kernel = nullptr; }

    // 分配新内存
    if (!d_img_seq) { cudaMalloc(&d_img_seq, new_img_seq_size); img_seq_size = new_img_seq_size; }
    if (!d_HR) {
        cudaMalloc(&d_HR, new_HR_size);
        cudaMalloc(&d_Z, new_HR_size);
        cudaMalloc(&d_HR_A, new_HR_size);
        HR_size = new_HR_size;
    }
    if (!d_Tvec) { cudaMalloc(&d_Tvec, new_Tvec_size); Tvec_size = new_Tvec_size; }
    if (!d_Rvec) { cudaMalloc(&d_Rvec, new_Rvec_size); Rvec_size = new_Rvec_size; }
    if (!d_D) { cudaMalloc(&d_D, new_D_size); D_size = new_D_size; }
    if (!d_kernel) { cudaMalloc(&d_kernel, new_kernel_size); kernel_size = new_kernel_size; }
    if (!stream) { cudaStreamCreate(&stream); }

    checkCudaError(cudaGetLastError(), "initCudaMemory 失败");
}

extern "C" void freeCudaMemory() {
    if (d_img_seq) cudaFree(d_img_seq);
    if (d_HR) cudaFree(d_HR);
    if (d_Z) cudaFree(d_Z);
    if (d_HR_A) cudaFree(d_HR_A);
    if (d_Tvec) cudaFree(d_Tvec);
    if (d_Rvec) cudaFree(d_Rvec);
    if (d_D) cudaFree(d_D);
    if (d_kernel) cudaFree(d_kernel);
    if (stream) cudaStreamDestroy(stream);

    d_img_seq = nullptr;
    d_HR = d_Z = d_HR_A = nullptr;
    d_Tvec = d_Rvec = d_kernel = nullptr;
    d_D = nullptr;
    stream = nullptr;
    img_seq_size = HR_size = Tvec_size = Rvec_size = D_size = kernel_size = 0;
}

// ==================== 核函数 ====================

__global__ void computeDisplacementKernel2(const float* Tvec, const float* Rvec,
    int* D, int num, int resFactor) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num) return;

    float tx = Tvec[i * 2 + 0];
    float ty = Tvec[i * 2 + 1];

    float dx_raw = tx * resFactor;
    float dy_raw = ty * resFactor;

    D[i * 2 + 0] = static_cast<int>(roundf(dx_raw)) % resFactor;
    D[i * 2 + 1] = static_cast<int>(roundf(dy_raw)) % resFactor;

    if (D[i * 2 + 0] < 0) D[i * 2 + 0] += resFactor;
    if (D[i * 2 + 1] < 0) D[i * 2 + 1] += resFactor;
}

__global__ void medianAndShiftKernel2(
    const unsigned char* img_seq,
    const int* D,
    float* HR,
    float* HR_A,
    unsigned char* S,
    int num,
    int lw,
    int lh,
    int resFactor)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= lw * resFactor || y >= lh * resFactor) return;

    // 计算子像素偏移和源图像坐标
    int dx = x % resFactor;
    int dy = y % resFactor;
    int src_x = x / resFactor;
    int src_y = y / resFactor;

    if (src_x >= lw || src_y >= lh) return;

    int idx = y * (lw * resFactor) + x;

    float sum = 0.0f;
    int count = 0;

    // 累加所有匹配当前子像素位置的帧
    for (int i = 0; i < num; i++) {
        if (D[i * 2 + 0] == dx && D[i * 2 + 1] == dy) {
            unsigned char pixel = img_seq[i * lw * lh + src_y * lw + src_x];
            sum += (float)pixel;
            count++;

            // 标记子像素覆盖
            int subpixel_idx = dy * resFactor + dx;
            S[subpixel_idx] = 1;
        }
    }

    // 如果找到匹配的帧，用平均值覆盖底图
    if (count > 0) {
        HR[idx] = sum / count;
        HR_A[idx] = 1.0f;  // 高置信度
    }
    // 如果 count == 0，保留底图的值（HR[idx] 不变）
}

// ==================== 主处理函数 ====================

extern "C" void cudaMedianAndShift(
    const unsigned char* img_seq,
    float* HR,
    float* HR_A,
    const int* D,
    int num,
    int lw,
    int lh,
    int resFactor)
{
    if (!img_seq || !HR || !HR_A || !D || num <= 0 || lw <= 0 || lh <= 0 || resFactor <= 0) {
        std::cerr << "无效的输入参数！" << std::endl;
        return;
    }

    size_t img_seq_bytes = num * lw * lh * sizeof(unsigned char);
    size_t HR_bytes = lw * resFactor * lh * resFactor * sizeof(float);
    size_t S_bytes = resFactor * resFactor * sizeof(unsigned char);

    unsigned char* d_S;
    float* d_HR_temp;
    float* d_HR_A_temp;

    cudaMalloc(&d_S, S_bytes);
    cudaMalloc(&d_HR_temp, HR_bytes);
    cudaMalloc(&d_HR_A_temp, HR_bytes);

    cudaMemsetAsync(d_S, 0, S_bytes, stream);

    // 复制输入数据到设备（包括底图）
    cudaMemcpyAsync(d_img_seq, img_seq, img_seq_bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_D, D, num * 2 * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_HR, HR, HR_bytes, cudaMemcpyHostToDevice, stream);  // 复制底图
    cudaMemsetAsync(d_HR_A, 0, HR_bytes, stream);

    // 启动核函数
    dim3 blockDim(16, 16);
    dim3 gridDim((lw * resFactor + blockDim.x - 1) / blockDim.x,
        (lh * resFactor + blockDim.y - 1) / blockDim.y);

    // 第一步：直接填充匹配的像素
    medianAndShiftKernel2 << <gridDim, blockDim, 0, stream >> > (
        d_img_seq, d_D, d_HR, d_HR_A, d_S, num, lw, lh, resFactor);
    checkCudaError(cudaGetLastError(), "medianAndShiftKernel2 失败");

    // 检查子像素覆盖情况
    std::vector<unsigned char> S(resFactor * resFactor);
    cudaMemcpyAsync(S.data(), d_S, S_bytes, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    std::cout << "\n========== 子像素覆盖矩阵 (" << resFactor << "x" << resFactor << ") ==========" << std::endl;
    int covered_count = 0;
    for (int i = 0; i < resFactor * resFactor; i++) {
        if (S[i] == 1) covered_count++;
    }

    float coverage_rate = (float)covered_count / (resFactor * resFactor) * 100.0f;
    std::cout << "子像素覆盖率: " << coverage_rate << "% ("
        << covered_count << "/" << resFactor * resFactor << ")" << std::endl;

    // 复制结果回主机
    cudaMemcpyAsync(HR, d_HR, HR_bytes, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(HR_A, d_HR_A, HR_bytes, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    cudaFree(d_S);
    cudaFree(d_HR_temp);
    cudaFree(d_HR_A_temp);
}

extern "C" void cudaFastRobustSR(
    const unsigned char* img_seq,
    float* HR,
    float* Z,
    float* HR_A,
    const float* Tvec,
    const float* Rvec,
    int num,
    int width,
    int height,
    int resFactor,
    const float* kernel,
    int kernel_size)
{
    if (!img_seq || !HR || !Z || !HR_A || !Tvec || !Rvec || num <= 0 ||
        width <= 0 || height <= 0 || resFactor <= 0) {
        std::cerr << "无效的输入参数！" << std::endl;
        return;
    }

    // 确保 GPU 内存已初始化
    if (!d_img_seq || !d_Tvec || !d_Rvec || !d_D || !stream) {
        std::cerr << "GPU 内存未初始化，正在初始化..." << std::endl;
        initCudaMemory(num, width, height, resFactor, kernel_size);
    }

    size_t img_seq_bytes = num * width * height * sizeof(unsigned char);
    size_t HR_bytes = width * resFactor * height * resFactor * sizeof(float);

    // 复制输入数据到设备
    cudaMemcpyAsync(d_img_seq, img_seq, img_seq_bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_Tvec, Tvec, num * 2 * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_Rvec, Rvec, num * sizeof(float), cudaMemcpyHostToDevice, stream);

    // 计算子像素位移矩阵
    dim3 blockDim(256);
    dim3 gridDim((num + blockDim.x - 1) / blockDim.x);
    computeDisplacementKernel2 << <gridDim, blockDim, 0, stream >> > (
        d_Tvec, d_Rvec, d_D, num, resFactor);
    checkCudaError(cudaGetLastError(), "computeDisplacementKernel2 失败");

    // 打印位移矩阵（调试用）
    std::vector<int> D_host(num * 2);
    cudaMemcpyAsync(D_host.data(), d_D, num * 2 * sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    std::cout << "\n========== 子像素位移矩阵 D ==========" << std::endl;
    for (int i = 0; i < num; i++) {
        std::cout << "帧 " << i << ": (" << D_host[i * 2 + 0] << ", "
            << D_host[i * 2 + 1] << ")" << std::endl;
    }

    // 执行超分辨率重建，HR包含底图
    cudaMedianAndShift(img_seq, HR, HR_A, D_host.data(), num, width, height, resFactor);

    // Z 复制 HR 的结果
    cudaMemcpyAsync(Z, HR, HR_bytes, cudaMemcpyHostToHost, stream);
    cudaStreamSynchronize(stream);
}