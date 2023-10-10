#include "cusnn.h"
#include "cuutils.h"

#include <cuda_runtime.h>

__global__ void cusnnLIFneuronForward(
    const float * x,         // 神经元输入电流，维度为TxNxD，分别指时间步数、batch size、特征维度
    float       * outu,      // 神经元每个时刻的膜电位，用于反向传播时计算代理梯度值
    float       * outs,      // 输出的脉冲序列，维度与输入维度相同
    const int     timesteps, // 输入数据的时间维度长度
    const int     n,         // 神经元数量，即NxD
    const float   tau,       // 膜电位衰减系数
    const float   threshold, // 脉冲发放阈值，某一时刻膜电位高于阈值则输出1，否则输出0
    const float   reset) 
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    float u = 0.0f;
    int posi = idx;
    if (idx < n) {
        for (int i = 0; i < timesteps; ++i, posi += n) {
            u = u / tau + x[posi];
            outu[posi] = u;
            if (u > threshold) {
                u = reset;
                outs[posi] = 1.0f;
            }
            else {
                outs[posi] = 0.0f;
            }
        }
    }
}

__global__ void cusnnLIFneuronBackward(
    const float * dout,      // 输出脉冲的导数，维度为TxNxD
    const float * outu,      // 神经元的膜电位值
    float       * dx,        // 反向传播计算得到的输入电流的导数，维度同dout
    const int     timesteps, // 输入数据的时间维度长度
    const int     n,         // 神经元数量，即NxD
    const float   tau,       // 膜电位衰减系数
    const float   threshold, // 脉冲发放阈值，某一时刻膜电位高于阈值则输出1，否则输出0
    const float   reset)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    float sg = 0.0f, u = 0.0f, du = 0.0f, s = 0.0f;
    int posi = (timesteps - 1) * n + idx;
    if (idx < n) {
        for (int i = timesteps - 1; i >= 0; --i, posi -= n) {
            u = outu[posi];
            s = (float)(u > threshold);
            sg = ArcTan(u - threshold, 4.0);
            du = dout[posi] * sg + du / tau * (1 - s + (reset - u) * sg);
            dx[posi] = du;
        }
    }
}

__global__ void cusnnAdamwGradientDescent(
    float       * params,    // 参数数组，长度为n
    const float * grad,      // 参数梯度数组
    float       * m,         // 一阶动量
    float       * v,         // 二阶动量
    const int     n,         // 数组长度
    const float   lr,        // 学习率
    const float   weight_decay,//l2正则
    const float   beta1,     // 一阶动量系数
    const float   beta2,     // 二阶动量系数
    const float   eps        // 加在分母避免除0
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    float deltaW = 0.0f, m1 = 0.0f, v1 = 0.0f;

    if (idx < n) {
        deltaW = grad[idx];

        m1 = beta1 * m[idx] + (1 - beta1) * deltaW;
        m[idx] = m1;

        v1 = beta2 * v[idx] + (1 - beta2) * powf(deltaW, 2);
        v[idx] = v1;

        deltaW = m1 / (1 - beta1) / (eps + sqrtf(v1 / (1 - beta2)));
        if (weight_decay != 0.0f) {
            deltaW += weight_decay * params[idx];
        }
        params[idx] -= lr * deltaW; 

    }
}

__global__ void cusnnSGDGradientDescent(
    float       * param, 
    const float * grad, 
    const int     n,
    const float   lr, 
    const float   wd)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    float delta_w;
    if (idx < n)
    {
        delta_w = grad[idx] + wd * param[idx];
        param[idx] -= delta_w * lr;
    }
}

__global__ void cusnnSquareError(
    const float * target,
    const float * output,
    float       * result,
    const int     n
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        result[idx] = powf(output[idx] - target[idx], 2);
    }
}

__global__ void cusnnMSEGradient(
    const float * target,
    const float * output,
    float       * result,
    const int     n
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        result[idx] = (output[idx] - target[idx]) * 2 / n;
    }
}

__global__ void cusnnFillConstant(float * array, const int n, float c) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) array[idx] = c;
}

__global__ void cusnnReadOut(
    const float * target,    // 样本标签，维度为TxNxC，分别是时间步数、batch size、类别数
    const float * output,    // 模型输出，维度为TxNxC
    float       * result,    // 模型预测结果，维度为N，表示每个样本预测结果，正确为1，错误为0
    const int     timesteps, // 时间步长
    const int     n_batch,   // batch size
    const int     n_class    // 类别数
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int cTarget = 0, cOutput = 0, n = n_class * n_batch;
    float sumOutput = 0.0f, maxOutput = 0.0f;

    if (idx < n_batch) {
        for (int i = 0; i < n_class; ++i) {
            sumOutput = 0.0f;
            for (int j = 0; j < timesteps; ++j) {
                sumOutput += output[j * n + idx * n_class + i];
            }
            if (sumOutput > maxOutput) {
                maxOutput = sumOutput;
                cOutput = i;
            }
            if (target[idx * n_class + i] == 1) {
                cTarget = i;
            }
        }
        result[idx] = cTarget == cOutput ? 1.0f : 0.0f;
    }
}

// x = x * (b - a) + a
__global__ void cusnnRescaleTensor(float * x, const int n, const float a, const float b) 
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) x[idx] = x[idx] * (b - a) + a;
}

// 
__global__ void cusnnReducedSum(const float * x, int n, float * result) 
{
    int stride = (int)((n + 1) / 2);
    extern __shared__ float sharedX[];

    if (threadIdx.x < stride) {
        sharedX[threadIdx.x] = x[threadIdx.x];
    }
    if ((threadIdx.x + stride) < n) {
        sharedX[threadIdx.x + stride] = x[threadIdx.x + stride];
    }
    __syncthreads();

    while (n > 1) {
        if ((threadIdx.x + stride) < n) {
            sharedX[threadIdx.x] += sharedX[threadIdx.x + stride];
        }
        n = stride;
        stride = (int)((stride + 1) / 2);
        __syncthreads();
    }
    if (threadIdx.x == 0) *result = sharedX[0];
}

__global__ void cusnnReLUforward(const float * x, float * y, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    float v;
    if (idx < n) 
    {
        v = x[idx];
        y[idx] = v > 0 ? v : 0.0f;
    }
}

__global__ void cusnnReLUbackward(const float * x, const float * dy, int n , float * dx)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n)
        dx[idx] = x[idx] >= 0 ? dy[idx] : 0.0f;
}