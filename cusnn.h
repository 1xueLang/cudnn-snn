#ifndef CUSNN_H
#define CUSNN_H

#include <cuda_runtime.h>

__global__ void cusnnLIFneuronForward(
    const float * x,         // 神经元输入电流，维度为TxNxD，分别指时间步数、batch size、特征维度
    float       * outu,      // 神经元每个时刻的膜电位，用于反向传播时计算代理梯度值
    float       * outs,      // 输出的脉冲序列，维度与输入维度相同
    const int     timesteps, // 输入数据的时间维度长度
    const int     n,         // 神经元数量，即NxD
    const float   tau,       // 膜电位衰减系数
    const float   threshold, // 脉冲发放阈值，某一时刻膜电位高于阈值则输出1，否则输出0
    const float   reset      // 发放脉冲后膜电位重置值
);

__global__ void cusnnLIFneuronBackward(
    const float * dout,      // 输出脉冲的导数，维度为TxNxD
    const float * outu,      // 神经元的膜电位值
    float       * dx,        // 反向传播计算得到的输入电流的导数，维度同dout
    const int     timesteps, // 输入数据的时间维度长度
    const int     n,         // 神经元数量，即NxD
    const float   tau,       // 膜电位衰减系数
    const float   threshold, // 脉冲发放阈值，某一时刻膜电位高于阈值则输出1，否则输出0
    const float   reset      // 发放脉冲后膜电位重置值
);

__global__ void cusnnAdamwGradientDescent(
    float       * param,     // 参数数组，长度为n
    const float * grad,      // 参数梯度数组
    float       * m,         // 一阶动量
    float       * v,         // 二阶动量
    const int     n,         // 数组长度
    const float   lr,        // 学习率
    const float   weight_decay,//l2正则
    const float   beta1,     // 一阶动量系数
    const float   beta2,     // 二阶动量系数
    const float   eps        // 加在分母避免除0
);

__global__ void cusnnSGDGradientDescent(
    float       * param, 
    const float * grad, 
    const int     n,
    const float   lr, 
    const float   wd
);

__global__ void cusnnSquareError(
    const float * target,
    const float * output,
    float       * result,
    const int     n
);

__global__ void cusnnMSEGradient(
    const float * target,
    const float * output,
    float       * result,
    const int     n
);

__global__ void cusnnFillConstant(float * array, const int n, float c);

__global__ void cusnnReadOut(
    const float * target,    // 样本标签，维度为TxNxC，分别是时间步数、batch size、类别数
    const float * output,    // 模型输出，维度为TxNxC
    float       * result,    // 模型预测结果，维度为N，表示每个样本预测结果，正确为1，错误为0
    const int     timesteps, // 时间步长
    const int     n_batch,   // batch size
    const int     n_class    // 类别数
);

// x = x * (b - a) + a
__global__ void cusnnRescaleTensor(float * x, const int n, const float a, const float b);

// 
__global__ void cusnnReducedSum(const float * x, const int n, float * result);
//
__global__ void cusnnReLUforward(const float * x, float * y, int n);
//
__global__ void cusnnReLUbackward(const float * x, const float * dy, int n , float * dx);


#endif