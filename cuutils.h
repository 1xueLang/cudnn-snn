#ifndef CUUTILS_H
#define CUUTILS_H

#include <iostream>
#include <sstream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include <curand.h>
#include <stdlib.h>

#define FatalError(s) do {                                                    \
    std::stringstream _where, _message;                                       \
    _where << __FILE__ << ':' << __LINE__;                                    \
    _message << std::string(s) + "\n" << _where.str();                        \
    std::cerr << _message.str() << "\nAborting...\n";                         \
    cudaDeviceReset();                                                        \
    exit(1);                                                                  \
} while(0)

#define cudnnSuccessfullyReturn(status) do {                                  \
    std::stringstream _error;                                                 \
    if (status != CUDNN_STATUS_SUCCESS) {                                     \
        _error << "CUDNN failure: " << cudnnGetErrorString(status);           \
        FatalError(_error.str());                                             \
    }                                                                         \
} while(0)

#define cudaSuccessfullyReturn(status) do {                                   \
    std::stringstream _error;                                                 \
    if (status != cudaSuccess) {                                              \
        _error << "Cuda failure: " << status;                                 \
        FatalError(_error.str());                                             \
    }                                                                         \
} while(0)

#define cublasSuccessfullyReturn(status) do {                                 \
    std::stringstream _error;                                                 \
    if (status != CUBLAS_STATUS_SUCCESS) {                                    \
        _error << "cublas failure: " << "cublasGetStatusString(status)";        \
        FatalError(_error.str());                                             \
    }                                                                         \
} while(0)

#define curandSuccessfullyReturn(status) do {                                 \
    std::stringstream _error;                                                 \
    if (status != CURAND_STATUS_SUCCESS) {                                    \
        _error << "curand failure: " << status;                               \
        FatalError(_error.str());                                             \
    }                                                                         \
} while(0)

#define CUDA_1D_BLOCK_SIZE 256
#define CUDA_1D_BLOCK_NUMS(n)                                                 \
    (int)((n + CUDA_1D_BLOCK_SIZE - 1) / CUDA_1D_BLOCK_SIZE)

#ifndef PI
#define PI 3.14159265358979323846
#endif

#define ArcTan(x, alpha) (alpha) / 2 / (1 + powf(PI / 2 * (alpha) * (x), 2))

#endif