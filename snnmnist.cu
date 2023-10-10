#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <stdlib.h>
#include <stdio.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <curand.h>

#include <cuda_runtime.h>

#include "readubyte.h"
#include "nn.h"
#include "cusnn.h"
#include "cuutils.h"

snn::Linear * get_linear(int in_features, int out_features, optim::Adamw * optimizer, cublasHandle_t * handle, int batch_size, float * onevec)
{
    snn::Linear * linear = new snn::Linear(in_features, out_features);

    linear->set_configure(handle);
    linear->set_configure(batch_size, onevec);
    linear->allocate_gradspace();

    linear->reset_parameters(2023);

    optimizer->add_parameter(linear->get_bias(), linear->get_dbias(), linear->get_bias_size());
    optimizer->add_parameter(linear->get_weight(), linear->get_dweight(), linear->get_weight_size());

    return linear;
}

snn::Conv2d * get_conv1(int in_channels, int out_channels, optim::Adamw * optimizer, cudnnHandle_t * handle, cudnnTensorDescriptor_t * inputTensDesc, int * conv_cfg)
{
    snn::Conv2d * conv = new snn::Conv2d(handle, inputTensDesc, in_channels, out_channels, 
    conv_cfg[0], conv_cfg[1], conv_cfg[2], conv_cfg[3], conv_cfg[4], conv_cfg[5]);

    conv->reset_parameters(2023);

    optimizer->add_parameter(conv->get_bias(), conv->get_dbias(), conv->get_bias_size());
    optimizer->add_parameter(conv->get_weight(), conv->get_dweight(), conv->get_weight_size());

    return conv;
}

snn::LIF * get_neuron(int timesteps, int n_neurons)
{
    snn::LIF * neuron = new snn::LIF();
    neuron->set_configure(timesteps, n_neurons);
    return neuron;
}

snn::BatchNorm * get_batchnorm(int * bn_cfg, cudnnHandle_t * handle, optim::Adamw * optimizer)
{
    snn::BatchNorm * bn = new snn::BatchNorm(bn_cfg[0], bn_cfg[1], bn_cfg[2], bn_cfg[3]);

    bn->set_configure(handle);
    optimizer->add_parameter(bn->get_bias(), bn->get_d_bias(), bn->get_bias_size());
    optimizer->add_parameter(bn->get_weight(), bn->get_d_weight(), bn->get_weight_size());

    return bn;
}


int main() 
{
    std::vector<float> tr_images(60000 * 784), ts_images(10000 * 784);
    std::vector<float> tr_labels(60000 * 10), ts_labels(10000 * 10);
    {   
        size_t width, height;
        std::vector<uint8_t> images_buffer(60000 * 784), labels_buffer(60000);
        ReadUByteDataset(
            "mnist/train-images.idx3-ubyte", 
            "mnist/train-labels.idx1-ubyte", 
            &images_buffer[0], 
            &labels_buffer[0], 
            width, 
            height
        );
        for (size_t i = 0; i < 60000 * 784; ++i) 
            tr_images[i] = (float)images_buffer[i] / 255.0f;
        for (size_t i = 0; i < 60000 * 10; ++i)
            tr_labels[i] = 0.0f;
        for (size_t i = 0; i < 60000; ++i)
            tr_labels[labels_buffer[i] + i * 10] = 1.0f;
        
        ReadUByteDataset(
            "mnist/t10k-images.idx3-ubyte", 
            "mnist/t10k-labels.idx1-ubyte", 
            &images_buffer[0], 
            &labels_buffer[0], 
            width, 
            height
        );
        for (size_t i = 0; i < 10000 * 784; ++i) 
            ts_images[i] = (float)images_buffer[i] / 255.0f;
        for (size_t i = 0; i < 10000 * 10; ++i)
            ts_labels[i] = 0.0f;
        for (size_t i = 0; i < 10000; ++i)
            ts_labels[labels_buffer[i] + i * 10] = 1.0f;
    }

    float lr = 1e-3;
    int batch_size = 32;
    int timesteps = 8;
    size_t n_workspace = 0;
    float * workspace;
    std::vector<network::Layer *> net;
    optim::Adamw optimizer(lr, 0);

    cudnnHandle_t cudnnHandle;
    cudnnSuccessfullyReturn(cudnnCreate(&cudnnHandle));
    cudnnTensorDescriptor_t inputTensDesc;
    cudnnSuccessfullyReturn(cudnnCreateTensorDescriptor(&inputTensDesc));
    cudnnSuccessfullyReturn(cudnnSetTensor4dDescriptor(
        inputTensDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size * timesteps, 1, 28, 28
    ));
    {
        int cfg[6] = {3, 3, 1, 1, 1, 1};
        int bn_cfg[4] = {batch_size * timesteps, 32, 28, 28};
        snn::Conv2d * c1 = get_conv1(1, 32, &optimizer, &cudnnHandle, &inputTensDesc, cfg);
        cfg[2] = cfg[3] = 2;
        snn::Conv2d * c2 = get_conv1(32, 32, &optimizer, &cudnnHandle, c1->get_outdesc(), cfg);

        n_workspace = std::max(n_workspace, c1->get_n_workspace());
        n_workspace = std::max(n_workspace, c2->get_n_workspace());
        cudaSuccessfullyReturn(cudaMalloc(&workspace, sizeof(float) * n_workspace));
        c1->set_configure(workspace, n_workspace);
        c2->set_configure(workspace, n_workspace);

        net.push_back(new network::Layer(c1));
        net.push_back(new network::Layer(get_batchnorm(bn_cfg, &cudnnHandle, &optimizer)));
        net.push_back(new network::Layer(get_neuron(timesteps, batch_size * 32 * 784)));
        net.push_back(new network::Layer(c2));
        bn_cfg[2] = bn_cfg[3] = 14;
        net.push_back(new network::Layer(get_batchnorm(bn_cfg, &cudnnHandle, &optimizer)));
        net.push_back(new network::Layer(get_neuron(timesteps, batch_size * 32 * 14 * 14)));
    }

    cublasHandle_t cublasHandle;
    cublasSuccessfullyReturn(cublasCreate(&cublasHandle));
    float * onevec;
    cudaSuccessfullyReturn(cudaMalloc(&onevec, sizeof(float) * batch_size * timesteps));
    cusnnFillConstant<<<CUDA_1D_BLOCK_NUMS(batch_size), CUDA_1D_BLOCK_SIZE>>>(onevec, batch_size, 1.0f);
    net.push_back(new network::Layer(get_linear(32 * 14 * 14, 800, &optimizer, &cublasHandle, batch_size * timesteps, onevec)));
    net.push_back(new network::Layer(get_neuron(timesteps, batch_size * 800)));
    net.push_back(new network::Layer(get_linear(800, 10, &optimizer, &cublasHandle, batch_size * timesteps, onevec)));

    
    float * input, * labels, * result;
    cudaSuccessfullyReturn(cudaMalloc(&input,  sizeof(float) * timesteps * batch_size * 784));
    cudaSuccessfullyReturn(cudaMalloc(&labels, sizeof(float) * timesteps * batch_size * 10));
    cudaSuccessfullyReturn(cudaMalloc(&result, sizeof(float) * batch_size));

    float loss = 0.0f, tloss = 0.0f, acc = 0.0f, tacc = 0.0f;
    
    for (int e = 0; e < 1000; ++e) 
    {   
        acc = tacc = loss = tloss = 0.0f;

        for (int i = 0; i < (int)(60000 / batch_size); ++i) 
        {   
            for (int t = 0; t < timesteps; ++t)
            {
                cudaSuccessfullyReturn(cudaMemcpyAsync(
                    input + t * batch_size * 784, &tr_images[i * batch_size * 784], sizeof(float) * batch_size * 784, cudaMemcpyHostToDevice
                ));
                cudaSuccessfullyReturn(cudaMemcpyAsync(
                    labels + t * batch_size * 10, &tr_labels[i * batch_size * 10], sizeof(float) * batch_size * 10, cudaMemcpyHostToDevice
                ));
            }

            net[0]->forward(input);
            for (int j = 1; j < net.size(); ++j)
            {
                net[j]->forward(net[j - 1]->get_out());
            }

            cusnnSquareError<<<CUDA_1D_BLOCK_NUMS(timesteps * batch_size * 10), CUDA_1D_BLOCK_SIZE>>>(
                labels, net.back()->get_out(), net.back()->get_dout(), timesteps * batch_size * 10
            );
            cublasSuccessfullyReturn(cublasSasum(cublasHandle, timesteps * batch_size * 10, net.back()->get_dout(), 1, &tloss));
            
            tloss /= timesteps * batch_size * 10;
            loss += tloss;

            cusnnMSEGradient<<<CUDA_1D_BLOCK_NUMS(timesteps * batch_size * 10), CUDA_1D_BLOCK_SIZE>>>(
                labels, net.back()->get_out(), net.back()->get_dout(), timesteps * batch_size * 10
            );
            
            for (int j = net.size() - 1; j > 0; --j)
            {
                net[j]->backward(net[j - 1]->get_out(), net[j -1]->get_dout());
            }
            net[0]->backward(input, nullptr);

            optimizer.step();

            cusnnReadOut<<<CUDA_1D_BLOCK_NUMS(batch_size), CUDA_1D_BLOCK_SIZE>>>(
                labels, net.back()->get_out(), result, timesteps, batch_size, 10
            );
            cublasSuccessfullyReturn(cublasSasum(cublasHandle, batch_size, result, 1, &tacc));
            tacc /= (float)batch_size;
            acc += tacc;

            if ((i + 1) % 100 == 0)
            {
                printf("tacc: %f acc: %f, tloss: %f, loss: %f\n", tacc, acc / (i + 1), tloss, loss / (i + 1));
            }

        }
    }
}