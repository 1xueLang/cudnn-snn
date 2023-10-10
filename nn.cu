#include "nn.h"
#include "cusnn.h"
#include "cuutils.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>

network::Layer::Layer(snn::BatchNorm * bn)
{
    this->bn_layer = bn;
    this->type = network::LayerType::BN;
}
network::Layer::Layer(snn::Conv2d * cv)
{
    this->conv_layer = cv;
    this->type = network::LayerType::CV;
}
network::Layer::Layer(snn::LIF * sn)
{
    this->neuron_layer = sn;
    this->type = network::LayerType::SN;
}
network::Layer::Layer(snn::Linear * fc)
{
    this->fc_layer = fc;
    this->type = network::LayerType::FC;
}
network::Layer::Layer(ann::ReLU * ac)
{
    this->act_layer = ac;
    this->type = network::LayerType::AC;
}

float * network::Layer::forward(const float * input)
{
    if (this->type == network::LayerType::AC)
    {
        return this->act_layer->forward(input);
    }
    else if (this->type == network::LayerType::BN)
    {
        return this->bn_layer->forward(input);
    }
    else if (this->type == network::LayerType::CV)
    {
        return this->conv_layer->forward(input);
    }
    else if (this->type == network::LayerType::FC)
    {
        return this->fc_layer->forward(input);
    }
    else if (this->type == network::LayerType::SN)
    {
        return this->neuron_layer->forward(input);
    }
    else 
    {
        return nullptr;
    }
}
float * network::Layer::backward(const float * input, float * dinput)
{
    if (this->type == network::LayerType::AC)
    {
        return this->act_layer->backward(input, dinput);
    }
    else if (this->type == network::LayerType::BN)
    {
        return this->bn_layer->backward(input, dinput);
    }
    else if (this->type == network::LayerType::CV)
    {
        return this->conv_layer->backward(input, dinput);
    }
    else if (this->type == network::LayerType::FC)
    {
        return this->fc_layer->backward(input, dinput);
    }
    else if (this->type == network::LayerType::SN)
    {
        return this->neuron_layer->backward(input, dinput);
    }
    else 
    {
        return nullptr;
    }
}
float * network::Layer::get_out()
{
    if (this->type == network::LayerType::AC)
    {
        return this->act_layer->get_out();
    }
    else if (this->type == network::LayerType::BN)
    {
        return this->bn_layer->get_out();
    }
    else if (this->type == network::LayerType::CV)
    {
        return this->conv_layer->get_out();
    }
    else if (this->type == network::LayerType::FC)
    {
        return this->fc_layer->get_out();
    }
    else if (this->type == network::LayerType::SN)
    {
        return this->neuron_layer->get_out();
    }
    else 
    {
        return nullptr;
    }
}
float * network::Layer::get_dout()
{
    if (this->type == network::LayerType::AC)
    {
        return this->act_layer->get_dout();
    }
    else if (this->type == network::LayerType::BN)
    {
        return this->bn_layer->get_dout();
    }
    else if (this->type == network::LayerType::CV)
    {
        return this->conv_layer->get_dout();
    }
    else if (this->type == network::LayerType::FC)
    {
        return this->fc_layer->get_dout();
    }
    else if (this->type == network::LayerType::SN)
    {
        return this->neuron_layer->get_dout();
    }
    else 
    {
        return nullptr;
    }
}

snn::Linear::Linear(int in_features, int out_features, bool bias, bool train)
{
    this->in_features  = in_features;
    this->out_features = out_features;
    this->use_bias     = bias;
    this->training     = train;
    this->n_weight     = in_features * out_features;
    this->n_out        = 0;

    cudaSuccessfullyReturn(cudaMalloc(&(this->weight), sizeof(float) * this->n_weight));
    if (this->use_bias)
    cudaSuccessfullyReturn(cudaMalloc(&(this->bias),   sizeof(float) * this->out_features));
}

void snn::Linear::set_configure(cublasHandle_t * handle)
{
    this->handle = handle;
}
void snn::Linear::set_configure(int batch_size, float * onevec)
{
    this->batch_size = batch_size;
    this->onevec     = onevec;
    this->n_out      = batch_size * this->out_features;
    cudaSuccessfullyReturn(cudaMalloc(&(this->out), sizeof(float) * this->n_out));
}
void snn::Linear::set_configure(bool training)
{
    this->training = training;
}

void snn::Linear::allocate_gradspace()
{   
    cudaSuccessfullyReturn(cudaMalloc(&(this->d_weight), sizeof(float) * this->n_weight));
    cudaSuccessfullyReturn(cudaMalloc(&(this->dout),     sizeof(float) * this->n_out));
    if (this->use_bias)
    cudaSuccessfullyReturn(cudaMalloc(&(this->d_bias),   sizeof(float) * this->out_features));
}
void snn::Linear::free_gradspace()
{
    cudaSuccessfullyReturn(cudaFree(this->d_weight));
    cudaSuccessfullyReturn(cudaFree(this->dout));
    if (this->use_bias)
    cudaSuccessfullyReturn(cudaFree(this->d_bias));
    this->training = false;
}

float * snn::Linear::get_out()
{
    return this->out;
}
float * snn::Linear::get_dout()
{
    return this->dout;
}
float * snn::Linear::get_weight()
{
    return this->weight;
}
float * snn::Linear::get_bias()
{
    return this->bias;
}
float * snn::Linear::get_dweight()
{
    return this->d_weight;
}
float * snn::Linear::get_dbias()
{
    return this->d_bias;
}

size_t snn::Linear::get_weight_size()
{
    return this->n_weight;
}
size_t snn::Linear::get_bias_size()
{
    return this->use_bias ? this->out_features : 0;
}
size_t snn::Linear::get_out_size()
{
    return this->n_out;
}

float * snn::Linear::forward(const float * input)
{
    float alpha = 1.0f, beta = 0.0f;

    cublasSuccessfullyReturn(cublasSgemm(
        *(this->handle),
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        this->out_features,
        this->batch_size,
        this->in_features,
        &alpha,
        this->weight,
        this->out_features,
        input,
        this->in_features,
        &beta,
        this->out,
        this->out_features
    ));

    if (this->use_bias == false) return this->out;

    cublasSuccessfullyReturn(cublasSger(
        *(this->handle),
        this->out_features,
        this->batch_size,
        &alpha,
        this->bias,
        1,
        this->onevec,
        1,
        this->out,
        this->out_features
    ));

    return this->out;
}
float * snn::Linear::backward(const float * input, float * dinput)
{
    float alpha = 1.0f, beta = 0.0f;
    // d_weight
    cublasSuccessfullyReturn(cublasSgemm(
        *(this->handle),
        CUBLAS_OP_N,
        CUBLAS_OP_T,
        this->out_features,
        this->in_features,
        this->batch_size,
        &alpha,
        this->dout,
        this->out_features,
        input,
        this->in_features,
        &beta,
        this->d_weight,
        this->out_features
    ));
    // d_input
    if (dinput != nullptr)
    cublasSuccessfullyReturn(cublasSgemm(
        *(this->handle),
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        this->in_features,
        this->batch_size,
        this->out_features,
        &alpha,
        this->weight,
        this->out_features,
        this->dout,
        this->out_features,
        &beta,
        dinput,
        this->in_features
    ));
    // d_bias
    if (this->use_bias == false) return dinput;

    cublasSuccessfullyReturn(cublasSgemv(
        *(this->handle),
        CUBLAS_OP_N,
        this->out_features,
        this->batch_size,
        &alpha,
        this->dout,
        this->out_features,
        this->onevec,
        1,
        &beta,
        this->d_bias,
        1
    ));

    return dinput;
}

float * snn::Linear::backward(const float * d_out, const float * input, float * dinput)
{
    float alpha = 1.0f, beta = 0.0f;
    // d_weight
    cublasSuccessfullyReturn(cublasSgemm(
        *(this->handle),
        CUBLAS_OP_N,
        CUBLAS_OP_T,
        this->out_features,
        this->in_features,
        this->batch_size,
        &alpha,
        d_out,
        this->out_features,
        input,
        this->in_features,
        &beta,
        this->d_weight,
        this->out_features
    ));
    // d_input
    if (dinput != nullptr)
    cublasSuccessfullyReturn(cublasSgemm(
        *(this->handle),
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        this->in_features,
        this->batch_size,
        this->out_features,
        &alpha,
        this->weight,
        this->out_features,
        d_out,
        this->out_features,
        &beta,
        dinput,
        this->in_features
    ));
    // d_bias
    if (this->use_bias == false) return dinput;

    cublasSuccessfullyReturn(cublasSgemv(
        *(this->handle),
        CUBLAS_OP_N,
        this->out_features,
        this->batch_size,
        &alpha,
        d_out,
        this->out_features,
        this->onevec,
        1,
        &beta,
        this->d_bias,
        1
    ));

    return dinput;
}



void snn::Linear::reset_parameters(int seed)
{   
    float gain = 1.0f / sqrt(this->in_features);
    curandGenerator_t generator;
    curandSuccessfullyReturn(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_MT19937));
    curandSuccessfullyReturn(curandSetPseudoRandomGeneratorSeed(generator, seed));

    curandSuccessfullyReturn(curandGenerateUniform(generator, this->weight, this->n_weight));
    cusnnRescaleTensor<<<CUDA_1D_BLOCK_NUMS(this->n_weight), CUDA_1D_BLOCK_SIZE>>>(
        this->weight, this->n_weight, -gain, gain
    );

    if (this->use_bias)
    {
        curandSuccessfullyReturn(curandGenerateUniform(
            generator, this->bias, this->out_features
        ));
        cusnnRescaleTensor<<<CUDA_1D_BLOCK_NUMS(this->out_features), CUDA_1D_BLOCK_SIZE>>>(
            this->bias, this->out_features, -gain, gain
        );
    }
    curandSuccessfullyReturn(curandDestroyGenerator(generator));
}

snn::Linear::~Linear()
{
    cudaSuccessfullyReturn(cudaFree(this->weight));
    cudaSuccessfullyReturn(cudaFree(this->out));
    if (this->use_bias)
    cudaSuccessfullyReturn(cudaFree(this->bias));
    if (this->training) this->free_gradspace();
}

snn::Conv2d::Conv2d(
    cudnnHandle_t * cudnnHandle, cudnnTensorDescriptor_t * inputTensDesc, 
    int in_channels, int out_channels, int kernel1, int kernel2, 
    int stride1, int stride2, int padding1, int padding2, bool train)
{   
    this->handle       = cudnnHandle;
    this->in_channels  = in_channels;
    this->out_channels = out_channels;
    this->kernel1      = kernel1;
    this->kernel2      = kernel2;
    this->stride1      = stride1;
    this->stride2      = stride2;
    this->padding1     = padding1;
    this->padding2     = padding2;
    this->training     = train;
    this->n_weight     = kernel1 * kernel2 * in_channels * out_channels;
    this->n_out        = 0;
    this->inputTensDesc = inputTensDesc;

    int n, c, h, w;

    cudnnSuccessfullyReturn(cudnnCreateConvolutionDescriptor(&(this->convDesc)));
    cudnnSuccessfullyReturn(cudnnSetConvolution2dDescriptor(
        this->convDesc, padding1, padding2, stride1, stride2, 1, 1,
        CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT
    ));

    cudnnSuccessfullyReturn(cudnnCreateFilterDescriptor(&(this->convFilterDesc)));
    cudnnSuccessfullyReturn(cudnnSetFilter4dDescriptor(
        this->convFilterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 
        out_channels, in_channels, kernel1, kernel2
    ));
    cudaSuccessfullyReturn(cudaMalloc(&(this->convFilter), sizeof(float) * this->n_weight));

    cudnnSuccessfullyReturn(cudnnCreateTensorDescriptor(&(this->convBiasTensDesc)));
    cudnnSuccessfullyReturn(cudnnSetTensor4dDescriptor(
        this->convBiasTensDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, out_channels, 1, 1
    ));
    cudaSuccessfullyReturn(cudaMalloc(&(this->convBias), sizeof(float) * out_channels));

    cudnnSuccessfullyReturn(cudnnGetConvolution2dForwardOutputDim(
        this->convDesc, *inputTensDesc, this->convFilterDesc, &n, &c, &h, &w
    ));
    cudnnSuccessfullyReturn(cudnnCreateTensorDescriptor(&(this->outTensDesc)));
    cudnnSuccessfullyReturn(cudnnSetTensor4dDescriptor(
        this->outTensDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w
    ));
    this->n_out = n * c * h * w;
    cudaSuccessfullyReturn(cudaMalloc(&(this->out), sizeof(float) * this->n_out));

    if (this->training)
    {
        cudaSuccessfullyReturn(cudaMalloc(&(this->dconvFilter), sizeof(float) * this->n_weight));
        cudaSuccessfullyReturn(cudaMalloc(&(this->dconvBias), sizeof(float) * this->out_channels));
        cudaSuccessfullyReturn(cudaMalloc(&(this->dout), sizeof(float) * this->n_out));
    }
    else
    {
        this->dconvFilter = this->dconvBias = this->dout = nullptr;
    }
    cudnnSuccessfullyReturn(cudnnGetConvolutionForwardAlgorithm(
        *cudnnHandle, *inputTensDesc, this->convFilterDesc, this->convDesc, this->outTensDesc, 
        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &(this->convFwdAlgo)
    ));
    cudnnSuccessfullyReturn(cudnnGetConvolutionBackwardDataAlgorithm(
        *cudnnHandle, this->convFilterDesc, this->outTensDesc, this->convDesc, *inputTensDesc,
        CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, &(this->convBwdDataAlgo)
    ));
    cudnnSuccessfullyReturn(cudnnGetConvolutionBackwardFilterAlgorithm(
        *cudnnHandle, *inputTensDesc, this->outTensDesc, convDesc, this->convFilterDesc, 
        CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, &(this->convBwdFilterAlgo)
    ));
    size_t size;
    cudnnSuccessfullyReturn(cudnnGetConvolutionForwardWorkspaceSize(
        *cudnnHandle, *inputTensDesc, this->convFilterDesc, this->convDesc, this->outTensDesc, this->convFwdAlgo, &size
    ));
    this->n_workspace = std::max(size, (size_t)0);
    cudnnSuccessfullyReturn(cudnnGetConvolutionBackwardDataWorkspaceSize(
        *cudnnHandle, this->convFilterDesc, this->outTensDesc, this->convDesc, *inputTensDesc, this->convBwdDataAlgo, &size
    ));
    this->n_workspace = std::max(size, this->n_workspace);
    cudnnSuccessfullyReturn(cudnnGetConvolutionBackwardFilterWorkspaceSize(
        *cudnnHandle, *inputTensDesc, this->outTensDesc, this->convDesc, this->convFilterDesc, this->convBwdFilterAlgo, &size
    ));
    this->n_workspace = std::max(size, this->n_workspace);

}

snn::Conv2d & snn::Conv2d::set_configure(cudnnHandle_t * cudnnHandle)
{
    this->handle = cudnnHandle;
    return *this;
}
snn::Conv2d & snn::Conv2d::set_configure(bool training)
{   
    if (this->training && !training)
    {
        cudaSuccessfullyReturn(cudaFree(this->dconvFilter));
        cudaSuccessfullyReturn(cudaFree(this->dconvBias));
        cudaSuccessfullyReturn(cudaFree(this->dout));
    }
    this->training = training;
    return *this;
}
snn::Conv2d & snn::Conv2d::set_configure(float * workspace, int n_workspace)
{
    this->workspace = workspace;
    this->n_workspace = n_workspace;
    return *this;
}

float * snn::Conv2d::forward(const float * input)
{
    float alpha = 1.0f, beta1 = 0.0f, beta2 = 1.0f;
    cudnnSuccessfullyReturn(cudnnConvolutionForward(
        *(this->handle),
        &alpha,
        *(this->inputTensDesc),
        input,
        this->convFilterDesc,
        this->convFilter,
        this->convDesc,
        this->convFwdAlgo,
        this->workspace,
        this->n_workspace,
        &beta1,
        this->outTensDesc,
        this->out
    ));

    cudnnSuccessfullyReturn(cudnnAddTensor(
        *(this->handle),
        &alpha,
        this->convBiasTensDesc,
        this->convBias,
        &beta2,
        this->outTensDesc,
        this->out
    ));

    return this->out;

}

float * snn::Conv2d::backward(const float * input, float * dinput)
{
    float alpha = 1.0f, beta = 0.0f;
    // dw
    cudnnSuccessfullyReturn(cudnnConvolutionBackwardFilter(
        *(this->handle),
        &alpha,
        *(this->inputTensDesc),
        input,
        this->outTensDesc,
        this->dout,
        this->convDesc,
        this->convBwdFilterAlgo,
        this->workspace,
        this->n_workspace,
        &beta,
        this->convFilterDesc,
        this->dconvFilter
    ));
    // db
    cudnnSuccessfullyReturn(cudnnConvolutionBackwardBias(
        *(this->handle),
        &alpha,
        this->outTensDesc,
        this->dout,
        &beta,
        this->convBiasTensDesc,
        this->dconvBias
    ));
    // dx
    if (dinput != nullptr)
    cudnnSuccessfullyReturn(cudnnConvolutionBackwardData(
        *(this->handle),
        &alpha,
        this->convFilterDesc,
        this->convFilter,
        this->outTensDesc,
        this->dout,
        this->convDesc,
        this->convBwdDataAlgo,
        this->workspace,
        this->n_workspace,
        &beta,
        *(this->inputTensDesc),
        dinput
    ));

    return dinput;
}

float * snn::Conv2d::get_out()
{
    return this->out;
}
float * snn::Conv2d::get_dout()
{
    return this->dout;
}
float * snn::Conv2d::get_weight()
{
    return this->convFilter;
}
float * snn::Conv2d::get_dweight()
{
    return this->dconvFilter;
}
size_t snn::Conv2d::get_weight_size()
{
    return this->n_weight;
}
float * snn::Conv2d::get_bias()
{
    return this->convBias;
}
float * snn::Conv2d::get_dbias()
{
    return this->dconvBias;
}
size_t snn::Conv2d::get_bias_size()
{
    return this->out_channels;
}
size_t snn::Conv2d::get_n_workspace()
{
    return this->n_workspace;
}
cudnnTensorDescriptor_t * snn::Conv2d::get_outdesc()
{
    return &(this->outTensDesc);
}

snn::Conv2d & snn::Conv2d::reset_parameters(int seed)
{
    float gain = 1.0f / sqrt((float)(this->in_channels * this->kernel1 * this->kernel2));
    curandGenerator_t curandGenerator;
    curandSuccessfullyReturn(curandCreateGenerator(&curandGenerator, CURAND_RNG_PSEUDO_MT19937));
    curandSuccessfullyReturn(curandSetPseudoRandomGeneratorSeed(curandGenerator, seed));
    curandSuccessfullyReturn(curandGenerateUniform(curandGenerator, this->convFilter, this->n_weight));
    cusnnRescaleTensor<<<CUDA_1D_BLOCK_NUMS(this->n_weight), CUDA_1D_BLOCK_SIZE>>>(
        this->convFilter, this->n_weight, -gain, gain
    );
    curandSuccessfullyReturn(curandGenerateUniform(curandGenerator, this->convBias, this->out_channels));
    cusnnRescaleTensor<<<CUDA_1D_BLOCK_NUMS(this->out_channels), CUDA_1D_BLOCK_SIZE>>>(
        this->convBias, this->out_channels, -gain, gain
    );
    curandSuccessfullyReturn(curandDestroyGenerator(curandGenerator));
    return *this;
}

snn::Conv2d::~Conv2d()
{
    cudaSuccessfullyReturn(cudaFree(this->convFilter));
    cudaSuccessfullyReturn(cudaFree(this->convBias));
    cudaSuccessfullyReturn(cudaFree(this->out));
    if (this->training)
    {
        cudaSuccessfullyReturn(cudaFree(this->dconvFilter));
        cudaSuccessfullyReturn(cudaFree(this->dconvBias));
        cudaSuccessfullyReturn(cudaFree(this->dout));
    }
    cudnnSuccessfullyReturn(cudnnDestroyConvolutionDescriptor(this->convDesc));
    cudnnSuccessfullyReturn(cudnnDestroyFilterDescriptor(this->convFilterDesc));
    cudnnSuccessfullyReturn(cudnnDestroyTensorDescriptor(this->convBiasTensDesc));
    cudnnSuccessfullyReturn(cudnnDestroyTensorDescriptor(this->outTensDesc));
}

snn::BatchNorm::BatchNorm(int n, int c, int h, int w, bool train)
{
    this->training = train;
    this->n_params = c;
    this->steps    = 0;

    if (h == 1 && w == 1) 
        this->mode = CUDNN_BATCHNORM_PER_ACTIVATION;
    else 
        this->mode = CUDNN_BATCHNORM_SPATIAL_PERSISTENT;

    cudnnSuccessfullyReturn(cudnnCreateTensorDescriptor(&(this->yDesc)));
    cudnnSuccessfullyReturn(cudnnSetTensor4dDescriptor(
        yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w
    ));
    cudaSuccessfullyReturn(cudaMalloc(&(this->out), sizeof(float) * n * c * h * w));

    cudnnSuccessfullyReturn(cudnnCreateTensorDescriptor(&(this->bnScaleBiasMeanVarDesc)));
    cudnnSuccessfullyReturn(cudnnSetTensor4dDescriptor(
        this->bnScaleBiasMeanVarDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, c, 1, 1
    ));
    // cudnnSuccessfullyReturn(cudnnDeriveBNTensorDescriptor(this->bnScaleBiasMeanVarDesc, this->yDesc, this->mode));

    cudaSuccessfullyReturn(cudaMalloc(&(this->bnScale), sizeof(float) * this->n_params));
    cusnnFillConstant<<<CUDA_1D_BLOCK_NUMS(this->n_params), CUDA_1D_BLOCK_SIZE>>>(
        this->bnScale, this->n_params, 1.0f
    );
    cudaSuccessfullyReturn(cudaMalloc(&(this->bnBias),  sizeof(float) * this->n_params));
    cusnnFillConstant<<<CUDA_1D_BLOCK_NUMS(this->n_params), CUDA_1D_BLOCK_SIZE>>>(
        this->bnBias, this->n_params, 0.0f
    );

    this->exponentialAverageFactor = 0.1;
    cudaSuccessfullyReturn(cudaMalloc(&(this->resultRunningMean), sizeof(float) * this->n_params));
    cusnnFillConstant<<<CUDA_1D_BLOCK_NUMS(this->n_params), CUDA_1D_BLOCK_SIZE>>>(
        this->resultRunningMean, this->n_params, 0.0f
    );
    cudaSuccessfullyReturn(cudaMalloc(&(this->resultRunningVariance), sizeof(float) * this->n_params));
    cusnnFillConstant<<<CUDA_1D_BLOCK_NUMS(this->n_params), CUDA_1D_BLOCK_SIZE>>>(
        this->resultRunningVariance, this->n_params, 0.0f
    );

    this->epsilon = 1e-5;
    cudaSuccessfullyReturn(cudaMalloc(&(this->resultSaveMean), sizeof(float) * this->n_params));
    cudaSuccessfullyReturn(cudaMalloc(&(this->resultSaveInvVariance), sizeof(float) * this->n_params));

    if (this->training)
    {
        cudaSuccessfullyReturn(cudaMalloc(&(this->dout), sizeof(float) * n * c * h * w));
        cudaSuccessfullyReturn(cudaMalloc(&(this->dbnBias), sizeof(float) * this->n_params));
        cudaSuccessfullyReturn(cudaMalloc(&(this->dbnScale), sizeof(float) * this->n_params));
    }
    else
    {
        this->dout = nullptr;
        this->dbnBias = nullptr;
        this->dbnScale = nullptr;
    }
}

float * snn::BatchNorm::forward(const float * x)
{   
    float alpha = 1.0f, beta = 0.0f;
    if (this->training)
    {
        // this->exponentialAverageFactor += 1.0;

        cudnnSuccessfullyReturn(cudnnBatchNormalizationForwardTraining(
            *(this->handle),
            this->mode,
            &alpha,
            &beta,
            this->yDesc,
            x,
            this->yDesc,
            this->out,
            this->bnScaleBiasMeanVarDesc,
            this->bnScale,
            this->bnBias,
            this->steps == 0 ? 1.0 : this->exponentialAverageFactor,
            // 1.0 / this->exponentialAverageFactor,
            this->resultRunningMean,
            this->resultRunningVariance,
            this->epsilon,
            this->resultSaveMean,
            this->resultSaveInvVariance
        ));
        
        this->steps += 1;
    }
    else
    {
        cudnnSuccessfullyReturn(cudnnBatchNormalizationForwardInference(
            *(this->handle),
            this->mode,
            &alpha,
            &beta,
            this->yDesc,
            x,
            this->yDesc,
            this->out,
            this->bnScaleBiasMeanVarDesc,
            this->bnScale,
            this->bnBias,
            this->resultRunningMean,
            this->resultRunningVariance,
            this->epsilon
        ));
    }
    return this->out;
}

float * snn::BatchNorm::backward(const float * x, float * dx)
{
    float alpha = 1.0f, beta = 0.0f;

    cudnnSuccessfullyReturn(cudnnBatchNormalizationBackward(
        *(this->handle),
        this->mode,
        &alpha,
        &beta,
        &alpha,
        &beta,
        this->yDesc,
        x,
        this->yDesc,
        this->dout,
        this->yDesc,
        dx,
        this->bnScaleBiasMeanVarDesc,
        this->bnScale,
        this->dbnScale,
        this->dbnBias,
        this->epsilon,
        this->resultSaveMean,
        this->resultSaveInvVariance
    ));
    
    return dx;
}

void snn::BatchNorm::set_configure(cudnnHandle_t * cudnnHandle)
{
    this->handle = cudnnHandle;
}
void snn::BatchNorm::set_configure(bool training)
{
    this->training = training;
}

float * snn::BatchNorm::get_out()
{
    return this->out;
}
float * snn::BatchNorm::get_dout()
{
    return this->dout;
}

float * snn::BatchNorm::get_weight()
{
    return this->bnScale;
}
float * snn::BatchNorm::get_d_weight()
{
    return this->dbnScale;
}
size_t snn::BatchNorm::get_weight_size()
{
    return this->n_params;
}
float * snn::BatchNorm::get_bias()
{
    return this->bnBias;
}
float * snn::BatchNorm::get_d_bias()
{
    return this->dbnBias;
}
size_t snn::BatchNorm::get_bias_size()
{
    return this->n_params;
}

snn::BatchNorm::~BatchNorm()
{
    cudaSuccessfullyReturn(cudaFree(this->out));
    cudaSuccessfullyReturn(cudaFree(this->bnScale));
    cudaSuccessfullyReturn(cudaFree(this->bnBias));
    cudaSuccessfullyReturn(cudaFree(this->resultRunningMean));
    cudaSuccessfullyReturn(cudaFree(this->resultRunningVariance));
    cudaSuccessfullyReturn(cudaFree(this->resultSaveMean));
    cudaSuccessfullyReturn(cudaFree(this->resultSaveInvVariance));

    if (this->training)
    {
        cudaSuccessfullyReturn(cudaFree(this->dout));
        cudaSuccessfullyReturn(cudaFree(this->dbnBias));
        cudaSuccessfullyReturn(cudaFree(this->dbnScale));
    }
}

snn::LIF & snn::LIF::set_configure(int timesteps, int n_neurons)
{
    this->timesteps = timesteps;
    this->n_neurons = n_neurons;

    cudaSuccessfullyReturn(cudaMalloc(&(this->psp), sizeof(float) * timesteps * n_neurons));
    cudaSuccessfullyReturn(cudaMalloc(&(this->out), sizeof(float) * timesteps * n_neurons));
    cudaSuccessfullyReturn(cudaMalloc(&(this->dout), sizeof(float) * timesteps * n_neurons));

    return *this;
}

float * snn::LIF::forward(const float * input)
{
    cusnnLIFneuronForward<<<CUDA_1D_BLOCK_NUMS(this->n_neurons), CUDA_1D_BLOCK_SIZE>>>(
        input, this->psp, this->out, this->timesteps, this->n_neurons, this->tau, this->thres, this->vre
    );

    return this->out;
}

float * snn::LIF::backward(const float * input, float * dinput)
{
    cusnnLIFneuronBackward<<<CUDA_1D_BLOCK_NUMS(this->n_neurons), CUDA_1D_BLOCK_SIZE>>>(
        this->dout, this->psp, dinput, this->timesteps, this->n_neurons, this->tau, this->thres, this->vre
    );

    return dinput;
}

float * snn::LIF::get_out()
{
    return this->out;
}
float * snn::LIF::get_dout()
{
    return this->dout;
}

snn::LIF::~LIF()
{
    cudaSuccessfullyReturn(cudaFree(this->psp));
    cudaSuccessfullyReturn(cudaFree(this->out));
    cudaSuccessfullyReturn(cudaFree(this->dout));
}
//////////////////////

ann::ReLU::ReLU(int n)
{
    this->n = n;
    cudaSuccessfullyReturn(cudaMalloc(&(this->out), sizeof(float) * n));
    cudaSuccessfullyReturn(cudaMalloc(&(this->dout), sizeof(float) * n));
}

float * ann::ReLU::get_out()
{
    return this->out;
}
float * ann::ReLU::get_dout()
{
    return this->dout;
}

float * ann::ReLU::forward(const float * input)
{
    cusnnReLUforward<<<CUDA_1D_BLOCK_NUMS(this->n), CUDA_1D_BLOCK_SIZE>>>(
        input, this->out, this->n
    );
    return this->out;
}
float * ann::ReLU::backward(const float * input, float * dinput)
{   
    cusnnReLUbackward<<<CUDA_1D_BLOCK_NUMS(this->n), CUDA_1D_BLOCK_SIZE>>>(
        input, this->dout, this->n, dinput
    );
    return dinput;
}

ann::ReLU::~ReLU()
{
    cudaSuccessfullyReturn(cudaFree(this->out));//, sizeof(float) * n));
    cudaSuccessfullyReturn(cudaFree(this->dout));//, sizeof(float) * n));
}



////////////
void optim::Adamw::add_parameter(float * param, float * grad, int n)
{   
    float * m0, * v0;
    cudaSuccessfullyReturn(cudaMalloc(&m0, sizeof(float) * n));
    cusnnFillConstant<<<CUDA_1D_BLOCK_NUMS(n), CUDA_1D_BLOCK_SIZE>>>(
        m0, n, 0.0f
    );
    cudaSuccessfullyReturn(cudaMalloc(&v0, sizeof(float) * n));
    cusnnFillConstant<<<CUDA_1D_BLOCK_NUMS(n), CUDA_1D_BLOCK_SIZE>>>(
        v0, n, 0.0f
    );

    this->params.push_back(param);
    this->grads.push_back(grad);
    this->m.push_back(m0);
    this->v.push_back(v0);
    this->sizes.push_back(n);
}

void optim::Adamw::step()
{   
    for (int i = 0; i < this->params.size(); ++i)
    {
        cusnnAdamwGradientDescent<<<CUDA_1D_BLOCK_NUMS(this->sizes[i]), CUDA_1D_BLOCK_SIZE>>>(
            this->params[i], this->grads[i], this->m[i], this->v[i], this->sizes[i], 
            this->lr, this->wd, this->beta1, this->beta2, this->eps
        );
    }
}

optim::Adamw::~Adamw()
{
    for (auto p : this->m) 
        cudaSuccessfullyReturn(cudaFree(p));
    for (auto p : this->v)
        cudaSuccessfullyReturn(cudaFree(p));
}

void optim::SGD::add_parameter(float * param, float * grad, int n)
{
    this->params.push_back(param);
    this->grads.push_back(grad);
    this->sizes.push_back(n);
}

void optim::SGD::step()
{
    for (int i = 0; i < this->params.size(); ++i)
    {
        cusnnSGDGradientDescent<<<CUDA_1D_BLOCK_NUMS(this->sizes[i]), CUDA_1D_BLOCK_SIZE>>>(
            this->params[i], this->grads[i], this->sizes[i], this->lr, this->wd
        );
    }
}