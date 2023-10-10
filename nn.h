#ifndef SNN_H
#define SNN_H

#include <vector>
#include <cublas_v2.h>
#include <cudnn.h>


namespace snn 
{
    class Linear
    {
        public:

        Linear(int in_features, int out_features, bool bias = true, bool train = true);
        //
        void set_configure(cublasHandle_t * handle);
        void set_configure(int batch_size, float * onevec);
        void set_configure(bool training);
        //
        void allocate_gradspace();
        void free_gradspace();
        //
        float * get_out();
        float * get_dout();
        float * get_weight();
        float * get_dweight();
        float * get_bias();
        float * get_dbias();
        size_t  get_weight_size();
        size_t  get_bias_size();
        size_t  get_out_size();
        //
        float * forward(const float * input);
        float * backward(const float * input, float * dinput);
        float * backward(const float * dout, const float * input, float * dinput);
        //
        void reset_parameters(int seed = 2023);
        //
        ~Linear();

        private:
        //
        int in_features;
        int out_features;
        int n_weight;
        int n_out;
        bool use_bias, training;

        int batch_size;
        float * onevec          {nullptr};
        //
        float * weight          {nullptr};
        float * bias            {nullptr};
        float * d_weight        {nullptr};
        float * d_bias          {nullptr};
        //
        float * out             {nullptr};
        float * dout            {nullptr};
        //
        cublasHandle_t * handle {nullptr};
    };

    class Conv2d
    {
        public:
        Conv2d(
            cudnnHandle_t * cudnnHandle,
            cudnnTensorDescriptor_t * inputTensDesc, 
            int in_channels, int out_channels, 
            int kernel1, int kernel2, 
            int stride1, int stride2, 
            int padding1, int padding2,
            bool train = true
        );

        snn::Conv2d & set_configure(cudnnHandle_t * cudnnHandle);
        snn::Conv2d & set_configure(bool training);
        snn::Conv2d & set_configure(float * workspace, int n_workspace);
        snn::Conv2d & reset_parameters(int seed = 2023);
        ~Conv2d();

        float * get_out();
        float * get_dout();
        float * get_weight();
        float * get_dweight();
        size_t  get_weight_size();
        float * get_bias();
        float * get_dbias();
        size_t  get_bias_size();
        size_t  get_n_workspace();
        cudnnTensorDescriptor_t * get_outdesc();

        float * forward(const float * input);
        float * backward(const float * input, float * dinput);

        private:
        int in_channels, out_channels;
        int kernel1, kernel2;
        int stride1, stride2;
        int padding1, padding2;
        bool training;
        size_t n_weight, n_out;

        cudnnHandle_t * handle;
        cudnnConvolutionDescriptor_t convDesc;

        cudnnFilterDescriptor_t convFilterDesc;
        float * convFilter, * dconvFilter;

        cudnnTensorDescriptor_t convBiasTensDesc;
        float * convBias, * dconvBias;

        cudnnTensorDescriptor_t outTensDesc;
        float * out, * dout;

        cudnnTensorDescriptor_t * inputTensDesc;

        cudnnConvolutionFwdAlgo_t convFwdAlgo;
        cudnnConvolutionBwdDataAlgo_t convBwdDataAlgo;
        cudnnConvolutionBwdFilterAlgo_t convBwdFilterAlgo;

        float * workspace;
        size_t n_workspace;

    };

    class BatchNorm
    {
        public:
        BatchNorm(int n, int c, int h, int w, bool train = true);

        float * forward(const float * x);
        float * backward(const float * x, float * dx);

        void set_configure(cudnnHandle_t * cudnnHandle);
        void set_configure(bool training);

        float * get_out();
        float * get_dout();
        float * get_weight();
        float * get_d_weight();
        size_t  get_weight_size();
        float * get_bias();
        float * get_d_bias();
        size_t  get_bias_size();

        ~BatchNorm();

        private:

        cudnnHandle_t * handle;
        cudnnBatchNormMode_t mode;

        cudnnTensorDescriptor_t yDesc;
        float * out, * dout;

        cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc;
        float * bnScale, * bnBias, * dbnScale, * dbnBias;

        double exponentialAverageFactor;
        float * resultRunningMean, * resultRunningVariance;

        double epsilon;
        float * resultSaveMean, * resultSaveInvVariance;

        bool training;
        unsigned int n_params, steps;

    };

    class LIF
    {
        public:
        LIF(float tau = 2.0f, float thres = 1.0f, float vre = 0.0f, float alpha = 4.0f)
            : tau(tau), thres(thres), vre(vre), alpha(alpha) {}

        snn::LIF & set_configure(int timesteps, int n_neurons);
        
        float * forward(const float * input);
        
        float * backward(const float * input, float * dinput);

        float * get_out();
        float * get_dout();

        ~LIF();

        private:
        float tau, thres, vre, alpha;

        int timesteps, n_neurons;

        float * psp;
        float * out;
        float * dout;
    };

}

namespace ann
{
    class ReLU
    {
        public:

        ReLU(int n);

        float * get_out();
        float * get_dout();
        float * forward(const float * input);
        float * backward(const float * input, float * dinput);

        ~ReLU();

        private:
        float * out, * dout;
        int n;

    };
}

namespace optim
{
    class Adamw
    {
        public:
        Adamw(float lr, float wd = 1e-2, float beta1 = 0.9, float beta2 = 0.999, float eps = 1e-8)
        : lr(lr), wd(wd), beta1(beta1), beta2(beta2), eps(eps) {}

        void add_parameter(float * param, float * grad, int n);
        void step();

        ~Adamw();

        private:

        float lr;
        float wd;
        float beta1;
        float beta2;
        float eps;
        
        std::vector<float *> params;
        std::vector<float *> grads;
        std::vector<float *> m;
        std::vector<float *> v;
        std::vector<int>     sizes;
    };

    class SGD
    {
        public:
        SGD(float lr, float wd = 1e-2) : lr(lr), wd(wd) {}
        void add_parameter(float * param, float * grad, int n);
        void step();

        private:
        float lr, wd;

        std::vector<float *> params, grads;
        std::vector<int> sizes;

    };

}

/////////////////////////////

namespace network 
{
    enum LayerType {
        BN,
        CV,
        FC,
        SN,
        AC,
    };

    class Layer
    {
        public:
        Layer(snn::BatchNorm * bn);
        Layer(snn::Conv2d * cv);
        Layer(snn::LIF * sn);
        Layer(snn::Linear * fc);
        Layer(ann::ReLU * ac);
        Layer(network::Layer &) = delete;

        float * forward(const float * input);
        float * backward(const float * input, float * dinput);

        float * get_out();
        float * get_dout();

        private:
        snn::BatchNorm * bn_layer;
        snn::Conv2d * conv_layer;
        snn::LIF * neuron_layer;
        snn::Linear * fc_layer;
        ann::ReLU * act_layer;

        network::LayerType type;

    };
}


#endif