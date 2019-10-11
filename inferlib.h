#ifndef _INFERLIB_H_
#define _INFERLIB_H_
#include <cuda_runtime_api.h>
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

class DAWNBench_Inference{
public:
    DAWNBench_Inference(const std::string &model_path, const std::string &int8_cache_path);
    void load_image_data(const cv::Mat &);
    std::vector<int> do_inference();
    ~DAWNBench_Inference();

private:
    cudaStream_t stream;
    cudaEvent_t end;
    int width;
    int height;
    int channel;
    void *context;
    float *input_cpu;
    float *input_gpu;
    int *output_cpu;
    int *output_gpu;
    void *buffer[2];
    std::string model_path;
    std::string int8_cache_path;
};

#endif