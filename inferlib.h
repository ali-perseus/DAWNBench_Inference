#ifndef _INFERLIB_H_
#define _INFERLIB_H_
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <opencv2/opencv.hpp>
#include <hgrt_c_api_pub.h>

class DAWNBench_Inference{
public:
    DAWNBench_Inference(const std::string &model_path);
    void preprocessing_data(char* data, const cv::Mat& img);
    void do_preload(char* input_data);
    std::vector<int> do_inference(char* input_data);
    ~DAWNBench_Inference();

private:
    hgExecutionContext context;
    hgHostTensorDesc hdesc;
    hgTensorDesc ddesc;
    hgTensor dtensor;
    char *output_cpu;
    void *buffer[2];
    std::string model_path;
    int width;
    int height;
    int channel;
};
#endif
