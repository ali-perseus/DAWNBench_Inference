#include <sys/time.h>

#include <chrono>
#include <fstream>
#include <vector>

#include "inferlib.h"

const std::vector<std::pair<std::string, int>> read_label(
    const std::string& base_dir = "./") {
    std::vector<std::pair<std::string, int>> rst;
    std::ifstream f_label(base_dir + "label.txt");
    while (f_label) {
        std::string temp_str;
        int label;
        f_label >> temp_str;
        f_label >> label;
        if (temp_str.size() > 0) {
            rst.push_back(std::make_pair(base_dir + temp_str, label));
        }
    }
    return rst;
}

int main(void) {
    DAWNBench_Inference inference("resnet26d.engine");
    std::vector<std::pair<std::string, int>> dataset = read_label("./");
    double totaltime = 0;
    int top5_pass_count = 0;
    int image_count = 0;
    char(*preprocess_data)[3 * 224 * 224] = new char[50000][3 * 224 * 224];
    int* label = new int[50000];
    // preprocess_data
    for (int i = 0; i < 50000; i++) {
        auto& data = dataset[i];
        cv::Mat img_bgr = cv::imread(data.first);
        inference.preprocessing_data(preprocess_data[i], img_bgr);
        label[i] = data.second;
        if (i % 1000 == 0) {
            std::cout << "loading: " << i << std::endl;
        }
    }
    // inference
    for (int image_idx = 0; image_idx < 50000; image_idx++) {
        // do inference
        auto tStart = std::chrono::high_resolution_clock::now();
        std::vector<int> rst =
            inference.do_inference(preprocess_data[image_idx]);
        auto tEnd = std::chrono::high_resolution_clock::now();
        double m_dectime =
            std::chrono::duration<float, std::milli>(tEnd - tStart).count();
        totaltime += m_dectime;
        // check top 5
        image_count += 1;
        for (int i = 0; i < 5; i++) {
            if (rst[i] == label[image_idx]) {
                top5_pass_count += 1;
                break;
            }
        }
        if (image_count % 100 == 0) {
            std::cout << "run at " << image_count
                      << ": Prec@5: " << (double)top5_pass_count / image_count
                      << std::endl;
            std::cout << "inference time at " << image_count << " : "
                      << totaltime / image_count << " ms" << std::endl;
        }
    }
    std::cout << "final inference time : " << totaltime / dataset.size()
              << " ms" << std::endl;
    std::cout << "final Prec@5: " << (double)top5_pass_count / image_count
              << std::endl;
    delete[] preprocess_data;
    delete[] label;
}
