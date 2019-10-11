#include "inferlib.h"
#include <fstream>
#include <vector>
#include <chrono>
#include <sys/time.h>

const std::vector<std::pair<std::string,int>> read_label(const std::string& base_dir = "./"){
    std::vector<std::pair<std::string,int>> rst;
    std::ifstream f_label(base_dir + "label.txt");
    while(f_label){
        std::string temp_str;
        int label;
        f_label >> temp_str;
        f_label >> label;
        if(temp_str.size() > 0){
            rst.push_back(std::make_pair(base_dir + temp_str,label));
        }
    }
    return rst;
}


int main(void){
    DAWNBench_Inference inference("resnet26d.onnx", "./INT8CacheFile");
    std::vector<std::pair<std::string,int>> dataset = read_label("./");
    double totaltime = 0;
    int top5_pass_count = 0;
    int image_count = 0;
    for(auto& data :dataset){
        cv::Mat img_bgr = cv::imread(data.first);
        inference.load_image_data(img_bgr);
        auto tStart = std::chrono::high_resolution_clock::now();
        std::vector<int> rst = inference.do_inference();
        auto tEnd = std::chrono::high_resolution_clock::now();
        double m_dectime = std::chrono::duration<float, std::milli>(tEnd - tStart).count();;
        totaltime += m_dectime;
        image_count +=1;
        for(int i=0;i<5;i++){
            if(rst[i] == data.second){
                top5_pass_count += 1;
                break;
            }
        }

        if (image_count%100 == 0){
            std::cout<<"run at "<<image_count<<": Prec@5: "<< (double)top5_pass_count/image_count << std::endl;
            std::cout << "inference time at "<<image_count<<" : " << totaltime/ image_count <<"ms"<< std::endl;
        }
    }
    std::cout<<"final Prec@5: "<< (double)top5_pass_count/image_count << std::endl;
    std::cout << "final inference time : " << totaltime/ image_count<<"ms"<< std::endl;
}