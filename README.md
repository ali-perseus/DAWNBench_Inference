# DAWNBench_Inference

## Resnet50 for DAWNBench inference task on ImageNet

We run Resnet50 on Alibaba Cloud ecs.gn5i-c8g1.2xlarge, which consists of 1 NVIDIA P4 GPU and 8 vCPUs.

The following instructions show how to achieve the performance that we submitted to DAWNBench step by step.

1. install CUDA 10 and CUDNN 7, TensorRT 5 and TensorFlow 1.12
```
   download and install CUDA 10 and driver (https://developer.nvidia.com/compute/cuda/10.0/Prod/local_installers/cuda_10.0.130_410.48_linux)

   download and install CUDNN 7 library for Linux (https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v7.5.1/prod/10.0_20190418/cudnn-10.0-linux-x64-v7.5.1.10.tgz)

   download and install TensorRT 5 for CentOS/RedHat 7 (https://developer.nvidia.com/compute/machine-learning/tensorrt/5.0/GA_5.0.2.6/tars/TensorRT-5.0.2.6.Red-Hat.x86_64-gnu.cuda-10.0.cudnn7.3.tar.gz)

   install TensorFlow 1.12 wheel package for CUDA 10 (pip install tensorflow_pkg/tensorflow-1.12.2-cp27-cp27mu-linux_x86_64.whl)
```
2. git clone DAWNBench_Inference code
```
   git lfs clone https://github.com/ali-perseus/DAWNBench_Inference.git
```
   Note: please install git-lfs before clone the DAWNBench_Inference. In CentOS, just running the following commands to install git-lfs:  
```
   yum install git-lfs  
   git lfs install  
```
3. run the following commands to replicate our results submitted to DAWNBench,  
```
   MODEL_DIR=`pwd`/frozen_int8_graph.pb
   DATA_DIR="directory for ImageNet evaluation tfrecord"

   python resnet50_inference.py --model $MODEL_DIR  --data_dir $DATA_DIR
```
Note: to create TFRecords files for ImageNet evaluation data set you can use,  
   https://github.com/tensorflow/models/blob/master/research/inception/inception/data/build_imagenet_data.py
