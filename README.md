# DAWNBench_Inference
## Resnet26d for DAWNBench inference task on ImageNet

We run Resnet26d on Alibaba Cloud ecs.gn6i-c8g1.2xlarge, which consists of 1 NVIDIA T4 GPU and 8 vCPUs.

The following instructions show how to achieve the performance that we submitted to DAWNBench step by step.

1. install CUDA 10 and CUDNN 7, TensorRT 6 and TensorFlow 1.12
```
    download CUDA 10.0.130 for CentOS/RedHat 7  (https://developer.nvidia.com/compute/cuda/10.0/Prod/local_installers/cuda_10.0.130_410.48_linux)

    download and install CUDNN 7.6.3.30 for CentOS/RedHat 7 (https://developer.nvidia.com/compute/machine-learning/cudnn/secure/7.6.3.30/Production/10.0_20190822/cudnn-10.0-linux-x64-v7.6.3.30.tgz)

    download and install TensorRT 6.0.1.5 for CentOS/RedHat 7 (https://developer.nvidia.com/compute/machine-learning/tensorrt/secure/6.0/GA_6.0.1.5/tars/TensorRT-6.0.1.5.CentOS-7.6.x86_64-gnu.cuda-10.0.cudnn7.6.tar.gz)
```

2. install gcc && opencv && python3 && pillow && torch
```shell
    yum -y install gcc+ gcc-c++
    yum -y install opencv-devel
    yum -y install python3
    yum -y install libSM
    pip3 install pillow opencv-python
    pip3 install torch torchvision
```

3. git clone DAWNBench_Inference code
```
   git lfs clone -b resnet26d https://github.com/ali-perseus/DAWNBench_Inference.git
```
   Note: please install git-lfs before clone the DAWNBench_Inference. In CentOS, just running the following commands to install git-lfs:  
```
   yum install git-lfs  
   git lfs install  
```

4. run the following commands to replicate our results submitted to DAWNBench,  
```shell
   ## make sure the gpu card mode is in Persistence mode.
   nvidia-smi -pm 1
   ## set mclk and pclk
   nvidia-smi -ac 5000,1590
   export LD_LIBRARY_PATH=/path/to/TensorRT-6.0.1.5/lib:$LD_LIBRARY_PATH
   ##resize and crop using torchvision
   ##edit preprocress.py if necessary
   python3 ./preprocress.py
   ##edit build.sh if necessary
   ##build
   sh ./build.sh
   ##run test
   ./inference_test
```

5.Congratulations! the result is as follows:
```shell
final inference time : 0.637316ms
final Prec@5: 0.93028
```