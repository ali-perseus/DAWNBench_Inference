# DAWNBench_Inference
## Resnet26d for DAWNBench inference task on ImageNet

We run Resnet26d on Alibaba Cloud ecs.ebman1.26xlarge, which consists of 4 npu core and 104 vCPUs.

The following instructions show how to achieve the performance that we submitted to DAWNBench step by step.

1. install miniconda and dependencies
```shell
   wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh
   sh Miniconda3-latest-Linux-x86_64.sh
   conda install python=3.6
   conda install glog=0.3.4
   conda install opencv
   pip install pillow opencv-python
   pip install torch torchvision
   pip install hgai-centos_rel_1.0.4.sp1.whl
```

2. git clone DAWNBench_Inference code
```
   git clone https://github.com/ali-perseus/DAWNBench_Inference.git
```

3. run the following commands to replicate our results submitted to DAWNBench,  
```shell
   ##resize and crop using torchvision
   ##edit preprocress.py if necessary
   python3 ./preprocress.py
   ##edit build.sh if necessary
   ##build
   sh ./build.sh
   ##run test
   sh ./test.sh
```

4.Congratulations! the result is as follows:
```shell
final inference time : 0.0739278 ms
final Prec@5: 0.93156
```
