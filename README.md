# DAWNBench_Inference

## Resnet50 for DAWNBench inference task on ImageNet

We run Resnet50 on Alibaba Cloud ecs.gn5i-c8g1.2xlarge, which consists of 1 NVIDIA P4 GPU and 8 vCPUs. 

The following instructions show how to achieve the performance that we submitted to DAWNBench step by step.
1, get Tensorflow from NVIDIA GPU Cloud, 
   docker pull nvcr.io/nvidia/tensorflow:18.09-py2
2, make a project directory
   mkdir dawn_bench && cd dawn_bench
3, clone tensorflow benchmark, and switch to 'cnn_tf_v1.10_compatible' branch
   git clone https://github.com/tensorflow/benchmarks.git   
   cd benchmarks && git checkout origin/cnn_tf_v1.10_compatible && cd ..  
4, clone optimized benchmark_cnn.py,
   git lfs clone https://github.com/ali-perseus/DAWNBench_Inference.git   
   Note: please install git-lfs before clone the DAWNBench_Inference. In Centos, just running the following commands to install git-lfs:
   yum install git-lfs
   git lfs install
5, prepare the benchmark code,
   cp DAWNBench_Inference/benchmark_cnn.py benchmarks/scripts/tf_cnn_benchmarks
   and overwrite the destination file  
6, prepare optimized tensorrt graph   
   cp DAWNBench_Inference/trt_int8_graph benchmarks/scripts/tf_cnn_benchmarks
7, start the docker container to create a tensorflow environment
   nvidia-docker run -v $path_to_dawn_bench/:/data0/ -v $path_to_imagenet_tf/:/ramdisk/ -it nvcr.io/nvidia/tensorflow:18.09-py2 bash
   ls /ramdisk/imagenet_tf/
   that would show 128 TFRecords files that include all the 50000 evaluation images
8, run the following commands in the docker container to replicate our results submitted to DAWNBench,
   cd /data0/benchmarks/scripts/tf_cnn_benchmarks/
   DATA_DIR=/ramdisk/imagenet_tf/
   python tf_cnn_benchmarks.py \
    --use_datasets=False \
    --data_format=NCHW \
    --batch_size=1 \
    --model=resnet50 \
    --data_dir=${DATA_DIR} \
    --nodistortions \
    --num_batches=50000 \
    --num_gpus=1 \
    --display_every=1000 \
    --eval=True \
    --eval_freeze=True \
    --load_frozen_graph=./trt_int8_graph

Note: to create TFRecords files for ImageNet evaluation data set you can use,
   https://github.com/tensorflow/models/blob/master/research/inception/inception/data/build_imagenet_data.py


