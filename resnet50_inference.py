# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
import time
import numpy as np

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94
_CHANNEL_MEANS = [_R_MEAN, _G_MEAN, _B_MEAN]

INPUT_SIZE = 224
CROP_PADDING = 16

def _mean_image_subtraction(image, means, num_channels):
    if image.get_shape().ndims != 3:
        raise ValueError('Input must be of size [height, width, C>0]')

    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')

    # We have a 1-D tensor of means; convert to 3-D.
    means = tf.expand_dims(tf.expand_dims(means, 0), 0)
    # image = tf.cast(image, dtype=tf.float32)

    return image - means


def _decode_and_center_crop(image_bytes, image_size=INPUT_SIZE):
    """Crops to center of image with padding then scales image_size."""
    shape = tf.image.extract_jpeg_shape(image_bytes)
    image_height = shape[0]
    image_width = shape[1]

    padded_center_crop_size = tf.cast(
        ((image_size / (image_size + CROP_PADDING)) *
          tf.cast(tf.minimum(image_height, image_width), tf.float32)),
          tf.int32)

    offset_height = ((image_height - padded_center_crop_size) + 1) // 2
    offset_width = ((image_width - padded_center_crop_size) + 1) // 2
    crop_window = tf.stack([offset_height, offset_width,
                            padded_center_crop_size, padded_center_crop_size])
    image = tf.image.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)
    image = tf.image.resize_bilinear([image], [image_size, image_size])[0]

    return image

def preprocess_image(image_buffer,
                     output_height,
                     output_width,
                     num_channels=3
                     ):
    # For validation, we want to decode, resize, then just crop the middle.
    image = _decode_and_center_crop(image_buffer)
    return _mean_image_subtraction(image, _CHANNEL_MEANS, num_channels)

class LoggerHook(tf.train.SessionRunHook):
    """Logs runtime of each iteration"""
    def __init__(self, batch_size, num_records, display_every):
        self.iter_times = []
        self.display_every = display_every
        self.num_steps = (num_records + batch_size - 1) / batch_size
        self.batch_size = batch_size

    def before_run(self, run_context):
        self.start_time = time.time()

    def after_run(self, run_context, run_values):
        current_time = time.time()
        duration = current_time - self.start_time
        self.start_time = current_time
        self.iter_times.append(duration)
        current_step = len(self.iter_times)
        if current_step % self.display_every == 0:
            print("    step %d/%d, iter_time(ms)=%.4f, images/sec=%d" % (
                current_step, self.num_steps, duration * 1000,
                self.batch_size / self.iter_times[-1]))

def run(frozen_graph, model, data_dir, data_files, batch_size,
        num_iterations, num_warmup_iterations, display_every=100):
    """Evaluates a frozen graph

    This function evaluates a graph on the ImageNet validation set.
    tf.estimator.Estimator is used to evaluate the accuracy of the model
    and a few other metrics. The results are returned as a dict.

    frozen_graph: GraphDef, a graph containing input node 'input' and outputs 'logits' and 'classes'
    model: string, the model name (see NETS table in graph.py)
    data_files: List of TFRecord files used for inference
    batch_size: int, batch size for TensorRT optimizations
    num_iterations: int, number of iterations(batches) to run for
    """
    # Define model function for tf.estimator.Estimator
    def model_fn(features, labels, mode):
        logits_out = tf.import_graph_def(frozen_graph,
                                         input_map={'input': features},
                                         return_elements=['logits:0'],
                                         name='')
        logits_out = tf.reshape(logits_out, [-1, 1001])
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits_out)

        labels = tf.reshape(labels, [-1])
        top5accuracy = tf.nn.in_top_k(predictions=logits_out, targets=labels, k=5, name='acc_op')
        top5accuracy = tf.cast(top5accuracy, tf.int32)
        top5accuracy = tf.metrics.mean(top5accuracy)

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode,
                loss=loss,
                eval_metric_ops={'accuracy': top5accuracy})

    # preprocess function for input data
    preprocess_fn = get_preprocess_fn(model)

    def get_tfrecords_count(files):
        num_records = 0
        for fn in files:
            for record in tf.python_io.tf_record_iterator(fn):
                num_records += 1
        return num_records

    # Define the dataset input function for tf.estimator.Estimator
    def eval_input_fn():
        dataset = tf.data.Dataset.list_files(os.path.join(data_dir, "validation*"))
        dataset = dataset.apply(
            tf.data.experimental.parallel_interleave(tf.data.TFRecordDataset, cycle_length=4, prefetch_input_elements=8))
        dataset = dataset.apply(
            tf.data.experimental.map_and_batch(map_func=preprocess_fn, batch_size=batch_size, num_parallel_calls=4))
        dataset = dataset.apply(tf.data.experimental.prefetch_to_device("/gpu:0", 100))
        iterator = dataset.make_one_shot_iterator()
        features, labels = iterator.get_next()
        return features, labels

    # Evaluate model
    logger = LoggerHook(
        display_every=display_every,
        batch_size=batch_size,
        num_records=get_tfrecords_count(data_files))
    tf_config = tf.ConfigProto()
    tf_config.intra_op_parallelism_threads = 4
    tf_config.inter_op_parallelism_threads = 4
    tf_config.gpu_options.allow_growth = True
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=tf.estimator.RunConfig(session_config=tf_config))
    results = estimator.evaluate(eval_input_fn, steps=num_iterations, hooks=[logger])

    # Gather additional results
    iter_times = np.array(logger.iter_times[num_warmup_iterations:])
    results['latency_mean'] = np.mean(iter_times) * 1000
    return results

def deserialize_image_record(record):
    feature_map = {
        'image/encoded': tf.FixedLenFeature([], tf.string, ''),
        'image/class/label': tf.FixedLenFeature([1], tf.int64, -1),
    }
    with tf.name_scope('deserialize_image_record'):
        obj = tf.parse_single_example(record, feature_map)
        imgdata = obj['image/encoded']
        label = tf.cast(obj['image/class/label'], tf.int32)
        return imgdata, label

def get_preprocess_fn(model, mode='classification'):
    def process(record):
        imgdata, label = deserialize_image_record(record)
        image = preprocess_image(imgdata, INPUT_SIZE, INPUT_SIZE)
        return image, label

    return process

def get_frozen_graph(model, batch_size=1):
    """Retreives a frozen GraphDef from model definitions in classification.py and applies TF-TRT

    model: str, the model name (see NETS table in classification.py)
    use_trt: bool, if true, use TensorRT
    precision: str, floating point precision (fp32, fp16, or int8)
    batch_size: int, batch size for TensorRT optimizations
    returns: tensorflow.GraphDef, the TensorRT compatible frozen graph
    """
    num_nodes = {}
    times = {}
    graph_sizes = {}

    # Load from pb file if frozen graph was already created and cached
    if os.path.isfile(model):
        print('Loading cached frozen graph from \'%s\'' % model)
        start_time = time.time()
        with tf.gfile.GFile(model, "rb") as f:
            frozen_graph = tf.GraphDef()
            frozen_graph.ParseFromString(f.read())
        times['loading_frozen_graph'] = time.time() - start_time
        num_nodes['loaded_frozen_graph'] = len(frozen_graph.node)
        num_nodes['trt_only'] = len([1 for n in frozen_graph.node if str(n.op) == 'TRTEngineOp'])
        graph_sizes['loaded_frozen_graph'] = len(frozen_graph.SerializeToString())
        return frozen_graph, num_nodes, times, graph_sizes

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate model')
    parser.add_argument('--model', type=str, required=True,
        help='Model file name')
    parser.add_argument('--data_dir', type=str, required=True,
        help='Directory containing validation set TFRecord files.')
    parser.add_argument('--batch_size', type=int, default=1,
        help='Number of images per batch.')
    parser.add_argument('--num_iterations', type=int, default=None,
        help='How many iterations(batches) to evaluate. If not supplied, the whole set will be evaluated.')
    parser.add_argument('--display_every', type=int, default=1000,
        help='Number of iterations executed between two consecutive display of metrics')
    parser.add_argument('--num_warmup_iterations', type=int, default=100,
        help='Number of initial iterations skipped from timing')
    args = parser.parse_args()

    def get_files(data_dir, filename_pattern):
        if data_dir == None:
            return []
        files = tf.gfile.Glob(os.path.join(data_dir, filename_pattern))
        if files == []:
            raise ValueError('Can not find any files in {} with pattern "{}"'.format(
                data_dir, filename_pattern))
        return files

    validation_files = get_files(args.data_dir, 'validation*')

    # Retreive graph using NETS table in graph.py
    frozen_graph, num_nodes, times, graph_sizes = get_frozen_graph(
        model=args.model,
        batch_size=args.batch_size)

    def print_dict(input_dict, str='', scale=None):
        for k, v in sorted(input_dict.items()):
            headline = '{}({}): '.format(str, k) if str else '{}: '.format(k)
            v = v * scale if scale else v
            print('{}{}'.format(headline, '%.1f' % v if type(v) == float else v))

    print_dict(vars(args))
    print_dict(num_nodes, str='num_nodes')
    print_dict(graph_sizes, str='graph_size(MB)', scale=1. / (1 << 20))
    print_dict(times, str='time(s)')

    # Evaluate model
    print('running inference...')
    results = run(
        frozen_graph,
        model=args.model,
        data_dir=args.data_dir,
        data_files=validation_files,
        batch_size=args.batch_size,
        num_iterations=args.num_iterations,
        num_warmup_iterations=args.num_warmup_iterations,
        display_every=args.display_every)

    # Display results
    print('results of {}:'.format(args.model))
    print('    accuracy: %.2f' % (results['accuracy'] * 100))
    print('    latency_mean(ms): %.2f' % results['latency_mean'])
