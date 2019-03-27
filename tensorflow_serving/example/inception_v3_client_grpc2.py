# Copyright 2016 Google Inc. All Rights Reserved.
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
# ==============================================================================
"""Send JPEG image to tensorflow_model_server loaded with ResNet model.

"""

from __future__ import print_function

# This is a placeholder for a Google-internal import.

import grpc
import requests
import tensorflow as tf
from PIL import Image
#from StringIO import StringIO
import numpy as np
from preprocessing import inception_preprocessing

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import time
import os
# The image URL is the location of the image we should send to the server

tf.app.flags.DEFINE_string('server', 'localhost:8555',
                           'PredictionService host:port')
#tf.app.flags.DEFINE_string('image', '/home/pmeng/images/test2.jpg', 'path to image in JPEG format')
tf.app.flags.DEFINE_string('image', '/lustre/dataset/tensorflow/imagenet/validation-00103-of-00128', 'path to image in JPEG format')
tf.app.flags.DEFINE_string('data_dir', '/lustre/dataset/tensorflow/imagenet/', 'path to image in JPEG format')
FLAGS = tf.app.flags.FLAGS


def main(_):
  '''
  if FLAGS.image:
    with open(FLAGS.image, 'rb') as f:
      data = f.read()
  print(1)

  image = np.array(Image.open(FLAGS.image))
  print(2)
  '''
  height = 299
  width = 299
  def preprocess_fn(path):
    image = tf.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = inception_preprocessing.preprocess_image(image, width, height, is_training=False)
    return image

  files = [os.path.join(path, name) for path, _, files in os.walk(FLAGS.data_dir) for name in files]
  dataset = tf.data.Dataset.from_tensor_slices(files)
  dataset = dataset.apply(tf.contrib.data.map_and_batch(map_func=preprocess_fn, batch_size=1, num_parallel_calls=8))
  dataset = dataset.repeat(count=1)
  iterator = dataset.make_one_shot_iterator()
  sess = tf.Session()
  for i in range(1,20):
      image = iterator.get_next()
      image = sess.run(image)
      print("Image shape:", image.shape)
      #print(image)
      channel = grpc.insecure_channel(FLAGS.server)
      stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
      # Send request
      # See prediction_service.proto for gRPC request/response details.
      
      request = predict_pb2.PredictRequest()
      request.model_spec.name = 'inception_v3'
      request.model_spec.signature_name = 'predict_images'
      request.inputs['image'].CopyFrom(
          #tf.contrib.util.make_tensor_proto(str(image), shape=[1]))
          tf.contrib.util.make_tensor_proto(image.astype(dtype=np.float32), shape=[1, 299, 299, 3])) #worked
          #tf.contrib.util.make_tensor_proto(image, dtype=tf.float32, shape=[1, 299, 299, 3]))
      start = time.time()
      print("send request")
      result = stub.Predict(request, 10)  # 10 secs timeout
      dur1 = time.time() - start
      print("Get Result time: %.6f" % dur1)
      #print(result.outputs['out'])
      
      dur2 = time.time() - start
      print("After Print Result time: %.6f" % dur2)



if __name__ == '__main__':
  tf.app.run()
