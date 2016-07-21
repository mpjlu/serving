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

#!/usr/bin/env python2.7

"""A client that talks to mnist_inference service.

The client downloads test images of mnist data set, queries the service with
such test images to get classification, and calculates the inference error rate.
Please see mnist_inference.proto for details.

Typical usage example:

    mnist_client.py --num_tests=100 --server=localhost:9000
"""

import sys
import threading
from timeit import default_timer as timer

# This is a placeholder for a Google-internal import.

from grpc.beta import implementations
import numpy
import tensorflow as tf

from tensorflow_serving.example import mnist_inference_pb2
from tensorflow_serving.example import mnist_input_data


tf.app.flags.DEFINE_integer('concurrency', 1,
                            'maximum number of concurrent inference requests')
tf.app.flags.DEFINE_integer('num_tests', 100, 'Number of test images')
tf.app.flags.DEFINE_string('server', '', 'mnist_inference service host:port')
tf.app.flags.DEFINE_string('work_dir', '/tmp', 'Working directory. ')
FLAGS = tf.app.flags.FLAGS


class InferenceStats(object):
  """Statistics useful for evaluating basic classification and
     runtime performance"""

  @staticmethod
  def print_summary(stats, percentiles=[50, 90, 99]):
    filtered = numpy.ma.masked_invalid(stats.timings).compressed() # remove NaNs

    print '\nInference error rate: %s%%' % (
        stats.classification_error * 100)

    print "Request error rate: %s%%" % (
        (1.0 - float(filtered.size) / stats.timings.size) * 100)

    print "Avg. Throughput: %s reqs/s" % (
        float(stats.num_tests) / stats.total_elapsed_time)

    if filtered.size > 0:
      print "Request Latency (percentiles):"
      for pc, x in zip(percentiles, numpy.percentile(filtered, percentiles)):
        print "  %ith ....... %ims" % (pc, x * 1000.0)

  def __init__(self, num_tests, classification_error,
               timings, total_elapsed_time):
    assert num_tests == timings.size
    self.num_tests = num_tests
    self.classification_error = classification_error
    self.timings = timings
    self.total_elapsed_time = total_elapsed_time


def do_inference(hostport, work_dir, concurrency, num_tests):
  """Tests mnist_inference service with concurrent requests.

  Args:
    hostport: Host:port address of the mnist_inference service.
    work_dir: The full path of working directory for test data set.
    concurrency: Maximum number of concurrent requests.
    num_tests: Number of test images to use.

  Returns:
    An instance of InferenceStats

  Raises:
    IOError: An error occurred processing test data set.
  """
  test_data_set = mnist_input_data.read_data_sets(work_dir).test
  host, port = hostport.split(':')
  channel = implementations.insecure_channel(host, int(port))
  stub = mnist_inference_pb2.beta_create_MnistService_stub(channel)
  cv = threading.Condition()
  result = {'active': 0, 'error': 0, 'done': 0}
  result_timing = numpy.zeros(num_tests, dtype=numpy.float64);
  def done(reqid, result_future, label):
    with cv:
      # Workaround for gRPC issue https://github.com/grpc/grpc/issues/7133
      try:
        exception = result_future.exception()
      except AttributeError:
        exception = None
      if exception:
        result_timing[reqid] = numpy.NaN  # ignore when evaluating time statistics
        result['error'] += 1
        print exception
      else:
        result_timing[reqid] = timer() - result_timing[reqid]
        sys.stdout.write('.')
        sys.stdout.flush()
        response = numpy.array(result_future.result().value)
        prediction = numpy.argmax(response)
        if label != prediction:
          result['error'] += 1
      result['done'] += 1
      result['active'] -= 1
      cv.notify()
  start_time = timer()
  for n in range(num_tests):
    request = mnist_inference_pb2.MnistRequest()
    image, label = test_data_set.next_batch(1)
    for pixel in image[0]:
      request.image_data.append(pixel.item())
    with cv:
      while result['active'] == concurrency:
        cv.wait()
      result['active'] += 1
    result_timing[n] = timer()
    result_future = stub.Classify.future(request, 5.0)  # 5 seconds
    result_future.add_done_callback(
        lambda result_future, l=label[0], n=n: done(n, result_future, l))  # pylint: disable=cell-var-from-loop
  with cv:
    while result['done'] != num_tests:
      cv.wait()

    return InferenceStats(num_tests,
      result['error'] / float(num_tests),
      result_timing,
      timer() - start_time)


def main(_):
  if FLAGS.num_tests > 10000:
    print 'num_tests should not be greater than 10k'
    return
  if not FLAGS.server:
    print 'please specify server host:port'
    return
  stats = do_inference(FLAGS.server, FLAGS.work_dir,
                       FLAGS.concurrency, FLAGS.num_tests)
  InferenceStats.print_summary(stats)


if __name__ == '__main__':
  tf.app.run()
