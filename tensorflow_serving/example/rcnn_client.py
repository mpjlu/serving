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

"""A client that talks to rcnn_detector service. Please see
rcnn_detector.proto for service API details.

Typical usage example:
    rcnn_client.py --server=localhost:9000 --gui --img=/path/to/image.jpg
"""

import sys
import threading
from timeit import default_timer as timer

from grpc.beta import implementations
from grpc.framework.interfaces.face.face import AbortionError

import numpy
import tensorflow as tf

from tensorflow_serving.example import rcnn_detector_pb2
import skimage.io as io
import skimage.transform as transform
import matplotlib.pyplot as plt

tf.app.flags.DEFINE_string('server', '', 'mnist_inference service host:port')
tf.app.flags.DEFINE_bool('gui', False, 'show detections in a gui')
tf.app.flags.DEFINE_string('img', 'https://upload.wikimedia.org/wikipedia/commons/4/4f/G8_Summit_working_session_on_global_and_economic_issues_May_19%2C_2012.jpg',
                           'url or path of an image to classify')

FLAGS = tf.app.flags.FLAGS

def connect(hostport):
  """
  Connect to the inference server
  """
  host, port = hostport.split(':')
  channel = implementations.insecure_channel(host, int(port))
  return rcnn_detector_pb2.beta_create_DetectService_stub(channel)

def handshake(stub):
  """
  No handshake required in this simple example; a more
  complete implementation may retrieve service-specific
  cofigurations such as input-image shape, number of
  channels etc.
  """
  return (600, 800, 3)

def im_transpose(im):
  """
  transpose input image to shape [c = 3, h, w] and BGR channel order.
  """
  R, G, B = im.transpose((2, 0, 1))
  return numpy.array((B,G,R), dtype=numpy.uint8)

def im_scale_to_fit(im, out_shape):
  """
  rescale an input image of arbitary dimensions to fit
  within out_shape, maintaining aspect ratio and not
  permitting cropping.

  Returns the rescaled image
  """
  h, w, c = im.shape
  out_h, out_w, _ = out_shape

  scale = min(float(out_h) / h, float(out_w) / w)
  im2 = transform.resize(im, numpy.floor((h * scale, w * scale)),
                         preserve_range=True)

  result = numpy.zeros([out_h, out_w, c], dtype=numpy.uint8)
  result[0:im2.shape[0], 0:im2.shape[1], ...] = im2
  return result

def do_inference(stub, im):
  cv = threading.Condition()
  result = []

  def done(result_future):
    with cv:
      try:
        res = result_future.result()
        result.append(res.detections)
      except AbortionError as e:
        print ("An RPC error occured: %s" % e)

      cv.notify()

  im_input = im_transpose(im)
  req = rcnn_detector_pb2.DetectRequest(image_data=bytes(im_input.data))

  result_future = stub.Detect.future(req, 30)
  result_future.add_done_callback(
      lambda result_future: done(result_future))

  with cv:
    cv.wait()

  return result[0]

def vis_detections(im, dets):
    """
    Draw image and detected bounding boxes.
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for det in dets:
      ax.text(det.roi_x1+3, det.roi_y1-3, "{:s} ({:.3f})".format(det.class_label, det.score),
              color='black', backgroundcolor='white',
              verticalalignment='bottom')

      ax.add_patch(
          plt.Rectangle((det.roi_x1, det.roi_y1),
                        det.roi_x2 - det.roi_x1,
                        det.roi_y2 - det.roi_y1,
                        fill=False, edgecolor='white',
                        linewidth=2))
    plt.tight_layout()
    plt.draw()

def main(_):
  if not FLAGS.server:
    print 'please specify server host:port'
    return

  stub = connect(FLAGS.server)
  input_shape = handshake(stub)

  im = im_scale_to_fit(io.imread(FLAGS.img), input_shape)
  result = do_inference(stub, im)

  for det in result:
     print(' {:s}\n  > score: {:.3f}\n  > bbox (x1, y1, x2, y2): ({:d}, {:d}, {:d}, {:d})\n'.format(
           det.class_label, det.score, det.roi_x1, det.roi_y1, det.roi_x2, det.roi_y2))

  if FLAGS.gui:
    vis_detections(im, result)
    plt.show()

if __name__ == '__main__':
  tf.app.run()
