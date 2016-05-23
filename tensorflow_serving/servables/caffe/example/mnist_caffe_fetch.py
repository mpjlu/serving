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

#!/usr/bin/python2.7

"""Functions for downloading and extracting pretrained MNIST caffe models."""
from __future__ import print_function

import tarfile
import os

from six.moves import urllib

SOURCE_URL = 'https://ibm.box.com/shared/static/gq1nqilju1zd8rjjkhdesvp6lh7f9udd.gz'
OUT_FILE = 'mnist_pretrained_caffe.gz'
DEST = '/tmp/mnist_caffe_fetch'

def maybe_download(url, filename, work_directory):
  """Download the data"""
  if not os.path.exists(work_directory):
    os.mkdir(work_directory)

  filepath = os.path.join(work_directory, filename)
  if not os.path.exists(filepath):
    filepath, _ = urllib.request.urlretrieve(SOURCE_URL, filepath)
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  
  return filepath

if __name__ == '__main__':
  print('Downloading...', SOURCE_URL)
  filename = maybe_download(SOURCE_URL, OUT_FILE, DEST)

  print('Extracting "%s" to "%s"' % (filename, DEST))
  with tarfile.open(filename) as tar:
    tar.extractall(path=DEST)
