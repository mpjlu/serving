# Copyright 2016 IBM Corp. All Rights Reserved.
#!/usr/bin/python2.7

"""Functions for downloading and extracting pretrained py-faster-rcnn caffe models
   and client utils."""
from __future__ import print_function

import argparse
import tarfile
import os

from six.moves import urllib

REPO_NAME = 'py-faster-rcnn'
COMMIT_SHA1 = 'd14cb16b78816cc5ab0f10283381cf2ff3c6a1af'
SOURCE_URL = 'https://github.com/Austriker/%s/archive/%s.tar.gz' % (REPO_NAME, COMMIT_SHA1)

DEMO_MODEL_URL = 'http://www.cs.berkeley.edu/~rbg/faster-rcnn-data/faster_rcnn_models.tgz'
DEMO_MODEL_FILE = 'faster_rcnn_models.tgz'

parser = argparse.ArgumentParser()
parser.add_argument("dl_path",
  help="location to download the repository")
parser.add_argument("--fetch-demo-models",
  dest="fetch_demo_models", action='store_true', help="download faster-rcnn demo models")

def maybe_download(url, filename, work_directory):
  if not os.path.exists(work_directory):
    os.mkdir(work_directory)

  filepath = os.path.join(work_directory, filename)
  if not os.path.exists(filepath):
    filepath, _ = urllib.request.urlretrieve(url, filepath)
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')

  return filepath

def untar(src, dst):
  print('Extracting "%s" => "%s"' % (src, dst))
  with tarfile.open(src) as tar:
      tar.extractall(path=dst)

if __name__ == '__main__':
  args = parser.parse_args()
  export_dir = args.dl_path

  if not os.path.exists(export_dir):
    os.makedirs(export_dir)

  # 1. Download py-faster-rcnn repository (as a tar.gz)
  print('Downloading repository...', SOURCE_URL)
  filename = maybe_download(SOURCE_URL, '%s.tar.gz' % REPO_NAME, export_dir)
  untar(filename, export_dir)

  src = os.path.join(export_dir, '%s-%s/' % (REPO_NAME, COMMIT_SHA1))
  dst = os.path.join(export_dir, REPO_NAME)
  print('Linking "%s" => "%s"' % (dst, src))

  if os.path.lexists(dst):
    os.unlink(dst)

  # 2. standardize the directory structure slightly
  os.symlink(src, dst)

  # 3. Download demo models
  if args.fetch_demo_models:
    print('Downloading Faster R-CNN demo models (695M)...')
    filename = maybe_download(DEMO_MODEL_URL, DEMO_MODEL_FILE, export_dir)
    dst = os.path.join(export_dir, REPO_NAME, 'data')
    untar(filename, dst)