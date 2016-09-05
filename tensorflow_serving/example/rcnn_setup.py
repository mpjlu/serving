#!/usr/bin/python2.7
# Copyright 2016 IBM Corp. All Rights Reserved.

"""
Functions for setting up an rcnn export from a source directory layout.
Produces the following layout:

  [export_path]/
    lib/
    [model_version]
      deploy.prototxt
      weights.caffemodel
      classlabels.txt

"""
from __future__ import print_function

import argparse
import tarfile
import importlib
import sys
import os
import subprocess

from six.moves import urllib
from os.path import join

parser = argparse.ArgumentParser()
parser.add_argument("--version", type=int, default=1, help="model version")
parser.add_argument("--force", action='store_true', help="overwrite existing export path")
parser.add_argument("repo_path", help="location to the repository")
parser.add_argument("export_path", help="location to build the export")

VERSION_FORMAT_SPECIFIER = "%08d"
# class labels from "py-faster-rcnn/tools/demo.py"
CLASSES = (
  '__background__',
  'aeroplane', 'bicycle', 'bird', 'boat',
  'bottle', 'bus', 'car', 'cat', 'chair',
  'cow', 'diningtable', 'dog', 'horse',
  'motorbike', 'person', 'pottedplant',
  'sheep', 'sofa', 'train', 'tvmonitor'
)

def check(path):
  if not os.path.exists(path):
    raise IOError('{:s} not found.\nDid you run ./rcnn_fetch.py ?'.format(path))

def link(src, dst):
  print('%s => %s' % (dst, src))
  check(src)

  if not os.path.exists(dst):
    os.symlink(src, dst)

if __name__ == '__main__':
  args = parser.parse_args()
  export_dir_base = args.export_path
  repo_path = args.repo_path

  # 0. Build py-faster-rcnn cython extensions.
  src = join(repo_path, 'py-faster-rcnn/lib/setup.py')
  check(src)

  ret = subprocess.call([sys.executable, src, 'build_ext', '--inplace'],
    cwd=join(repo_path, 'py-faster-rcnn/lib'))

  if ret == 1:
    raise RuntimeError('Failed to build py-faster-rcnn')

  # 1. create TFS export path
  export_dir = join(export_dir_base, VERSION_FORMAT_SPECIFIER % args.version)
  if not os.path.exists(export_dir):
    os.makedirs(export_dir)
  elif not args.force:
    raise RuntimeError('Overwriting exports can cause corruption and are '
                       'not allowed. Duplicate export dir: %s' % export_dir)

  # 2. write the classlabels file
  print ('writing class labels')
  with open(join(export_dir, 'classlabels.txt'), 'w') as f:
    for _, cls in enumerate(CLASSES):
      f.write(cls + '\n')

  # 3. setup symlinks
  src = join(repo_path, 'py-faster-rcnn/data/faster_rcnn_models/VGG16_faster_rcnn_final.caffemodel')
  dst = join(export_dir, 'weights.caffemodel')

  link(src, dst)

  src = join(repo_path, 'py-faster-rcnn/models/pascal_voc/VGG16/faster_rcnn_alt_opt/faster_rcnn_test.pt')
  dst = join(export_dir, 'deploy.prototxt')

  link(src, dst)

  src = join(repo_path, 'py-faster-rcnn/lib')
  dst = join(export_dir_base, 'lib')

  link(src, dst)
