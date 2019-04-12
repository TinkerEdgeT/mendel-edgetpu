# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Test utils for benchmark and manual tests."""

import collections
import contextlib
import csv
import os
import platform
import random
import urllib.parse

import numpy as np
from PIL import Image

TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             '..', 'test_data')

def TestDataPath(path, *paths):
  """Returns absolute path for a given test file."""
  return os.path.abspath(os.path.join(TEST_DATA_DIR, path, *paths))


@contextlib.contextmanager
def TestImage(path, *paths):
  """Returns opened test image."""
  with open(TestDataPath(path, *paths), 'rb') as f:
    with Image.open(f) as image:
      yield image


def GetModelList():
  """Gets all tflite models for benchmark."""
  return (f for f in os.listdir(TEST_DATA_DIR) if f.endswith('.tflite'))


def GenerateRandomInput(seed, n):
  """Generates a list with n uint8 numbers."""
  random.seed(a=seed)
  return [random.randint(0, 255) for _ in range(n)]


def PrepareImages(image_list, directory, shape):
  """Reads images and converts them to numpy array with specified shape.

  Args:
    image_list: a list of strings storing file names.
    directory: string, path of directory storing input images.
    shape: a 2-D tuple represents the shape of required input tensor.
  Returns:
    A list of numpy.array.
  """
  ret = []
  for filename in image_list:
    with  open(os.path.join(directory, filename), 'rb') as f:
      with Image.open(f) as img:
        img = img.resize(shape, Image.NEAREST)
        ret.append(np.asarray(img).flatten())
  return np.array(ret)


def ReadLabelFile(file_path):
  """Reads labels from txt file.

  Each line contains a pair of id and description such as:
    1 cat
    2 dog
    ...

  Args:
    file_path: string, path to the labels file.
  Returns:
    {int : string}, a dict maps label id to label.
  """
  with open(file_path, 'r') as f:
    lines = f.readlines()
  ret = {}
  for line in lines:
    pair = line.strip().split(maxsplit=1)
    ret[int(pair[0])] = pair[1].strip()
  return ret


def Area(box):
  """Calculates area of a given bounding box."""
  return float((box[1][0]-box[0][0]) * (box[1][1] - box[0][1]))


def IOU(box_a, box_b):
  """Calculates intersection area / union area for two bounding boxes."""
  intersect = np.array(
      [[max(box_a[0][0], box_b[0][0]), max(box_a[0][1], box_b[0][1])],
       [min(box_a[1][0], box_b[1][0]), min(box_a[1][1], box_b[1][1])]])
  return Area(intersect) / (Area(box_a) + Area(box_b) - Area(intersect))


def PrepareDataSetFromDirectory(path, test_ratio, fixed_order):
  """Parses data set from given directory, split them into train/test sets.

  Args:
    path: string, path of the data set. Images are stored in sub-directory
      named by category.
    test_ratio: float in (0,1), ratio of data used for testing.
    fixed_order: bool, whether to spilt data set in fixed order.

  Returns:
    (train_set, test_set), A tuple of two OrderedDicts. Keys are the categories
    and values are lists of image file names.
  """
  train_set = collections.OrderedDict()
  test_set = collections.OrderedDict()
  sub_dirs = os.listdir(path)
  if fixed_order:
    sub_dirs.sort()
  for category in sub_dirs:
    category_dir = os.path.join(path, category)
    if os.path.isdir(category_dir):
      images = [f for f in os.listdir(category_dir)
                if os.path.isfile(os.path.join(category_dir, f))]
      if images:
        if fixed_order:
          images.sort()
        k = int(test_ratio * len(images))
        test_set[category] = images[:k]
        assert test_set[category], 'No images to test [{}]'.format(category)
        train_set[category] = images[k:]
        assert train_set[category], 'No images to train [{}]'.format(category)
  return train_set, test_set
