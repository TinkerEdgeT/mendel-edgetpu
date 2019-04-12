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

"""Evaluates the accuracy of imprinting based transfer learning model."""

import collections
import contextlib
import os
import unittest

from edgetpu.basic.basic_engine import BasicEngine
from edgetpu.classification.engine import ClassificationEngine
from edgetpu.learn.imprinting.engine import ImprintingEngine
from PIL import Image

from . import test_utils

@contextlib.contextmanager
def TestImage(path):
  with open(path, 'rb') as f:
    with Image.open(f) as image:
      yield image

class ImprintingEngineEvaluationTest(unittest.TestCase):

  def _GetInputTensorShape(self, model_path):
    """Gets input tensor shape of given model.

    Args:
      model_path: string, path of the model.

    Returns:
      List of integers.
    """
    tmp = BasicEngine(model_path)
    shape = tmp.get_input_tensor_shape()
    return shape.copy()

  def _TransferLearnAndEvaluate(self, extractor_path, dataset_path,
                                test_ratio, top_k_range):
    """Transfer-learns with given params and returns the evaluatoin result.

    Args:
      extractor_path: string, path of embedding extractor.
      dataset_path: string, path to the directory of dataset. The images
        should be put under sub-directory named by category.
      test_ratio: float, the ratio of images used for test.
      top_k_range: int, top_k range to be evaluated. The function will return
        accuracy from top 1 to top k.

    Returns:
      list of float numbers.
    """
    print('---------------      Parsing dataset      ----------------')
    print('Dataset path:', dataset_path)

    # Train in fixed order, this is because many test images get same score for
    # several categories. If we don't fix the order, the evaluation result will
    # be slightly different.
    #
    # For example, a real daffodil image is classified as:
    # 1.pansy(0.0625) 2.daffodil(0.0625) 3.crocus(0.0625) 4.iris(0.0625) ...
    #
    # The order of them depends on the training order. Hence if we train
    # daffodil first, we'll get:
    # 1.daffodil(0.0625) 2.pansy(0.0625) ..
    #
    # The top 1 accuracy increased while top 2 accuracy decreased.
    train_set, test_set = test_utils.PrepareDataSetFromDirectory(
        dataset_path, test_ratio, True)

    print('Image list successfully parsed! Number of Categories = ',
          len(train_set))
    input_shape = self._GetInputTensorShape(extractor_path)
    required_image_shape = (input_shape[2], input_shape[1])  # (width, height)
    print('---------------  Processing training data ----------------')
    print('This process may take more than 30 seconds.')
    train_input = collections.OrderedDict()
    for category, image_list in train_set.items():
      print('Processing {} ({} images)'.format(category, len(image_list)))
      train_input[category] = test_utils.PrepareImages(
          image_list,
          os.path.join(dataset_path, category),
          required_image_shape
      )

    # Train
    print('----------------      Start training     -----------------')
    imprinting_engine = ImprintingEngine(extractor_path)
    labels_map = imprinting_engine.TrainAll(train_input)
    print('----------------     Training finished   -----------------')
    imprinting_engine.SaveModel('model_for_evaluation.tflite')

    # Evaluate
    print('----------------     Start evaluating    -----------------')
    classification_engine = ClassificationEngine('model_for_evaluation.tflite')
    # top[i] represents number of top (i+1) correct inference.
    top_k_correct_count = [0] * top_k_range
    image_num = 0
    for category, image_list in test_set.items():
      n = len(image_list)
      print('Evaluating {} ({} images)'.format(category, n))
      for image_name in image_list:
        with TestImage(os.path.join(dataset_path, category, image_name)) as raw_image:
          # Set threshold as a negative number to ensure we get top k candidates
          # even if its score is 0.
          candidates = classification_engine.ClassifyWithImage(
              raw_image, threshold=-0.1, top_k=top_k_range)
          for i in range(len(candidates)):
            if labels_map[candidates[i][0]] == category:
              top_k_correct_count[i] += 1
              break
      image_num += n
    for i in range(1, top_k_range):
      top_k_correct_count[i] += top_k_correct_count[i-1]

    return [top_k_correct_count[i] / image_num for i in range(top_k_range)]

  def testOxford17Flowers(self):
    expected = [0.79, 0.89, 0.92, 0.93, 0.95]
    top_k_range = len(expected)

    ret = self._TransferLearnAndEvaluate(
        test_utils.TestDataPath(
            'imprinting/mobilenet_v1_1.0_224_quant_embedding_extractor.tflite'),
        test_utils.TestDataPath('oxford_17flowers'),
        0.25,
        top_k_range
    )
    for i in range(top_k_range):
      self.assertGreaterEqual(ret[i], expected[i])

if __name__ == '__main__':
  unittest.main()
