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

import unittest
from edgetpu.classification.engine import ClassificationEngine
import numpy as np
from PIL import Image

from . import test_utils

def mobilenet_v1_engine():
  return ClassificationEngine(
      test_utils.TestDataPath('mobilenet_v1_1.0_224_quant_edgetpu.tflite'))

class TestClassificationEnginePythonAPI(unittest.TestCase):

  def _TestClassifyCat(self, model_name, expected):
    labels = test_utils.ReadLabelFile(test_utils.TestDataPath('imagenet_labels.txt'))
    engine = ClassificationEngine(test_utils.TestDataPath(model_name))
    with test_utils.TestImage('cat.bmp') as img:
      ret = engine.ClassifyWithImage(img, top_k=1)
      self.assertEqual(len(ret), 1)
      # Some models recogize it as egyptian cat while others recognize it as
      # tabby cat.
      self.assertTrue(labels[ret[0][0]] == 'tabby, tabby cat' or
                      labels[ret[0][0]] == 'Egyptian cat')
      ret = engine.ClassifyWithImage(img, top_k=3)
      self.assertEqual(len(expected), len(ret))
      for i in range(len(expected)):
        # Check label.
        self.assertEqual(labels[ret[i][0]], expected[i][0])
        # Check score.
        self.assertGreater(ret[i][1], expected[i][1])

  def testRandomInput(self):
    engine = mobilenet_v1_engine()
    random_input = test_utils.GenerateRandomInput(1, 224 * 224 * 3)
    ret = engine.ClassifyWithInputTensor(random_input, top_k=1)
    self.assertEqual(len(ret), 1)
    ret = engine.ClassifyWithInputTensor(random_input, threshold=1.0)
    self.assertEqual(len(ret), 0)

  def testTopK(self):
    engine = mobilenet_v1_engine()
    random_input = test_utils.GenerateRandomInput(1, 224 * 224 * 3)
    engine.ClassifyWithInputTensor(random_input, top_k=1)
    # top_k = number of categories
    engine.ClassifyWithInputTensor(random_input, top_k=1001)
    # top_k > number of categories
    engine.ClassifyWithInputTensor(random_input, top_k=1234)

  def testImageObject(self):
    engine = mobilenet_v1_engine()
    with test_utils.TestImage('cat.bmp') as img:
      ret = engine.ClassifyWithImage(img, threshold=0.4, top_k=10)
      self.assertEqual(len(ret), 1)
      self.assertEqual(ret[0][0], 286)  # Egyptian cat
      self.assertGreater(ret[0][1], 0.79)
      # Try with another resizing method.
      ret = engine.ClassifyWithImage(
          img, threshold=0.4, top_k=10, resample=Image.BICUBIC)
      self.assertEqual(len(ret), 1)
      self.assertEqual(ret[0][0], 286)  # Egyptian cat
      self.assertGreater(ret[0][1], 0.83)

  def testRawInput(self):
    engine = mobilenet_v1_engine()
    with test_utils.TestImage('cat.bmp') as img:
      img = img.resize((224, 224), Image.NEAREST)
      input_tensor = np.asarray(img).flatten()
      ret = engine.ClassifyWithInputTensor(input_tensor, threshold=0.4, top_k=10)
      self.assertEqual(len(ret), 1)
      self.assertEqual(ret[0][0], 286)  # Egyptian cat
      self.assertGreater(ret[0][1], 0.79)

  def testGetRawOuput(self):
    engine = mobilenet_v1_engine()
    with test_utils.TestImage('cat.bmp') as img:
      engine.ClassifyWithImage(img, top_k=3)
    raw_output = engine.get_raw_output()
    self.assertGreater(raw_output[282], 0.05)  # tabby, tabby cat
    self.assertGreater(raw_output[283], 0.12)  # tiger cat
    self.assertGreater(raw_output[286], 0.79)  # Egyptian cat

  def testVariousModels(self):
    # Mobilenet V1
    self._TestClassifyCat(
        'mobilenet_v1_1.0_224_quant_edgetpu.tflite',
        [('Egyptian cat', 0.78), ('tiger cat', 0.128)]
    )
    # Mobilenet V2
    self._TestClassifyCat(
        'mobilenet_v2_1.0_224_quant_edgetpu.tflite',
        [('Egyptian cat', 0.84)]
    )
    # Inception V1.
    self._TestClassifyCat(
        'inception_v1_224_quant_edgetpu.tflite',
        [('tabby, tabby cat', 0.41),
         ('Egyptian cat', 0.35),
         ('tiger cat', 0.156)]
    )
    # Inception V2.
    self._TestClassifyCat(
        'inception_v2_224_quant_edgetpu.tflite',
        [('Egyptian cat', 0.85)]
    )
    # Inception V3.
    self._TestClassifyCat(
        'inception_v3_299_quant_edgetpu.tflite',
        [('tabby, tabby cat', 0.45),
         ('Egyptian cat', 0.2),
         ('tiger cat', 0.15)]
    )
    # Inception V4.
    self._TestClassifyCat(
        'inception_v4_299_quant_edgetpu.tflite',
        [('Egyptian cat', 0.45),
         ('tabby, tabby cat', 0.3),
         ('tiger cat', 0.15)]
    )

if __name__ == '__main__':
  unittest.main()
