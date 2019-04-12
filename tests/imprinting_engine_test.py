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

import os
import tempfile
import unittest
from . import test_utils
from edgetpu.classification.engine import ClassificationEngine
from edgetpu.learn.imprinting.engine import ImprintingEngine
from PIL import Image


class TestImprintingEnginePythonAPI(unittest.TestCase):

  _EXTRACTOR_LIST = [
      'imprinting/mobilenet_v1_1.0_224_quant_embedding_extractor.tflite',
      'imprinting/mobilenet_v1_1.0_224_quant_embedding_extractor_edgetpu.tflite'
  ]

  def _ClassifyImage(self, engine, data_dir, image_name, label_id, score):
    with Image.open(os.path.join(data_dir, image_name)) as img:
      ret = engine.ClassifyWithImage(img, top_k=1)
      self.assertEqual(len(ret), 1)
      self.assertEqual(ret[0][0], label_id)
      self.assertGreater(ret[0][1], score)

  def testTrainingFromScratch(self):
    for extractor in self._EXTRACTOR_LIST:
      with self.subTest():
        with tempfile.NamedTemporaryFile(suffix='.tflite') as output_model_path:
          # Train.
          engine = ImprintingEngine(test_utils.TestDataPath(extractor))
          shape = (224, 224)
          train_set = {
              'cat': ['cat_train_0.bmp'],
              'dog': ['dog_train_0.bmp'],
              'hot_dog': ['hotdog_train_0.bmp', 'hotdog_train_1.bmp']
          }
          data_dir = test_utils.TestDataPath('imprinting')
          self.assertEqual(engine.Train(
              test_utils.PrepareImages(train_set['cat'], data_dir, shape)), 0)
          self.assertEqual(engine.Train(
              test_utils.PrepareImages(train_set['dog'], data_dir, shape)), 1)
          self.assertEqual(engine.Train(
              test_utils.PrepareImages(train_set['hot_dog'], data_dir, shape)), 2)
          engine.SaveModel(output_model_path.name)

          # Test.
          engine = ClassificationEngine(output_model_path.name)
          self.assertEqual(1, engine.get_num_of_output_tensors())
          self.assertEqual(3, engine.get_output_tensor_size(0))

          self._ClassifyImage(engine, data_dir, 'cat_test_0.bmp', 0, 0.38)
          self._ClassifyImage(engine, data_dir, 'dog_test_0.bmp', 1, 0.38)
          self._ClassifyImage(engine, data_dir, 'hotdog_test_0.bmp', 2, 0.38)

  def testIncrementalTraining(self):
    for extractor in [
        'imprinting/retrained_mobilenet_v1_cat_only.tflite',
        'imprinting/retrained_mobilenet_v1_cat_only_edgetpu.tflite']:
      with self.subTest():
        with tempfile.NamedTemporaryFile(suffix='.tflite') as output_model_path:
          # Train.
          engine = ImprintingEngine(test_utils.TestDataPath(extractor))
          shape = (224, 224)
          train_set = {
              'dog': ['dog_train_0.bmp'],
              'hot_dog': ['hotdog_train_0.bmp', 'hotdog_train_1.bmp']
          }
          data_dir = test_utils.TestDataPath('imprinting')
          self.assertEqual(engine.Train(
              test_utils.PrepareImages(train_set['dog'], data_dir, shape)), 1)
          self.assertEqual(engine.Train(
              test_utils.PrepareImages(train_set['hot_dog'], data_dir, shape)), 2)
          engine.SaveModel(output_model_path.name)

          # Test.
          engine = ClassificationEngine(output_model_path.name)
          self.assertEqual(1, engine.get_num_of_output_tensors())
          self.assertEqual(3, engine.get_output_tensor_size(0))

          self._ClassifyImage(engine, data_dir, 'cat_test_0.bmp', 0, 0.38)
          self._ClassifyImage(engine, data_dir, 'dog_test_0.bmp', 1, 0.38)
          self._ClassifyImage(engine, data_dir, 'hotdog_test_0.bmp', 2, 0.38)

  def testTrainAll(self):
    for extractor in self._EXTRACTOR_LIST:
      with self.subTest():
        with tempfile.NamedTemporaryFile(suffix='.tflite') as output_model_path:
          data_dir = test_utils.TestDataPath('imprinting')
          engine = ImprintingEngine(test_utils.TestDataPath(extractor))

          # Train.
          shape = (224, 224)
          train_set = {
              'cat': ['cat_train_0.bmp'],
              'dog': ['dog_train_0.bmp'],
              'hot_dog': ['hotdog_train_0.bmp', 'hotdog_train_1.bmp']
          }
          train_input = {}
          for category, image_list in train_set.items():
            train_input[category] = test_utils.PrepareImages(
                image_list, data_dir, shape)
          id_to_label_map = engine.TrainAll(train_input)
          label_to_id_map = {v: k for k, v in id_to_label_map.items()}
          engine.SaveModel(output_model_path.name)

          # Test.
          engine = ClassificationEngine(output_model_path.name)
          self.assertEqual(1, engine.get_num_of_output_tensors())
          self.assertEqual(3, engine.get_output_tensor_size(0))

          self._ClassifyImage(
              engine, data_dir, 'cat_test_0.bmp', label_to_id_map['cat'], 0.38)
          self._ClassifyImage(
              engine, data_dir, 'dog_test_0.bmp', label_to_id_map['dog'], 0.38)
          self._ClassifyImage(
              engine, data_dir, 'hotdog_test_0.bmp', label_to_id_map['hot_dog'],
              0.38)


if __name__ == '__main__':
  unittest.main()
