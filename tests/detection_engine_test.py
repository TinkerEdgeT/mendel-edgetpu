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
from edgetpu.detection.engine import DetectionEngine
import numpy as np
from PIL import Image

from . import test_utils

def mobilenet_ssd_v1_coco_engine():
  return  DetectionEngine(
      test_utils.TestDataPath('mobilenet_ssd_v1_coco_quant_postprocess.tflite'))

class TestDetectionEnginePythonAPI(unittest.TestCase):

  def _TestCat(self, model_name):
    engine = DetectionEngine(test_utils.TestDataPath(model_name))
    with test_utils.TestImage('cat.bmp') as img:
      ret = engine.DetectWithImage(img, top_k=1)
      self.assertEqual(len(ret), 1)
      self.assertEqual(ret[0].label_id, 16)  # cat
      self.assertGreater(ret[0].score, 0.7)
      self.assertGreater(
          test_utils.IOU(
              np.array([[0.1, 0.1], [0.7, 1.0]]), ret[0].bounding_box), 0.88)
      # Check coordinates in pixels.
      ret = engine.DetectWithImage(img, top_k=1, relative_coord=False)
      self.assertGreater(
          test_utils.IOU(
              np.array([[60, 40], [420, 400]]), ret[0].bounding_box), 0.88)

  def _TestFace(self, model_name):
    engine = DetectionEngine(test_utils.TestDataPath(model_name))
    with test_utils.TestImage('face.jpg') as img:
      ret = engine.DetectWithImage(img, top_k=1)
      self.assertEqual(len(ret), 1)
      self.assertEqual(ret[0].label_id, 0)
      self.assertGreater(ret[0].score, 0.95)
      self.assertGreater(
          test_utils.IOU(
              np.array([[0.41, 0.07], [0.78, 0.49]]), ret[0].bounding_box), 0.9)

      ret = engine.DetectWithImage(img, top_k=1, relative_coord=False)
      # Check coordinates in pixels.
      self.assertGreater(
          test_utils.IOU(
              np.array([[427, 53], [801, 354]]), ret[0].bounding_box), 0.9)

  def _TestPet(self, model_name):
    engine = DetectionEngine(test_utils.TestDataPath(model_name))
    with test_utils.TestImage('cat.bmp') as img:
      ret = engine.DetectWithImage(img, top_k=1)
      self.assertEqual(len(ret), 1)
      self.assertEqual(ret[0].label_id, 0)  # Abyssinian
      self.assertGreater(ret[0].score, 0.9)
      self.assertGreater(
          test_utils.IOU(
              np.array([[0.35, 11], [0.7, 0.66]]), ret[0].bounding_box), 0.84)

      ret = engine.DetectWithImage(img, top_k=1, relative_coord=False)
      # Check coordinates in pixels.
      self.assertGreater(
          test_utils.IOU(
              np.array([[211, 47], [415, 264]]), ret[0].bounding_box), 0.84)

  def testImageObject(self):
    engine = mobilenet_ssd_v1_coco_engine()
    with test_utils.TestImage('cat.bmp') as img:
      ret = engine.DetectWithImage(img, top_k=1)
      self.assertEqual(len(ret), 1)
      self.assertEqual(ret[0].label_id, 16)  # cat
      self.assertGreater(ret[0].score, 0.79)
      self.assertGreater(
          test_utils.IOU(
              np.array([[0.1, 0.1], [0.7, 1.0]]), ret[0].bounding_box), 0.9)

      # Detect with different resample algorithm.
      ret = engine.DetectWithImage(
          img, top_k=1, resample=Image.BICUBIC)
      self.assertEqual(len(ret), 1)
      self.assertEqual(ret[0].label_id, 16)  # cat
      self.assertGreater(ret[0].score, 0.79)
      self.assertGreater(
          test_utils.IOU(
              np.array([[0.1, 0.1], [0.7, 1.0]]), ret[0].bounding_box), 0.91)

      # No error when top_k > number limit of detection candidates.
      engine.DetectWithImage(img, top_k=100000)

  def testImageObjectWithoutLabels(self):
    engine = mobilenet_ssd_v1_coco_engine()
    with test_utils.TestImage('cat.bmp') as img:
      ret = engine.DetectWithImage(img, top_k=1)
      self.assertEqual(len(ret), 1)
      self.assertEqual(ret[0].label_id, 16)  # cat
      self.assertGreater(ret[0].score, 0.79)
      self.assertGreater(
          test_utils.IOU(
              np.array([[0.1, 0.1], [0.7, 1.0]]), ret[0].bounding_box), 0.9)

  def testRawInput(self):
    engine = mobilenet_ssd_v1_coco_engine()
    with test_utils.TestImage('cat.bmp') as img:
      input_tensor = np.asarray(img.resize((300, 300), Image.NEAREST)).flatten()
      ret = engine.DetectWithInputTensor(input_tensor, top_k=1)
      self.assertEqual(len(ret), 1)
      self.assertEqual(ret[0].label_id, 16)  # cat
      self.assertGreater(ret[0].score, 0.79)
      self.assertGreater(
          test_utils.IOU(
              np.array([[0.1, 0.1], [0.7, 1.0]]), ret[0].bounding_box), 0.9)

  def testVariousModelsWithCat(self):
    for model in ['mobilenet_ssd_v1_coco_quant_postprocess.tflite',
                  'mobilenet_ssd_v1_coco_quant_postprocess_edgetpu.tflite',
                  'mobilenet_ssd_v2_coco_quant_postprocess.tflite',
                  'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite']:
      self._TestCat(model)

  def testFaceDetection(self):
    for model in ['mobilenet_ssd_v2_face_quant_postprocess.tflite',
                  'mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite']:
      self._TestFace(model)

  def testPetDetection(self):
    for model in ['ssd_mobilenet_v1_fine_tuned.tflite',
                  'ssd_mobilenet_v1_fine_tuned_edgetpu.tflite']:
      self._TestPet(model)


if __name__ == '__main__':
  unittest.main()
