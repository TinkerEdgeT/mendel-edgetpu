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

import threading
import unittest

from . import test_utils
from edgetpu.basic import edgetpu_utils
from edgetpu.classification.engine import BasicEngine
from edgetpu.classification.engine import ClassificationEngine
from edgetpu.detection.engine import DetectionEngine
import numpy as np


class MultipleTpusTest(unittest.TestCase):

  def testCreateBasicEngineWithSpecificPath(self):
    edge_tpus = edgetpu_utils.ListEdgeTpuPaths(
        edgetpu_utils.EDGE_TPU_STATE_UNASSIGNED)
    self.assertGreater(len(edge_tpus), 0)
    model_path = test_utils.TestDataPath(
        'mobilenet_v1_1.0_224_quant_edgetpu.tflite')
    basic_engine = BasicEngine(model_path, edge_tpus[0])
    self.assertEqual(edge_tpus[0], basic_engine.device_path())

  def testRunClassificationAndDetectionEngine(self):
    def classification_task(num_inferences):
      tid = threading.get_ident()
      print('Thread: %d, %d inferences for classification task' %
            (tid, num_inferences))
      labels = test_utils.ReadLabelFile(
          test_utils.TestDataPath('imagenet_labels.txt'))
      model_name = 'mobilenet_v1_1.0_224_quant_edgetpu.tflite'
      engine = ClassificationEngine(test_utils.TestDataPath(model_name))
      print('Thread: %d, using device %s' % (tid, engine.device_path()))
      with test_utils.TestImage('cat.bmp') as img:
        for _ in range(num_inferences):
          ret = engine.ClassifyWithImage(img, top_k=1)
          self.assertEqual(len(ret), 1)
          self.assertEqual(labels[ret[0][0]], 'Egyptian cat')
      print('Thread: %d, done classification task' % tid)

    def detection_task(num_inferences):
      tid = threading.get_ident()
      print('Thread: %d, %d inferences for detection task' %
            (tid, num_inferences))
      model_name = 'mobilenet_ssd_v1_coco_quant_postprocess_edgetpu.tflite'
      engine = DetectionEngine(test_utils.TestDataPath(model_name))
      print('Thread: %d, using device %s' % (tid, engine.device_path()))
      with test_utils.TestImage('cat.bmp') as img:
        for _ in range(num_inferences):
          ret = engine.DetectWithImage(img, top_k=1)
          self.assertEqual(len(ret), 1)
          self.assertEqual(ret[0].label_id, 16)  # cat
          self.assertGreater(ret[0].score, 0.7)
          self.assertGreater(
              test_utils.IOU(
                  np.array([[0.1, 0.1], [0.7, 1.0]]), ret[0].bounding_box),
              0.88)
      print('Thread: %d, done detection task' % tid)

    num_inferences = 10
    t1 = threading.Thread(target=classification_task, args=(num_inferences,))
    t2 = threading.Thread(target=detection_task, args=(num_inferences,))

    t1.start()
    t2.start()

    t1.join()
    t2.join()


if __name__ == '__main__':
  unittest.main()
