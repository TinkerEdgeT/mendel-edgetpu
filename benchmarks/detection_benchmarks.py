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

"""Benchmark for Detection Engine Python API."""

import time
import timeit

from edgetpu.detection.engine import DetectionEngine
from PIL import Image
import test_utils


def _RunBenchmarkForModel(model_name, image):
  """Benchmarks model with given image.

  Args:
    model_name: string, file name of the model.
    image: string, name of the image used for test.

  Returns:
    float, average inference time.
  """
  print('Benchmark for [', model_name, '] on ', image)
  engine = DetectionEngine(test_utils.TestDataPath(model_name))
  iterations = 200 if ('edgetpu' in model_name) else 10

  with Image.open(test_utils.TestDataPath(image)) as img_obj:
    benchmark_time = timeit.timeit(
        lambda: engine.DetectWithImage(img_obj, threshold=0.4, top_k=10),
        number=iterations)

  time_per_inference = (benchmark_time / iterations) * 1000
  return time_per_inference

if __name__ == '__main__':
  args = test_utils.ParseArgs()
  images_for_tests = ['cat.bmp', 'cat_720p.jpg', 'cat_1080p.jpg']
  machine = test_utils.MachineInfo()
  test_utils.CheckCpuScalingGovernorStatus()
  model_list, reference = test_utils.ReadReference(
      'detection_reference_%s.csv' % machine)
  total_models = len(model_list)
  results = [('MODEL', 'IMAGE_NAME', 'INFERENCE_TIME')]
  for cnt, model in enumerate(model_list, start=1):
    print('-------------- Model ', cnt, '/', total_models, ' ---------------')
    for img in images_for_tests:
      results.append((model, img, _RunBenchmarkForModel(model, img)))
  test_utils.SaveAsCsv('detection_benchmarks_%s_%s.csv' %
                       (machine, time.strftime('%Y%m%d-%H%M%S')), results)
  test_utils.CheckResult(reference, results, args.enable_assertion)
