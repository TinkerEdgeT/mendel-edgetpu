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

"""Benchmark of models.

Benchmark are measured with CPU 'performance' mode. To enable it, you need to
install 'cpupower' and run:
sudo cpupower frequency-set --governor performance

The reference number is measured on:
  - 'x86_64': Intel Xeon E5-1650 v3(3.50GHz) + EdgeTPU accelarator + USB 3.0
  - 'rp3b': Raspberry Pi 3 B (version1.2)+ EdgeTPU accelarator + USB 2.0
  - 'rp3b+': Raspberry Pi 3 B+ (version1.3)+ EdgeTPU accelarator + USB 2.0
  - 'aarch64': EdgeTPU dev board.
"""

import time
import timeit

from edgetpu.basic.basic_engine import BasicEngine
import numpy as np
import test_utils


def _RunBenchmarkForModel(model_name):
  """Runs benchmark for given model with a random input.

  Args:
    model_name: string, file name of the model.

  Returns:
    float, average inference time.
  """
  iterations = 200 if ('edgetpu' in model_name) else 20
  print('Benchmark for [', model_name, ']')
  print('model path = ', test_utils.TestDataPath(model_name))
  engine = BasicEngine(test_utils.TestDataPath(model_name))
  print('Shape of input tensor : ', engine.get_input_tensor_shape())

  # Prepare a random generated input.
  input_size = engine.required_input_array_size()
  random_input = test_utils.GenerateRandomInput(1, input_size)

  # Convert it to a numpy.array.
  input_data = np.array(random_input, dtype=np.uint8)

  benchmark_time = timeit.timeit(
      lambda: engine.RunInference(input_data),
      number=iterations)

  # Time consumed for each iteration (milliseconds).
  time_per_inference = (benchmark_time / iterations) * 1000
  print(time_per_inference, 'ms (iterations = ', iterations, ')')
  return time_per_inference


if __name__ == '__main__':
  args = test_utils.ParseArgs()
  machine = test_utils.MachineInfo()
  test_utils.CheckCpuScalingGovernorStatus()
  # Read references from csv file.
  model_list, reference = test_utils.ReadReference(
      'basic_engine_reference_%s.csv' % machine)
  total_models = len(model_list)
  # Put column names in first row.
  results = [('MODEL', 'INFERENCE_TIME')]
  for cnt, model in enumerate(model_list, start=1):
    print('-------------- Model ', cnt, '/', total_models, ' ---------------')
    results.append((model, _RunBenchmarkForModel(model)))
  test_utils.SaveAsCsv(
      'basic_engine_benchmarks_%s_%s.csv' % (
          machine, time.strftime('%Y%m%d-%H%M%S')),
      results)
  test_utils.CheckResult(reference, results, args.enable_assertion)
