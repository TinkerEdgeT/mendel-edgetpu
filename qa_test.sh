#!/bin/bash
#
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

set -e
set -x

date

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
DEFAULT='\033[0m' # No Color

RUNTIME_PERF=throttled
AUTO_CONFIRM=n
TEST_TYPE=local

PYTHON=${PYTHON:-python3}
BENCHMARK_OPTIONS= # Empty by default

while :; do
  case $1 in
    -y|-Y) AUTO_CONFIRM=y  # Automatic confirm.
    ;;
    -m) RUNTIME_PERF=direct  # Use maximum frequency.
    ;;
    -i) TEST_TYPE=installed  # Test installed library.
    ;;
    *) break
  esac
  shift
done

cpu_arch=$(uname -m)
os_version=$(uname -v)

if [[ "$cpu_arch" == "x86_64" ]] && [[ "$os_version" == *"Debian"* || "$os_version" == *"Ubuntu"* ]]; then
  platform="x86_64_linux"
  echo -e "${GREEN}Recognized as Linux on x86_64!${DEFAULT}"
elif [[ "$cpu_arch" == "armv7l" ]]; then
  board_version=$(cat /proc/device-tree/model)
  if [[ "$board_version" == "Raspberry Pi 3 Model B Rev"* ]]; then
    platform="raspberry_pi_3b"
    echo -e "${GREEN}Recognized as Raspberry Pi 3 B!${DEFAULT}"
  elif [[ "$board_version" == "Raspberry Pi 3 Model B Plus Rev"* ]]; then
    platform="raspberry_pi_3b+"
    echo -e "${GREEN}Recognized as Raspberry Pi 3 B+!${DEFAULT}"
  fi
elif [[ -f /etc/mendel_version ]]; then
  platform="edgetpu_devboard"
  echo -e "${GREEN}Recognized as Edgetpu DevBoard!${DEFAULT}"
else
  echo -e "${RED}Platform not supported!${DEFAULT}"
  exit 1
fi

if [[ "$platform" == "edgetpu_devboard" ]]; then
  LD_LIBRARY_PATH="${SCRIPT_DIR}/libedgetpu/${RUNTIME_PERF}/aarch64-linux-gnu"
  if [[ "${RUNTIME_PERF}" == "direct" ]]; then
    BENCHMARK_OPTIONS="--enable_assertion"
  fi
elif [[ "$platform" == "raspberry_pi_3b" ]] || [[ "$platform" == "raspberry_pi_3b+" ]]; then
  LD_LIBRARY_PATH="${SCRIPT_DIR}/libedgetpu/${RUNTIME_PERF}/arm-linux-gnueabihf"
elif [[ "$platform" == "x86_64_linux" ]]; then
  LD_LIBRARY_PATH="${SCRIPT_DIR}/libedgetpu/${RUNTIME_PERF}/x86_64-linux-gnu"
fi

function run_test {
  if [[ "${TEST_TYPE}" == "installed" ]]; then
    pushd /
      MPLBACKEND=agg python3 -m unittest -v "${SCRIPT_DIR}/tests/$1.py"
    popd
  else
    sudo env MPLBACKEND=agg LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" PYTHONPATH="${SCRIPT_DIR}" \
      `which ${PYTHON}` -m unittest -v "${SCRIPT_DIR}/tests/$1.py"
  fi
}

function run_benchmark {
  if [[ "${TEST_TYPE}" == "installed" ]]; then
    pushd /
      python3 "${SCRIPT_DIR}/benchmarks/$1.py" ${BENCHMARK_OPTIONS}
    popd
  else
    sudo env LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" PYTHONPATH="${SCRIPT_DIR}" \
        `which ${PYTHON}` "${SCRIPT_DIR}/benchmarks/$1.py" ${BENCHMARK_OPTIONS}
  fi
}

sudo apt-get update
sudo apt-get install -y libc++1 libc++abi1 libgcc1 libc6 python3-numpy python3-pil libunwind8

if [[ "$platform" == "x86_64_linux" ]]; then
  rm -rf "${SCRIPT_DIR}/.env"

  if [[ "${TEST_TYPE}" == "installed" ]]; then
    python3 -m pip install virtualenv
    python3 -m virtualenv --system-site-packages "${SCRIPT_DIR}/.env"
    source "${SCRIPT_DIR}/.env/bin/activate"
  else
    ${PYTHON} -m pip install virtualenv
    ${PYTHON} -m virtualenv "${SCRIPT_DIR}/.env"
    source "${SCRIPT_DIR}/.env/bin/activate"
    pip install numpy Pillow
  fi
fi

if [[ "$platform" == "raspberry_pi_3b" ]] || [[ "$platform" == "raspberry_pi_3b+" ]] || [[ "$platform" == "edgetpu_devboard" ]]; then
  echo -e "${GREEN}--------------- Enable CPU performance mode -----------------${DEFAULT}"
  sudo apt-get install -y linux-cpupower
  sudo cpupower frequency-set --governor performance
fi

if [[ "$platform" != "edgetpu_devboard" ]] && [[ "$AUTO_CONFIRM" != "y" ]]; then
  echo -e "${GREEN}Plug in USB Accelerator and press 'Enter' to continue.${DEFAULT}"
  read LINE
fi

echo -e "${BLUE}Test Exceptions${DEFAULT}"
run_test exception_test

echo -e "${BLUE}Unit test of BasicEngine"
echo -e "Run unit test with BasicEngine. It will run inference on all models once.${DEFAULT}"
run_test basic_engine_test

echo -e "${BLUE}Benchmark of BasicEngine"
echo -e "Benchmark all supported models with BasicEngine.${DEFAULT}"
echo -e "${YELLOW}This test will take long time.${DEFAULT}"
run_benchmark basic_engine_benchmarks

echo -e "${BLUE}ClassificationEngine"
echo -e "Now we'll run unit test of ClassificationEngine${DEFAULT}"
run_test classification_engine_test

echo -e "${BLUE}Multiple Edge TPUs test${DEFAULT}"
run_test multiple_tpus_test

echo -e "${BLUE}Edge TPU utils test${DEFAULT}"
run_test edgetpu_utils_test

echo -e "${BLUE}Benchmark for ClassificationEngine"
echo -e "Benchmark all classification models with different image size.${DEFAULT}"
echo -e "${YELLOW}This test will take long time.${DEFAULT}"
run_benchmark classification_benchmarks

echo -e "${BLUE}DetectionEngine"
echo -e "Now we'll run unit test of DetectionEngine${DEFAULT}"
run_test detection_engine_test

echo -e "${BLUE}Benchmark for DetectionEngine"
echo -e "Benchmark all detection models with different image size.${DEFAULT}"
echo -e "${YELLOW}This test will take long time.${DEFAULT}"
run_benchmark detection_benchmarks

echo -e "${BLUE}COCO test for DetectionEngine"
if [[ "$platform" == "x86_64_linux" ]]; then
  # Takes a long time.
  echo -e "${YELLOW}This test will take long time.${DEFAULT}"
  echo -e "${GREEN}Download dependent libraries.${DEFAULT}"
  sudo apt-get install -y libfreetype6-dev libpng-dev libqhull-dev libagg-dev python3-dev pkg-config
  python3 -m pip install matplotlib
  python3 -m pip install cython
  python3 -m pip install git+https://github.com/cocodataset/cocoapi#subdirectory=PythonAPI

  echo -e "${GREEN}Download coco data set.${DEFAULT}"
  ${SCRIPT_DIR}/test_data/download_coco_val_data.sh

  echo -e "${GREEN}Start tests.${DEFAULT}"
  run_test coco_object_detection_test
else
  echo -e "${YELLOW}Skip.${DEFAULT}"
fi

echo -e "${BLUE}ImprintingEngine"
echo -e "Now we'll run unit test of ImprintingEngine${DEFAULT}"
run_test imprinting_engine_test

echo -e "${BLUE}Benchmark for ImprintingEngine"
echo -e "Benchmark speed of transfer learning with Imprinting Engine.${DEFAULT}"
echo -e "${YELLOW}This test will take long time.${DEFAULT}"
${SCRIPT_DIR}/test_data/download_imprinting_test_data.sh
run_benchmark imprinting_benchmarks

echo -e "${BLUE}Evaluation for ImprintingEngine${DEFAULT}"
if [[ "$platform" == "x86_64_linux" ]]; then
  ${SCRIPT_DIR}/test_data/download_oxford_17flowers.sh
  run_test imprinting_evaluation_test
else
  echo -e "${YELLOW}Skip.${DEFAULT}"
fi
