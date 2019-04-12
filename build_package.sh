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
#
# This script will generate edgetpu_api.tar.gz which contains Edge TPU
# Python API.

set -e
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERSION=$(cd "${SCRIPT_DIR}" && python3 -c "print(__import__('edgetpu').__version__)")
(cd "${SCRIPT_DIR}" && python3 setup.py bdist_wheel)

mkdir -p "${SCRIPT_DIR}/dist/edgetpu_api"
mkdir -p "${SCRIPT_DIR}/dist/edgetpu_api/libedgetpu"

cp ${SCRIPT_DIR}/libedgetpu/*.so \
   "${SCRIPT_DIR}/dist/edgetpu_api/libedgetpu"

cp ${SCRIPT_DIR}/dist/edgetpu-*.whl \
   "${SCRIPT_DIR}/99-edgetpu-accelerator.rules" \
   "${SCRIPT_DIR}/install.sh" \
   "${SCRIPT_DIR}/uninstall.sh" \
   "${SCRIPT_DIR}/dist/edgetpu_api"

tar -C "${SCRIPT_DIR}/dist" -zcvf "edgetpu_api_${VERSION}.tar.gz" edgetpu_api
