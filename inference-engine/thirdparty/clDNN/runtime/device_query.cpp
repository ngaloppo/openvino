/*
// Copyright (c) 2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "cldnn/runtime/device_query.hpp"
#include "ocl/ocl_device_detector.hpp"
#ifdef CLDNN_WITH_SYCL
#include "sycl/sycl_device_detector.hpp"
#endif

#include <map>
#include <string>

namespace cldnn {

// We use runtime_type to filter out the same device with different execution runtime in order to skip it in devices list
// So we can have 2 logical devices for single physical GPU for OCL and L0 runtimes
// but user always see it as single device (e.g. GPU or GPU.0), and the actual runtime is specified in plugin config.
// Need to make sure that this is a good way to handle different backends from the users perspective and
// ensure that correct physical device is always selected for L0 case.
device_query::device_query(runtime_types runtime_type, void* user_context, void* user_device) {
#ifdef CLDNN_WITH_SYCL
    sycl::sycl_device_detector sycl_detector;
    auto sycl_devices = sycl_detector.get_available_devices(runtime_type, user_context, user_device);
    _available_devices.insert(sycl_devices.begin(), sycl_devices.end());
#else
    gpu::ocl_device_detector ocl_detector;
    _available_devices = ocl_detector.get_available_devices(user_context, user_device);
#endif
}
}  // namespace cldnn
