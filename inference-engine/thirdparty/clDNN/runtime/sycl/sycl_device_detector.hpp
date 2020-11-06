/*
// Copyright (c) 2018-2021 Intel Corporation
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
#pragma once

#include "cldnn/runtime/device.hpp"
#include "cldnn/runtime/engine_configuration.hpp"
#include "sycl_common.hpp"

#include <list>
#include <string>
#include <utility>
#include <vector>
#include <map>

namespace cldnn {
namespace sycl {

class sycl_device_detector {
private:
    const cl::sycl::info::device_type dev_type = cl::sycl::info::device_type::gpu;  // cldnn supports only gpu devices
    const uint32_t dev_vendor = 0x8086;  // Intel vendor
public:
    sycl_device_detector() = default;

    std::map<std::string, device::ptr> get_available_devices(runtime_types runtime_type, void* user_context, void* user_device) const;
private:
    bool does_device_match_config(const cl::sycl::device& device) const;
    std::vector<device::ptr> create_device_list(runtime_types runtime_type) const;
    std::vector<device::ptr> create_device_list_from_user_context(void* user_context) const;
    std::vector<device::ptr> create_device_list_from_user_device(void* user_device) const;
};

}  // namespace sycl
}  // namespace cldnn
