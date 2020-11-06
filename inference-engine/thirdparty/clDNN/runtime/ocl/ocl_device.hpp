/*
// Copyright (c) 2019-2021 Intel Corporation
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

#include "cldnn/runtime/engine.hpp"
#include "cldnn/runtime/device.hpp"
#include "ocl_common.hpp"

#include <map>
#include <string>
#include <vector>
#include <algorithm>

namespace cldnn {
namespace gpu {

struct ocl_device : public device {
public:
    ocl_device(const cl::Device dev, const cl::Context& ctx, const cl_platform_id platform);

    device_info get_info() const override { return _info; }
    memory_capabilities get_mem_caps() const override { return _mem_caps; }

    const cl::Device& get_device() const { return _device; }
    cl::Device& get_device() { return _device; }
    const cl::Context& get_context() const { return _context; }
    cl_platform_id get_platform() const { return _platform; }

    ~ocl_device() = default;

private:
    cl::Context _context;
    cl::Device _device;
    cl_platform_id _platform;
    device_info _info;
    memory_capabilities _mem_caps;
};

}  // namespace gpu
}  // namespace cldnn
