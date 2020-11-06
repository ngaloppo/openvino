// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "cldnn/runtime/device.hpp"
#include "ocl_common.hpp"

#include <list>
#include <string>
#include <utility>
#include <vector>
#include <map>

namespace cldnn {
namespace gpu {

class ocl_device_detector {
private:
    const uint32_t dev_type = CL_DEVICE_TYPE_GPU;  // cldnn supports only gpu devices
    const uint32_t dev_vendor = 0x8086;  // Intel vendor
public:
    ocl_device_detector() = default;

    std::map<std::string, device::ptr> get_available_devices(void* user_context, void* user_device) const;
private:
    bool does_device_match_config(bool out_of_order, const cl::Device& device) const;
    std::vector<device::ptr> create_device_list(bool out_out_order) const;
    std::vector<device::ptr> create_device_list_from_user_context(bool out_out_order, void* user_context) const;
    std::vector<device::ptr> create_device_list_from_user_device(bool out_out_order, void* user_device) const;
};

}  // namespace gpu
}  // namespace cldnn
