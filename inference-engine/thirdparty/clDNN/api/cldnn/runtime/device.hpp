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

#include "device_info.hpp"
#include "memory_caps.hpp"

#include <memory>

namespace cldnn {

struct device {
public:
    using ptr = std::shared_ptr<device>;
    virtual device_info get_info() const = 0;
    virtual memory_capabilities get_mem_caps() const = 0;

    virtual ~device() = default;
};

}  // namespace cldnn
