// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_args.hpp"
#include "event.hpp"

#include <memory>
#include <vector>

namespace cldnn {

class kernel {
public:
    using ptr = std::shared_ptr<kernel>;
    virtual std::shared_ptr<kernel> clone() const = 0;
    virtual ~kernel() = default;
};

}  // namespace cldnn
