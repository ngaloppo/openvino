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
#pragma once

#include "memory.hpp"

#include <memory>
#include <vector>

namespace cldnn {

struct work_group_sizes {
    std::vector<size_t> global;
    std::vector<size_t> local;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Scalar
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct scalar_desc {
    union ValueT {
        uint8_t u8;
        uint16_t u16;
        uint32_t u32;
        uint64_t u64;
        int8_t s8;
        int16_t s16;
        int32_t s32;
        int64_t s64;
        float f32;
        double f64;
    };

    enum class Types {
        UINT8,
        UINT16,
        UINT32,
        UINT64,
        INT8,
        INT16,
        INT32,
        INT64,
        FLOAT32,
        FLOAT64,
    };

    Types t;
    ValueT v;
};

using scalars_desc = std::vector<scalar_desc>;


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// ArgumentDescpirtor
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct argument_desc {
    enum class Types {
        INPUT,
        OUTPUT,
        WEIGHTS,
        BIAS,
        SCALE_TABLE,
        SLOPE,
        SPLIT,
        INTERNAL_BUFFER,
        SCALAR,
        RECURRENT,  // RNN/LSTM/GRU recurrent weights
        HIDDEN,     // RNN/LSTM/GRU hidden input
        CELL,       // LSTM cell input
        LSTM_PACK,  // LSTM packed output
        WEIGHTS_ZERO_POINTS,
        ACTIVATIONS_ZERO_POINTS,
        COMPENSATION,
        INPUT_OF_FUSED_PRIMITIVE
    };

    Types t;
    uint32_t index;
};

using arguments_desc = std::vector<argument_desc>;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// KernelParams
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct kernel_arguments_desc {
    work_group_sizes workGroups;
    arguments_desc arguments;
    scalars_desc scalars;
    std::string layerID;
};

struct kernel_arguments_data {
    std::vector<memory::cptr> inputs;
    std::vector<memory::cptr> intermediates;
    memory::cptr output;
    memory::cptr weights;
    memory::cptr recurrent;
    memory::cptr hidden;
    memory::cptr cell;
    memory::cptr bias;
    memory::cptr weights_zero_points;
    memory::cptr activations_zero_points;
    memory::cptr compensation;
    memory::cptr lookup_table;
    memory::cptr scale_table;
    memory::cptr slope;

    std::vector<memory::cptr> fused_op_inputs;
    int32_t split = 0;
    const scalars_desc* scalars = nullptr;
};

}  // namespace cldnn
