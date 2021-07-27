// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <map>
#include <set>
#include <memory>
#include <string>
#include <utility>
#include "ie_blob.h"
#include "cpp/ie_cnn_network.h"
#include <cpp_interfaces/impl/ie_executable_network_thread_safe_default.hpp>
#include "cldnn_graph.h"
#include "cldnn_config.h"
#include "cldnn_remote_context.h"
#include "cldnn_extension_manager.h"

namespace CLDNNPlugin {

class CLDNNExecNetwork : public InferenceEngine::ExecutableNetworkThreadSafeDefault {
public:
    typedef std::shared_ptr<CLDNNExecNetwork> Ptr;

    CLDNNExecNetwork(InferenceEngine::CNNNetwork &network,
                     InferenceEngine::RemoteContext::Ptr context,
                     Config config,
                     GPUExtensionManager::Ptr extensionManager);

    InferenceEngine::CNNNetwork GetExecGraphInfo() override;
    InferenceEngine::IInferRequestInternal::Ptr CreateInferRequest() override;
    InferenceEngine::IInferRequestInternal::Ptr CreateInferRequestImpl(InferenceEngine::InputsDataMap networkInputs,
                                                                       InferenceEngine::OutputsDataMap networkOutputs) override;

    InferenceEngine::Parameter GetMetric(const std::string &name) const override;
    InferenceEngine::Parameter GetConfig(const std::string &name) const override;
    InferenceEngine::RemoteContext::Ptr GetContext() const override;

    std::vector<std::shared_ptr<CLDNNGraph>> m_graphs;
    InferenceEngine::gpu::ClContext::Ptr m_context;
    Config m_config;
    InferenceEngine::ITaskExecutor::Ptr m_taskExecutor;
    GPUExtensionManager::Ptr m_extensionManager;
};

};  // namespace CLDNNPlugin
