#include <CL/sycl.hpp>
#include "ocl_ext.hpp"

#include <regex>

// nGraph op
NGRAPH_RTTI_DEFINITION(OCLLayerOp, "OCLLayerOp", 0);

OCLLayerOp::OCLLayerOp(const ngraph::Output<ngraph::Node>& arg) : Op({arg}) {
    constructor_validate_and_infer_types();
}

void OCLLayerOp::validate_and_infer_types() {
    // Operation doesn't change shapes and element type
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

inline std::shared_ptr<ngraph::Node> getNodeByName(const std::string& name,
                                                   const std::shared_ptr<ngraph::Function> func,
                                                   bool namePattern = true) {
    for (const auto& node : func->get_ops()) {
        bool match = namePattern ? node->get_friendly_name().find(name) == 0 :
                     node->get_friendly_name() == name;
        if (match) {
            return node;
        }
    }
    std::cout << "Could not find a node: " + name << std::endl;
    return nullptr;
}

std::shared_ptr<ngraph::Node> OCLLayerOp::clone_with_new_inputs(const ngraph::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<OCLLayerOp>(new_args.at(0));
}

bool OCLLayerOp::visit_attributes(ngraph::AttributeVisitor& visitor) {
    return true;
}

std::vector<std::string> InfEngineNgraphExtension::getImplTypes(const std::shared_ptr<ngraph::Node>& node) {
    return {"CPU"};
}

InferenceEngine::ILayerImpl::Ptr
InfEngineNgraphExtension::getImplementation(const std::shared_ptr<ngraph::Node>& node,
                                            const std::string& implType) {
    if (std::dynamic_pointer_cast<OCLLayerOp>(node) && implType == "CPU") {
        return std::make_shared<OCLLayerImpl>(node);
    }
    return nullptr;
}

OCLLayerImpl::OCLLayerImpl(const std::shared_ptr<ngraph::Node>& node)
{
    inpShapes.resize(node->get_input_size());
    for (size_t i = 0; i < inpShapes.size(); ++i)
        inpShapes[i] = node->get_input_shape(i);
    outShape = node->get_output_shape(0);
}

InferenceEngine::StatusCode OCLLayerImpl::init(InferenceEngine::LayerConfig& config,
                                    InferenceEngine::ResponseDesc *resp) noexcept
{
    return InferenceEngine::StatusCode::OK;
}

InferenceEngine::StatusCode
OCLLayerImpl::getSupportedConfigurations(std::vector<InferenceEngine::LayerConfig>& conf,
                                          InferenceEngine::ResponseDesc* resp) noexcept
{
    std::vector<InferenceEngine::DataConfig> inDataConfig;
    std::vector<InferenceEngine::DataConfig> outDataConfig;

    // Allow any offset before data
    size_t offset((std::numeric_limits<size_t>::max)());

    // Input shape
    for (const auto& shape : inpShapes)
    {
        InferenceEngine::SizeVector order(shape.size());
        std::iota(order.begin(), order.end(), 0);

        InferenceEngine::DataConfig inpConf;
        inpConf.desc = InferenceEngine::TensorDesc(InferenceEngine::Precision::FP32, shape, {shape, order, offset});
        inDataConfig.push_back(inpConf);
    }

    // Output shape
    InferenceEngine::SizeVector order(outShape.size());
    std::iota(order.begin(), order.end(), 0);

    InferenceEngine::DataConfig outConf;
    outConf.desc = InferenceEngine::TensorDesc(InferenceEngine::Precision::FP32, outShape, {outShape, order, offset});
    outDataConfig.push_back(outConf);

    InferenceEngine::LayerConfig layerConfig;
    layerConfig.inConfs = inDataConfig;
    layerConfig.outConfs = outDataConfig;

    conf.push_back(layerConfig);
    return InferenceEngine::StatusCode::OK;
}

InferenceEngine::StatusCode OCLLayerImpl::execute(std::vector<InferenceEngine::Blob::Ptr>& inputs,
                                                   std::vector<InferenceEngine::Blob::Ptr>& outputs,
                                                   InferenceEngine::ResponseDesc *resp) noexcept
{
    std::cout << "execute" << std::endl;
    const float* inp = inputs[0]->cbuffer().as<float*>();
    float* out = outputs[0]->buffer().as<float*>();
    for (size_t i = 0; i < inputs[0]->size(); ++i) {
        out[i] = inp[i] + 1.0f;
    }
    return InferenceEngine::OK;
}
