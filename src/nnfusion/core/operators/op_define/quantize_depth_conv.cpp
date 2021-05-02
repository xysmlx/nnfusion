// Microsoft (c) 2019, NNFusion Team

#include <functional>
#include <memory>
#include <utility>

#include "nnfusion/core/graph/gnode.hpp"
#include "quantize_depth_conv.hpp"

using namespace std;
using namespace nnfusion::op;


QuantizeDepthwiseConv2dNative::QuantizeDepthwiseConv2dNative(size_t quantize_bit)
    : Op("QuantizeDepthwiseConv2dNative")
// Initialize the SparseDot Op according to the original Dot Op
{
    m_quantize_bit = quantize_bit;

}

void QuantizeDepthwiseConv2dNative::validate_and_infer_types(std::shared_ptr<graph::GNode> gnode)
{
    // Only set the output type, no need to infer the output
    // shape again
    auto result_shape = gnode->get_output_shape(0);
    nnfusion::element::Type result_et;
    switch (m_quantize_bit){
        case 8:
            // result_et = nnfusion::element::from<int8_t>();
            result_et = nnfusion::element::from<float>();
            break;
        case 16:
            // result_et = nnfusion::element::from<nnfusion::element::half>();
            result_et = nnfusion::element::from<float>();
            break;
        default:
            throw "Not supported quantize config";
            break;
    }

    gnode->set_output_type_and_shape(0, result_et, result_shape);
}
