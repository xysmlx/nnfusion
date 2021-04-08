// Microsoft (c) 2019, NNFusion Team

#include <functional>
#include <memory>
#include <utility>

#include "nnfusion/core/graph/gnode.hpp"
#include "quantizedot.hpp"

using namespace std;
using namespace nnfusion::op;


QuantizeDot::QuantizeDot(shared_ptr<Dot> ori_dot, size_t quantize_bit)
    : Op("QuantizeDot")
// Initialize the SparseDot Op according to the original Dot Op
{
    bool trans_A = ori_dot->get_transpose_A();
    bool trans_B = ori_dot->get_transpose_B();
    set_transpose(trans_A, trans_B);
    m_reduction_axes_count = ori_dot->get_reduction_axes_count();
    m_quantize_bit = quantize_bit;

}

void QuantizeDot::validate_and_infer_types(std::shared_ptr<graph::GNode> gnode)
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
