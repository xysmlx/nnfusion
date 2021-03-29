// Microsoft (c) 2019, NNFusion Team

#include <functional>
#include <memory>
#include <utility>

#include "nnfusion/core/graph/gnode.hpp"
#include "quantizedot.hpp"

using namespace std;
using namespace nnfusion::op;


QuantizeDot8bit::QuantizeDot8bit(shared_ptr<Dot> ori_dot)
    : Op("QuantizeDot8bit")
// Initialize the SparseDot Op according to the original Dot Op
{

}

void QuantizeDot8bit::validate_and_infer_types(std::shared_ptr<graph::GNode> gnode)
{
    // No need to infer the output shape and type again, should
    // throw "Shape inference of sparsedot is not implemented yet";
    return;
}
