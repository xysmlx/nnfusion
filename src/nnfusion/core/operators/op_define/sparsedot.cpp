// Microsoft (c) 2019, NNFusion Team

#include <functional>
#include <memory>
#include <utility>

#include "sparsedot.hpp"
#include "nnfusion/core/graph/gnode.hpp"

using namespace std;
using namespace nnfusion::op;

SparseDot::SparseDot()
    : SparseDot(0, false)
{
}

SparseDot::SparseDot(size_t reduction_axes_count, bool has_reduction_axes_count, bool trans_a, bool trans_b)
    : Op("SparseDot")
    , m_reduction_axes_count(reduction_axes_count)
    , m_has_reduction_axes_count(has_reduction_axes_count)
    , m_transpose_A(trans_a)
    , m_transpose_B(trans_b)
{
}

void SparseDot::validate_and_infer_types(std::shared_ptr<graph::GNode> gnode)
{
    // No need to infer the output shape and type again, should
    // throw "Shape inference of sparsedot is not implemented yet";
    return;
}
