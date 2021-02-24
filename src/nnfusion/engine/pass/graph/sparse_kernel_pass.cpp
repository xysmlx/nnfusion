// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "sparse_kernel_pass.hpp"
#include <queue>
#include "kernel_selection.hpp"
#include "nnfusion/core/graph/gnode.hpp"
#include "nnfusion/core/graph/graph.hpp"
#include "nnfusion/core/kernels/kernel_registration.hpp"
#include "nnfusion/core/operators/op_define/broadcast.hpp"
#include "nnfusion/core/operators/op_define/noop.hpp"
#include "nnfusion/core/operators/op_define/reshape.hpp"
#include "nnfusion/core/operators/util/elementwise_arithmetic.hpp"

#include "gflags/gflags.h"

using namespace nnfusion::graph;
using namespace nnfusion::pass::graph;
using namespace nnfusion::kernels;


bool SparseKernelPass::run_on_graph(std::shared_ptr<Graph>& graph){
    
    return true;
}