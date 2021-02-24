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

DEFINE_bool(sparse_kernel, false, "Sparse Kernel.");

using namespace nnfusion::graph;
using namespace nnfusion::pass::graph;
using namespace nnfusion::kernels;

class SparseKernelOptimizer
{
public:
    SparseKernelOptimizer(std::shared_ptr<Graph> g)
        : m_graph(g)
    {
    }
    bool Optimize(){
        auto gnodes = m_graph->get_ordered_ops();
        for(auto node: gnodes){
            // traverse all nodes and check whether is valuable to
            // to use the sparse kernel

        }
        return true;
    }
private:
    std::shared_ptr<Graph> m_graph;
};

bool SparseKernelPass::run_on_graph(std::shared_ptr<Graph>& graph){
    bool enable_sparse_kernel = FLAGS_sparse_kernel;
    if(!enable_sparse_kernel)
        return true;

    SparseKernelOptimizer optimizer(graph);
    return optimizer.Optimize();
}