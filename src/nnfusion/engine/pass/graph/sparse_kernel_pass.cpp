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
    bool Optimize()
    {
        auto gnodes = m_graph->get_ordered_ops();
        for (auto node : gnodes)
        {
            // traverse all nodes and check whether is valuable to
            // to use the sparse kernel
            if (node->get_op_type() == "Dot")
            {
                DotSparseOptimize(node);
            }
            else if (node->get_op_type() == "Conv2d")
            {
                // ???
            }
        }
        return true;
    }

private:
    void DotSparseOptimize(std::shared_ptr<GNode> cur_node)
    {
        bool has_constant = false;

        for (auto in_edge : cur_node->get_in_edges())
        {
            auto src_node = in_edge->get_src();
            if (src_node->is_constant())
            {
                // Get the sparsity ratio, if sparsity ratio is larger than a threshold
                auto weight_constant = std::dynamic_pointer_cast<nnfusion::op::Constant> (src_node->get_op_ptr());
                auto data_ptr = weight_constant->get_data_ptr();
                auto m_shape = weight_constant->get_shape();
                float sparsity_ratio = get_sparsity_ratio<float>(std::static_cast<float*>(data_ptr), nnfusion::shape_size(m_shape), 1e-6);
                assert(data_ptr != nullptr);
                has_constant = true;
                break;
            }
        }
        if (!has_constant)
            // on the fly convertion is to expensive
            return;
    }
    template <typename scalar_t>
    float get_sparsity_ratio(scalar_t* data, size_t n, scalar_t threshold )
    {
        int count = 0;
        for (int i = 0; i < n; i++) {
            if(data[i]<=threshold)
                count++;
        }
        return count*1.0/n;
    }
    template <typename scalar_t>
    tuple<vector<int>, vector<int>, vector<scalar_t>> convert_to_csr(scalar_t *data, const nnfusion::Shape & m_shape, scalar_t threshold){
        assert(m_shape.size()==2);
        vector<int> row_idx;
        vector<int> col_idx,
        vector<scalar_t> values;
        for(int i=0;i<m_shape[0];i++){
            row_idx.push_back(values.size());
            for(int j=0;j<m_shape[1];j++){
                size_t pos = i * m_shape[1] + j;
                if(data<threshold){
                    // sparsity
                    continue;
                }else{
                    values.push_back(data[pos]);
                    col_idx.push_back(j);
                }
            }
        }
        row_idx.push_back(values.size());
        return std::make_tuple(row_idx, col_idx, values);
    }

    std::shared_ptr<Graph> m_graph;
};

bool SparseKernelPass::run_on_graph(std::shared_ptr<Graph>& graph)
{
    bool enable_sparse_kernel = FLAGS_sparse_kernel;
    if (!enable_sparse_kernel)
        return true;
    NNFUSION_LOG(INFO) << "Enable the Sparse kernels";
    SparseKernelOptimizer optimizer(graph);
    return optimizer.Optimize();
}