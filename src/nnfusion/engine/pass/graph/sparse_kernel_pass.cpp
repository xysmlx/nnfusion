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

DEFINE_bool(fsparse_kernel, false, "Sparse Kernel.");

using namespace nnfusion::graph;
using namespace nnfusion::pass::graph;
using namespace nnfusion::kernels;
using namespace nnfusion::element;

class SparseKernelOptimizer
{
public:
    SparseKernelOptimizer(std::shared_ptr<Graph> g, float threshold)
        : m_graph(g)
        , sparse_threshold(threshold)
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
                auto weight_constant =
                    std::dynamic_pointer_cast<nnfusion::op::Constant>(src_node->get_op_ptr());
                auto data_ptr = weight_constant->get_data_ptr();
                assert(data_ptr != nullptr);
                auto m_shape = weight_constant->get_shape();
                float sparsity_ratio =
                    get_sparsity_ratio<float>(static_cast<const float*>(data_ptr),
                                              nnfusion::shape_size(m_shape),
                                              sparse_threshold);
                // if sparsity ratio is too low then it's not worth
                if (sparsity_ratio < 0.9)
                    continue;
                std::shared_ptr<vector<int32_t>> row_idx, col_idx;
                std::shared_ptr<vector<float>> values;
                std::tie(row_idx, col_idx, values) = convert_to_csr<float>(
                    static_cast<const float*>(data_ptr), m_shape, sparse_threshold);
                insert_sparse_dot<float>(m_graph, in_edge, row_idx, col_idx, values);
                has_constant = true;
                break;
            }
        }
        if (!has_constant)
            // on the fly convertion is to expensive
            return;
    }
    template <typename scalar_t>
    float get_sparsity_ratio(const scalar_t* data, size_t n, scalar_t threshold)
    {
        int count = 0;
        for (int i = 0; i < n; i++)
        {
            if (data[i] <= threshold)
                count++;
        }
        return count * 1.0 / n;
    }

    template <typename scalar_t>
    tuple<std::shared_ptr<vector<int32_t>>,
          std::shared_ptr<vector<int32_t>>,
          std::shared_ptr<vector<scalar_t>>>
        convert_to_csr(const scalar_t* data, const nnfusion::Shape& m_shape, scalar_t threshold)
    {
        assert(m_shape.size() == 2);
        auto row_idx = std::make_shared<vector<int32_t>>();
        auto col_idx = std::make_shared<vector<int32_t>>();
        auto values = std::make_shared<vector<scalar_t>>();

        for (int i = 0; i < m_shape[0]; i++)
        {
            row_idx->push_back(values->size());
            for (int j = 0; j < m_shape[1]; j++)
            {
                size_t pos = i * m_shape[1] + j;
                if (data[pos] < threshold)
                {
                    // sparsity
                    continue;
                }
                else
                {
                    values->push_back(data[pos]);
                    col_idx->push_back(j);
                }
            }
        }
        row_idx->push_back(values->size());
        return std::make_tuple(row_idx, col_idx, values);
    }
    template <typename scalar_t>
    void insert_sparse_dot(std::shared_ptr<Graph> pgraph,
                           std::shared_ptr<Edge> in_edge,
                           std::shared_ptr<vector<int32_t>> row_idx,
                           std::shared_ptr<vector<int32_t>> col_idx,
                           std::shared_ptr<vector<scalar_t>> values)
    {
        std::shared_ptr<GNode> src_node = in_edge->get_src();
        std::shared_ptr<GNode> dst_node = in_edge->get_dst();
        //create the constant nodes for the csr format weight
        auto row_idx_cons = std::make_shared<op::Constant>(
            from<int32_t>(), nnfusion::Shape({row_idx->size()}), (void*)row_idx->data());
        auto row_idx_node = std::make_shared<GNode>(row_idx_cons, GNodeVector({}));
        row_idx_node->get_op_ptr()->revalidate_and_infer_types(row_idx_node->shared_from_this());

        auto col_idx_cons = std::make_shared<op::Constant>(
            from<int32_t>(), nnfusion::Shape({col_idx->size()}), (void*)col_idx->data());
        auto col_idx_node = std::make_shared<GNode>(col_idx_cons, GNodeVector({}));
        col_idx_node->get_op_ptr()->revalidate_and_infer_types(col_idx_node->shared_from_this());
        auto values_cons = std::make_shared<op::Constant>(
            from<scalar_t>(), nnfusion::Shape({values->size()}), (void*)values->data());
        auto values_node = std::make_shared<GNode>(values_cons, GNodeVector({}));
        values_node->get_op_ptr()->revalidate_and_infer_types(values_node->shared_from_this());
        auto dense_op = std::dynamic_pointer_cast<op::Dot>(
            dst_node->get_op_ptr()); // The original dense gemm op

        pgraph->add_node(row_idx_node);
        pgraph->add_node(col_idx_node);
        pgraph->add_node(values_node);

        auto dst_pos = in_edge->get_dst_input();
        size_t other_input = 1 - dst_pos;
        auto other_node = dst_node->get_in_edge(other_input)->get_src();
        GNodeVector input_gv({row_idx_node, col_idx_node, values_node, other_node});
        auto sparse_op = std::make_shared<op::SparseDot>(
            dense_op, dst_pos, values->size(), src_node->get_output_shape(0));

        auto sparse_node = std::make_shared<GNode>(sparse_op, input_gv);
        for (int i = 0; i < input_gv.size(); i++)
        {
            pgraph->add_edge(input_gv.at(i), 0, sparse_node, i);
        }

        // pgraph->add_node_and_edge(sparse_node, GNodeVector({row_idx_node, col_idx_node, values_node}));
        auto ori_output = dst_node->get_outputs();
        // just copy the output from the original dense node
        for (int i = 0; i < ori_output.size(); i++)
            sparse_node->set_output(i, ori_output[i]);
        // insert the sparse node into the original graph
        pgraph->replace_node(dst_node, sparse_node, false);
        pgraph->remove_node(src_node);
    }

    std::shared_ptr<Graph> m_graph;
    float sparse_threshold;
};

bool SparseKernelPass::run_on_graph(std::shared_ptr<Graph>& graph)
{
    bool enable_sparse_kernel = FLAGS_fsparse_kernel;
    if (!enable_sparse_kernel)
        return true;
    NNFUSION_LOG(INFO) << "Enable the Sparse kernels";
    SparseKernelOptimizer optimizer(graph, 1e-6);
    return optimizer.Optimize();
}
