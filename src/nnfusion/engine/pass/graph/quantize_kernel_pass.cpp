// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "quantize_kernel_pass.hpp"
#include <map>
#include <queue>
#include <string>
#include <vector>
#include "gflags/gflags.h"
#include "kernel_selection.hpp"
#include "nnfusion/core/graph/gnode.hpp"
#include "nnfusion/core/graph/graph.hpp"
#include "nnfusion/core/kernels/kernel_registration.hpp"
#include "nnfusion/core/operators/op_define/broadcast.hpp"
#include "nnfusion/core/operators/op_define/noop.hpp"
#include "nnfusion/core/operators/op_define/reshape.hpp"
#include "nnfusion/core/operators/util/elementwise_arithmetic.hpp"

DEFINE_string(fquantize_cfg, "", "Quantize cfg for the target model");
using namespace nnfusion::graph;
using namespace nnfusion::pass::graph;
using namespace nnfusion::kernels;
using namespace nnfusion::element;
// using namespace std;

class QuantizeKernelOptimizer
{
public:
    QuantizeKernelOptimizer(std::shared_ptr<Graph> g, const std::string& cfg_path)
    {
        this->m_graph = g;
        this->cfg_path = cfg_path;
        load_quantize_cfg();
        cache_manager = std::make_shared<nnfusion::cache::KernelCacheManager>();
    }
    void load_quantize_cfg()
    {
        // Load the quantize config
        // pytorch name, onnx name, quantize bit
        ifstream cfg_file(cfg_path.c_str());
        assert(cfg_file.good());
        string line;
        while (std::getline(cfg_file, line))
        {
            std::istringstream iss(line);
            string name;
            int n_bit;
            iss >> name >> n_bit;
            quantize_cfg[name] = n_bit;
        }
    }

    vector<std::shared_ptr<GNode>> find_successors(std::shared_ptr<GNode> gnode)
    {
        vector<std::shared_ptr<GNode>> successors;
        const std::set<std::shared_ptr<nnfusion::graph::Edge>>& out_edges = gnode->get_out_edges();
        for (auto edge : out_edges)
        {
            successors.push_back(edge->get_dst());
        }
        return successors;
    }

    bool optimize()
    {
        if (!cache_manager->is_valid())
        {
            NNFUSION_LOG(INFO) << "No valid kernel cache, cannot find quantized kernel";
            return true;
        }
        auto gnodes = m_graph->get_ordered_ops();
        for (auto node : gnodes)
        {
            std::cout << node->get_unique_name() << " " << node->get_name() << std::endl;
            if (quantize_cfg.count(node->get_name()) == 0)
                continue;
            int quantize_bit = quantize_cfg[node->get_name()];
            std::shared_ptr<KernelContext> ctx(new KernelContext(node));
            std::string identifier = ctx->generate_identifier();
            std::cout << "Identifier: " << identifier << std::endl;
            std::vector<std::shared_ptr<GNode>> successors = find_successors(node);
            int succ_bit = 32;
            int succ_quantized = 0;
            for (auto succ_node : successors)
            {
                std::string succ_name = succ_node->get_name();
                if (quantize_cfg.count(succ_name))
                {
                    if (succ_quantized)
                        // following nodes should have the same quantize config
                        NNFUSION_CHECK(succ_bit == quantize_cfg[succ_name]);
                    succ_quantized += 1;
                    succ_bit = quantize_cfg[succ_name];
                }
            }
            NNFUSION_CHECK(succ_quantized == 0 || succ_quantized == successors.size());
            identifier += "quantize_" + to_string(quantize_bit) + "bit_" + to_string(succ_bit) + "bit";
            std::cout << "New Identifier:" << identifier<<std::endl;
            if (node->get_op_type() == "Dot")
            {
                // update the model graph
                if (quantize_bit == 8)
                {
                    // DotQuantizeOptimize8bit(node);
                }
                else
                {
                }
            }
            else if (node->get_op_type() == "Conv2d") {}
        }
        return true;
    }
    void DotQuantizeOptimize8bit(std::shared_ptr<GNode> cur_node)
    {
        std::cout << "In DotQuantizeOptimize 8bit" << std::endl;
        bool has_constant = false;
        for (auto in_edge : cur_node->get_in_edges())
        {
            auto src_node = in_edge->get_src();
            if (src_node->is_constant())
            {
                auto weight_constant =
                    std::dynamic_pointer_cast<nnfusion::op::Constant>(src_node->get_op_ptr());
                auto w_shape = weight_constant->get_shape();
                int weight_count = 1, out_count = 1;
                for (int i : w_shape)
                    weight_count *= i;
                auto out_shape = cur_node->get_output_shape(0);
                for (int i : out_shape)
                    weight_count *= i;
                // TODO unique_name vs name
                int quantize_bit = quantize_cfg[cur_node->get_unique_name()];
                // we filled the ramdom data temporarily
                float* tmp_weight = (float*)malloc(sizeof(float) * weight_count);
                float* tmp_out = (float*)malloc(sizeof(float) * out_count);
                float* tmp_scale = (float*)malloc(sizeof(float));
                auto dense_op = std::dynamic_pointer_cast<op::Dot>(cur_node->get_op_ptr());
                // quantized weight of the weight
                // The values is not right and will be filled after nnfusion.
                auto quan_w = std::make_shared<op::Constant>(
                    from<float>(), nnfusion::Shape(w_shape), static_cast<void*>(tmp_out));
                auto quan_w_node = std::make_shared<GNode>(quan_w, GNodeVector({}));
                //update the output
                quan_w_node->get_op_ptr()->revalidate_and_infer_types(
                    quan_w_node->shared_from_this());

                // Weight * input zero point
                auto w_mul_zp = std::make_shared<op::Constant>(
                    from<float>(), nnfusion::Shape(out_shape), static_cast<void*>(tmp_out));
                auto w_mul_zp_node = std::make_shared<GNode>(w_mul_zp, GNodeVector({}));
                w_mul_zp->revalidate_and_infer_types(w_mul_zp_node->shared_from_this());

                // zero points of the weight tensor
                auto w_zp = std::make_shared<op::Constant>(
                    from<float>(), nnfusion::Shape(w_shape), static_cast<void*>(tmp_weight));
                auto w_zp_node = std::make_shared<GNode>(w_zp, GNodeVector({}));
                w_zp->revalidate_and_infer_types(w_zp_node);

                auto zp_acc = std::make_shared<op::Constant>(
                    from<float>(), nnfusion::Shape(out_shape), static_cast<void*>(tmp_out));
                auto zp_acc_node = std::make_shared<GNode>(zp_acc, GNodeVector({}));
                zp_acc->revalidate_and_infer_types(zp_acc_node->shared_from_this());

                auto scale_integer =
                    std::make_shared<op::Constant>(from<float>(),
                                                   nnfusion::Shape(vector<size_t>({1})),
                                                   static_cast<void*>(tmp_scale));
                auto scale_integer_node = std::make_shared<GNode>(scale_integer, GNodeVector({}));
                scale_integer->revalidate_and_infer_types(scale_integer_node->shared_from_this());

                auto scale_shift =
                    std::make_shared<op::Constant>(from<float>(),
                                                   nnfusion::Shape(vector<size_t>({1})),
                                                   static_cast<void*>(tmp_scale));
                auto scale_shift_node = std::make_shared<GNode>(scale_shift, GNodeVector({}));
                scale_shift->revalidate_and_infer_types(scale_shift_node->shared_from_this());

                m_graph->add_node(quan_w_node);
                m_graph->add_node(w_mul_zp_node);
                m_graph->add_node(w_zp_node);
                m_graph->add_node(zp_acc_node);
                m_graph->add_node(scale_integer_node);
                m_graph->add_node(scale_shift_node);

                auto quan_dot = std::make_shared<op::QuantizeDot>(dense_op, 8);
                auto quan_dot_node = std::make_shared<GNode>(quan_dot, GNodeVector({}));
                auto ori_outputs = cur_node->get_outputs();
                //???
                for (int i = 0; i < ori_outputs.size(); i++)
                {
                    quan_dot_node->set_output(i, ori_outputs[i]);
                }
                // replace node will revalidate and infer the output tensor
                m_graph->replace_node(cur_node, quan_dot_node, false);
                m_graph->remove_node(src_node);

                has_constant = true;
            }
        }
    }

private:
    std::shared_ptr<Graph> m_graph;
    std::string cfg_path;
    std::map<string, int> quantize_cfg;
    std::shared_ptr<nnfusion::cache::KernelCacheManager> cache_manager;
};

bool QuantizeKernelPass::run_on_graph(std::shared_ptr<Graph>& graph)
{
    bool enable_quantize_kernel = FLAGS_fquantize_cfg.size() > 0;
    if (!enable_quantize_kernel)
        return true;
    NNFUSION_LOG(INFO) << "Enable the Quantized kernels";
    QuantizeKernelOptimizer optimizer(graph, FLAGS_fquantize_cfg);
    return optimizer.optimize();
}