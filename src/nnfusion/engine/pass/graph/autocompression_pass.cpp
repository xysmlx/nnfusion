// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "autocompression_pass.hpp"
#include "nnfusion/common/common.hpp"
#include "nnfusion/core/graph/gedge.hpp"
#include "nnfusion/core/kernels/cuda_gpu/cuda_emitter.hpp"
#include "nnfusion/engine/cache/manager.hpp"
#include "nnfusion/engine/op.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;
using namespace nnfusion::graph;
using namespace nnfusion::pass::graph;

DEFINE_bool(fautocompression, false, "Auto Compression");
DEFINE_string(fautocompression_config,
              "",
              "Auto compression config for the target model in json format");

class AutoCompressionOptimizer
{
public:
    AutoCompressionOptimizer(std::shared_ptr<nnfusion::graph::Graph> g, const std::string& cfg_path)
        : m_graph(g)
    {
        if (cfg_path != "")
        {
            std::ifstream fin;
            fin.open(cfg_path);

            std::stringstream ss;
            ss << fin.rdbuf();
            m_config = nlohmann::json::parse(ss.str());
        }
    }

    // void load_sparse_config()
    // {
    //     // load sparse config
    // }

    pair<NNFusion_DeviceType, kernels::KernelEmitter::Pointer>
        fetch_sparse_kernel(shared_ptr<cache::KernelCacheManager> cache_manager,
                            shared_ptr<GNode> gnode,
                            NNFusion_DeviceType devtype)
    {
        // std::vector<shared_ptr<const KernelRegistration>> kernel_regs =
        //     KernelRegistry::Global()->FindKernelRegistrations(
        //         gnode->get_op_type(), devtype, element::f32);
        shared_ptr<KernelContext> ctx(new KernelContext(gnode));
        std::vector<nlohmann::json> functions;

        std::string identifier = ctx->generate_identifier();
        // Todo: platform interface to be coordinated with nnfusion devtype
        const std::vector<std::string> SUPPORT_PLATFORM = {"CUDA_GPU"};

        if (identifier != "" &&
            find(SUPPORT_PLATFORM.begin(), SUPPORT_PLATFORM.end(), get_device_str(devtype)) !=
                SUPPORT_PLATFORM.end())
        {
            // fetch all available kernel entries from kernel cache DB
            auto fetched = cache_manager->fetch_all(identifier, get_device_str(devtype));

            // emit External kernels
            {
                for (auto kernel_entry : fetched)
                {
                    if (kernel_entry->source == "Compression" &&
                        kernel_entry->miscs["compression"]["gnode"] == gnode->get_name())
                    {
                        NNFUSION_CHECK(kernel_entry->tags.find("CudaEmitter") !=
                                       kernel_entry->tags.end());
                        auto kernel =
                            std::make_shared<kernels::cuda::CacheCudaEmitter>(ctx, kernel_entry);
                        if (kernel->get_or_emit_source())
                        {
                            return std::make_pair(devtype, kernel);
                        }
                    }
                }
            }

            // for bert
            {
                for (auto kernel_entry : fetched)
                {
                    if (kernel_entry->source == "Compression")
                    {
                        NNFUSION_CHECK(kernel_entry->tags.find("CudaEmitter") !=
                                       kernel_entry->tags.end());
                        auto kernel =
                            std::make_shared<kernels::cuda::CacheCudaEmitter>(ctx, kernel_entry);
                        if (kernel->get_or_emit_source())
                        {
                            return std::make_pair(devtype, kernel);
                        }
                    }
                }
            }
        }
        return std::make_pair(devtype, nullptr);
    }

    void process_mlp_mnist()
    {
        std::vector<std::shared_ptr<GNode>> nodes = m_graph->get_nodes();

        for (auto it : nodes)
        {
            if (it->get_name() == "Dot_11")
            {
                // transpose the input due to the layout requirement
            }

            if (it->get_name() == "Dot_29")
            {
                // transpose the output due to the layout requirement
            }
        }
    }

    bool Optimize()
    {
        if (m_config.find("model_name") != m_config.end())
        {
            if (m_config["model_name"] == "mlp_mnist")
            {
                process_mlp_mnist();
            }
        }

        auto cache_manager = std::make_shared<cache::KernelCacheManager>();
        if (!cache_manager->is_valid())
        {
            NNFUSION_LOG(INFO) << "No valid kernel cache, FetchBasedSelector will be skipped";
            return true;
        }
        // auto dev_name = FLAGS_fdefault_device.c_str();
        // NNFusion_DeviceType default_device = nnfusion::get_device_type(dev_name);

        std::vector<std::shared_ptr<GNode>> nodes = m_graph->get_nodes();
        for (auto it : nodes)
        {
            if (!(*it)["Kernel_Selection_Result"].is_valid())
            {
                if (!(*it)["DeviceType"].is_valid())
                {
                    NNFUSION_CHECK_FAIL()
                        << "GNode DeviceType should be assigned before this pass："
                        << it->get_name();
                }
                auto n_device_type = (*it)["DeviceType"].as<NNFusion_DeviceType>();
                NNFUSION_CHECK(n_device_type != UNKNOWN);
                auto ans = fetch_sparse_kernel(cache_manager, it, n_device_type);

                if (ans.second != nullptr)
                    (*it)["Kernel_Selection_Result"] = ans;
            }
        }

        return true;
    }

private:
    std::shared_ptr<nnfusion::graph::Graph> m_graph;
    nlohmann::json m_config;
};

bool AutoCompressionPass::run_on_graph(std::shared_ptr<nnfusion::graph::Graph>& graph)
{
    if (FLAGS_fautocompression == false)
    {
        return true;
    }

    AutoCompressionOptimizer autocompression_optimizer(graph, FLAGS_fautocompression_config);
    autocompression_optimizer.Optimize();

    return true;
}