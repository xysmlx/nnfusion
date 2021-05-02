
// Microsoft (c) 2019, NNFusion Team

#pragma once

#include "nnfusion/core/operators/op.hpp"
#include "nnfusion/core/operators/op_define/dot.hpp"

namespace nnfusion
{
    namespace op
    {
        class QuantizeDepthwiseConv2dNative : public Op
        {
        public:

            QuantizeDepthwiseConv2dNative(size_t qunatizebit);

            void validate_and_infer_types(std::shared_ptr<graph::GNode>) override;

        protected:
            int stridw_h;
            int stride_w;
            int kernel_h;
            int kernel_w;
            size_t m_quantize_bit;
        };
        
    } // namespace op
} // namespace nnfusion