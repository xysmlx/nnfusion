
// Microsoft (c) 2019, NNFusion Team

#pragma once

#include "nnfusion/core/operators/op.hpp"
#include "nnfusion/core/operators/op_define/dot.hpp"

namespace nnfusion
{
    namespace op
    {
        class BitConverter : public Op
        {
        public:

            BitConverter(size_t inbit, size_t outbit);

            void validate_and_infer_types(std::shared_ptr<graph::GNode>) override;
 

        protected:
            size_t inbit;
            size_t outbit;
        };
        
    } // namespace op
} // namespace nnfusion