
// Microsoft (c) 2019, NNFusion Team

#pragma once

#include "nnfusion/core/operators/op.hpp"
#include "nnfusion/core/operators/op_define/dot.hpp"

namespace nnfusion
{
    namespace op
    {
        class QuantizeDot : public Op
        {
        public:

            QuantizeDot(std::shared_ptr<Dot> ori_dot, size_t qunatizebit);

            void validate_and_infer_types(std::shared_ptr<graph::GNode>) override;
            size_t get_reduction_axes_count() const { return m_reduction_axes_count; }
            void set_transpose(bool trans_a, bool trans_b)
            {
                m_transpose_A = trans_a;
                m_transpose_B = trans_b;
            }

            bool& get_transpose_A() { return m_transpose_A; }
            bool& get_transpose_B() { return m_transpose_B; }
            size_t get_quantize_bit() {return m_quantize_bit; }

        protected:
            size_t m_reduction_axes_count;
            bool m_has_reduction_axes_count;
            bool m_transpose_A = false;
            bool m_transpose_B = false;
            size_t m_quantize_bit;
        };
        
    } // namespace op
} // namespace nnfusion