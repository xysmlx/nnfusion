
// Microsoft (c) 2019, NNFusion Team

#pragma once

#include "nnfusion/core/operators/op.hpp"
#include "nnfusion/core/operators/op_define/dot.hpp"

namespace nnfusion{
    namespace op{
        class SparseDot: public Op
        {
        public:
            SparseDot(size_t reduction_axes_count,
            bool has_reduction_axes_count = true,
            bool trans_a = false,
            bool trans_b = false);
            SparseDot();
            SparseDot(Dot ori_dot);
    
            void validate_and_infer_types(std::shared_ptr<graph::GNode>) override;
            size_t get_reduction_axes_count() const { return m_reduction_axes_count; }
            void set_transpose(bool trans_a, bool trans_b)
            {
                m_transpose_A = trans_a;
                m_transpose_B = trans_b;
            }

            bool& get_transpose_A() { return m_transpose_A; }
            bool& get_transpose_B() { return m_transpose_B; }
            size_t& get_sparse_index() {return sparse_index; }
        protected:
            size_t m_reduction_axes_count;
            bool m_has_reduction_axes_count;
            bool m_transpose_A = false;
            bool m_transpose_B = false;
            size_t sparse_index;
        };
    }
}