#pragma once
#include "../cuda_emitter.hpp"
#include "../cuda_langunit.hpp"

namespace nnfusion
{
    namespace kernels
    {
        namespace cuda
        {
            class QuantizeDot: public CudaLibEmitter
            {
            public:
                QuantizeDot(shared_ptr<KernelContext> ctx);

                LanguageUnit_p emit_function_body() override;
                LanguageUnit_p emit_dependency() override;
                LanguageUnit_p emit_function_signature() override;
                LanguageUnit_p emit_comments() override;
            private:
                // call the corresponding emitter according to the quantize config
                shared_ptr<CudaLibEmitter> concrete_emitter;
            };  

            class QuantizeDot8bit : public CudaLibEmitter
            {
            public:
                QuantizeDot8bit(shared_ptr<KernelContext> ctx);

                LanguageUnit_p emit_function_body() override;
                LanguageUnit_p emit_dependency() override;
                LanguageUnit_p emit_function_signature() override;
                LanguageUnit_p emit_comments() override;
                bool require_cusparse_handle() override { return false; }
            private:
                shared_ptr<KernelContext> kernel_ctx;
                size_t reduction_axes;
            };
        } // namespace cuda
    }     // namespace kernels
} // namespace nnfusion
