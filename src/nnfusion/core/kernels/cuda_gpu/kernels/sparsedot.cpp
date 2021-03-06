// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "sparsedot.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;

cuda::SparseDot::SparseDot(shared_ptr<KernelContext> ctx)
    : CudaLibEmitter(ctx)
{
    auto dot_op = static_pointer_cast<nnfusion::op::SparseDot>(ctx->gnode->get_op_ptr());

    // reduction_axes = dot_op->get_reduction_axes_count();
    // arg0_shape = nnfusion::Shape(ctx->inputs[0]->get_shape());
    // arg1_shape = nnfusion::Shape(ctx->inputs[1]->get_shape());
    // out_shape = nnfusion::Shape(ctx->outputs[0]->get_shape());
    // dtype = nnfusion::element::Type(ctx->outputs[0]->get_element_type());

    std::stringstream tag;
    tag << "SparseDot initilization";
    custom_tag = tag.str();
}

LanguageUnit_p cuda::SparseDot::emit_function_body()
{
    auto& ctx = m_context;
    // auto gemm = static_pointer_cast<nnfusion::op::Dot>(ctx->gnode->get_op_ptr());
    // auto trans_A = gemm->get_transpose_A();
    // auto trans_B = gemm->get_transpose_B();
    // auto dtype = ctx->outputs[0]->get_element_type();

    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
    auto& lu = *_lu;

    // function signature:
    // void kernel(m_context->dtypes[0]* input0, m_context->dtypes[0]* input1, m_context->dtypes[2]* output0)

    //lu.block_begin();
    lu<<"SparseDot function body here";
    //lu.block_end();
    return _lu;
}

LanguageUnit_p cuda::SparseDot::emit_comments()
{
    auto& ctx = m_context;
    // auto gemm = static_pointer_cast<nnfusion::op::Dot>(ctx->gnode->get_op_ptr());
    // auto trans_A = gemm->get_transpose_A();
    // auto trans_B = gemm->get_transpose_B();
    // auto dtype = ctx->outputs[0]->get_element_type();

    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
    auto& lu = *_lu;

    // function signature:
    // void kernel(m_context->dtypes[0]* input0, m_context->dtypes[0]* input1, m_context->dtypes[2]* output0)

    //lu.block_begin();
    lu<<"//SparseDot function commments here";
    //lu.block_end();
    return _lu;
}

LanguageUnit_p cuda::SparseDot::emit_dependency()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
    _lu->require(header::cuda);
    _lu->require(header::cublas);
    _lu->require(header::stdexcept);
    _lu->require(header::sstream);
    _lu->require(macro::CUSPARSE_SAFE_CALL);
    _lu->require(macro::CUDA_SAFE_CALL);

    // _lu->require(declaration::cuda_fp16_scale);
    //_lu->require(declaration::cublas_handle);
    return _lu;
}

LanguageUnit_p cuda::SparseDot::emit_function_signature()
{
    LanguageUnit_p _lu(new LanguageUnit(this->m_kernel_name + "_sig"));
    auto& lu = *_lu;

    vector<string> params;
    for (size_t i = 0; i < m_context->inputs.size(); i++)
    {
        stringstream ss;
        ss << m_context->inputs[i]->get_element_type().c_type_string() << "* ";
        ss << "input" << i;
        params.push_back(ss.str());
    }

    for (size_t i = 0; i < m_context->outputs.size(); i++)
    {
        stringstream ss;
        ss << m_context->outputs[i]->get_element_type().c_type_string() << "* ";
        ss << "output" << i;
        params.push_back(ss.str());
    }
    lu << "void "
    << "(cublasHandle_t cublas_handle, " << join(params, ", ") << ")";
    return _lu;

}

REGISTER_KERNEL_EMITTER(
    "SparseDot",                                                                   // op_name
    Device(CUDA_GPU).TypeConstraint(element::f32).Tag("cusparse").Priority(2), // attrs
    cuda::SparseDot)                                                               // constructor

REGISTER_KERNEL_EMITTER(
    "SparseDot",                                                                   // op_name
    Device(CUDA_GPU).TypeConstraint(element::f16).Tag("cusparse").Priority(2), // attrs
    cuda::SparseDot)                                                               // constructor

REGISTER_KERNEL_EMITTER(
    "SparseDot",                                                                   // op_name
    Device(ROCM_GPU).TypeConstraint(element::f32).Tag("cusparse").Priority(2), // attrs
    cuda::SparseDot)                                                               // constructor
