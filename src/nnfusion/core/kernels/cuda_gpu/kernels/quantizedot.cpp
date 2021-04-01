// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "quantizedot.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;

cuda::QuantizeDot::QuantizeDot(shared_ptr<KernelContext> ctx)
    : CudaLibEmitter(ctx)
{
    auto quan_node = ctx->gnode;
    auto quan_dot = static_pointer_cast<nnfusion::op::QuantizeDot>(quan_node->get_op_ptr());
    size_t n_quan_bit = quan_dot->get_quantize_bit();
    switch(n_quan_bit){
        case 8:
            concrete_emitter = dynamic_pointer_cast<cuda::CudaLibEmitter>(make_shared<cuda::QuantizeDot8bit>(ctx));
            break;
        case 16:
            // TODO add Quantize16bit here
            break;
        default:
            throw "This quantize config is not supported yet!";
    }
}

LanguageUnit_p cuda::QuantizeDot::emit_function_body()
{
    return concrete_emitter->emit_function_body();
}

LanguageUnit_p cuda::QuantizeDot::emit_comments()
{
    return concrete_emitter->emit_comments();
}

LanguageUnit_p cuda::QuantizeDot::emit_dependency()
{
    return concrete_emitter->emit_dependency();
}


LanguageUnit_p cuda::QuantizeDot::emit_function_signature()
{
    return concrete_emitter->emit_function_signature();
}

/// 8 bit Quantized kernel implementation
cuda::QuantizeDot8bit::QuantizeDot8bit(shared_ptr<KernelContext> ctx)
    : CudaLibEmitter(ctx)
{
    auto quan_node = ctx->gnode;
    auto quan_dot = static_pointer_cast<nnfusion::op::QuantizeDot>(quan_node->get_op_ptr());
    
    std::stringstream tag;
    tag << "QuantizeDot8bit initilization";
    custom_tag = tag.str();

}

LanguageUnit_p cuda::QuantizeDot8bit::emit_function_body()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
    auto& lu = *_lu;
    lu << "Quantize kernel body here!";
    return _lu;
}

LanguageUnit_p cuda::QuantizeDot8bit::emit_comments()
{
    auto& ctx = m_context;


    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
    auto& lu = *_lu;


    lu<<"//Quantize Dot function commments here\n";
    //lu.block_end();
    return _lu;
}

LanguageUnit_p cuda::QuantizeDot8bit::emit_dependency()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
    _lu->require(header::cuda);
    _lu->require(header::cublas);
    _lu->require(header::stdexcept);
    _lu->require(header::sstream);

    _lu->require(macro::CUDA_SAFE_CALL);

    // _lu->require(declaration::cuda_fp16_scale);
    //_lu->require(declaration::cublas_handle);
    return _lu;
}


LanguageUnit_p cuda::QuantizeDot8bit::emit_function_signature()
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
    << "(" << join(params, ", ") << ")";
    return _lu;

}

REGISTER_KERNEL_EMITTER(
    "QuantizeDot",                                                                   // op_name
    Device(CUDA_GPU).TypeConstraint(element::f32).Tag("cuda").Priority(2), // attrs
    cuda::QuantizeDot)                                                               // constructor

REGISTER_KERNEL_EMITTER(
    "QuantizeDot",                                                                   // op_name
    Device(CUDA_GPU).TypeConstraint(element::i8).Tag("cuda").Priority(2), // attrs
    cuda::QuantizeDot)                                                               // constructor

REGISTER_KERNEL_EMITTER(
    "QuantizeDot",                                                                   // op_name
    Device(CUDA_GPU).TypeConstraint(element::f16).Tag("cuda").Priority(2), // attrs
    cuda::QuantizeDot)      

REGISTER_KERNEL_EMITTER(
    "QuantizeDot",                                                                   // op_name
    Device(ROCM_GPU).TypeConstraint(element::f32).Tag("cuda").Priority(2), // attrs
    cuda::QuantizeDot)                                                               // constructor
