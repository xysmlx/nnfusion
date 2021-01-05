// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

REGISTER_OP(ScatterND).infershape([](std::shared_ptr<graph::GNode> gnode) -> void {
    NNFUSION_CHECK(gnode->get_input_size() == 3);
    auto ref_type = gnode->get_input_element_type(0);
    auto ref_shape = gnode->get_input_shape(0);
    auto indicies_type = gnode->get_input_element_type(1);
    auto indicies_shape = gnode->get_input_shape(1);
    auto update_type = gnode->get_input_element_type(2);
    auto update_shape = gnode->get_input_shape(2);

    auto generic_op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(gnode->get_op_ptr());

    NNFUSION_CHECK(ref_type == update_type) << "Variable should have same datetype as Update.";
    NNFUSION_CHECK(ref_shape.size() >= 1) << "Data shape must have rank at least one.";
    NNFUSION_CHECK(indicies_shape.size() >= 1) << "Indices shape must have rank at least one.";
    NNFUSION_CHECK(update_shape.size() >= 1) << "Updates shape must have rank at least one.";

    gnode->set_output_type_and_shape(0, ref_type, ref_shape);
});