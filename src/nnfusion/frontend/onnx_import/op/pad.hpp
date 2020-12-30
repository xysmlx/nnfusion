//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

//----------------------------------------------------------------------------------------------
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See License.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include "../core/node.hpp"
#include "../util/util.hpp"

namespace nnfusion
{
    namespace frontend
    {
        namespace onnx_import
        {
            namespace set_1
            {
                NamedNodeVector TranslatePadOp(const onnx::NodeProto& node_proto,
                                               const NodeMap& all_ng_nodes,
                                               std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    auto input_gnode = GetInputNode(all_ng_nodes, node_proto, 0);
                    Node node(node_proto);

                    // Parse ONNX op attributes
                    auto pad_mode = node.get_attribute_value<std::string>("mode", "");
                    NNFUSION_CHECK(pad_mode == "constant") << "only support constant padding yet.";
                    NNFUSION_CHECK(abs(node.get_attribute_value<float>("value", 0)) < 1e-6)
                        << "only support padding value 0 yet.";

                    NNFUSION_CHECK(node.has_attribute("pads")) << "do not support this opset yet.";
                    auto pads_int64 = node.get_attribute_value<std::vector<int64_t>>("pads");
                    NNFUSION_CHECK(pads_int64.size() % 2 == 0);

                    // Convert padding from pads_int64 to Shape objects
                    Shape padding_below_shape{std::begin(pads_int64),
                                              std::begin(pads_int64) + pads_int64.size() / 2};
                    std::reverse(padding_below_shape.begin(), padding_below_shape.end());
                    Shape padding_above_shape{std::begin(pads_int64) + pads_int64.size() / 2,
                                              std::end(pads_int64)};
                    std::reverse(padding_above_shape.begin(), padding_above_shape.end());
                    NNFUSION_CHECK(padding_below_shape.size() == padding_above_shape.size());
                    Shape padding_interior(padding_below_shape.size());
                    for (size_t i = 0; i < padding_interior.size(); i++)
                    {
                        padding_interior[i] = 0;
                    }

                    std::shared_ptr<op::Op> pad_op = std::make_shared<op::Pad>(
                        padding_below_shape, padding_above_shape, padding_interior);
                    pad_op->set_name(node_proto.output(0));

                    auto pad_val_op =
                        std::make_shared<op::Constant>(input_gnode->get_element_type(),
                                                       nnfusion::Shape{},
                                                       std::vector<std::string>{"0"});
                    auto pad_val_gnode = m_graph->add_node_and_edge(pad_val_op, GNodeVector({}));

                    auto pad_gnode =
                        m_graph->add_node_and_edge(pad_op, {input_gnode, pad_val_gnode});

                    NamedNodeVector ret{{node_proto.output(0), pad_gnode}};
                    return ret;
                }

            } // namespace set_1
        }     // namespace onnx_import
    }         // namespace frontend
} // namespace nnfusion
