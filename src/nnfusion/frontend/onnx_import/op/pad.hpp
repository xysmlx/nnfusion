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
            namespace set_2
            {
                NamedNodeVector TranslatePadOp(const onnx::NodeProto& node_proto,
                                               const NodeMap& all_ng_nodes,
                                               std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    auto input_gnode = GetInputNode(all_ng_nodes, node_proto, 0);
                    Node node(node_proto);

                    auto mode = node.get_attribute_value<std::string>("mode", "");
                    NNFUSION_CHECK(mode == "constant")
                        << "NNFusion only supports constant mode for ONNX pad op yet.";

                    auto pads = node.get_attribute_value<std::vector<int64_t>>(
                        "pads", std::vector<int64_t>(input_gnode->get_shape().size() * 2, 0));

                    auto padding_value = node.get_attribute_value<float>("value", 0.0f);
                    NNFUSION_CHECK(fabs(padding_value) < 1e-5)
                        << "NNFusion only supports padding value 0 for ONNX pad op yet.";

                    Shape padding_above = Shape(pads.begin(), pads.begin() + pads.size() / 2);
                    Shape padding_below = Shape(pads.begin() + pads.size() / 2, pads.end());
                    Shape padding_interior = Shape(std::vector<size_t>(pads.size() / 2, 0));

                    auto pad_val_op =
                        std::make_shared<op::Constant>(input_gnode->get_element_type(),
                                                       nnfusion::Shape{},
                                                       std::vector<std::string>{"0"});
                    auto pad_val_gnode = m_graph->add_node_and_edge(pad_val_op, GNodeVector({}));

                    auto pad_op =
                        std::make_shared<op::Pad>(padding_below, padding_above, padding_interior);
                    pad_op->set_name(node_proto.output(0));

                    auto pad_gnode =
                        m_graph->add_node_and_edge(pad_op, {input_gnode, pad_val_gnode});

                    NamedNodeVector ret{{node_proto.output(0), pad_gnode}};
                    return ret;
                }

            } // namespace set_2
        }     // namespace onnx_import
    }         // namespace frontend
} // namespace nnfusion
