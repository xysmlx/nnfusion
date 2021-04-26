import torchvision
import torch
import onnx
import onnx.numpy_helper
"""
The main function of this page is to convert pytorch model to onnx model.
Convertion from pytorch model to onnx model is primary so that a critical 
problem is caused that Layer name of pytorch model fail to convert to onnx 
layer name directly. To solve it, we wrap pytorch model in new wrapper which 
multiply bit number and input before computation of each op. Only in this 
way can onnx model get bit number of corresponded layer.
"""
 
class LayernameModuleWrapper(torch.nn.Module):
    def __init__(self, module, module_bit) -> None:
        super().__init__()
        self.module = module
        self.module_bit = module_bit
    
    def forward(self, inputs):
        inputs = inputs*self.module_bit
        inputs = self.module(inputs)
        return inputs
 
def _setattr(model, name, module):
    name_list = name.split(".")
    for name in name_list[:-1]:
        model = getattr(model, name)
    setattr(model, name_list[-1], module)
 
def unwrapper(model_onnx, index2name, config):
    """
    Fill onnx config and remove wrapper node in onnx
    """
    # Support Gemm and Conv
    support_op = ['Gemm', 'Conv', 'Clip']
    idx = 0
    onnx_config = {}
    while idx < len(model_onnx.graph.node):
        nd = model_onnx.graph.node[idx]
        if nd.name[0:4] in support_op and  idx > 1:
            # Grad constant node and multiply node
            const_nd = model_onnx.graph.node[idx-2]
            mul_nd = model_onnx.graph.node[idx-1]
            # Get index number which is transferred by constant node
            index = int(onnx.numpy_helper.to_array(const_nd.attribute[0].t))
            if index != -1:
                name = index2name[index]
                onnx_config[nd.name] = config[name]
            nd.input[0] = mul_nd.input[0]
            # Remove constant node and multiply node
            model_onnx.graph.node.remove(const_nd)
            model_onnx.graph.node.remove(mul_nd)
            idx = idx-2
        idx = idx+1
    return model_onnx, onnx_config
 
def torch_to_onnx(model, config, input_shape, model_path, cfg_path):
    """
    Convert torch model to onnx model and get layer bit config of onnx model.
    """
    # Support Gemm and Conv
    support_op = [torch.nn.Conv2d, torch.nn.Linear, torch.nn.ReLU6]
    # Transfer bit number to onnx layer by using wrapper
    index2name = {}
    name2index = {}
    for i, name in enumerate(config.keys()):
        index2name[i] = name
        name2index[name] = i
    for name, module in model.named_modules():
        if config is not None and name in config:
            assert type(module) in support_op
            wrapper_module = LayernameModuleWrapper(module, name2index[name])
            _setattr(model, name, wrapper_module)
        elif type(module) in support_op:
            wrapper_module = LayernameModuleWrapper(module, -1)
            _setattr(model, name, wrapper_module)
    # Convert torch model to onnx model and save it in model_path
    dummy_input = torch.randn(input_shape)
    model.to('cpu')
    torch.onnx.export(model, dummy_input, model_path, verbose=False, export_params=True)
 
    # Load onnx model
    model_onnx = onnx.load(model_path)
    model_onnx, onnx_config = unwrapper(model_onnx, index2name, config)
    onnx.save(model_onnx, model_path)
 
    onnx.checker.check_model(model_onnx)
    with open(cfg_path, 'w') as cfg_f:
        for key, value in onnx_config.items():
            cfg_f.write('{} {}\n'.format(key, value))



 