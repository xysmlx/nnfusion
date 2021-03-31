import torch
import torchvision
from torchquantize import *

model = torchvision.models.resnet18()
cfg = {'conv1':8}
input_names = ["input1"]
output_names = ["output1"]
_, onnx_cfg = torch_to_onnx(model, cfg, (1,3,224,224),  './model_onnx', input_names, output_names)
print(onnx_cfg)