import torch
import torchvision
from torchquantize import *

model = torchvision.models.resnet18()
cfg = {'conv1':8, 'layer1.0.conv1':12}
input_names = ["input1"]
output_names = ["output1"]
torch_to_onnx(model, cfg, (1,3,224,224),  './model_onnx', './quan_cfg')
