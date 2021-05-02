#nnfusion efficient_ori.onnx -f onnx  -fblockfusion_level=0 -fquantize_cfg efficientnetb1_quantize_cfg -fbatchnorm_inference_folding=true
nnfusion efficientb1_ori.onnx -f onnx  -fblockfusion_level=0 -fquantize_cfg efficientnetb1_quantize_cfg -fbatchnorm_inference_folding=true
