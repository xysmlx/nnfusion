nnfusion mobilenet_ori.onnx -f onnx  -fblockfusion_level=0 -fquantize_cfg mobilenet_quantize_cfg -fbatchnorm_inference_folding=false
