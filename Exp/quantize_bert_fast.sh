nnfusion bert_original.onnx -f onnx  -fblockfusion_level=0 -fquantize_cfg bert_cfg -fenable_all_bert_fusion=true -fgelu_fusion=true
