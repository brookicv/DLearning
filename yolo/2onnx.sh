rm models/*
python 2onnx.py
python3 -m onnxsim models/yolov3_608.onnx models/sim_yolov3_608.onnx
~/onnx2ncnn models/sim_yolov3_608.onnx models/yolov3.param models/yolov3.bin