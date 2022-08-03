<!--- SPDX-License-Identifier: MIT -->

# Tiny YOLOv3

## Description

This model is a neural network for real-time object detection that detects 80
different classes. It is very fast and accurate. It is a smaller version of
YOLOv3 model.

## Model

|Model              |Download                       |mAP    |
|-------------------|:------------------------------|:------|
|Tiny YOLOv3        |[34 MB](tiny-yolov3-11.onnx)   |0.331  |

## Dataset

We use pretrained weights from pjreddie.com [here](https://pjreddie.com/media/files/yolov3-tiny.weights).

## References

* This model is converted from a keras model [repository](https://github.com/qqwweee/keras-yolo3)
  using keras2onnx converter [repository](https://github.com/onnx/keras-onnx).
* [onnx/models](https://github.com/onnx/models)

## License

MIT License
