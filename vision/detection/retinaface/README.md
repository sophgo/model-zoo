<!--- SPDX-License-Identifier: MIT License -->

# retinaface

## Description

RetinaFace is a deep learning based cutting-edge facial detector
for Python coming with facial landmarks.
Its detection performance is amazing even in the crowd as
shown in the following illustration.
RetinaFace is the face detection module of insightface project.
The original implementation is mainly based on mxnet.
Then,its tensorflow based re-implementation is published by Stanislas Bertrand.
So, this repo is heavily inspired from the study of Stanislas Bertrand.
Its source code is simplified and it is transformed to pip compatible but
the main structure of the reference model and its pre-trained weights are same.

## Model

|Model                |Download                              |Shape(hw)     |AP                |
|---------------------|:-------------------------------------|:-------------|:------------------|
|retinaface_resnet50(onnx)        |[104 MB](retinaface_resnet50.onnx.onnx)                 |839 840       |94.1               |

## Dataset

[WIDER FACE](http://shuoyang1213.me/WIDERFACE/) by Microsoft.

## References

* [retinaface](https://github.com/biubug6/Pytorch_Retinaface)

## License

MIT License
retinaface_resnet50.onnx     (http://219.142.246.77:65000/fsdownload/m6cELCAx7/retinaface)
retinaface（https://github.com/serengil/retinaface）
