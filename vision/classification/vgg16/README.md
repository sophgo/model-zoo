<!--- SPDX-License-Identifier: Apache-2.0 -->

# VGG16

## Description

VGG models perform image classification - they take images as input and classify
the major object in the image into a set of pre-defined classes. They are
trained on ImageNet dataset which contains images from 1000 classes.

VGG models provide very high accuracies but at the cost of increased model
sizes. They are ideal for cases when high accuracy of classification is
essential and there are limited constraints on model sizes.

## Model

|Model          |Download                   |Top-1 accuracy (%) |Top-5 accuracy (%) |
|:--------------|:--------------------------|:------------------|:------------------|
|VGG 16-fp32    |[527.8 MB](vgg16-12.onnx)  |72.38              |91.00              |

## Dataset

[ImageNet (ILSVRC2012)](http://www.image-net.org/challenges/LSVRC/2012/).

## References

* **VGG 16** and **VGG 19** are from the paper
  [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)

* **VGG 16_bn** and **VGG 19_bn** are the same models as above but with batch
  normalization applied after each convolution layer

* [MXNet](http://mxnet.incubator.apache.org),
  [Gluon model zoo](https://cv.gluon.ai/model_zoo/index.html)
  [GluonCV](https://gluon-cv.mxnet.io)

* [Intel® Neural Compressor](https://github.com/intel/neural-compressor)

* [https://github.com/onnx/models](https://github.com/onnx/models/tree/main/vision/classification/vgg)

## Contributors

* [abhinavs95](https://github.com/abhinavs95) (Amazon AI)
* [ankkhedia](https://github.com/ankkhedia) (Amazon AI)
* [mengniwang95](https://github.com/mengniwang95) (Intel)
* [airMeng](https://github.com/airMeng) (Intel)
* [ftian1](https://github.com/ftian1) (Intel)
* [hshen14](https://github.com/hshen14) (Intel)

## License

Apache 2.0
