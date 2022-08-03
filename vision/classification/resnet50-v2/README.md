<!--- SPDX-License-Identifier: Apache-2.0 -->

# ResNet50-v2

## Description

Deeper neural networks are more difficult to train. Residual learning framework
ease the training of networks that are substantially deeper. The research
explicitly reformulate the layers as learning residual functions with reference
to the layer inputs, instead of learning unreferenced functions. It also provide
comprehensive empirical evidence showing that these residual networks are easier
to optimize, and can gain accuracy from considerably increased depth. On the
ImageNet dataset the residual nets were evaluated with a depth of up to 152
layers — 8× deeper than VGG nets but still having lower complexity.

## Model

|Model          |Download                       |ONNX version   |Opset version  |Top-1 accuracy (%) |Top-5 accuracy (%) |
|---------------|:------------------------------|:--------------|:--------------|:------------------|:------------------|
|ResNet50-v2    |[97.7 MB](resnet50-v2-7.onnx)  |1.2.1          |7              |75.81              |92.82              |

## Dataset

[ImageNet (ILSVRC2012)](http://www.image-net.org/challenges/LSVRC/2012/).

## References

* **ResNetv2**
  [Identity mappings in deep residual networks](https://arxiv.org/abs/1603.05027)
  He, Kaiming, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.
  In European Conference on Computer Vision, pp. 630-645. Springer, Cham, 2016.
* [MXNet](http://mxnet.incubator.apache.org),
  [Gluon model zoo](https://cv.gluon.ai/model_zoo/index.html),
  [GluonCV](https://gluon-cv.mxnet.io)
* [Intel® Neural Compressor](https://github.com/intel/neural-compressor)
* [onnx/models](https://github.com/onnx/models/tree/main/vision/classification/resnet)

## Contributors

* [ankkhedia](https://github.com/ankkhedia) (Amazon AI)
* [abhinavs95](https://github.com/abhinavs95) (Amazon AI)
* [mengniwang95](https://github.com/mengniwang95) (Intel)
* [airMeng](https://github.com/airMeng) (Intel)
* [ftian1](https://github.com/ftian1) (Intel)
* [hshen14](https://github.com/hshen14) (Intel)

## License

Apache 2.0
