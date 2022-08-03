<!--- SPDX-License-Identifier: Apache-2.0 -->

# SqueezeNet

## Description

SqueezeNet is a small CNN which achieves AlexNet level accuracy on ImageNet with
50x fewer parameters. SqueezeNet requires less communication across servers
during distributed training, less bandwidth to export a new model from the cloud
to an autonomous car and more feasible to deploy on FPGAs and other hardware
with limited memory.

## Model

|Model          |Download                               | ONNX version  |Opset version  |Top-1 accuracy (%) |Top-5 accuracy (%) |
|---------------|:--------------------------------------|:--------------|:--------------|:------------------|:------------------|
|SqueezeNet 1.0 | [5 MB](squeezenet1.0-12.onnx)         |  1.9          | 12            |56.85              |79.87              |

## Dataset

[ImageNet (ILSVRC2012)](http://www.image-net.org/challenges/LSVRC/2012/).

## References

* **SqueezeNet1.0** Model from the paper [SqueezeNet](https://arxiv.org/abs/1602.07360)
* [MXNet](http://mxnet.incubator.apache.org),
  [Gluon model zoo](https://cv.gluon.ai/model_zoo/index.html),
  [GluonCV](https://gluon-cv.mxnet.io)
* [Intel® Neural Compressor](https://github.com/intel/neural-compressor)
* [onnx/models](https://github.com/onnx/models)

## Contributors

* [abhinavs95](https://github.com/abhinavs95) (Amazon AI)
* [ankkhedia](https://github.com/ankkhedia) (Amazon AI)
* [mengniwang95](https://github.com/mengniwang95) (Intel)
* [airMeng](https://github.com/airMeng) (Intel)
* [ftian1](https://github.com/ftian1) (Intel)
* [hshen14](https://github.com/hshen14) (Intel)

## License

Apache 2.0
