<!--- SPDX-License-Identifier: Apache-2.0 -->

# SqueezeNet

## Description

SqueezeNet is a small CNN which achieves AlexNet level accuracy on ImageNet with
50x fewer parameters. SqueezeNet requires less communication across servers
during distributed training, less bandwidth to export a new model from the cloud
to an autonomous car and more feasible to deploy on FPGAs and other hardware
with limited memory.

## Model

|                 | SqueezeNet v1.0                  |
| :-------------  |:-------------:                   |
| conv1:          | 96 filters of resolution 7x7     |
| pooling layers: | pool_{1,4,8}                     |
| computation     | 1.72 GFLOPS/image                |
|ImageNet accuracy| >= 80.3% top-5                   |

## Dataset

[ImageNet (ILSVRC2012)](http://www.image-net.org/challenges/LSVRC/2012/).

## References

* [forresti/SqueezeNet v1.0](https://github.com/forresti/SqueezeNet/tree/master/SqueezeNet_v1.0).
* [SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and< 0.5MB model size](https://arxiv.org/pdf/1602.07360v3.pdf)

## License

Apache 2.0