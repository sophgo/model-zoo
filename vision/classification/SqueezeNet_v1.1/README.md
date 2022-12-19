<!--- SPDX-License-Identifier: Apache-2.0 -->

# SqueezeNet_v1.1

## Description

SqueezeNet v1.1 has 2.4x less computation than v1.0, without sacrificing accuracy.

## Model

**What's new in SqueezeNet v1.1?**

|                 | SqueezeNet v1.1                  |
| :-------------  | :-----:                          |
| conv1:          | 64 filters of resolution 3x3     |
| pooling layers: | pool_{1,3,5}                     |
| computation     | 0.72 GFLOPS/image                |
|ImageNet accuracy| >= 80.3% top-5                   |

## Dataset

[ImageNet (ILSVRC2012)](http://www.image-net.org/challenges/LSVRC/2012/).

## References

* [forresti/SqueezeNet v1.1](https://github.com/forresti/SqueezeNet/tree/master/SqueezeNet_v1.1)
* [SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and< 0.5MB model size](https://arxiv.org/pdf/1602.07360v3.pdf)

## License

Apache 2.0