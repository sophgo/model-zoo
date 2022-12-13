<!--- SPDX-License-Identifier: Wide Resnet50 v2 -->

# wide_resnet_50

## Description

Deep residual networks were shown to be able to scale up to thousands of layers and still have improving performance. However, each fraction of a percent of improved accuracy costs nearly doubling the number of layers, and so training very deep residual networks has a problem of diminishing feature reuse, which makes these networks very slow to train. To tackle these problems, in this paper we conduct a detailed experimental study on the architecture of ResNet blocks, based on which we propose a novel architecture where we decrease depth and increase width of residual networks. We call the resulting network structures wide residual networks (WRNs) and show that these are far superior over their commonly used thin and very deep counterparts. For example, we demonstrate that even a simple 16-layer-deep wide residual network outperforms in accuracy and efficiency all previous deep residual networks, including thousand-layer-deep networks, achieving new state-of-the-art results on CIFAR, SVHN, COCO, and significant improvements on ImageNet. Our code and models are available at this https URL


## Model

|Model          |Download                       |pythorch version   |Top-1 accuracy (%) |Top-5 accuracy (%) |
|---------------|:------------------------------|:--------------|:------------------|:------------------|
|Wide Resnet50 v2 |[275.4 MB](wide-resnet-50-2-export-5ae25d50.pth)  |1.8.0          |78.00              |93.95              |

## Dataset

[ImageNet (ILSVRC2012)](http://www.image-net.org/challenges/LSVRC/2012/).

## References

* **WideResnet50**
  [Wide Residual Networks](https://arxiv.org/abs/1605.07146)
  Sergey Zagoruyko, Nikos Komodakis.
* [pytorch](https://github.com/szagoruyko/functional-zoo/blob/master/wide-resnet-50-2-export.ipynb)
* [lua](https://github.com/szagoruyko/wide-residual-networks/tree/master/pretrained)

## Contributors

## License

Apache 2.0
