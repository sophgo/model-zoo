<!--- SPDX-License-Identifier: Apache-2.0 -->

# Wrn50

## Description

Deep residual networks were shown to be able to scale up to thousands of layers and still have improving performance. However, each fraction of a percent of improved accuracy costs nearly doubling the number of layers, and so training very deep residual networks has a problem of diminishing feature reuse, which makes these networks very slow to train.

To tackle these problems, in this work we conduct a detailed experimental study on the architecture of ResNet blocks, based on which we propose a novel architecture where we decrease depth and increase width of residual networks. We call the resulting network structures wide residual networks (WRNs) and show that these are far superior over their commonly used thin and very deep counterparts

## Model

|Model                |Download                                                       |Top-1 accuracy (%) |Top-5 accuracy (%) |
|---------------------|:--------------------------------------------------------------|:------------------|:------------------|
| Wrn50               |[model](deploy_wrn50-2.prototxt) [weight](wrn50-2.caffemodel)  | 77.87             | 93.87             |


## References

* This code was used for experiments with Wide Residual Networks by Sergey Zagoruyko and Nikos Komodakis
* [Wide Residual Networks(BMVC 2016)](http://arxiv.org/abs/1605.07146)


## License
MIT License
