<!--- SPDX-License-Identifier: Apache-2.0 -->

# MobileNet

[Original](https://github.com/onnx/models/commit/cbda9ebd037241c6c6a0826971741d5532af8fa4)

## Use cases

MobileNet models perform image classification - they take images as input and
classify the major object in the image into a set of pre-defined classes. They
are trained on ImageNet dataset which contains images from 1000 classes.
MobileNet models are also very efficient in terms of speed and size and hence
are ideal for embedded and mobile applications.

## Description

MobileNet improves the state-of-the-art performance of mobile models on multiple
tasks and benchmarks as well as across a spectrum of different model sizes.
MobileNet is based on an inverted residual structure where the shortcut
connections are between the thin bottleneck layers. The intermediate expansion
layer uses lightweight depthwise convolutions to filter features as a source of
non-linearity. Additionally, it removes non-linearities in the narrow layers in
order to maintain representational power.

## Model

MobileNet reduces the dimensionality of a layer thus reducing the dimensionality
of the operating space. The  trade off between computation and accuracy is
exploited in Mobilenet via a width multiplier parameter approach which allows
one to reduce the dimensionality of the activation space until the manifold of
interest spans this entire space.

The below model is using multiplier value as 1.0.

| Model            | Download                      | ONNX version | Opset version | Top-1 accuracy (%) | Top-5 accuracy (%) |
|:-----------------|:------------------------------|:-------------|:--------------|:-------------------|:-------------------|
| MobileNet v2-1.0 | [13.6 MB](mobilenetv2-7.onnx) |        1.2.1 |             7 |              70.94 |              89.99 |

### Input

All pre-trained models expect input images normalized in the same way, i.e.
mini-batches of 3-channel RGB images of shape (N x 3 x H x W), where N is the
batch size, and H and W are expected to be at least 224.

### Preprocessing

The images have to be loaded in to a range of [0, 1] and then normalized using
mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]. The transformation
should preferrably happen at preprocessing.

### Output

The model outputs image scores for each of the 1000 classes of ImageNet.

### Postprocessing

The post-processing involves calculating the softmax probablility scores for
each class and sorting them to report the most probable classes.

## Dataset

Dataset used for train and validation: [ImageNet (ILSVRC2012)](http://www.image-net.org/challenges/LSVRC/2012/).

## Validation accuracy

The accuracies obtained by the model on the validation set are mentioned above.
The accuracies have been calculated on center cropped images with a maximum
deviation of 1% (top-1 accuracy) from the paper.

## References

* **MobileNet-v2** Model from the paper
  [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)
* [MXNet](http://mxnet.incubator.apache.org),
  [Gluon model zoo](https://cv.gluon.ai/model_zoo/index.html),
  [GluonCV](https://gluon-cv.mxnet.io)

## Contributors

* [ankkhedia](https://github.com/ankkhedia) (Amazon AI)
* [abhinavs95](https://github.com/abhinavs95) (Amazon AI)

## License

Apache 2.0
