<!--- SPDX-License-Identifier: Apache-2.0 -->

# Facenet

## Description

This is a implementation of the face recognizer described in the paper
["FaceNet: A Unified Embedding for Face Recognition and Clustering"](http://arxiv.org/abs/1503.03832). The project also uses ideas from the paper ["Deep Face Recognition"](http://www.robots.ox.ac.uk/~vgg/publications/2015/Parkhi15/parkhi15.pdf) from the [Visual Geometry Group](http://www.robots.ox.ac.uk/~vgg/) at Oxford.

## Model

| Model name      | LFW accuracy | Training dataset | Architecture        |
|-----------------|--------------|------------------|-------------------- |
| 20180408-102900 | 0.9905       | CASIA-WebFace    | Inception ResNet v1 |
| 20180402-114759 | 0.9965       | VGGFace2         | Inception ResNet v1 |

## Dataset

The CASIA-WebFace dataset has been used for training. This training set consists of total of 453 453 images over 10 575 identities after face detection. Some performance improvement has been seen if the dataset has been filtered before training. Some more information about how this was done will come later.
The best performing model has been trained on the [VGGFace2](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/) dataset consisting of ~3.3M faces and ~9000 classes.

## References

* [Face Recognition using Tensorflow](https://github.com/davidsandberg/facenet#face-recognition-using-tensorflow-)
* [Inception ResNet v1](https://github.com/davidsandberg/facenet/blob/master/src/models/inception_resnet_v1.py)
* [20180408-102900](https://drive.google.com/open?id=1R77HmFADxe87GmoLwzfgMu_HY0IhcyBz)
* [20180402-114759](https://drive.google.com/open?id=1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-)

## License

Apache-2.0