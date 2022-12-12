<!--- SPDX-License-Identifier: Apache-2.0 -->

# ppyoloe

## Description

PP-YOLOE is an excellent single-stage anchor-free model based on PP-YOLOv2,
surpassing a variety of popular YOLO models. PP-YOLOE has a series of models,
named s/m/l/x, which are configured through width multiplier and depth multiplier.
PP-YOLOE avoids using special operators, such as Deformable Convolution or
Matrix NMS, to be deployed friendly on various hardware.

## Model

| Model                       | Download                                 | Shape(hw) |
| --------------------------- |:---------------------------------------- |:--------- |
| ppyoloe_crn_x_300e_coco     | [341MB](ppyoloe_crn_x_300e_coco.zip)     | 640 640   |
| ppyoloe_plus_crn_x_80e_coco | [345MB](ppyoloe_plus_crn_x_80e_coco.zip) | 640 640   |

## Dataset

* [coco](http://images.cocodataset.org/zips/val2017.zip)

## References

* [PP-YOLOE: An evolved version of YOLO](https://arxiv.org/abs/2203.16250)
* [PaddlePaddle/ppyoloe](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.5/configs/ppyoloe)

## License

Apache 2.0
