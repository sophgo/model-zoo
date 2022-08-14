<!--- SPDX-License-Identifier: Apache-2.0 -->

# SqueezeNet

## Use cases
Computationally efficient CNN architecture designed specifically for mobile devices with very limited computing power.

## Description

ShuffleNet is a deep convolutional network for image classification. [ShuffleNetV2](https://pytorch.org/hub/pytorch_vision_shufflenet_v2/) is an improved architecture that is the state-of-the-art in terms of speed and accuracy tradeoff used for image classification.

PyTorch ShuffleNet-v2 ==> ONNX ShuffleNet-v2

## Model

|Model        |Download  |ONNX version|Opset version|Top-1 error |Top-5 error |
|-------------|:--------------|:--------------|:--------------|:--------------|:--------------|
|ShuffleNet-v2-fp32 |[8.79MB](./shufflenet-v2-12.onnx) |1.9 |12 |33.65 |13.43|

### Input to model
Input to the model are 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224.

data_0: float[1, 3, 224, 224]

### Preprocessing steps
All pre-trained models expect input images normalized in the same way. The images have to be loaded in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].

```python
input_image = Image.open(filename)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
```
Create a mini-batch as expected by the model.
```python
input_batch = input_tensor.unsqueeze(0)
```

## Dataset

[ImageNet (ILSVRC2012)](http://www.image-net.org/challenges/LSVRC/2012/).

## References
* Ningning Ma, Xiangyu Zhang, Hai-Tao Zheng, Jian Sun. ShuffleNet V2: Practical Guidelines for EfficientCNN Architecture Design. 2018.

* huffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices]

* [Intel® Neural Compressor](https://github.com/intel/neural-compressor)
<hr>

## Contributors
* Ksenija Stanojevic
* [mengniwang95](https://github.com/mengniwang95) (Intel)
* [airMeng](https://github.com/airMeng) (Intel)
* [ftian1](https://github.com/ftian1) (Intel)
* [hshen14](https://github.com/hshen14) (Intel)
<hr>

## License
BSD 3-Clause License
<hr>
