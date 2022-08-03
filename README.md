# Open Pre-Trained Models

## Usage - Git LFS

On default, cloning this repository will not download any ONNX models. Install
Git LFS with `pip install git-lfs`.

To download a specific model:
`git lfs pull --include="[path to model].onnx" --exclude=""`

To download all models:
`git lfs pull --include="*" --exclude=""`

## Usage - Model visualization

You can see visualizations of each model's network architecture by using [Netron](https://github.com/lutzroeder/Netron).
