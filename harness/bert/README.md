# Bert Benchmark mlperf

-----------------------------------

## 数据集
下载squad-v1.1数据集
```shell
wget https://raw.githubusercontent.com/nate-parrott/squad/master/data/dev-v1.1.json
```
或者
```
pip install dfn
python -m dfn --url=http://219.142.246.77:65000/sharing/jr3evzZlC
```

## 依赖

1、编译libpipeline.so

2、安装 `pip install tpu_perf-1.0.11-py3-none-manylinux2014_x86_64.whl`
## Inference

```shell
./scripts/run.sh
```
参数说明：
--accuracy: 是否打开计算精度 

--count: 运行的样本数，计算精度时最多为10833


