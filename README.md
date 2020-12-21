# resnet-pytorch-tf-mxnet-scratch

In this repo i introduce to you how to create a resnet model from scratch. The intuitively tutorial refer to [ResNet Model - Khanh's Blog](https://phamdinhkhanh.github.io/2020/12/19/Resnet.html)

# Architecture

The repository tree is designed as in below:

```
├── block
│   ├── res_block_mx.py
│   ├── res_block_pt.py
│   └── res_block_tf.py
├── models
│   ├── mxnet
│   │   └── resnet_mx_18.py
│   ├── pytorch
│   │   └── resnet_pt_18.py
│   └── tf
│       └── resnet_tf_18.py
├── test_mx_2.py
├── test_mx.py
├── test_pt.py
└── test_tf.py
```

* `block`: folder include those residual resnet blocks in `tensorflow, pytorch, mxnet`.
* `model`: resnet model in `pytorch, tensorflow, mxnet`.
* `test_mx.py, test_pt.py, test_tf.py`: files test model corresponding with DL frameworks.

# Install DL frameworks

Firstly, you should install `tensorflow, pytorch, mxnet` base on DL frameworks you want to test. Version as bellow:

* `tensorflow`: >=2.0.0
* `pytorch`: >=1.4.0
* `mxnet-cuxxx`: mxnet on cuda. For example `mxnet-cu101`. You run `nvcc --version` to find cuda version.

You can run `pip install requirements.txt` to install the whole frameworks.

# Run model

To test model on specific framework:

* tensorflow:

```
python3 test_tf.py
```

* pytorch:

```
python3 test_pt.py
```

* mxnet:

```
python3 test_mx.py
```
