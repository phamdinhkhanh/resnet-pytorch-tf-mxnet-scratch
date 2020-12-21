import mxnet as mx
from mxnet.gluon import nn as mxnn

class ResidualBlockMxnet(mxnn.HybridSequential):
  def __init__(self, num_channels, output_channels, strides=1, is_used_conv11=False, **kwargs):
    super(ResidualBlockMxnet, self).__init__(**kwargs)
    self.is_used_conv11 = is_used_conv11
    self.conv1 = mxnn.Conv2D(num_channels, padding=1, 
                           kernel_size=3, strides=1)
    self.batch_norm = mxnn.BatchNorm()
    self.conv2 = mxnn.Conv2D(num_channels, padding=1, 
                           kernel_size=3, strides=1)
    if self.is_used_conv11:
      self.conv3 = mxnn.Conv2D(num_channels, padding=0, 
                           kernel_size=1, strides=1)
    self.conv4 = mxnn.Conv2D(output_channels, padding=0, 
                           kernel_size=1, strides=strides)
    self.relu = mxnn.Activation('relu')
    
  def forward(self, X):
    if self.is_used_conv11:
      Y = self.conv3(X)
    else:
      Y = X
    X = self.conv1(X)
    X = self.relu(X)
    X = self.batch_norm(X)
    X = self.relu(X)
    X = self.conv2(X)
    X = self.batch_norm(X)
    X = self.relu(X+Y)
    X = self.conv4(X)
    return X

X = mx.nd.random.uniform(shape=(4, 1, 28, 28)) # shape=(batch_size, channels, width, height)
res_block = ResidualBlockMxnet(num_channels=1, output_channels=64, strides=2, is_used_conv11=True)
# you must initialize parameters for mxnet block.
res_block.initialize()
X = res_block(X)
print(X.shape)


import mxnet as mx
from mxnet.gluon import nn as mxnn

class ResNet18Mxnet(mxnn.HybridSequential):
  def __init__(self, residual_blocks, output_shape, **kwargs):
    super(ResNet18Mxnet, self).__init__(**kwargs)
    self.conv1 = mxnn.Conv2D(channels=64, padding=3, 
                           kernel_size=7, strides=2)
    self.batch_norm = mxnn.BatchNorm()
    self.max_pool = mxnn.MaxPool2D(pool_size=3)
    self.relu = mxnn.Activation('relu')
    self.residual_blocks = residual_blocks
    self.global_avg_pool = mxnn.GlobalAvgPool2D()
    self.dense = mxnn.Dense(units=output_shape)
    self.blk = mxnn.HybridSequential()
    for residual_block in self.residual_blocks:
      self.blk.add(residual_block)
  
  def forward(self, X):
    X = self.conv1(X)
    X = self.batch_norm(X)
    X = self.relu(X)
    X = self.max_pool(X)
    X = self.blk(X)
    X = self.global_avg_pool(X)
    X = self.dense(X)
    return X

residual_blocks = [
    # Two start conv mapping
    ResidualBlockMxnet(num_channels=64, output_channels=64, strides=2, is_used_conv11=False),
    ResidualBlockMxnet(num_channels=64, output_channels=64, strides=2, is_used_conv11=False),
    # Next three [conv mapping + identity mapping]
    ResidualBlockMxnet(num_channels=64, output_channels=128, strides=2, is_used_conv11=True),
    ResidualBlockMxnet(num_channels=128, output_channels=128, strides=2, is_used_conv11=False),
    ResidualBlockMxnet(num_channels=128, output_channels=256, strides=2, is_used_conv11=True),
    ResidualBlockMxnet(num_channels=256, output_channels=256, strides=2, is_used_conv11=False),
    ResidualBlockMxnet(num_channels=256, output_channels=512, strides=2, is_used_conv11=True),
    ResidualBlockMxnet(num_channels=512, output_channels=512, strides=2, is_used_conv11=False)
]

mxmodel = ResNet18Mxnet(residual_blocks, output_shape=10)
mxmodel.hybridize()

mx.viz.print_summary(
    mxmodel(mx.sym.var('data')), 
    shape={'data':(4, 1, 28, 28)}, #set your shape here
)


from mxnet import nd, gluon, init, autograd, gpu, cpu
from mxnet.gluon import nn
from mxnet.gluon.data.vision import datasets, transforms
import matplotlib.pyplot as plt
import time

mnist_train = datasets.MNIST(train=True)
mnist_val = datasets.MNIST(train=False)

transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.05, 0.05)])

mnist_train = mnist_train.transform_first(transformer)
mnist_val = mnist_val.transform_first(transformer)

batch_size = 32
train_data = gluon.data.DataLoader(
    mnist_train, batch_size=batch_size, shuffle=True, num_workers=4)
valid_data = gluon.data.DataLoader(
    mnist_val, batch_size=batch_size, shuffle=True, num_workers=4)

use_gpu = True
if use_gpu:
  # incase you have more than one GPU, you can add gpu(1), gpu(2),...
  devices = [gpu(0)]
else:
  devices = [cpu()]
print('devices: ', devices)

# net = mx.gluon.model_zoo.vision.mobilenet1_0(pretrained=True)
# net.hybridize()
# net(mx.nd.ones((1,3,224,224)))
# net.export('mobilenet1_0')
mxmodel = ResNet18Mxnet(residual_blocks, output_shape=10)
mxmodel.initialize(init=init.Xavier(), ctx=devices, force_reinit=True)
mxmodel.hybridize()
mxmodel.collect_params()
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(mxmodel.collect_params(), 'adam', {'learning_rate': 0.001})
                        
def acc(output, label):
    # output: (batch, num_output) float32 ndarray
    # label: (batch, ) int32 ndarray
    return (output.argmax(axis=1) ==
            label.astype('float32')).mean().asscalar()

for epoch in range(10):
    train_loss, train_acc, valid_acc = 0., 0., 0.
    tic = time.time()
    for i, (inputs, labels) in enumerate(train_data):
        actual_batch_size = inputs.shape[0]
        # Split data among GPUs. Since split_and_load is a deterministic function
        # inputs and labels are going to be split in the same way between GPUs.
        inputs = mx.gluon.utils.split_and_load(inputs, ctx_list=devices, even_split=False)
        labels = mx.gluon.utils.split_and_load(labels, ctx_list=devices, even_split=False)
        with mx.autograd.record():
          for input, label in zip(inputs, labels):
            output = mxmodel(input)
            loss = softmax_cross_entropy(output, label)

        loss.backward()
        # update parameters
        trainer.step(batch_size)
        mxmodel.hybridize()
        mxmodel.export("output", epoch=1)
        # calculate training metrics
        train_loss += loss.mean().asscalar()
        train_acc += acc(output, label)
        if i % 500 == 499:
          print("Epoch %d: Step %d: loss %.3f, train acc %.3f" % (
              epoch, i+1, train_loss/i, train_acc/i))
         
    # calculate validation accuracy
    for inputs, labels in valid_data:
        actual_batch_size = inputs.shape[0]
        # Split data among GPUs. Since split_and_load is a deterministic function
        # inputs and labels are going to be split in the same way between GPUs.
        inputs = mx.gluon.utils.split_and_load(inputs, ctx_list=devices, even_split=False)
        labels = mx.gluon.utils.split_and_load(labels, ctx_list=devices, even_split=False)
        for input, label in zip(inputs, labels):
          output = mxmodel(input)
          valid_acc += acc(output, label)

    print("Epoch %d: loss %.3f, train acc %.3f, test acc %.3f, in %.1f sec" % (
            epoch, train_loss/len(train_data), train_acc/len(train_data),
            valid_acc/len(valid_data), time.time()-tic))