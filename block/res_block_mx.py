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

# if __name__ == "__main__":
#     X = mx.nd.random.uniform(shape=(4, 1, 28, 28)) # shape=(batch_size, channels, width, height)
#     res_block = ResidualBlockMxnet(num_channels=1, output_channels=64, strides=2, is_used_conv11=True)
#     # you must initialize parameters for mxnet block.
#     res_block.initialize()
#     X = res_block(X)
#     print(X.shape)