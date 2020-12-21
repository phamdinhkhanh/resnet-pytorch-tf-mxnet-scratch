import mxnet as mx
from mxnet.gluon import nn as mxnn
from block.res_block_mx import ResidualBlockMxnet

class ResNet18Mxnet(mxnn.HybridSequential):
  def __init__(self, output_shape, **kwargs):
    super(ResNet18Mxnet, self).__init__(**kwargs)
    self.conv1 = mxnn.Conv2D(channels=64, padding=3, 
                           kernel_size=7, strides=2)
    self.batch_norm = mxnn.BatchNorm()
    self.max_pool = mxnn.MaxPool2D(pool_size=3)
    self.relu = mxnn.Activation('relu')
    self.global_avg_pool = mxnn.GlobalAvgPool2D()
    self.dense = mxnn.Dense(units=output_shape)
    self.blk = mxnn.HybridSequential()
    self.residual_blocks = [
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

# if __name__ == "__main__":
#     mxmodel = ResNet18Mxnet(output_shape=10)
#     mxmodel.hybridize()

#     mx.viz.print_summary(
#         mxmodel(mx.sym.var('data')), 
#         shape={'data':(4, 1, 28, 28)}, #set your shape here
#     )