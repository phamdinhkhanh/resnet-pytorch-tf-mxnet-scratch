import torch
from torch import nn
from torchsummary import summary
from block.res_block_pt import ResidualBlockPytorch

class ResNet18PyTorch(nn.Module):
  def __init__(self, output_shape):
    super(ResNet18PyTorch, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3)
    self.batch_norm = nn.BatchNorm2d(64)
    self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.relu = nn.ReLU()
    self.residual_list_blocks = [
        # Two start conv mapping
        ResidualBlockPytorch(num_channels=64, output_channels=64, strides=2, is_used_conv11=False),
        ResidualBlockPytorch(num_channels=64, output_channels=64, strides=2, is_used_conv11=False),
        # Next three [conv mapping + identity mapping]
        ResidualBlockPytorch(num_channels=64, output_channels=128, strides=2, is_used_conv11=True),
        ResidualBlockPytorch(num_channels=128, output_channels=128, strides=2, is_used_conv11=False),
        ResidualBlockPytorch(num_channels=128, output_channels=256, strides=2, is_used_conv11=True),
        ResidualBlockPytorch(num_channels=256, output_channels=256, strides=2, is_used_conv11=False),
        ResidualBlockPytorch(num_channels=256, output_channels=512, strides=2, is_used_conv11=True),
        ResidualBlockPytorch(num_channels=512, output_channels=512, strides=2, is_used_conv11=False)
    ]
    self.residual_blocks = nn.Sequential(*self.residual_list_blocks)
    self.global_avg_pool = nn.Flatten()
    self.dense = nn.Linear(in_features=512, out_features=output_shape)

  def forward(self, X):
    X = self.conv1(X)
    X = self.batch_norm(X)
    X = self.relu(X)
    X = self.max_pool(X)
    X = self.residual_blocks(X)
    X = self.global_avg_pool(X)
    X = self.dense(X)
    return X

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ptmodel = ResNet18PyTorch(residual_blocks, output_shape=10)
    ptmodel.to(device)
    summary(ptmodel, (1, 28, 28))