import torch
from torch import nn

class ResidualBlockPytorch(nn.Module):
  def __init__(self, num_channels, output_channels, strides=1, is_used_conv11=False, **kwargs):
    super(ResidualBlockPytorch, self).__init__(**kwargs)
    self.is_used_conv11 = is_used_conv11
    self.conv1 = nn.Conv2d(num_channels, num_channels, padding=1, 
                           kernel_size=3, stride=1)
    self.batch_norm = nn.BatchNorm2d(num_channels)
    self.conv2 = nn.Conv2d(num_channels, num_channels, padding=1, 
                           kernel_size=3, stride=1)
    if self.is_used_conv11:
      self.conv3 = nn.Conv2d(num_channels, num_channels, padding=0, 
                           kernel_size=1, stride=1)
    # Last convolutional layer to reduce output block shape.
    self.conv4 = nn.Conv2d(num_channels, output_channels, padding=0, 
                           kernel_size=1, stride=strides)
    self.relu = nn.ReLU(inplace=True)
    
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
#     X = torch.rand((4, 1, 28, 28)) # shape=(batch_size, channels, width, height)
#     X = ResidualBlockPytorch(num_channels=1, output_channels=64, strides=2, is_used_conv11=True)(X)
#     print(X.shape)