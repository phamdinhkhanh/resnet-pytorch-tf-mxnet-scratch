import tensorflow as tf
from block.res_block_tf import ResidualBlockTF

class ResNet18TF(tf.keras.Model):
  def __init__(self, output_shape):
    super(ResNet18TF, self).__init__()
    self.conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=7, strides=2, padding='same')
    self.batch_norm = tf.keras.layers.BatchNormalization()
    self.max_pool = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='same')
    self.relu = tf.keras.layers.ReLU()
    self.residual_blocks = [
      # Two start conv mapping
      ResidualBlockTF(num_channels=64, output_channels=64, strides=2, is_used_conv11=False),
      ResidualBlockTF(num_channels=64, output_channels=64, strides=2, is_used_conv11=False),
      # Next three [conv mapping + identity mapping]
      ResidualBlockTF(num_channels=64, output_channels=128, strides=2, is_used_conv11=True),
      ResidualBlockTF(num_channels=128, output_channels=128, strides=2, is_used_conv11=False),
      ResidualBlockTF(num_channels=128, output_channels=256, strides=2, is_used_conv11=True),
      ResidualBlockTF(num_channels=256, output_channels=256, strides=2, is_used_conv11=False),
      ResidualBlockTF(num_channels=256, output_channels=512, strides=2, is_used_conv11=True),
      ResidualBlockTF(num_channels=512, output_channels=512, strides=2, is_used_conv11=False)
    ]
    self.global_avg_pool = tf.keras.layers.GlobalAvgPool2D()
    self.dense = tf.keras.layers.Dense(units=output_shape)

  def call(self, X):
    X = self.conv1(X)
    X = self.batch_norm(X)
    X = self.relu(X)
    X = self.max_pool(X)
    for residual_block in self.residual_blocks:
      X = residual_block(X)
    X = self.global_avg_pool(X)
    X = self.dense(X)
    return X


if __name__ == "__main__":
  tfmodel = ResNet18TF(residual_blocks, output_shape=10)
  tfmodel.build(input_shape=(None, 28, 28, 1))
  tfmodel.summary()