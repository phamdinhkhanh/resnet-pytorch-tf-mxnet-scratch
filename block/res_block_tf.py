import tensorflow as tf

class ResidualBlockTF(tf.keras.layers.Layer):
  def __init__(self, num_channels, output_channels, strides=1, is_used_conv11=False, **kwargs):
    super(ResidualBlockTF, self).__init__(**kwargs)
    self.is_used_conv11 = is_used_conv11
    self.conv1 = tf.keras.layers.Conv2D(num_channels, padding='same', 
                                        kernel_size=3, strides=1)
    self.batch_norm = tf.keras.layers.BatchNormalization()
    self.conv2 = tf.keras.layers.Conv2D(num_channels, padding='same', 
                                        kernel_size=3, strides=1)
    if self.is_used_conv11:
      self.conv3 = tf.keras.layers.Conv2D(num_channels, padding='same', 
                                          kernel_size=1, strides=1)
    # Last convolutional layer to reduce output block shape.
    self.conv4 = tf.keras.layers.Conv2D(output_channels, padding='same',
                                        kernel_size=1, strides=strides)
    self.relu = tf.keras.layers.ReLU()

  def call(self, X):
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
#   X = tf.random.uniform((4, 28, 28, 1)) # shape=(batch_size, width, height, channels)
#   X = ResidualBlockTF(num_channels=1, output_channels=64, strides=2, is_used_conv11=True)(X)
#   print(X.shape)