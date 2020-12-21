from models.tf.resnet_tf_18 import ResNet18TF
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers import Adam

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train/255.0
X_test = X_test/255.0
X_train = np.reshape(X_train, (-1, 28, 28, 1))
X_test = np.reshape(X_test, (-1, 28, 28, 1))
# Convert data type bo be adaptable to tensorflow computation engine
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
# print(X_test.shape, X_train.shape)

opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.99)
model = ResNet18TF(output_shape=10)
model.compile(optimizer=opt, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
model.fit(X_train, y_train,
        validation_data = (X_test, y_test), 
        batch_size=32,
        epochs=10)
model = ResNet18TF(output_shape=10)