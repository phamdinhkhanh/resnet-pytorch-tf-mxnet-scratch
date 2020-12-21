from models.mxnet.resnet_mx_18 import ResNet18Mxnet
from mxnet import nd, gluon, init, autograd, gpu, cpu
from mxnet.gluon import nn
import mxnet as mx
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

model = ResNet18Mxnet(output_shape=10)
model.hybridize()
model.collect_params()
model.initialize(init=init.Xavier(), ctx=devices, force_reinit=True)

# mx.viz.print_summary(
#     model(mx.sym.var('data')), 
#     shape={'data':(4, 1, 28, 28)}, #set your shape here
# )

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': 0.001})
                        
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
                output = model(input)
                loss = softmax_cross_entropy(output, label)

            loss.backward()
            # update parameters
            trainer.step(batch_size)
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
            output = model(input)
            valid_acc += acc(output, label)

    print("Epoch %d: loss %.3f, train acc %.3f, test acc %.3f, in %.1f sec" % (
            epoch, train_loss/len(train_data), train_acc/len(train_data),
            valid_acc/len(valid_data), time.time()-tic))