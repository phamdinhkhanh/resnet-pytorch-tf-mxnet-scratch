from models.pytorch.resnet_pt_18 import ResNet18PyTorch
import torch.optim as optim
import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet18PyTorch(output_shape=10)
model.to(device)

transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.05), (0.05))])

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                        shuffle=True, num_workers=8)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                    download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                        shuffle=False, num_workers=8)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.99))
     
def acc(output, label):
    # output: (batch, num_output) float32 ndarray
    # label: (batch, ) int32 ndarray
    return (torch.argmax(output, axis=1)==label).float().mean()

for epoch in range(10):  # loop over the dataset multiple times
    total_loss = 0.0
    tic = time.time()
    tic_step = time.time()
    train_acc = 0.0
    valid_acc = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        train_acc += acc(outputs, labels)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        total_loss += loss.item()
        if i % 500 == 499:
            print("iter %d: loss %.3f, train acc %.3f in %.1f sec" % (
                i+1, total_loss/i, train_acc/i, time.time()-tic_step))
            tic_step = time.time()

    # calculate validation accuracy
    for i, data in enumerate(testloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        valid_acc += acc(model(inputs), labels)

    print("Epoch %d: loss %.3f, train acc %.3f, test acc %.3f, in %.1f sec" % (
            epoch, total_loss/len(trainloader), train_acc/len(trainloader),
            valid_acc/len(testloader), time.time()-tic))
print('Finished Training')