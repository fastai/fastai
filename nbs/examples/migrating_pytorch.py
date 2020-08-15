import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class Flatten(nn.Module):
    def forward(self, x): return x.view(x.size(0), -1)

class Net(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(1, 32, 3, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1), nn.MaxPool2d(2), nn.Dropout2d(0.25),
            Flatten(), nn.Linear(9216, 128), nn.ReLU(), nn.Dropout2d(0.5),
            nn.Linear(128, 10), nn.LogSoftmax(dim=1) )

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx*len(data), len(train_loader.dataset),
                100. * batch_idx/len(train_loader), loss.item()))

def test(model, device, test_loader):
    model.eval()
    test_loss,correct = 0,0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss/len(test_loader.dataset), correct, len(test_loader.dataset),
        100. * correct/len(test_loader.dataset)))

batch_size,test_batch_size = 256,512
epochs,lr = 1,1e-2

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True}
transform=transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.1307,), (0.3081,))])
train_loader = DataLoader(
    datasets.MNIST('../data', train=True, download=True, transform=transform),
                   batch_size=batch_size, shuffle=True, **kwargs)
test_loader = DataLoader(
    datasets.MNIST('../data', train=False, transform=transform),
                   batch_size=test_batch_size, shuffle=True, **kwargs)

model = Net().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

if __name__ == '__main__':
    for epoch in range(1, epochs+1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

