# using CIFAR-10 for training the model

from torchvision.datasets import CIFAR10
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

testset = CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)


trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)
testloader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)
