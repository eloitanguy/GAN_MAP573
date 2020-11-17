import torch
from torch.nn import Module, Linear
import torch.nn.functional as F


class Generator(Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = Linear(100, 256)
        self.fc2 = Linear(256, 512)
        self.fc3 = Linear(512, 1024)
        self.fc4 = Linear(1024, 28 * 28)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.dropout(x, 0.3)
        return torch.tanh(self.fc4(x))


class Discriminator(Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = Linear(28*28, 1024)
        self.fc2 = Linear(1024, 512)
        self.fc3 = Linear(512, 256)
        self.fc4 = Linear(256, 1)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.dropout(x, 0.3)
        return torch.sigmoid(self.fc4(x))
