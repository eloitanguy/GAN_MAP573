from torchvision.datasets import MNIST
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from models import Generator, Discriminator
from torchvision.utils import save_image


class AverageMeter(object):
    def __init__(self):
        self.number = 0.
        self.sum = 0.
        self.avg = 0.

    def update(self, value):
        self.number += 1.
        self.sum += value
        self.avg = self.sum / self.number

    def reset(self):
        self.number, self.sum, self.avg = 0., 0., 0.


def preprocess_batch(batch):
    batch_size = batch[0].shape[0]
    net_in = batch[0].reshape(batch_size, 28 * 28)
    net_in = net_in
    return net_in.cuda()


def train_D_batch(batch):
    D_optimiser.zero_grad()
    batch_size = batch.shape[0]  # we expect a batch of [batch_size, 28*28]

    # Train the discriminator on the true example
    D_out = D(batch)
    loss = loss_function(D_out, torch.ones(batch_size, 1).cuda())  # the dataset examples are all real examples (1)

    # Train the discriminator on generated examples
    generated = G(torch.randn(batch_size, 100).cuda())
    D_out = D(generated)
    loss = loss + loss_function(D_out, torch.zeros(batch_size, 1).cuda())  # we want D to say that the examples are fake
    loss.backward()
    D_optimiser.step()

    return loss.item()


def train_G_batch(batch):
    G_optimiser.zero_grad()
    batch_size = batch.shape[0]  # we expect a batch of [batch_size, 28*28]
    generated = G(torch.randn(batch_size, 100).cuda())
    loss = loss_function(D(generated), torch.ones(batch_size, 1).cuda())  # We want G to fool D, ie to answer 1
    loss.backward()
    G_optimiser.step()

    return loss.item()


train_dataset = MNIST(root='./mnist_data/', train=True, download=True,
                      transform=transforms.ToTensor())
train_loader = DataLoader(train_dataset, batch_size=2048, num_workers=6, shuffle=True)

G = Generator().cuda()
D = Discriminator().cuda()

loss_function = torch.nn.BCELoss()
G_optimiser = torch.optim.Adam(G.parameters(), lr=1e-4)
D_optimiser = torch.optim.Adam(D.parameters(), lr=1e-4)

EPOCHS = 50
K = 1

for epoch in range(1, EPOCHS+1):
    D_loss, G_loss = AverageMeter(), AverageMeter()
    for idx, batch in enumerate(train_loader):
        x = preprocess_batch(batch)
        D_loss_batch = train_D_batch(x)
        D_loss.update(D_loss_batch)

        if idx % K == 0:
            G_loss_batch = train_G_batch(x)
            G_loss.update(G_loss_batch)

    print('[{}/{}]\tD: {}\tG: {}'.format(epoch, EPOCHS, D_loss.avg, G_loss.avg))

print('Saving models ...')
torch.save(D.state_dict(), 'D.pth')
torch.save(G.state_dict(), 'G.pth')

print('Outputting example ...')
generated = G(torch.randn(100, 100).cuda())
save_image(generated.view(100, 1, 28, 28), 'sample.png')
