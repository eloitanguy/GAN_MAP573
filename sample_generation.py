import torch
from models import Generator
from torchvision.utils import save_image

G = Generator()
G_checkpoint = torch.load('G.pth')
G = G.eval().cuda()
G.load_state_dict(G_checkpoint)

g = G(torch.randn(100, 100).cuda())
save_image((g.view(100, 1, 28, 28) + 1)/2, 'sample.png')
