import torch
from models import Generator
from torchvision.utils import save_image

G = Generator()
G_checkpoint = torch.load('G.pth')
G = G.cuda()
G.load_state_dict(G_checkpoint)

generated = G(torch.randn(100, 100).cuda())
save_image(generated.view(100, 1, 28, 28), 'sample.png')
