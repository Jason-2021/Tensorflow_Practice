import torch
from torchvision import transforms
from p1_model import ConDiffussionModel
from MyUNet import UNet
import sys
from torchvision.utils import save_image
import numpy as np
import os
import random


myseed = 42  # set a random seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

random.seed(myseed)
np.random.seed(myseed)

torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_dir = sys.argv[1]

    checkpoint = torch.load('./p1.ckpt', map_location='cpu')
    unet = UNet(class_embeded=checkpoint['noise'].to(device), device=device).to(device)
    model = ConDiffussionModel(model=unet, timestep=500, mode='generate', device=device).to(device)
    model.load_state_dict(checkpoint['model'])

    tfm = transforms.Compose([
        transforms.Normalize(mean=[0.,0.,0.], std=[1/0.5,1/0.5,1/0.5]),
        transforms.Normalize(mean=[-0.5,-0.5,-0.5], std=[1.,1.,1.]),
        transforms.Resize(28),
    ])
    
    model.eval()
    label = []
    for i in range(10):
        for j in range(100):
            label.append(i)
    
    label = torch.tensor(label, device=device)
    with torch.no_grad():
        out1 = model([900, 32], label=label[:900]).cpu()
        out2 = model([100, 32], label=label[900:]).cpu()
    out = torch.concatenate((out1, out2), dim=0)
    out = tfm(out)
    for i in range(10):
        for j in range(1, 101):
            save_image(out[100*i + (j-1)], os.path.join(save_dir, f'{i}_{j:03d}.png'), normalize=True, range=(-1,1))
    

    

    
    