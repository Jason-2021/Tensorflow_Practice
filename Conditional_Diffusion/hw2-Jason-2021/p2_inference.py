import torch
from torchvision.utils import save_image
from p1_model import DiffussionModel
from UNet import UNet
import os
import sys
from torch.utils.data import DataLoader, Dataset

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    noise_path = sys.argv[1]
    image_out = sys.argv[2]
    checkpoint_path = sys.argv[3]
    
    noise_list = sorted([i for i in os.listdir(noise_path) if i.endswith('.pt')])
    noise = torch.empty((0,3,256,256))
    for n in noise_list:
        tmp = torch.load(os.path.join(noise_path, n), map_location='cpu')
        noise = torch.concatenate((noise, tmp), dim=0)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    unet = UNet()
    unet.load_state_dict(checkpoint)
    DDIM = DiffussionModel(unet, mode='ger', device=device).to(device)

    DDIM.eval()
    with torch.no_grad():
        input = [noise.to(device), 0]
        out = DDIM(input, model_mode='DDIM')
        
        for i in range(len(out)):
            file_name = noise_list[i].rsplit('.', 1)[0]
            save_image(out[i], os.path.join(image_out, f"{file_name}.png"), normalize=True, range=(-1,1))
            


    
    

        

