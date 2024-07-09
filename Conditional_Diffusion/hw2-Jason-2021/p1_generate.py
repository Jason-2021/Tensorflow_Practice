import torch
from torchvision import transforms
from p1_model import DiffussionModel
from UNet import UNet
from PIL import Image
import os


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    target = [100, 32]  # [batch_size, image_size]
    checkpoint = torch.load("/home/r12922169/course/dlcv/hw2-Jason-2021/ckpt/p1_nor_last.ckpt", map_location='cpu')
    unet = UNet()
    model = DiffussionModel(model=unet, mode='generate', device=device).to(device)
    model.load_state_dict(checkpoint['model'])

    tfm = transforms.Compose([
        transforms.Normalize(mean=[0.,0.,0.], std=[1/0.229,1/0.224,1/0.225]),
        transforms.Normalize(mean=[-0.485,-0.456,-0.406], std=[1.,1.,1.]),
        transforms.ToPILImage(),
        transforms.Resize(28),
        
    ])
    save_dir = '/home/r12922169/course/dlcv/hw2-Jason-2021/p1test_result'
    model.eval()
    with torch.no_grad():
        out = model(target).cpu()
    for i in range(target[0]):
        image = tfm(out[i])
        image = image.save(os.path.join(save_dir, f"{i}.png"))
    
