import torch
from torchvision import transforms
from p1_model import ConDiffussionModel
from MyUNet import UNet
from PIL import Image
import os
from torchvision.utils import save_image

myseed = 913  # set a random seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    target = [10, 32]  # [batch_size, image_size]
    checkpoint = torch.load("/home/r12922169/course/dlcv/hw2-Jason-2021/ckpt/p1_500_last.ckpt", map_location='cpu')
    unet = UNet(class_embeded=checkpoint['noise'].to(device)).to(device)
    model = ConDiffussionModel(model=unet, mode='generate', device=device).to(device)
    model.load_state_dict(checkpoint['model'])

    tfm = transforms.Compose([
        transforms.Normalize(mean=[0.,0.,0.], std=[1/0.5,1/0.5,1/0.5]),
        transforms.Normalize(mean=[-0.5,-0.5,-0.5], std=[1.,1.,1.]),
        # transforms.Normalize(mean=[0.,0.,0.], std=[1/0.229,1/0.224,1/0.225]),
        # transforms.Normalize(mean=[-0.485,-0.456,-0.406], std=[1.,1.,1.]),
        # transforms.ToPILImage(),
        transforms.Resize(28),
        
    ])
    save_dir = '/home/r12922169/course/dlcv/hw2-Jason-2021/p1_judge'
    model.eval()
    image = torch.empty((0, 3, 28, 28))
    for label in range(10):
        with torch.no_grad():
            if label == 0:
                out, report = model(target, label, report=True)
                out = tfm(out.cpu())

                image = torch.concatenate((image, out), dim=0)
            else:
                out = model(target, label)
                out = tfm(out.cpu())
                image = torch.concatenate((image, out), dim=0)
        # for i in range(target[0]):
        #     image = tfm(out[i])
        #     #image = image.save(os.path.join(save_dir, f"{label}_{i+1:03d}.png"))
        #     save_image(image, os.path.join(save_dir, f"{label}_{i+1:03d}.png"), normalize=True, range=(-1,1))
    

    save_image(image, os.path.join(save_dir, "p1_report_1.png"),nrow=10, normalize=True, range=(-1,1))
    save_image(tfm(report).cpu(), os.path.join(save_dir, "p1_report_2.png"),nrow=6, normalize=True, range=(-1,1))




    # with torch.no_grad():
    #     out = model(target, 1).cpu()
    # for i in range(target[0]):
    #     image = tfm(out[i])
    #     image = image.save(os.path.join(save_dir, f"{i}.png"))
    
