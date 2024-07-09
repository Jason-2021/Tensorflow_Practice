import torch
import os
from p1_model import DiffussionModel
from UNet import UNet
from torchvision.utils import save_image
from torchvision.io import read_image
import sys




if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mode = sys.argv[1]
    
    if mode == "1":

        noise_path = '/home/r12922169/course/dlcv/hw2-Jason-2021/hw2_data/face/noise'
        image_out = '/home/r12922169/course/dlcv/hw2-Jason-2021/p2out'

        noise_list = sorted([i for i in os.listdir(noise_path) if i.endswith('.pt')])
        noise = torch.empty((0,3,256,256))
        for n in noise_list[:4]:
            tmp = torch.load(os.path.join(noise_path, n), map_location='cpu')
            noise = torch.concatenate((noise, tmp), dim=0)
        
        checkpoint = torch.load('/home/r12922169/course/dlcv/hw2-Jason-2021/hw2_data/face/UNet.pt')
        unet = UNet()
        unet.load_state_dict(checkpoint)
        DDIM = DiffussionModel(unet, mode='ger').to(device)

        etas = [0.25*i for i in range(5)]

        result = torch.empty((0, 3, 256, 256))

        for eta in etas:
            with torch.no_grad():
                input = [noise.to(device), eta]
                out = DDIM(input, model_mode='DDIM')
                result = torch.concatenate((result, out.cpu()))
        
        save_image(result, "p2_report_graph_1.png", nrow=4, normalize=True, range=(-1,1))
    
    elif mode == "2":
        
        x0 = torch.load('/home/r12922169/course/dlcv/hw2-Jason-2021/hw2_data/face/noise/00.pt', map_location='cpu')
        x1 = torch.load('/home/r12922169/course/dlcv/hw2-Jason-2021/hw2_data/face/noise/01.pt', map_location='cpu')

        alphas = [0.1*i for i in range(11)]
        
        #theta = torch.arccos(torch.nn.functional.cosine_similarity(x0.clone().flatten(), x1.clone().flatten()))
        t = torch.nn.functional.cosine_similarity(x0.clone().flatten(), x1.clone().flatten(), dim=0)
        theta = torch.arccos(t)
        print(theta)
        result1 = torch.empty((0, 3, 256, 256))
        result2 = torch.empty((0, 3, 256, 256))
        for alpha in alphas:
            tmp1 = torch.sin((1-alpha)*theta) * x0 / torch.sin(theta) + torch.sin(alpha*theta) * x1 / torch.sin(theta)
            tmp2 = (1-alpha)*x0 + alpha*x1

            result1 = torch.concatenate((result1, tmp1), dim=0)
            result2 = torch.concatenate((result2, tmp2), dim=0)
        
        checkpoint = torch.load('/home/r12922169/course/dlcv/hw2-Jason-2021/hw2_data/face/UNet.pt')
        unet = UNet()
        unet.load_state_dict(checkpoint)
        DDIM = DiffussionModel(unet, mode='ger', device=device).to(device)

        DDIM.eval()
        with torch.no_grad():
            result1 = DDIM([result1.to(device), 0], model_mode='DDIM')
            result2 = DDIM([result2.to(device), 0], model_mode='DDIM')
        

        save_image(result1.cpu(), "p2_slert_true.png", nrow=11, normalize=True, range=(-1,1))
        save_image(result2.cpu(), "p2_linear_true.png", nrow=11, normalize=True, range=(-1,1))
        



