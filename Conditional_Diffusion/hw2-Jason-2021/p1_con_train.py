import torch
from torch import nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
from torchvision import transforms
import os
from MyUNet import UNet
from p1_model import ConDiffussionModel



class p1_data(Dataset):
    def __init__(self, path, csv_path):
        super().__init__()
        self.path = path
        self.filename_list = pd.read_csv(csv_path).values.tolist()
    def __len__(self):
        return len(self.filename_list)
    def __getitem__(self, index):
        tfm = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            
        ])
        image = Image.open(os.path.join(self.path, self.filename_list[index][0]))
        label = int(self.filename_list[index][1])

        return tfm(image), label
        

if __name__ == '__main__':
    exp_name = 'p1_500'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    embedding_noise = torch.empty((0, 1, 32, 32))
    
    for i in range(10):
        embedding_noise = torch.concatenate((embedding_noise, torch.randn((1,1,32,32))), dim=0)
        

    # with open(f"log/{exp_name}", 'a', newline='') as f:
    #     f.write(f"lr = 0.00015\n")
    data_path = "/home/r12922169/course/dlcv/hw2-Jason-2021/hw2_data/digits/mnistm/data"
    csv_path = "/home/r12922169/course/dlcv/hw2-Jason-2021/hw2_data/digits/mnistm/train.csv"

    dataset = p1_data(data_path, csv_path)
    dataloader = DataLoader(dataset, batch_size=500, shuffle=True, pin_memory=True, num_workers=12)

    #checkpoint = torch.load('/home/r12922169/course/dlcv/hw2-Jason-2021/ckpt/p1_new_nor_last.ckpt', map_location='cpu')
    checkpoint = None
    num_epoch = 300
    learning_rate = 0.0001
    unet = UNet(class_embeded=embedding_noise.to(device), device=device).to(device)
    DDPM = ConDiffussionModel(model=unet, timestep=500, device=device).to(device)
    optimizer = torch.optim.Adam(DDPM.parameters(), lr=learning_rate)

    if checkpoint is not None:
        DDPM.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optim'])
        embedding_noise = checkpoint['noise']
    
    min_loss = np.inf
    start_epoch = 1 if checkpoint is None else checkpoint['epoch'] + 1
    
    best_loss = 1
    for epoch in range(start_epoch, num_epoch+1):
        optimizer.param_groups[0]['lr'] = learning_rate * (1 - epoch / num_epoch)
        DDPM.train()
        train_loss = []
        for batch in tqdm(dataloader):
            imgs, labels = batch[0].to(device), batch[1].to(device)

            loss = DDPM(imgs, labels)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            train_loss.append(loss.item())
        
        train_loss = sum(train_loss) / len(train_loss)
        state = {
            'model': DDPM.state_dict(),
            'optim': optimizer.state_dict(),
            'epoch': epoch,
            'noise': embedding_noise
        }
        # save model
        torch.save(state, f"ckpt/{exp_name}_last.ckpt")
        # if min_loss > train_loss:
        #     min_loss = train_loss
        #     torch.save(state, f"ckpt/{exp_name}_best_epoxh{epoch}_loss{train_loss:3f}")
        if epoch % 20 == 0:
            torch.save(state, f"ckpt/{exp_name}_{epoch}.ckpt")
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(state, f"ckpt/{exp_name}_best.ckpt")
            with open(f"log/{exp_name}", 'a', newline='') as f:
                f.write(f"best: {epoch}\n")
        
        print(f"Epoch {epoch}: loss = {train_loss}")
        with open(f"log/{exp_name}", 'a', newline='') as f:
            f.write(f"Epoch {epoch}: loss = {train_loss}\n")
        




