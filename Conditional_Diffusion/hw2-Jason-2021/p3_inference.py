import torch
from p3_model import DANN
import os
import sys
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import numpy as np


class Digit(Dataset):
    def __init__(self, path, tfm):
        super().__init__()
        self.path = path
        self.tfm = tfm
        self.filename = sorted([x for x in os.listdir(self.path) if x.endswith('png')])
    def __len__(self):
        return len(self.filename)
    def __getitem__(self, index):
        name = self.filename[index]
        image = Image.open(os.path.join(self.path, name)).copy().convert('RGB')
        return self.tfm(image), name


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_path = sys.argv[1]
    pred_path = sys.argv[2]
    
    with open(pred_path, 'w', newline='') as f:
        f.write('image_name,label\n')

    if 'usps' in data_path:
        checkpoint = torch.load('./p3_usps.ckpt', map_location='cpu')
    else:
        checkpoint = torch.load('./p3_svhn.ckpt', map_location='cpu')
    print(checkpoint['epoch'])
    tfm = transforms.Compose([
        transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    data = Digit(data_path, tfm)

    loader = DataLoader(data, 5000)

    model = DANN().to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    tmp_lambda = torch.tensor(1, device=device)
    for batch in loader:
        images, names = batch[0].to(device), batch[1]
        with torch.no_grad():
            out, _ = model(images, tmp_lambda)
        out = out.cpu().numpy()
        pred = np.argmax(out, axis=-1)
        for i in range(len(names)):
            with open(pred_path, 'a', newline='') as f:
                f.write(f"{names[i]},{pred[i]}\n")
    
        

    