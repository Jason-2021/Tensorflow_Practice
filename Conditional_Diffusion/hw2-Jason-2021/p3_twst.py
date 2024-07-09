import torch
import os
from torchvision import transforms
import sys
from p3_model import digit_classifier
from p3_data import Digit_data
#from digit_classifier import Classifier
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    dataset = sys.argv[1]
    checkpoint = torch.load('/home/r12922169/course/dlcv/hw2-Jason-2021/ckpt/mnistm_train.ckpt')

    if dataset == 'usps':
        valid_csv_path = '/home/r12922169/course/dlcv/hw2-Jason-2021/hw2_data/digits/usps/val.csv'
        valid_data_path = '/home/r12922169/course/dlcv/hw2-Jason-2021/hw2_data/digits/usps/data'
    elif dataset == 'svhn':
        valid_csv_path = '/home/r12922169/course/dlcv/hw2-Jason-2021/hw2_data/digits/svhn/val.csv'
        valid_data_path = '/home/r12922169/course/dlcv/hw2-Jason-2021/hw2_data/digits/svhn/data'
    else:
        raise ValueError(f"No dataset named: '{dataset}'")
    
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    digit = Digit_data(valid_csv_path, valid_data_path, tfm, True)
    loader = DataLoader(digit, 2000)

    model = digit_classifier().to(device)
    model.load_state_dict(checkpoint['model'])

    model.eval()
    val_accu = []
    for batch in tqdm(loader):
        img, label = batch[0], batch[1]
        with torch.no_grad():
            out = model(img.to(device))

        val_accu.append(((out.argmax(dim=-1).cpu()) == label).float().mean())
    print(sum(val_accu) / len(val_accu))
        


