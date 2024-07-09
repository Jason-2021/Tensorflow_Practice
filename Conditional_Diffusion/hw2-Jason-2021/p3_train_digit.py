import torch
import os
from torchvision import transforms
import sys
from p3_model import digit_classifier
from p3_data import Digit_data
#from digit_classifier import Classifier
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

if __name__ == "__main__":
    data_set_name = sys.argv[1]
    exp_name = data_set_name + '_train_2'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    
    if data_set_name == 'svhn':
        data_path = '/home/r12922169/course/dlcv/hw2-Jason-2021/hw2_data/digits/svhn/data'
        train_csv_path = '/home/r12922169/course/dlcv/hw2-Jason-2021/hw2_data/digits/svhn/train.csv'
        valid_csv_path = '/home/r12922169/course/dlcv/hw2-Jason-2021/hw2_data/digits/svhn/val.csv'
    
    elif data_set_name == 'usps':
        data_path = '/home/r12922169/course/dlcv/hw2-Jason-2021/hw2_data/digits/usps/data'
        train_csv_path = '/home/r12922169/course/dlcv/hw2-Jason-2021/hw2_data/digits/usps/train.csv'
        valid_csv_path = '/home/r12922169/course/dlcv/hw2-Jason-2021/hw2_data/digits/usps/val.csv'
    elif data_set_name == 'mnistm':
        data_path = '/home/r12922169/course/dlcv/hw2-Jason-2021/hw2_data/digits/mnistm/data'
        train_csv_path = '/home/r12922169/course/dlcv/hw2-Jason-2021/hw2_data/digits/mnistm/train.csv'
        valid_csv_path = '/home/r12922169/course/dlcv/hw2-Jason-2021/hw2_data/digits/mnistm/val.csv'
    else:
        raise ValueError(f"No dataset named: '{data_set_name}'")

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # config
    batch_size = 5000
    num_epoch = 100
    checkpoint = None
    start_epoch = 1 if checkpoint is None else checkpoint['epoch']+1
    lr = 0.001


    train_dataset = Digit_data(train_csv_path, data_path, tfm, True)
    valid_dataset = Digit_data(valid_csv_path, data_path, tfm, True)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, pin_memory=True, num_workers=12)
    valid_loader = DataLoader(valid_dataset, batch_size, shuffle=True, pin_memory=True, num_workers=12)

    model = digit_classifier().to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    criterian = torch.nn.CrossEntropyLoss()

    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optim'])
    
    for epoch in range(1, num_epoch+1):
        # train
        model.train()
        train_loss = []
        train_accu = []
        for batch in tqdm(train_loader):
            imgs, labels = batch[0].to(device), batch[1].to(device)

            out = model(imgs)
            loss = criterian(out, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            train_accu.append((out.argmax(dim=-1) == labels).float().mean())
        
        train_loss = sum(train_loss) / len(train_loss)
        train_accu = sum(train_accu) / len(train_accu)

        print(f"Epoch {epoch:03d}: Train Loss = {train_loss:.3f}")
        print(f"Epoch {epoch:03d}: Train Accu = {train_accu:.3f}")
        with open(f"log/{exp_name}.txt", 'a',  newline='') as f:
            f.write(f"Epoch {epoch:03d}: Train Loss = {train_loss:.3f}\n")
            f.write(f"Epoch {epoch:03d}: Train Accu = {train_accu:.3f}\n")

        # valid
        model.eval()
        valid_loss = []
        valid_accu = []
        for batch in tqdm(valid_loader):
            imgs, labels = batch[0].to(device), batch[1].to(device)
            with torch.no_grad():
                out = model(imgs)
            
            loss = criterian(out, labels)

            valid_loss.append(loss.item())
            valid_accu.append((out.argmax(dim=-1) == labels).float().mean())
        
        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_accu = sum(valid_accu) / len(valid_accu)

        print(f"Epoch {epoch:03d}: Valid Loss = {valid_loss:.3f}")
        print(f"Epoch {epoch:03d}: Valid Accu = {valid_accu:.3f}")
        with open(f"log/{exp_name}.txt", 'a',  newline='') as f:
            f.write(f"Epoch {epoch:03d}: Valid Loss = {valid_loss:.3f}\n")
            f.write(f"Epoch {epoch:03d}: Valid Accu = {valid_accu:.3f}\n")

    state = {
        'model': model.state_dict(),
        'epoch': num_epoch,
        'optim': optimizer.state_dict()
    }

    torch.save(state, f"{exp_name}.ckpt")
        
        



    
