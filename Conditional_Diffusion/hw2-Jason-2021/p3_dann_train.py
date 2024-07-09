import torch
import os
from torchvision import transforms
import sys
from p3_model import DANN
from p3_data import Digit_data, domain_data
#from digit_classifier import Classifier
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def optim_lr_schedule(optim, start, end, num_epoch, epoch):
    space = num_epoch - 1
    interval = (start - end) / space
    for p in optim.param_groups:
        p['lr'] = start - interval * (epoch - 1)
    return optim

if __name__ == '__main__':
    target_name = sys.argv[1]
    tsne_threshold = float(sys.argv[2])
    exp_name = "DANN_6_" + target_name

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    if target_name == 'svhn':
        target_data_path = '/home/r12922169/course/dlcv/hw2-Jason-2021/hw2_data/digits/svhn/data'
        target_csv_path = '/home/r12922169/course/dlcv/hw2-Jason-2021/hw2_data/digits/svhn/train.csv'
        valid_csv_path = '/home/r12922169/course/dlcv/hw2-Jason-2021/hw2_data/digits/svhn/val.csv'
    
    elif target_name == 'usps':
        target_data_path = '/home/r12922169/course/dlcv/hw2-Jason-2021/hw2_data/digits/usps/data'
        target_csv_path = '/home/r12922169/course/dlcv/hw2-Jason-2021/hw2_data/digits/usps/train.csv'
        valid_csv_path = '/home/r12922169/course/dlcv/hw2-Jason-2021/hw2_data/digits/usps/val.csv'
    
    source_data_path = '/home/r12922169/course/dlcv/hw2-Jason-2021/hw2_data/digits/mnistm/data'
    source_csv_path = '/home/r12922169/course/dlcv/hw2-Jason-2021/hw2_data/digits/mnistm/train.csv'
    source_valid_csv_path = '/home/r12922169/course/dlcv/hw2-Jason-2021/hw2_data/digits/mnistm/val.csv'


    tfm = transforms.Compose([
        transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    batch_size = 5000
    num_epoch = 1001
    checkpoint = None
    start_epoch = 1 if checkpoint is None else checkpoint['epoch']+1
    lr = 5e-3

    source_train_set = Digit_data(source_csv_path, source_data_path, tfm, True)
    target_train_set = Digit_data(target_csv_path, target_data_path, tfm, True)
    #DANN_data = domain_data(source_csv_path, source_data_path, target_csv_path, target_data_path, tfm)
    target_valid_set = Digit_data(valid_csv_path, target_data_path, tfm, True)
    source_valid_set = Digit_data(source_valid_csv_path, source_data_path, tfm, True)
    
    num_source = len(source_train_set)
    num_target = len(target_train_set)
    source_batch_size = 5000
    target_batch_size = int(5000 / (num_source / num_target))


    source_loader = DataLoader(source_train_set, source_batch_size, shuffle=True, num_workers=12, pin_memory=True)
    target_loader = DataLoader(target_train_set, target_batch_size, shuffle=True, num_workers=12, pin_memory=True)
    #DANN_loader = DataLoader(DANN_data, batch_size, shuffle=True, num_workers=12, pin_memory=True)
    valid_loader = DataLoader(target_valid_set, batch_size, shuffle=True, num_workers=12, pin_memory=True)
    source_valid_loader = DataLoader(source_valid_set, batch_size, shuffle=True)

    model = DANN().to(device)
    # optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, momentum=0.9)
    criterian_class = torch.nn.CrossEntropyLoss()
    criterian_domain = torch.nn.CrossEntropyLoss()

    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optim'])
    plot_train_class_accu = []
    plot_train_domain_accu = []
    plot_valid_accu = []
    plot_epoch = []
    best_accu = 0
    for epoch in range(1, num_epoch+1):
        # 1 ~ 2 epoch only train class_classifier
        plot_epoch.append(epoch)
        model.train()
        iter = 0
        optimizer = optim_lr_schedule(optimizer, lr, 0.0001, num_epoch, epoch)
        train_class_accu = []
        train_domain_accu = []
        for s_bat, t_bat in tqdm(zip(source_loader, target_loader)):
            if target_name == 'usps':
                p = float(iter + epoch * len(source_loader)) / num_epoch / len(source_loader)
                alpha = 2. / (1. + np.exp(-10 * p)) - 1
            else:
                alpha = np.log(1.02 + 1.7*epoch/200)
                if alpha >=1:
                    alpha = 1
            alpha = torch.tensor(alpha)

            s_imgs, s_labels = s_bat[0].to(device), s_bat[1].to(device)
            t_imgs = t_bat[0].to(device)
            s_domain_label = torch.zeros(len(s_imgs)).long().to(device)
            t_domain_label = torch.ones(len(t_imgs)).long().to(device)

            source_class_out, source_domain_out = model(s_imgs, alpha)

            source_class_loss = criterian_class(source_class_out, s_labels)
            source_domain_loss = criterian_domain(source_domain_out, s_domain_label)

            _, target_domain_out = model(t_imgs, alpha)
            target_domain_loss = criterian_domain(target_domain_out, t_domain_label)

            losses = source_class_loss + source_domain_loss + target_domain_loss

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            iter += 1

            train_class_accu.append((source_class_out.argmax(dim=-1) == s_labels).float().mean())
            train_domain_accu.append(
                (torch.sum((source_domain_out.argmax(dim=-1) == s_domain_label).float()) + \
                torch.sum((target_domain_out.argmax(dim=-1) == t_domain_label).float())) / \
                (len(s_imgs) + len(t_imgs))
            )

        train_class_accu = sum(train_class_accu) / len(train_class_accu)
        train_domain_accu = sum(train_domain_accu) / len(train_domain_accu)

        plot_train_class_accu.append(train_class_accu.cpu())
        plot_train_domain_accu.append(train_domain_accu.cpu())

        print(f"Epoch {epoch:03d}: Train class accu: {train_class_accu:.3f}")
        print(f"Epoch {epoch:03d}: Train domain accu: {train_domain_accu:.3f}")
        with open(f'./log/{exp_name}.txt', 'a', newline='') as f:
            f.write(f"Epoch {epoch:03d}: Train class accu: {train_class_accu:.3f}\n")
            f.write(f"Epoch {epoch:03d}: Train domain accu: {train_domain_accu:.3f}\n")

        
        model.eval()
        val_accu = []
        for batch in tqdm(valid_loader):
            imgs, labels = batch[0].to(device), batch[1].to(device)
            with torch.no_grad():
                out, _ = model(imgs, alpha)
            val_accu.append((out.argmax(dim=-1) == labels).float().mean())
        val_accu = sum(val_accu) / len(val_accu)
        plot_valid_accu.append(val_accu.cpu())
        
        print(f"Epoch {epoch:03d}: Valid class accu: {val_accu:.3f}")
        with open(f'./log/{exp_name}.txt', 'a', newline='') as f:
            f.write(f"Epoch {epoch:03d}: Valid class accu: {val_accu:.3f}\n")

        # save model_last
        state = {
            'model': model.state_dict(),
            'optim': optimizer.state_dict(),
            'epoch': epoch
        }
        torch.save(state, f'ckpt/{exp_name}_last.ckpt')

        # plot loss
        plt.figure(figsize=(15,10))
        plt.title(f"{exp_name} Accuracy")
        plt.plot(plot_epoch, plot_train_class_accu, label='train class')
        plt.plot(plot_epoch, plot_train_domain_accu, label='train domain')
        plt.plot(plot_epoch, plot_valid_accu, label='valid class')
        plt.xlabel("Epoch")
        plt.ylabel('Accu')
        plt.legend(loc="best")
        plt.savefig(f'{exp_name}_accu.png')
        
        
        
        if (val_accu > best_accu and val_accu > tsne_threshold) or epoch == 3 or epoch % 200 == 0:
            if val_accu > best_accu:
                with open(f'./log/{exp_name}.txt', 'a', newline='') as f:
                    f.write(f"Best found at epoch {epoch:03d}, best accu: {val_accu:.3f}\n")
                
                torch.save(state, f'ckpt/{exp_name}_best.ckpt')
                best_accu = val_accu
            
            all_x = None
            all_x_label = None
            all_y = None
            all_y_label = None

            for batch in source_valid_loader:
                img, label = batch[0].to(device), batch[1]
                out = model.get_embeded(img)
                if all_x is None:
                    all_x = out.detach().cpu().numpy()[:1000]
                    all_x_label = label.numpy()[:1000]
                    break
                    
                else:
                    all_x = np.concatenate((all_x, out.detach().cpu().numpy()), axis=0)
                    all_x_label = np.concatenate((all_x_label, label.numpy()), axis=0)
            
            for batch in valid_loader:
                img, label = batch[0].to(device), batch[1]
                out = model.get_embeded(img)
                if all_y is None:
                    all_y = out.detach().cpu().numpy()[:1000]
                    all_y_label = label.numpy()[:1000]
                    break
                else:
                    all_y = np.concatenate((all_y, out.detach().cpu().numpy()), axis=0)
                    all_y_label = np.concatenate((all_y_label, label.numpy()), axis=0)
            domain_x = np.zeros(len(all_x))
            domain_y = np.ones(len(all_y))
            
            tsne = TSNE(n_components=2, init='pca')
            data_x = tsne.fit_transform(all_x)
            data_y = tsne.fit_transform(all_y)
            plt.figure(figsize=(15, 10))
            
            plt.title(f"t-SNE of {epoch} epoch (by class)")
            plt.scatter(np.hstack((data_x[:,0], data_y[:,0])), np.hstack((data_x[:,1],data_y[:, 1])), c=np.hstack((all_x_label, all_y_label)))
            plt.savefig(f'./p3tsne/t-SNE_{exp_name}_{epoch}_class.png')

            plt.figure(figsize=(15,10))
            plt.title(f"t-SNE of {epoch} epoch (by domain)")
            plt.scatter(np.hstack((data_x[:,0], data_y[:,0])), np.hstack((data_x[:,1],data_y[:, 1])), c=np.hstack((domain_x, domain_y)))
            plt.savefig(f'./p3tsne/t-SNE_{exp_name}_{epoch}_domain.png')

        

        