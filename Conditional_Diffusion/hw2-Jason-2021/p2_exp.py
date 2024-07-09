import torch
from torch.nn.functional import mse_loss
import os
import torchvision
from PIL import Image

gt_path = "/home/r12922169/course/dlcv/hw2-Jason-2021/hw2_data/face/GT"
my_path = '/home/r12922169/course/dlcv/hw2-Jason-2021/p2out'

gt = sorted([x for x in os.listdir(gt_path)])
my = sorted([x for x in os.listdir(my_path)])
print(gt)
accu = 0
for i in range(10):
    img1 = torchvision.io.read_image(os.path.join(gt_path, gt[i])).float()
    img2 = torchvision.io.read_image(os.path.join(my_path, my[i])).float()
    # img1 = Image.open(os.path.join(gt_path, gt[i]))
    # img2 = Image.open(os.path.join(my_path, my[i]))

    accu += mse_loss(img1, img2) / 10
    
print(accu)