import numpy as np
import scipy.misc
import imageio
import argparse
import os

def read_masks(filepath):
    '''
    Read masks from directory and tranform to categorical
    '''
    file_list = [file for file in os.listdir(filepath) if file.endswith('.png')]
    file_list.sort()
    n_masks = len(file_list)
    masks = np.empty((n_masks, 512, 512))

    for i, file in enumerate(file_list):
        mask = imageio.imread(os.path.join(filepath, file))
        mask = (mask >= 128).astype(int)
        mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]
        masks[i, mask == 3] = 0  # (Cyan: 011) Urban land 
        masks[i, mask == 6] = 1  # (Yellow: 110) Agriculture land 
        masks[i, mask == 5] = 2  # (Purple: 101) Rangeland 
        masks[i, mask == 2] = 3  # (Green: 010) Forest land 
        masks[i, mask == 1] = 4  # (Blue: 001) Water 
        masks[i, mask == 7] = 5  # (White: 111) Barren land 
        masks[i, mask == 0] = 6  # (Black: 000) Unknown 

    return masks

def mean_iou_score(pred_path, labels_path):
    '''
    Compute mean IoU score over 6 classes
    '''
    pred = read_masks(pred_path)
    labels = read_masks(labels_path)
    
    mean_iou = 0
    iou_list = [0] * 6
    for i in range(6):
        tp_fp = np.sum(pred == i)
        tp_fn = np.sum(labels == i)
        tp = np.sum((pred == i) * (labels == i))
        
        # if (tp_fp + tp_fn - tp) == 0:
        #     iou = 0
        # else:
        iou = tp / (tp_fp + tp_fn - tp)
        iou_list[i] += iou
        mean_iou += iou / 6
        

    return mean_iou# , iou_list



if __name__ == '__main__':
    pass