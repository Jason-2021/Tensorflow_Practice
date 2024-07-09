import pandas as pd
import sys
from pprint import pprint

if __name__ == '__main__':
    dataset_name = sys.argv[1]

    if dataset_name == 'usps':
        gt_train = '/home/r12922169/course/dlcv/hw2-Jason-2021/hw2_data/digits/usps/train.csv'
        gt_valid = '/home/r12922169/course/dlcv/hw2-Jason-2021/hw2_data/digits/usps/val.csv'
        pred_path = './p3_csv_usps.csv'
    elif dataset_name == 'svhn':
        gt_train = '/home/r12922169/course/dlcv/hw2-Jason-2021/hw2_data/digits/svhn/train.csv'
        gt_valid = '/home/r12922169/course/dlcv/hw2-Jason-2021/hw2_data/digits/svhn/val.csv'
        pred_path = './p3_csv_svhn.csv'
    
    gt_train_csv = pd.read_csv(gt_train).values.tolist()
    gt_valid_csv = pd.read_csv(gt_valid).values.tolist()
    pred = pd.read_csv(pred_path).values.tolist()

    gt = gt_train_csv + gt_valid_csv
    
    
    # sorted(gt, key=lambda x: (x[0]))
    gt.sort(key=lambda x: (x[0]))

    # accu = 0
    # for i in range(len(gt)):
    #     if gt[i][1] == pred[i][1]:
    #         accu += 1
    # print(f'accu: {accu}/{len(gt)} = {accu/len(gt):.3f}')
    # # print(len(gt), len(pred))
    # pprint(gt_valid_csv)
    accu = 0
    for i in range(len(gt_valid_csv)):
        for j in range(i, len(pred)):
            if gt_valid_csv[i][0] == pred[j][0]:
                if gt_valid_csv[i][1] == pred[j][1]:
                    accu += 1
                break
    print(f"accu: {accu}/{len(gt_valid_csv)} = {accu / len(gt_valid_csv)}")
    