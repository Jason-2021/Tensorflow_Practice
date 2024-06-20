import tensorflow as tf
import argparse
from dataset_utils import get_dataset
from models import UNet
import os
from tqdm.auto import tqdm
import numpy as np
# from mean_iou_tool import mean_iou_score


def save_model(model, optimizer, epoch, exp_name, dir='checkpoint'):
    checkpoint_prefix = os.path.join(dir, exp_name, f'{epoch}')

    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    checkpoint.save(file_prefix=checkpoint_prefix)


def print_info(filename, message=''):
    
    with open(filename, 'a', newline='') as f:
        f.write(f'{message}\n')


def mean_iou(preds, labels):
    # preds = tf.make_ndarray(preds)
    # labels = tf.make_ndarray(labels)

    preds = np.argmax(preds, axis=-1)
    mean_iou = 0
    for i in range(6):
        tp_fp = np.sum(preds == i)
        tp_fn = np.sum(labels == i)
        tp = np.sum((preds == i) * (labels == i))
        iou = tp / (tp_fp + tp_fn - tp)
        mean_iou += iou / 6
    
    return mean_iou


def valid(valid_dir, model, record_name):
    @tf.function
    def valid_step(image, label):
        
        logits = model(image, training=False)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=label,
                                                                y_pred=logits,
                                                                from_logits=True,
                                                                axis=-1)
        return loss, logits
    
    valid_dataset = get_dataset(dir_path=valid_dir, batchsize=args.batch_size, train=False)
    valid_loss = []
    valid_preds = np.empty((0, 512, 512, 7))
    valid_labels = np.empty((0, 512, 512))
    for image, label in tqdm(valid_dataset):
        loss, logits = valid_step(image, label)
        valid_loss.append(loss)
        valid_preds = np.concatenate((valid_preds, logits.numpy()), axis=0)
        valid_labels = np.concatenate((valid_labels, label.numpy()),axis=0)
    valid_loss = sum(valid_loss) / len(valid_loss)
    mean_iou_score = mean_iou(valid_preds, valid_labels)

    message = f"Validation ...\n" + \
        f"Loss: {valid_loss}\n" + \
        f"Mean IoU: {mean_iou_score}\n"
    print_info(record_name, message)


def train(args, train_dir, valid_dir):
    @tf.function
    def train_step(image, label):
        with tf.GradientTape() as tape:
            logits = model(image, training=True)
            loss = loss_fn(label, logits)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss, logits

    # set record information
    record_name = os.path.join('record', f'{args.exp_name}.txt')
    with open(record_name, 'w', newline='') as f:
        f.write("")
    # data 
    train_dataset = get_dataset(dir_path=train_dir, batchsize=args.batch_size, train=True)
    

    # model
    model = UNet()

    # loss function
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)

    for epoch in range(args.epochs):
        train_preds = np.empty((0, 512, 512, 7))
        train_labels = np.empty((0, 512, 512))
        train_loss = []
        iter = 0
        for image, label in tqdm(train_dataset):
            loss, logits = train_step(image, label)
            train_loss.append(loss)
            # train_preds = np.concatenate((train_preds, tf.make_ndarray(logits)), axis=0)
            # train_labels = np.concatenate((train_labels, tf.make_ndarray(label)), axis=0)
            train_preds = np.concatenate((train_preds, logits.numpy()), axis=0)
            train_labels = np.concatenate((train_labels, label.numpy()), axis=0)
            iter += 1
            if iter == 50:
                break
        
        train_loss = sum(train_loss) / len(train_loss)
        mean_iou_score = mean_iou(train_preds, train_labels)
        message = f"Epoch: {epoch:02d}\n" + \
            f"Loss: {train_loss}\n" + \
            f"Mean IoU: {mean_iou_score}\n"
        print_info(record_name, message)

        if epoch % 3 == 0:
            valid(valid_dir, model, record_name)
        
        if epoch % 10 == 0:
            save_model(model, optimizer, epoch, args.exp_name)

        
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name',
                        type=str,
                        default='UNet')
    parser.add_argument('--epochs',
                        type=int,
                        default=80)
    parser.add_argument('--lr',
                        type=float,
                        default=0.001),
    parser.add_argument('--batch_size',
                        type=int,
                        default=4)
    args = parser.parse_args()
    
    train_dir = '/data/jason/hw1_data/p3_data/train/'
    valid_dir = '/data/jason/hw1_data/p3_data/validation/'
    valid_out_dir = './'

    train(args, train_dir, valid_dir)

