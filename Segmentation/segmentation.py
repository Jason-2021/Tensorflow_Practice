import tensorflow as tf
import argparse
from dataset_utils import get_dataset
from models import UNet
import os
from tqdm.auto import tqdm
import numpy as np
# from mean_iou_tool import mean_iou_score
import tensorflow_hub as hub
from unet import Unet


def save_model(model, optimizer, epoch, exp_name, dir='checkpoint'):
    checkpoint_prefix = os.path.join(dir, exp_name, f'{epoch}')

    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    checkpoint.save(file_prefix=checkpoint_prefix)


def print_info(filename, message=''):
    
    with open(filename, 'a', newline='') as f:
        f.write(f'{message}\n')


def mean_iou(preds, labels, numerator, denominator):
    # preds = tf.make_ndarray(preds)
    # labels = tf.make_ndarray(labels)

    preds = tf.math.argmax(preds, axis=-1, output_type=tf.int32)
    mean_iou = 0
    num_class = 6
    # numerator = np.zeros((6,))  # child
    # denominator = np.zeros((6,))  # mom
    for i in range(6):
        tp_fp = tf.math.reduce_sum(tf.cast(preds == i, dtype=tf.int32))
        tp_fn = tf.math.reduce_sum(tf.cast(labels == i, dtype=tf.int32))
        tp = tf.math.reduce_sum(tf.cast((preds == i) & (labels == i), dtype=tf.int32))
        # if tp_fp + tp_fn - tp == 0:
        #     num_class -= 1
        # else:
        #     iou = tp / (tp_fp + tp_fn - tp)
        #     mean_iou += iou
        numerator[i] += tp
        denominator[i] += tp_fp + tp_fn - tp
        
        
    
    return None # mean_iou / num_class


def valid(valid_dataset, model, record_name):
    @tf.function
    def valid_step(image, label):
        
        logits = model(image, training=False)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=label,
                                                                y_pred=logits,
                                                                from_logits=True,
                                                                axis=-1)
        return loss, logits
    
    
    valid_loss = []
    mean_iou_score = []
    numerator = np.zeros((6, ))
    denominator = np.zeros((6, ))
    for image, label in tqdm(valid_dataset):
        loss, logits = valid_step(image, label)
        
        valid_loss.append(tf.math.reduce_mean(loss))
        # mean_iou_score.append(mean_iou(logits, label))
        mean_iou(logits, label, numerator, denominator)
    
    valid_loss = sum(valid_loss) / len(valid_loss)
    # mean_iou_score = sum(mean_iou_score) / len(mean_iou_score)
    mean_iou_score = np.mean(numerator / denominator)

    message = f"    Validation ...\n" + \
        f"    Loss: {valid_loss}\n" + \
        f"    Mean IoU: {mean_iou_score}\n"
    print_info(record_name, message)


# def weighted_sparse_categorical_crossentropy(y_true, y_pred, class_weight):
#     preds = tf.math.argmax(preds, axis=-1, output_type=tf.int32)
#     weights = tf.gather(class_weight, preds)
#     print(tf.shape(weights))
#     loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
#     return loss * weights, weights


def train(args, train_dir, valid_dir):
    @tf.function
    def train_step(image, label, class_weight):
        with tf.GradientTape() as tape:
            logits = model(image, training=True)

            # loss = loss_fn(label, logits)
            # preds = tf.math.argmax(logits, axis=-1, output_type=tf.int32)
            weights = tf.gather(class_weight, label)
            # print(tf.shape(weights))
            loss = tf.keras.losses.sparse_categorical_crossentropy(label, logits, from_logits=True)
            loss = loss * weights

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss, logits

    # set record information
    record_name = os.path.join('record', f'{args.exp_name}.txt')
    with open(record_name, 'w', newline='') as f:
        f.write("")
    # data 
    train_dataset = get_dataset(dir_path=train_dir, batchsize=args.batch_size, train=True)
    valid_dataset = get_dataset(dir_path=valid_dir, batchsize=args.batch_size, train=False)

    # model
    model = UNet()
    #model = Unet(input_size=(512, 512, 3), classes=7)
    #model = Unet(input_size=(512, 512, 3), classes=7)
    # loss function
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)

    weight = tf.Variable([1]*7, dtype=tf.float32)

    for epoch in range(args.epochs):
        
        train_loss = []
        mean_iou_score = []
        numerator = np.zeros((6, ))
        denominator = np.zeros((6, ))
        for image, label in tqdm(train_dataset):
            loss, logits = train_step(image, label, weight)
            train_loss.append(tf.math.reduce_mean(loss))
            
            mean_iou(logits, label, numerator, denominator)
            
            # break
            
        
        train_loss = sum(train_loss) / len(train_loss)
        iou_scores = numerator / denominator
        mean_iou_score = np.mean(iou_scores)  #mean_iou(train_preds, train_labels)
        
        # iou_scores[iou_scores == 0] = 0.1
        if args.weight:
            # iou_scores = 6 * (iou_scores / np.sum(iou_scores))
            iou_scores = np.exp(iou_scores) / sum(np.exp(iou_scores))
            iou_scores = 1 / iou_scores
            iou_scores = 6 * (np.exp(iou_scores) / sum(np.exp(iou_scores)))
            weight.assign(np.append(iou_scores, 0.25))
        else:
            pass

        message = f"Epoch: {epoch:02d}\n" + \
            f"Loss: {train_loss}\n" + \
            f"Mean IoU: {mean_iou_score}\n"
        print_info(record_name, message)

        if epoch % 3 == 0:
            valid(valid_dataset, model, record_name)
        
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
                        default=10)
    parser.add_argument('--weight',
                        action='store_true')
    args = parser.parse_args()
    
    train_dir = '/data4/jason/hw1_data/p3_data/train/'
    valid_dir = '/data4/jason/hw1_data/p3_data/validation/'
    valid_out_dir = './'

    train(args, train_dir, valid_dir)

