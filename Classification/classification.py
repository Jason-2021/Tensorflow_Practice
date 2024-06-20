import tensorflow as tf
from dataset_utils import get_dataset
import argparse
from tqdm.auto import tqdm
from models import BasicModel, Resnet50_based
import os


def save_model(model, optimizer, epoch, exp_name, dir='checkpoint'):
    checkpoint_prefix = os.path.join(dir, exp_name, f'{epoch}')

    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    checkpoint.save(file_prefix=checkpoint_prefix)


def print_info(filename, message=''):
    
    with open(filename, 'a', newline='') as f:
        f.write(f'{message}\n')


def train(train_dir, valid_dit, args):
    @tf.function
    def train_step(image, label):
        with tf.GradientTape() as tape:
            logits = model(image, training=True)
            loss = loss_fn(label, logits)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss, logits

    record_name = os.path.join('record', f'{args.exp_name}.txt')
    with open(record_name, 'w', newline='') as f:
        f.write("")
    # data
    train_dataset = get_dataset(dir_path=train_dir, batchsize=args.batch_size)
    valid_dataset = get_dataset(dir_path=valid_dir, batchsize=args.batch_size)

    # model
    if args.exp_name != 'ResNet':
        model = BasicModel(global_pool=args.global_pool)
    else:
        model = Resnet50_based()
    
    # optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    
    # loss
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    for epoch in range(args.epochs):
        loss_list = []
        acc_list = []
        for image, label in tqdm(train_dataset):
            loss, logit = train_step(image, label)
            loss_list.append(loss)
            acc_list.append(tf.math.reduce_mean(tf.cast(tf.math.argmax(logit, output_type=tf.int32, axis=-1) == label, dtype=tf.float32)))

        message = f"Epoch {epoch:03d}\n" + \
            f"loss: {sum(loss_list) / len(loss_list)}\n" + \
            f"acc:  {sum(acc_list) / len(acc_list)}\n"
        print_info(record_name, message)
        if epoch % 10 == 0:
            save_model(model, optimizer, epoch, args.exp_name)
        
        




if __name__ == '__main__':
    train_dir = '/data/jason/hw1_data/p1_data/train_50'
    valid_dir = '/data/jason/hw1_data/p1_data/val_50'

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',
                        type=int,
                        default=50)
    parser.add_argument('--batch_size',
                        type=int,
                        default=128)
    parser.add_argument('--lr',
                        type=float,
                        default=1e-3)
    parser.add_argument('--exp_name',
                        type=str,
                        default='basic_cnn')
    parser.add_argument('--global_pool',
                        action='store_true')
    args = parser.parse_args()
    train(train_dir, valid_dir, args)
    
    
    