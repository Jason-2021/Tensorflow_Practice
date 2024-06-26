import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten
from keras import Sequential
from digit_dataset import get_dataset
import argparse
import os
from tqdm.auto import tqdm



class CNN(tf.keras.Model):
    def __init__(self):
        super().__init__(self)
        # self.net = Sequential([
        #     Conv2D(6, (5,5)),
        #     MaxPool2D(),
        #     Conv2D(16, (5,5)),
        #     MaxPool2D(),
        #     Flatten(),
        #     Dense(128, activation='relu'),
        #     Dense(64, activation='relu'),
        #     Dense(10, activation='softmax')
        # ])
        self.conv1 = Conv2D(6, (5,5), activation='relu')
        self.pool = MaxPool2D()
        self.conv2 = Conv2D(16, (5,5), activation='relu')
        self.flat = Flatten()
        self.fc1 = Dense(128, activation='relu')
        self.fc2 = Dense(64, activation='relu')
        self.fc3 = Dense(10, activation='softmax')

    def call(self, x):
        #return self.net(x)
        #print(tf.shape(x))
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        #print(tf.shape(x))
        x = self.flat(x)
        #print(tf.shape(x))
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x        



def save_model(model, optimizer, epoch, exp_name, dir='checkpoint'):
    checkpoint_prefix = os.path.join(dir, exp_name, f'{epoch}')

    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    checkpoint.save(file_prefix=checkpoint_prefix)


def print_info(filename, message='', init=False):
    
    if init:
        with open(filename, 'w', newline='') as f:
            f.write("")
    else:
        with open(filename, 'a', newline='') as f:
            f.write(f'{message}\n')

def train(dir, args):
    @tf.function
    def train_step(image, label):
        with tf.GradientTape() as tape:
            logits = model(image, training=True)
            loss = loss_fn(label, logits)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss, logits
    
    record_name = os.path.join('record', f'{args.exp_name}.txt')
    print_info(record_name, '', True)

    train_data = get_dataset(dir, 'train', args.batch_size)
    valid_data = get_dataset(dir, 'val', args.batch_size)
    model = CNN()
    # model = keras.models.Sequential([
    #             Conv2D(6, (5,5)),
    #             MaxPool2D((2, 2)),
    #             Conv2D(16, (5,5)),
    #             MaxPool2D((2, 2)),
    #             Flatten(),
    #             Dense(128, activation='relu'),
    #             Dense(64, activation='relu'),
    #             Dense(10, activation='softmax')
    #         ])
    
    optimizer = tf.optimizers.Adam()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    for epoch in range(args.epochs):
        loss_list = []
        acc_list = []
        for image, label in tqdm(train_data):
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',
                        type=str)
    parser.add_argument('--exp_name',
                        type=str,
                        default='digit_classifier')
    parser.add_argument('--epochs',
                        type=int,
                        default=80)
    parser.add_argument('--batch_size',
                        type=int,
                        default=128)
    parser.add_argument('--lr',
                        type=float,
                        default=1e-3)
    args = parser.parse_args()
    dir = '/data4/jason/hw2_data/digits/mnistm'
    train(dir, args)