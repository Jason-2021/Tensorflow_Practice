import tensorflow as tf
import os
import pandas as pd
import numpy as np


def get_dataset(dir='', mode='train', batch_size=128):  
    mean = tf.constant([0.485, 0.456, 0.406])
    std = tf.constant([0.229, 0.224, 0.225])
    def load_data(filepath):
        image = tf.io.read_file(filepath)
        image = tf.image.decode_png(image, channels=3)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = (image - mean) / std

        return image


    df = pd.read_csv(os.path.join(dir, mode+'.csv'))
    df = df.to_numpy()
    image_dataset = tf.data.Dataset.from_tensor_slices([os.path.join(dir, 'data', x[0]) for x in df])
    # print([os.path.join(dir, 'data', x[0]) for x in df])
    label_dataset = tf.data.Dataset.from_tensor_slices([x[1] for x in df])
    # print([x[1] for x in df])
    image_dataset = image_dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = tf.data.Dataset.zip((image_dataset, label_dataset))

    dataset = dataset.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset

if __name__ == '__main__':
    dir = '/data4/jason/hw2_data/digits/mnistm'
    print(get_dataset(dir=dir))