import tensorflow as tf
import os
import imageio.v2 as imageio
import numpy as np


def get_sat(dir_path, batchsize=16):
    mean = tf.constant([0.485, 0.456, 0.406])
    std = tf.constant([0.229, 0.224, 0.225])
    def load_image(filepath):
        img = tf.io.read_file(filepath)  # type
        img = tf.image.decode_jpeg(img, channels=3)  # type
        
        # img = tf.image.resize(img, train_image_size)
        img = tf.image.convert_image_dtype(img, dtype=tf.float32)
        img = (img - mean) / std
        
        return img

    image_paths = [x for x in os.listdir(dir_path) if x.endswith('.jpg')]
    image_paths.sort()
    image_paths = [os.path.join(dir_path, x) for x in image_paths]

    path_dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    sat_dataset = path_dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return sat_dataset


def get_label(dir_path, batchsize=16):
    # label = tf.Variable(tf.zeros((512, 512), dtype=tf.int32))
    def load_image(filepath):
        mask = tf.io.read_file(filepath)  # type
        mask = tf.image.decode_jpeg(mask, channels=3)

        with tf.init_scope():
            label = tf.Variable(tf.zeros((512, 512), dtype=tf.int32))
        # label = tf.Variable(tf.zeros_like(label, dtype=tf.int32))
        
        mask = tf.cast((mask > 128), dtype=tf.int32)
        mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + 1 * mask[:, :, 2]

        label.assign(tf.where(mask == 3, 0, label))
        label.assign(tf.where(mask == 6, 1, label))
        label.assign(tf.where(mask == 5, 2, label))
        label.assign(tf.where(mask == 2, 3, label))
        label.assign(tf.where(mask == 1, 4, label))
        label.assign(tf.where(mask == 7, 5, label))
        label.assign(tf.where(mask == 0, 6, label))
        
        return label
    

    # def load_image_wrapper(filepath):
    #     image = tf.py_function(func=load_image, inp=[filepath], Tout=tf.float32)
    #     return image

    label_paths = [x for x in os.listdir(dir_path) if x.endswith('.png')]
    label_paths.sort()
    label_paths = [os.path.join(dir_path, x) for x in label_paths]

    path_dataset = tf.data.Dataset.from_tensor_slices(label_paths)
    label_dataset = path_dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return label_dataset


def get_dataset(dir_path, batchsize=16, train=True):
    def augment(image, label):
        if tf.random.uniform(()) > 0.5:
            image = tf.image.flip_left_right(image)
            label = tf.reverse(label, axis=[1])
        
        if tf.random.uniform(()) > 0.5:
            image = tf.image.flip_up_down(image)
            label = tf.reverse(label, axis=[0])
        
        return (image, label)
    
    sat = get_sat(dir_path=dir_path, batchsize=batchsize)
    label = get_label(dir_path=dir_path, batchsize=batchsize)

    dataset = tf.data.Dataset.zip((sat, label))
    
    if train:
        dataset = dataset.map(augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.shuffle(buffer_size=len(dataset), reshuffle_each_iteration=False)
    
    dataset = dataset.batch(batchsize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset