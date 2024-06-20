import tensorflow as tf
import os
import numpy as np




def get_dataset(dir_path, train_image_size=(224, 224), batchsize=64):
    mean = tf.constant([0.485, 0.456, 0.406])
    std = tf.constant([0.229, 0.224, 0.225])
    def load_image(filepath):
        img = tf.io.read_file(filepath)  # type
        img = tf.image.decode_png(img, channels=3)  # type
        
        img = tf.image.resize(img, train_image_size)
        img = tf.image.convert_image_dtype(img, dtype=tf.float32)
        img = (img - mean) / std
        a = tf.constant([1])
        
        
        
        return img
    

    def augment_image(image):
        # image, label = data[0], data[1]
        image = tf.image.random_flip_left_right(image)
        # image = tf.image.random_flip_up_down(image)
        # image = tf.image.rot90(image, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.resize_with_crop_or_pad(image, 250, 250)
        image = tf.image.random_crop(image, size=[224, 224, 3])
        return image
    
    
    image_paths = [os.path.join(dir_path, x) for x in os.listdir(dir_path) if x.endswith('.png')]
    
    path_dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    image_dataset = path_dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    image_dataset = image_dataset.map(augment_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    label = tf.data.Dataset.from_tensor_slices([np.int32(x.split('/')[-1].split('_')[0]) for x in image_paths])
    
    dataset = tf.data.Dataset.zip((image_dataset, label))
    
    dataset = dataset.shuffle(buffer_size=len(dataset), reshuffle_each_iteration=False)
    dataset = dataset.batch(batchsize)
    

    return dataset



if __name__ == '__main__':
    image_dir = '/data/jason/hw1_data/p1_data/train_50'
    imgs = [os.path.join(image_dir, x) for x in os.listdir(image_dir) if x.endswith('.png')]

    dataset = get_dataset(imgs)
    for img, label in dataset:
        print(type(img))
        # print(label)
        break
    
    # get_dataset(imgs[0])

