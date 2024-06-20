import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv2D, Conv2DTranspose, MaxPool2D, Dense, BatchNormalization, ReLU

class UNet(keras.Model):
    def __init__(self):
        super().__init__(self)
        self.block1 = self.Block(64)
        self.block2 = self.Block(128)
        self.block3 = self.Block(256)
        self.block4 = self.Block(512)
        self.block5 = self.Block(1024)
        self.block6 = self.Block(512)
        self.block7 = self.Block(256)
        self.block8 = self.Block(128)
        self.block9 = self.Block(64)

        self.max_pool = [MaxPool2D() for _ in range(4)]

        self.conv_T1 = Conv2DTranspose(512, (2,2), (2,2), 'same')
        self.conv_T2 = Conv2DTranspose(156, (2,2), (2,2), 'same')
        self.conv_T3 = Conv2DTranspose(128, (2,2), (2,2), 'same')
        self.conv_T4 = Conv2DTranspose(64, (2,2), (2,2), 'same')

        self.out = Conv2D(7, (1,1), (1,1), 'same')
    

    def Block(self, filters):
        return keras.Sequential([
            Conv2D(filters, (3,3), (1,1), 'same'),
            BatchNormalization(),
            ReLU(),
            Conv2D(filters, (3,3), (1,1), 'same'),
            BatchNormalization(),
            ReLU()
        ])
    

    def call(self, input, training=False):
        # down
        x = self.block1(input, training=training)
        encoder_64 = tf.identity(x)
        x = self.max_pool[0](x)
        

        x = self.block2(x, training=training)
        encoder_128 = tf.identity(x)
        x = self.max_pool[1](x)
        

        x = self.block3(x, training=training)
        encoder_256 = tf.identity(x)
        x = self.max_pool[2](x)
        

        x = self.block4(x, training=training)
        encoder_512 = tf.identity(x)
        x = self.max_pool[3](x)

        # bottleneck
        x = self.block5(x, training=training)
    
        # up
        x = self.conv_T1(x)
        x = self.block6(tf.concat([x, encoder_512], axis=-1), training=training)

        x = self.conv_T2(x)
        x = self.block7(tf.concat([x, encoder_256], axis=-1), training=training)

        x = self.conv_T3(x)
        x = self.block8(tf.concat([x, encoder_128], axis=-1), training=training)

        x = self.conv_T4(x)
        x = self.block9(tf.concat([x, encoder_64], axis=-1), training=training)

        x = self.out(x)

        return x

    
