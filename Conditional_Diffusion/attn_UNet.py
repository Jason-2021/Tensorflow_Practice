import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv2D, Conv2DTranspose, Attention, BatchNormalization, Dropout, MaxPool2D
from keras import Sequential


class Down(keras.layers.Layer):
    def __init__(self, filters, drop_rate, num_conv_per_block):
        super().__init__(self)
        self.conv_block = Conv_Block(filters, drop_rate, num_conv_per_block)
        self.pool = MaxPool2D()
    
    def call(self, input, training=False):
        x = self.conv_block(input, training=training)
        return self.pool(x)
    

class Up(keras.layers.Layer):
    def __init__(self, filters, drop_rate, num_conv_per_block):
        super().__niit__(self)
        self.conv_T = Conv2DTranspose(filters, (2,2), (2,2), 'same')
        self.conv_block = Conv_Block(filters, drop_rate, num_conv_per_block)
    
    def call(self, residual_input, input, training=False):
        x = self.conv_T(input)
        x = self.conv_block(tf.concat([residual_input, x], axis=-1), training=training)
        return x


class Conv_Block(keras.layers.Layer):
    def __init__(self, filters, drop_rate, num_conv_per_block):
        super().__init__(self)
        self.net = Sequential([])
        for _ in range(num_conv_per_block):
            self.net.add(Conv2D(filters, (3,3), (1,1), 'same', activation='relu'))
            self.net.add(BatchNormalization())
            self.net.add(Dropout(drop_rate))
    
    def call(self, input, training):
        return self.net(input, training=training)


class SelfAttention(keras.layers.Layer):
    def __init__(self):
        super.__init__(self)


class attn_UNet(keras.Model):
    def __init__(self, drop_rate, num_conv_per_block):
        super().__init__(self)
        self.drop_rate = drop_rate
        self.num_conv_per_block = num_conv_per_block
    
    def Conv_Block(self, filters):
        net = Sequential([])
        
        for _ in range(self.num_conv_per_block):
            net.add(Conv2D(filters, (3,3), (1,1), 'same', activation='relu'))
            net.add(BatchNormalization())
            net.add(Dropout(self.drop_rate))
        
        return net
    

    def Down(self, filters):
        net = self.Conv_Block(filters)
        return net.add(MaxPool2D())
    

    