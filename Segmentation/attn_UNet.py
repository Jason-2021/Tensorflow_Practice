import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv2D, Conv2DTranspose, Attention, BatchNormalization, Dropout, MaxPool2D, MultiHeadAttention, LayerNormalization, Dense
from keras import Sequential
import numpy as np


class Down(keras.layers.Layer):
    def __init__(self, filters, drop_rate, num_conv_per_block, con_dim, key_dim):
        super(Down, self).__init__()
        self.conv_block = Conv_Block(filters, drop_rate, num_conv_per_block)
        self.pool = MaxPool2D()
        self.mha = AttentionLayer(filters, con_dim=con_dim, drop_rate=drop_rate, key_dim=key_dim)
    
    def call(self, input, condition=None, training=False):
        x = self.conv_block(input, training=training)
        
        if condition is None:
            x = self.mha(x, x, training=training)
        else:
            x = self.mha(x, condition, training=training)
        
        
        return self.pool(x), x
    

class Up(keras.layers.Layer):
    def __init__(self, filters, drop_rate, num_conv_per_block, con_dim, key_dim):
        super(Up, self).__init__()
        self.conv_T = Conv2DTranspose(filters, (2,2), (2,2), 'same')
        self.conv_block = Conv_Block(filters, drop_rate, num_conv_per_block)
        #self.mha = AttentionLayer(filters, con_dim=con_dim, drop_rate=drop_rate, key_dim=key_dim)
    
    def call(self, residual_input, input, condition=None, training=False):
        x = self.conv_T(input)
        x = self.conv_block(tf.concat([residual_input, x], axis=-1), training=training)
        # if condition is None:
        #     x = self.mha(x, x, training=training)
        # else:
        #     x = self.mha(x, condition, training=training)

        return x


class Conv_Block(keras.layers.Layer):
    def __init__(self, filters, drop_rate, num_conv_per_block):
        super(Conv_Block, self).__init__()
        self.net = Sequential([])
        for _ in range(num_conv_per_block):
            self.net.add(Conv2D(filters, (3,3), (1,1), 'same', activation='relu'))
            self.net.add(BatchNormalization())
            self.net.add(Dropout(drop_rate))
        # self.net = Sequential([
        #     Conv2D(filters, (3,3), (1,1), 'same', activation='relu'),
        #     BatchNormalization(),
        #     Dropout(drop_rate),
        #     Conv2D(filters, (3,3), (1,1), 'same', activation='relu'),
        #     BatchNormalization(),
        #     Dropout(drop_rate)
        # ])
        #self.net.build((None, None, None, input_dim))
    
    def call(self, input, training):
        return self.net(input, training=training)


class AttentionLayer(keras.layers.Layer):
    def __init__(self, channels, con_dim, drop_rate, key_dim):
        super(AttentionLayer, self).__init__()
        self.mha = MultiHeadAttention(num_heads=4, key_dim=key_dim, value_dim=con_dim, dropout=drop_rate)
        self.fc = Sequential([
            LayerNormalization(),
            Dense(channels, activation='gelu'),
            # tf.keras.activations.gelu(),
            Dense(channels)
        ])
        self.fc.build((None, None, None, channels))
    def call(self, x, condition, training=False):
        x = tf.transpose(x, perm=[0, 3, 1, 2])
        condition = tf.transpose(condition, perm=[0, 3, 1, 2])
        
        x = self.mha(query=x, value=condition, training=training)
       
        x = tf.transpose(x, perm=[0, 2, 3, 1])
        
        
        x = self.fc(x, training=training)

        

        return x
        
        



class attn_UNet(keras.Model):
    def __init__(self, image_size, out_channel, drop_rate, num_down=4, num_conv_per_block=2, con_dim=None):
        super(attn_UNet, self).__init__()
        self.drop_rate = drop_rate
        self.num_conv_per_block = num_conv_per_block
        self.num_down = num_down


        # check
        if image_size[0] != image_size[1]:
            raise ValueError("Input image should be square")
        if image_size[0] % (2**num_down) != 0:
            raise ValueError("image_size % (2**num_down) != 0")
        
        image_size = np.array(image_size)
        if con_dim is None:
            self.down_list = [Down(filters=64*(2**i), drop_rate=drop_rate, num_conv_per_block=num_conv_per_block, con_dim=64*(2**i), key_dim=image_size/(2**i)) for i in range(0, num_down)]
            self.up_list = [Up(filters=64*(2**i), drop_rate=drop_rate, num_conv_per_block=num_conv_per_block, con_dim=64*(2**i), key_dim=image_size/(2**i)) for i in range(0, num_down)]
        else:
            self.down_list = [Down(filters=64*(2**i), drop_rate=drop_rate, num_conv_per_block=num_conv_per_block, con_dim=con_dim, key_dim=image_size/(2**i)) for i in range(0, num_down)]
            self.up_list = [Up(filters=64*(2**i), drop_rate=drop_rate, num_conv_per_block=num_conv_per_block, con_dim=con_dim, key_dim=image_size/(2**i)) for i in range(0, num_down)]
        
        self.res_block = Conv_Block(filters=64*(2**num_down), drop_rate=drop_rate, num_conv_per_block=num_conv_per_block)
        self.out = Conv2D(out_channel, (1,1), (1,1), 'same')
        

    def call(self, x, condition=None, training=False):
        skip_list = []

        for i in range(self.num_down):
            x, res_x = self.down_list[i](x, condition, training=training)
            skip_list.append(res_x)
        
        x = self.res_block(x, training=training)

        for i in range(self.num_down-1, -1, -1):
            x = self.up_list[i](skip_list[i], x, condition, training=training)

        out = self.out(x)

        return out


    
    

    