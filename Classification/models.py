import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv2D, Dense, Dropout, BatchNormalization, MaxPool2D, ReLU, Flatten, GlobalAveragePooling2D



class BasicModel(tf.keras.Model):
    def __init__(self, global_pool=False):
        super().__init__(self)
        
        self.conv_1 = Conv2D(filters=64, kernel_size=(3,3), strides=1, padding='same')
        self.conv_2 = Conv2D(filters=128, kernel_size=(3,3), strides=1, padding='same')
        self.conv_3 = Conv2D(filters=256, kernel_size=(3,3), strides=1, padding='same')
        self.conv_4 = Conv2D(filters=512, kernel_size=(3,3), strides=1, padding='same')
        self.conv_5 = Conv2D(filters=512, kernel_size=(3,3), strides=1, padding='same')

        self.bn = [BatchNormalization() for _ in range(5)]

        self.relu = [ReLU() for _ in range(5)]

        self.pool = [MaxPool2D((2,2)) for _ in range(5)]

        if not global_pool:
            self.flatten = Flatten()
        else:
            self.flatten = GlobalAveragePooling2D()

        
        self.dn_1 = Dense(1024, activation='relu')
        self.dn_2 = Dense(512, activation='relu')
        self.dn_3 = Dense(256, activation='relu')
        self.dn_4 = Dense(128, activation='relu')
        self.dn_5 = Dense(50)

        self.drop = [Dropout(0.3) for _ in range(4)]

    def call(self, input, training=False):
        # feature extractor
        x = self.conv_1(input)
        x = self.bn[0](x, training=training)
        x = self.relu[0](x)
        x = self.pool[0](x)

        x = self.conv_2(x)
        x = self.bn[1](x, training=training)
        x = self.relu[1](x)
        x = self.pool[1](x)

        x = self.conv_3(x)
        x = self.bn[2](x, training=training)
        x = self.relu[2](x)
        x = self.pool[2](x)

        x = self.conv_4(x)
        x = self.bn[3](x, training=training)
        x = self.relu[3](x)
        x = self.pool[3](x)

        x = self.conv_5(x)
        x = self.bn[4](x, training=training)
        x = self.relu[4](x)
        x = self.pool[4](x)

        # classifier
        x = self.flatten(x)

        x = self.dn_1(x)
        x = self.drop[0](x, training=training)

        x = self.dn_2(x)
        x = self.drop[1](x, training=training)

        x = self.dn_3(x)
        x = self.drop[2](x, training=training)

        x = self.dn_4(x)
        x = self.drop[3](x, training=training)

        x = self.dn_5(x)

        return x


class Resnet50_based(tf.keras.Model):
    def __init__(self):
        super().__init__(self)
        self.feature_extractor = tf.keras.applications.ResNet50(
            include_top=False,
            weights='imagenet',
            input_shape=(224, 224, 3)
        )
        self.global_pool = GlobalAveragePooling2D()

        self.dn_0 = Dense(1024, activation='relu')
        self.dn_1 = Dense(512, activation='relu')
        self.dn_2 = Dense(256, activation='relu')
        self.dn_3 = Dense(128, activation='relu')
        self.dn_4 = Dense(50)

        self.drop = [Dropout(0.3) for _ in range(4)]
    

    def call(self, input, training=False):
        x = self.feature_extractor(input, training=training)

        x = self.global_pool(x)
        
        x = self.dn_0(x)
        x = self.drop[0](x, training=training)

        x = self.dn_1(x)
        x = self.drop[1](x, training=training)

        x = self.dn_2(x)
        x = self.drop[2](x, training=training)

        x = self.dn_3(x)
        x = self.drop[3](x, training=training)

        x = self.dn_4(x)

        return x

if __name__ == '__main__':
    
    model = tf.keras.applications.ResNet50()
    print(model.summary)


        