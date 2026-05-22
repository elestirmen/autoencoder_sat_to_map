import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.utils import Sequence
from sklearn.model_selection import train_test_split
import cv2
import datetime
from tensorflow.keras.models import load_model
an = datetime.datetime.now()
zaman=datetime.datetime.strftime(an, '%d_%m_%Y__%H_%M')

class CustomDataGenerator(Sequence):
    def __init__(self, image_paths, batch_size=16, dim=(544, 544), shuffle=True):
        self.image_paths = image_paths
        self.batch_size = batch_size
        self.dim = dim
        self.shuffle = shuffle
        self.on_epoch_end()

    def on_epoch_end(self):
        
        self.indexes = np.arange(len(self.image_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.floor(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_image_paths = [self.image_paths[k] for k in indexes]
        X, y = self.__data_generation(batch_image_paths)
        
        return X, y

    def __data_generation(self, batch_image_paths):
        X = np.empty((self.batch_size, *self.dim, 1))
        y = np.empty((self.batch_size, *self.dim, 1))

        for i, image_path in enumerate(batch_image_paths):
            img = load_img(image_path, color_mode='grayscale')

            
            img = img_to_array(img)
           
           
                     
            input_img = img[:, :512]
            label_img = img[:, 512:]
            
            # cv2.imwrite('input_img1.jpg', input_img)
            # cv2.imwrite('label_img1.jpg', label_img)
            
            input_img = cv2.resize(input_img, (544, 544),interpolation=cv2.INTER_NEAREST)
            label_img = cv2.resize(label_img, (544, 544),interpolation=cv2.INTER_NEAREST)
            
            # input_img = input_img.astype(np.float32) / 255.0
            # label_img = label_img.astype(np.float32) / 255.0
            
            input_img = ((input_img - 127.5) / 127.5)
            label_img = ((label_img - 127.5) / 127.5)
            
            input_img=input_img.reshape(544,544,1)
            label_img=label_img.reshape(544,544,1)
            
            """
            print(input_img.shape)
            print(label_img.shape)
            
            
            
            cv2.imwrite('input_img2.jpg', input_img)
            cv2.imwrite('label_img2.jpg', label_img)
            input("pause")
            """
            X[i,] = input_img
            y[i,] = label_img

        return X, y

#%%
# Veri kümesi yolu
data_dir = "out_ps"

image_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.jpg')]

# Eğitim ve doğrulama setlerini ayırma
train_image_paths, val_image_paths = train_test_split(image_paths, test_size=0.2, random_state=42)

# DataGenerator örnekleri oluşturma
train_generator = CustomDataGenerator(train_image_paths)
val_generator = CustomDataGenerator(val_image_paths)


import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Input
from tensorflow.keras.models import Model


def create_autoencoder_model(input_shape, activation_func='sigmoid', filter_count=32, kernel_size=(3, 3), strides=(1, 1)):
    input_layer = Input(shape=input_shape)
    x = Conv2D(filter_count, kernel_size=kernel_size, strides=strides, activation='elu', padding='same', kernel_initializer='he_normal')(input_layer)
    x = Conv2D(filter_count // 2, kernel_size=kernel_size, strides=strides, activation='elu', padding='same', kernel_initializer='he_normal')(x)
    x = Conv2D(filter_count // 4, kernel_size=kernel_size, strides=strides, activation='elu', padding='same', kernel_initializer='he_normal')(x)
    x = Conv2DTranspose(filter_count // 4, kernel_size=kernel_size, strides=strides, activation='elu', padding='same', kernel_initializer='he_normal')(x)
    x = Conv2DTranspose(filter_count // 2, kernel_size=kernel_size, strides=strides, activation='elu', padding='same', kernel_initializer='he_normal')(x)
    x = Conv2DTranspose(filter_count, kernel_size=kernel_size, strides=strides, activation='elu', padding='same', kernel_initializer='he_normal')(x)
    output_layer = Conv2D(1, kernel_size=kernel_size, strides=(1, 1), activation=activation_func, padding='same')(x)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    return model

input_shape = (544, 544, 1)
activation_func='sigmoid'
filtre_adet=32
kernel_size=(3, 3)
strds=(1, 1)
hizlandirici="GPU"


tf.config.run_functions_eagerly(True)

model = create_autoencoder_model(input_shape)

model = load_model("son_model.h5")

adam = tf.keras.optimizers.Adam(lr=0.001)  

model.compile(optimizer='adam', loss='mse', metrics=['accuracy'],run_eagerly=True)
model.summary()


#%%
from tensorflow.keras.callbacks import ModelCheckpoint

strds = str(strds).replace(",", "_")
checkpoint = ModelCheckpoint('_'+str(zaman)+"_"+hizlandirici+'_model_f'+str(filtre_adet)+"_k"+str(kernel_size[0])+'_epoch_{epoch:05d}_'+activation_func+'_'+str(strds)+'_.h5', period=1, save_best_only=False)

# Modeli eğitme
hist=model.fit(train_generator, validation_data=val_generator, epochs=10, callbacks=[checkpoint])

model.save("son_model.h5")