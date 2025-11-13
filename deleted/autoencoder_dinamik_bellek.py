import datetime
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from os import listdir
from numpy import asarray, vstack, savez_compressed, load
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, npz_filename, batch_size, size_, start=None, end=None):
        self.npz_filename = npz_filename
        self.batch_size = batch_size
        self.size_ = size_
        self.start = start
        self.end = end
        self.src_images, self.tar_images = self.__load_data()
        self.indexes = np.arange(self.start if self.start is not None else 0, self.end if self.end is not None else len(self.src_images))


    def __len__(self):
        if self.start is not None and self.end is not None:
            return int(np.ceil((self.end - self.start) / self.batch_size))
        else:
            return int(np.ceil(len(self.src_images) / self.batch_size))


    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        x_train, x_test = self.__data_generation(indexes)
        return x_train, x_test

    def __load_data(self):
        data = load(self.npz_filename)
        src_images, tar_images = data['arr_0'], data['arr_1']
        if self.start is not None and self.end is not None:
            src_images, tar_images = src_images[self.start:self.end], tar_images[self.start:self.end]
        return src_images, tar_images

    def __data_generation(self, indexes):
        src_images, tar_images = self.src_images[indexes], self.tar_images[indexes]
        x_train = src_images.astype('float32') / 255.0
        x_test = tar_images.astype('float32') / 255.0
        x_train = np.array(x_train).reshape(-1, self.size_, self.size_, 1)
        x_test = np.array(x_test).reshape(-1, self.size_, self.size_, 1)
        return x_train, x_test




strds = (1,1)
hizlandirici = "GPU"
filtre_adet=48
kernel_size=(3,3)
activation_func="sigmoid"




def create_autoencoder_model(input_shape, activation_func=activation_func, filtre_adet=32, kernel_size=kernel_size, strds=(1, 1)):
    model = tf.keras.Sequential()
    model.add(Conv2D(filtre_adet, kernel_size=kernel_size, strides=strds, activation='elu', padding='same', kernel_initializer='he_normal', input_shape=input_shape))
    model.add(Conv2D(filtre_adet/2, kernel_size=kernel_size, strides=strds, activation='elu', padding='same', kernel_initializer='he_normal'))
    model.add(Conv2D(filtre_adet/4, kernel_size=kernel_size, strides=strds, activation='elu', padding='same', kernel_initializer='he_normal'))
    model.add(Conv2DTranspose(filtre_adet/4, kernel_size=kernel_size, strides=strds, activation='elu', padding='same', kernel_initializer='he_normal'))
    model.add(Conv2DTranspose(filtre_adet/2, kernel_size=kernel_size, strides=strds, activation='elu', padding='same', kernel_initializer='he_normal'))
    model.add(Conv2DTranspose(filtre_adet, kernel_size=kernel_size, strides=strds, activation='elu', padding='same', kernel_initializer='he_normal'))
    model.add(Conv2D(1, kernel_size=kernel_size, strides=(1, 1), activation=activation_func, padding='same'))
    return model

def plot_history(hist):
    plt.figure()
    plt.plot(hist.history["loss"], label="Train Loss")
    plt.plot(hist.history["val_loss"], label="Validation Loss")
    plt.legend()
    plt.show()

size_ = 544
kanal = 1

an = datetime.datetime.now()
zaman = datetime.datetime.strftime(an, '%d_%m_%Y__%H_%M')

# tmp=np.load('merged_maps.npz')

# src_images, tar_images = tmp['arr_0'], tmp['arr_1']

batch_size = 1
train_data_generator = DataGenerator('merged_maps.npz', batch_size, size_, end=1000)
val_data_generator   = DataGenerator('maps_val_544.npz', batch_size, size_)

input_shape = (size_, size_, 1)
model = create_autoencoder_model(input_shape)
#model = load_model("son_model.h5")
model.compile(optimizer='adam', loss='BinaryCrossentropy', metrics=['accuracy'])
model.summary()


strds = str((1,1)).replace(",", "_")

checkpoint = ModelCheckpoint('' + str(zaman) + "" + hizlandirici + 'model_f' + str(filtre_adet) + "k" + str(kernel_size[0]) + 'epoch{epoch:05d}' + activation_func + '' + str(strds) + '_.h5', period=1, save_best_only=True)

hist = model.fit(train_data_generator,
                epochs=10,
                validation_data=val_data_generator,
                callbacks=[checkpoint])

model.save("son_model.h5")

plot_history(hist)
