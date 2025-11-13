import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D,UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model

from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from numpy import asarray
from numpy import vstack
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from numpy import savez_compressed

import datetime
an = datetime.datetime.now()
zaman=datetime.datetime.strftime(an, '%d_%m_%Y__%H_%M')



size_=544
kanal=1

#%% load, split and scale the maps dataset ready for training
from os import listdir
from numpy import asarray
from numpy import vstack
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from numpy import savez_compressed

"""
# load all images in a directory into memory
def load_images(path, size=(size_,size_*2)):
 	src_list, tar_list = list(), list()
 	i=0# enumerate filenames in directory, assume all are images
    
 	for filename in listdir(path):
         
		# load and resize the image
 		 pixels = load_img(path + filename, target_size=size,color_mode = "grayscale")
		# convert to numpy array
 		 pixels = img_to_array(pixels)
		# split into satellite and map
 		 sat_img, map_img = pixels[:, :size_], pixels[:, size_:]
 		 src_list.append(sat_img)
 		 tar_list.append(map_img)
 		 i+=1
        
 		 if i==1000:    #2200                                       #kaç görüntü alınacağı ayarlanır
  			 break
 	return [asarray(src_list), asarray(tar_list)]

# dataset path
path = "maps_zoom_level_19/sonuclar_19/"

# load dataset
[src_images, tar_images] = load_images(path)
print('Loaded: ', src_images.shape, tar_images.shape)
# save as compressed numpy array
filename = 'maps_val_'+str(size_)+'.npz'
savez_compressed(filename, src_images, tar_images)
print('Saved dataset: ', filename)

src_list = []
tar_list = []
src_images = []
tar_images = []
pixels = []

input("pause")
"""
#%%


# load the prepared dataset
from numpy import load
from matplotlib import pyplot
# load the dataset
data = load('maps_4_'+str(size_)+'.npz')
data_val = load('maps_val_544.npz')

src_images, tar_images = data['arr_0'], data['arr_1']
src_images_val, tar_images_val = data_val['arr_0'], data_val['arr_1']

katsayi=1
yigin=3100

src_images=src_images[(katsayi-1)*yigin:katsayi*yigin]
tar_images=tar_images[(katsayi-1)*yigin:katsayi*yigin]
del data
del data_val

print('Loaded: ', src_images.shape, tar_images.shape)
#plot source images
n_samples = 3
for i in range(n_samples):
 	pyplot.subplot(2, n_samples, 1 + i)
 	pyplot.axis('off')
 	pyplot.imshow(src_images[i].astype('uint8'))
# plot target image
for i in range(n_samples):
 	pyplot.subplot(2, n_samples, 1 + n_samples + i)
 	pyplot.axis('off')
 	pyplot.imshow(tar_images[i].astype('uint8'))
pyplot.show()
data=[]


#input("pause")
#%% Veri Kümesini oluşturmak ve ayırmak 


x_train_ = src_images.astype('float32')/255.0
x_train_val = src_images_val.astype('float32')/255.0

del src_images
del src_images_val

x_test_ = tar_images.astype('float32')/255.0

x_test_val = tar_images_val.astype('float32')/255.0

del tar_images
del tar_images_val

x_train_ = np.array(x_train_).reshape(-1,int(size_),int(size_),1)
x_test_  = np.array(x_test_).reshape(-1,size_,size_,1)

x_train_val = np.array(x_train_val).reshape(-1,int(size_),int(size_),1)
x_test_val  = np.array(x_test_val).reshape(-1,size_,size_,1)



#%%


import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Conv2DTranspose


  
activation_func='sigmoid'
    
filtre_adet=32
kernel_size=(3,3)
strds=(1,1)
input_shape =(size_,size_,1)


model = tf.keras.Sequential()
model.add(Conv2D(filtre_adet, kernel_size=kernel_size, strides=strds, activation='elu', padding='same', kernel_initializer='he_normal', input_shape=input_shape))
model.add(Conv2D(filtre_adet/2, kernel_size=kernel_size, strides=strds, activation='elu', padding='same', kernel_initializer='he_normal'))
model.add(Conv2D(filtre_adet/4, kernel_size=kernel_size, strides=strds, activation='elu', padding='same', kernel_initializer='he_normal'))

model.add(Conv2DTranspose(filtre_adet/4, kernel_size=kernel_size, strides=strds, activation='elu', padding='same', kernel_initializer='he_normal'))
model.add(Conv2DTranspose(filtre_adet/2, kernel_size=kernel_size, strides=strds, activation='elu', padding='same', kernel_initializer='he_normal'))
model.add(Conv2DTranspose(filtre_adet, kernel_size=kernel_size, strides=strds, activation='elu', padding='same', kernel_initializer='he_normal'))
model.add(Conv2D(1, kernel_size=kernel_size, strides=(1, 1), activation=activation_func, padding='same'))


model = load_model("son_model.h5")

adam = tf.keras.optimizers.Adam(lr=0.0003)  

model.compile(optimizer='adam', loss='BinaryCrossentropy', metrics=['accuracy'])
model.summary()


#%% pretrained modeli yüklemek için





#%%

from tensorflow.keras.callbacks import ModelCheckpoint
#train model normally
#model.fit(training_dataset, epochs=EPOCHS, steps_per_epoch=…)

#BATCH_SIZE = 128 * tpu_strategy.num_replicas_in_sync

strds = str(strds).replace(",", "_")


#print(BATCH_SIZE)
hizlandirici="GPU"

checkpoint = ModelCheckpoint('_'+str(zaman)+"_"+hizlandirici+'_model_f'+str(filtre_adet)+"_k"+str(kernel_size[0])+'_epoch_{epoch:05d}_'+activation_func+'_'+str(strds)+'_.h5', period=1, save_best_only=True)
hist = model.fit(x_train_, x_test_,
                epochs=10,
                batch_size=1,
                validation_data=(x_train_val, x_test_val), callbacks=[checkpoint])
"""

checkpointer = ModelCheckpoint(filepath='eniyi.h5',verbose=1,save_best_only=True)

hist =  model.fit(x_train_, x_test_,
        epochs=100,
        batch_size=8,
        #validation_split=0.1,
        validation_data = (x_train_, x_test_),                 
        shuffle=True,callbacks=[checkpointer])           #Shuffle data for each epoch

"""

model.save("son_model.h5")


#%%

print(hist.history.keys())
plt.figure()
plt.plot(hist.history["loss"],label="Train Loss")
plt.plot(hist.history["val_loss"],label="Validation Loss")
plt.legend()
plt.show()

"""

#ram'i serbest bırakmak icin
from numba import cuda 
device = cuda.get_current_device()
device.reset()
"""