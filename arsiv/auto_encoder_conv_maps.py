# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 17:09:17 2021

@author: ertug
"""

from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D,UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt

from numpy import load
from os import listdir
from numpy import asarray
from numpy import vstack
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from numpy import savez_compressed




import os

dirname = os.path.dirname(__file__)   #relatif path için kullanılır


#%%


size_=256               #görüntünün bir kenarının uzunluğunu belirler

                     
#%%
"""
# load all images in a directory into memory
def load_images(path, size=(size_,size_*2)):
    src_list, tar_list = list(), list()
    # enumerate filenames in directory, assume all are images
    for filename in listdir(path):
        # load and resize the image
        pixels = load_img(path + filename, target_size=size)
        # convert to numpy array
        pixels = img_to_array(pixels)
        # split into satellite and map
        sat_img, map_img = pixels[:, :size_], pixels[:, size_:]
        src_list.append(sat_img)
        tar_list.append(map_img)
    return [asarray(src_list), asarray(tar_list)]




# dataset path
path = dirname+'/maps/train/'
# load dataset
[src_images, tar_images] = load_images(path)
print('Loaded: ', src_images.shape, tar_images.shape)
# save as compressed numpy array
filename = 'maps'+str(size_)+'.npz'
savez_compressed(filename, src_images, tar_images)
print('Saved dataset: ', filename)

"""

#%%# load the prepared dataset


# load the dataset
data = load(dirname+'/maps_'+str(size_)+'.npz')
src_images, tar_images = data['arr_0'], data['arr_1']
print('Loaded: ', src_images.shape, tar_images.shape)
# plot source images
n_samples = 3
for i in range(n_samples):
    plt.subplot(2, n_samples, 1 + i)
    plt.axis('off')
    plt.imshow(src_images[i].astype('uint8'))
# plot target image
for i in range(n_samples):
    plt.subplot(2, n_samples, 1 + n_samples + i)
    plt.axis('off')
    plt.imshow(tar_images[i].astype('uint8'))
plt.show()


src_images=[]
tar_images=[]

#%% Veri Kümesini oluşturmak ve ayırmak 

(x_train,x_test) = data['arr_0'], data['arr_1']

x_train_ = x_train.astype('float32')/255.0
x_test_ = x_test.astype('float32')/255.0

x_train_=np.array(x_train_).reshape(-1,size_,size_,3)
x_test_=np.array(x_test_).reshape(-1,size_,size_,3)


x_train=[]
x_test=[]



#%%



import tensorflow as tf
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose


model = tf.keras.Sequential()
model.add(Conv2D(512, (5, 5), strides=(2, 2),activation='relu', padding='same', input_shape=(size_,size_,3)))

model.add(Conv2D(256, (5, 5), strides=(2, 2), activation='relu', padding='same'))


model.add(Conv2D(128, (5, 5), strides=(2, 2), activation='relu', padding='same'))

model.add(Conv2D(64, (5, 5), strides=(2, 2), activation='relu', padding='same'))
 
model.add(Conv2D(32, (5, 5), strides=(1, 1), activation='relu', padding='same'))

     
model.add(Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))

model.add(Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))

model.add(Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False))

model.add(Conv2DTranspose(512, (5, 5), strides=(2, 2), padding='same', use_bias=False))

model.add(Conv2D(3, (3, 3), activation='relu', padding='same'))

model.compile(optimizer='adam', loss='MeanSquaredError', metrics=['accuracy'])
model.summary()


#%%

checkpointer = ModelCheckpoint(filepath=dirname+'/eniyi.h5',verbose=1,save_best_only=True)

hist = model.fit(x_train_, x_test_,
        epochs=10,
        batch_size=2,
        validation_data = (x_train_, x_test_),                 
        shuffle=True,callbacks=[checkpointer])           #Shuffle data for each epoch


model.save("son_model.h5")



#%%


print(hist.history.keys())

plt.plot(hist.history["loss"],label="Train Loss")
plt.plot(hist.history["val_loss"],label="Validation Loss")
plt.legend()
plt.show()


#%%

model = tf.keras.models.load_model(dirname+'/eniyi.h5')


#%%
encoded_img = model.predict(x_train_[0:100][:])

plt.imshow(x_test_[1000].reshape(size_,size_,3))
plt.axis("off")
plt.show()

# plt.imshow(encoded_img[1000].reshape(8,8))
# plt.axis("off")
# plt.show()


#%%

print(x_train_[0].shape)

print(x_train_[0].T.shape)


#ert=x_train_[0:100][:]

decoded_imgs=model.predict(x_train_[0:5][:])



"""
plt.imshow(decoded_imgs[26].reshape(size_,size_,3))
 

plt.figure()
plt.imshow(x_train_[26].reshape(256,256,3))
  
    
plt.show()

"""

n_goruntu=0
n_samples = 3

for i in range(n_samples):
    plt.subplot(3, n_samples, 1  + i)
    plt.axis('off')
    plt.imshow(x_train_[i+n_goruntu].reshape(size_,size_,3))
    
for i in range(n_samples):
    plt.subplot(3, n_samples, 1 +n_samples+ i)
    plt.axis('off')
    plt.imshow(x_test_[i+n_goruntu].reshape(size_,size_,3))
# plot target image
for i in range(n_samples):
    plt.subplot(3, n_samples, 1 + n_samples*2 + i)
    plt.axis('off')
    plt.imshow(decoded_imgs[i+n_goruntu].reshape(size_,size_,3))
plt.show()

#%%

model = tf.keras.models.load_model(dirname+'/eniyi.h5')
def load_image(filename, size=(256,256,3)):
	# load image with the preferred size
	pixels = load_img(filename, target_size=size)
	# convert to numpy array
	uretilen=model.predict(pixels)
	return uretilen

resim=load_img(r"C:\Users\ertug\OneDrive - Çukurova Üniversitesi\Terrain Relative Navigation\AutoEncoder_pix2pix\test\yerleske.jpg",target_size=(256,256,3))

resim = np.asarray(resim)
resim=resim.astype('float32')/255.0
resim=model.predict(resim.reshape(-1,size_,size_,3))
print(resim.shape)
plt.imshow(resim[0])