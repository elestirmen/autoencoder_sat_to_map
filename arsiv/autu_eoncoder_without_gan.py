# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 17:04:00 2021

@author: ertug
"""

from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D,UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard

from tensorflow.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from numpy import asarray
from numpy import vstack
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from numpy import savez_compressed



#%%
# load all images in a directory into memory
def load_images(path, size=(256,512)):
	src_list, tar_list = list(), list()
	# enumerate filenames in directory, assume all are images
	for filename in listdir(path):
		# load and resize the image
		pixels = load_img(path + filename, target_size=size)
		# convert to numpy array
		pixels = img_to_array(pixels)
		# split into satellite and map
		sat_img, map_img = pixels[:, :256], pixels[:, 256:]
		src_list.append(sat_img)
		tar_list.append(map_img)
	return [asarray(src_list), asarray(tar_list)]

# dataset path
path = 'C:/Users/ertug\OneDrive - Çukurova Üniversitesi/Terrain Relative Navigation/AutoEncoder_pix2pix/maps/train/'
# load dataset
[src_images, tar_images] = load_images(path)
print('Loaded: ', src_images.shape, tar_images.shape)
# save as compressed numpy array
filename = 'maps_256.npz'
savez_compressed(filename, src_images, tar_images)
print('Saved dataset: ', filename)



#%%

# load the prepared dataset
from numpy import load
from matplotlib import pyplot
# load the dataset
data = load(r'C:\Users\ertug\OneDrive - Çukurova Üniversitesi\Terrain Relative Navigation\AutoEncoder_pix2pix\maps_256.npz')
src_images, tar_images = data['arr_0'], data['arr_1']
print('Loaded: ', src_images.shape, tar_images.shape)
# plot source images
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


#%% Veri Kümesini oluşturmak ve ayırmak 

(x_train,x_test) = data['arr_0'], data['arr_1']

x_train = x_train.astype('float32')/255.0
x_test = x_test.astype('float32')/255.0

x_train = np.reshape(x_train,(len(x_train),256*256*3))
x_test = np.reshape(x_test,  (len(x_test),256*256*3))

#%%

input_img = Input(shape=(256*256*3,))


encoded = Dense(1024,activation="relu")(input_img)
encoded = Dense(512,activation="relu")(encoded)

decoded = Dense(256,activation="relu")(encoded)
decoded = Dense(512,activation="relu")(decoded)
decoded = Dense(1024,activation="relu")(decoded)

decoded = Dense(196608,activation="sigmoid")(decoded)


autoencoder = Model(input_img,decoded)

autoencoder.compile(optimizer="rmsprop",loss="binary_crossentropy")

hist = autoencoder.fit(x_train,
                       x_test,
                       epochs=10,
                       batch_size=1,
                       shuffle=True,
                       validation_data = (x_train,x_test))



autoencoder.save_weights("autoencoder_model.h5")


#%%

print(hist.history.keys())

plt.plot(hist.history["loss"],label="Train Loss")
plt.plot(hist.history["val_loss"],label="Validation Loss")
plt.legend()
plt.show()


#%%

encoder = Model(input_img,encoded)
encoded_img = encoder.predict(x_train)

plt.imshow(x_test[1000].reshape(256,256,3))
plt.axis("off")
plt.show()

# plt.imshow(encoded_img[1000].reshape(8,8))
# plt.axis("off")
# plt.show()


#%%

print(x_train[0].shape)

print(x_train[0].T.shape)


ert=x_train[0:100][:]

decoded_imgs=autoencoder.predict(x_train[0:100][:])

print(decoded_imgs.shape)


plt.imshow(decoded_imgs[99].reshape(256,256,3))
 

#plt.figure()
#plt.imshow(x_train[10].reshape(256,256,3))
    
   
    
plt.show()
















