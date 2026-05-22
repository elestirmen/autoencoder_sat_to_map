import cv2
import numpy as np
import datetime
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Input,UpSampling2D,MaxPooling2D,Dropout, BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K




import os

an = datetime.datetime.now()
zaman = datetime.datetime.strftime(an, '%d_%m_%Y__%H_%M')

# Özel callback sınıfını tanımlayın
class PeriodicSave(tf.keras.callbacks.Callback):
    def __init__(self, save_every=1000):
        self.save_every = save_every
        super(PeriodicSave, self).__init__()

    def on_train_batch_end(self, batch, logs=None):
        if batch % self.save_every == 0:
            self.model.save(f'{model_adi}_step_{batch}.h5')

    def on_epoch_end(self, epoch, logs=None):
        self.model.save(f'{model_adi}_epoch_{epoch}.h5')
        
        
        
def ssim_loss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))        
        

        
# Görüntü yollarını al
all_image_paths = ["C:\\d_surucusu\\satnap\\output_ps\\" + fname for fname in os.listdir("C:\\d_surucusu\\satnap\\output_ps")]

#all_image_paths = ["C:\\d_surucusu\\sonuclar_19\\" + fname for fname in os.listdir("C:\\d_surucusu\\sonuclar_19")]


np.random.seed(21)

# Görüntü yollarını karıştır
np.random.shuffle(all_image_paths)

# Eğitim ve doğrulama setlerini ayır
split_at = int(len(all_image_paths) * 0.9)  # %80 eğitim, %20 doğrulama
train_image_paths = all_image_paths[:split_at]
val_image_paths = all_image_paths[split_at:]        



def load_and_preprocess(image_path):
    try:
        # Dosyayı oku ve decode et
        img_raw = tf.io.read_file(image_path)
        img = tf.image.decode_image(img_raw, channels=1)

        # Veri tipini float32'ye çevir
        img = tf.cast(img, tf.float32)

        # Boyutları al
        shape = tf.shape(img)
        height = shape[0]
        width = shape[1] // 2  # Girdi ve etiket yan yana olduğu için genişliği yarıya böl

        # Girdi ve etiketi ayır
        input_img = tf.slice(img, [0, 0, 0], [height, width, 1])
        label_img = tf.slice(img, [0, width, 0], [height, width, 1])

        # Boyutlandırma
        input_img = tf.image.resize(input_img, [544, 544], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        label_img = tf.image.resize(label_img, [544, 544], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        # Normalizasyon
        input_img = (input_img - 127.5) / 127.5
        label_img = (label_img - 127.5) / 127.5

        return (input_img, label_img)

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return (tf.zeros([544, 544, 1], dtype=tf.float32), tf.zeros([544, 544, 1], dtype=tf.float32))




def create_autoencoder_model(input_shape, activation_func='sigmoid', filter_count=32, kernel_size=(5, 5), strides=(1, 1)):
    input_layer = Input(shape=input_shape)
    x = Conv2D(filter_count, kernel_size=kernel_size, strides=strides,
               activation='elu', padding='same', kernel_initializer='he_normal')(input_layer)
    x = Conv2D(filter_count // 2, kernel_size=kernel_size, strides=strides,
               activation='elu', padding='same', kernel_initializer='he_normal')(x)
    x = Conv2D(filter_count // 4, kernel_size=kernel_size, strides=strides,
               activation='elu', padding='same', kernel_initializer='he_normal')(x)

    x = Conv2DTranspose(filter_count // 4, kernel_size=kernel_size, strides=strides,
                        activation='elu', padding='same', kernel_initializer='he_normal')(x)
    x = Conv2DTranspose(filter_count // 2, kernel_size=kernel_size, strides=strides,
                        activation='elu', padding='same', kernel_initializer='he_normal')(x)
    x = Conv2DTranspose(filter_count, kernel_size=kernel_size, strides=strides,
                        activation='elu', padding='same', kernel_initializer='he_normal')(x)
    
    x = Conv2DTranspose(filter_count, kernel_size=kernel_size, strides=strides,
                        activation='elu', padding='same', kernel_initializer='he_normal')(x)
    
    output_layer = Conv2D(1, kernel_size=kernel_size, strides=(
        2, 2), activation=activation_func, padding='same')(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model


def create_autoencoder_model_backup(input_shape, activation_func='sigmoid', filter_count=32, kernel_size=(5, 5), strides=(1, 1)):
    input_layer = Input(shape=input_shape)
    x = Conv2D(filter_count, kernel_size=kernel_size, strides=strides,
               activation='elu', padding='same', kernel_initializer='he_normal')(input_layer)
    x = Conv2D(filter_count // 2, kernel_size=kernel_size, strides=strides,
               activation='elu', padding='same', kernel_initializer='he_normal')(x)
    x = Conv2D(filter_count // 4, kernel_size=(3,3), strides=(1,1),
               activation='elu', padding='same', kernel_initializer='he_normal')(x)

    x = Conv2D(filter_count // 4, kernel_size=(1,1), strides=(2,2),
               activation='elu', padding='same', kernel_initializer='he_normal')(x)

    x = Conv2DTranspose(filter_count // 4, kernel_size=(3,3), strides=(2,2),
                        activation='elu', padding='same', kernel_initializer='he_normal')(x)
    x = Conv2DTranspose(filter_count // 2, kernel_size=kernel_size, strides=strides,
                        activation='elu', padding='same', kernel_initializer='he_normal')(x)
    output_layer = Conv2DTranspose(1, kernel_size=kernel_size, strides=strides,
                        activation=activation_func, padding='same', kernel_initializer='he_normal')(x)
    
    
    model = Model(inputs=input_layer, outputs=output_layer)
    return model





def create_upsampled_autoencoder(input_shape, activation_func='sigmoid', filter_count=32, kernel_size=(5, 5), strides=(1, 1)):
    input_layer = Input(shape=input_shape)
    
    # Encoder
    x = Conv2D(filter_count, kernel_size=kernel_size, strides=strides, activation='elu', padding='same', kernel_initializer='he_normal')(input_layer)
    x = Conv2D(filter_count // 2, kernel_size=kernel_size, strides=strides, activation='elu', padding='same', kernel_initializer='he_normal')(x)
    x = Conv2D(filter_count // 4, kernel_size=kernel_size, strides=strides, activation='elu', padding='same', kernel_initializer='he_normal')(x)
    x = Conv2D(filter_count // 8, kernel_size=(2, 2), strides=(1, 1), activation='elu', padding='same', kernel_initializer='he_normal')(x)
    
    # Decoder
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(filter_count // 4, kernel_size=kernel_size, strides=(1, 1), activation='elu', padding='same', kernel_initializer='he_normal')(x)
    
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(filter_count // 2, kernel_size=kernel_size, strides=(1, 1), activation='elu', padding='same', kernel_initializer='he_normal')(x)
    
    x = UpSampling2D(size=(2, 2))(x)
    output_layer = Conv2D(1, kernel_size=kernel_size, strides=(1, 1), activation='elu', padding='same', kernel_initializer='he_normal')(x)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    return model



def create_advanced_autoencoder(input_shape, activation_func='relu', filter_count=32, kernel_size=(3, 3), strides=(1, 1)):
    input_layer = Input(shape=input_shape)
    
    # Encoder
    x = Conv2D(filter_count, kernel_size=kernel_size, strides=strides, activation=activation_func, padding='same')(input_layer)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(0.3)(x)
    
    x = Conv2D(filter_count * 2, kernel_size=kernel_size, strides=strides, activation=activation_func, padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    
    # Decoder
    x = Conv2DTranspose(filter_count * 2, kernel_size=kernel_size, strides=strides, activation=activation_func, padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Dropout(0.3)(x)
    
    x = Conv2DTranspose(filter_count, kernel_size=kernel_size, strides=strides, activation=activation_func, padding='same')(x)
   
    
    output_layer = Conv2DTranspose(1, kernel_size=kernel_size, strides=strides, activation=activation_func, padding='same')(x)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model



def create_gpt_autoencoder(input_shape, activation_func='relu', filter_count=32, kernel_size=(3, 3), strides=(1, 1)):  #favori
    
    # Encoder
    input_layer = Input(shape=input_shape)
    
    x = Conv2D(filter_count, kernel_size, activation='elu', padding='same',activity_regularizer=regularizers.l1(1e-5))(input_layer)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(0.4)(x)
    
    x = Conv2D(filter_count*2, kernel_size, activation='elu', padding='same',activity_regularizer=regularizers.l1(1e-5))(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(0.4)(x)
    
    # Decoder
    x = Conv2DTranspose(filter_count*2, kernel_size, activation='elu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Dropout(0.4)(x)
    
    x = Conv2DTranspose(filter_count, kernel_size, activation='elu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Dropout(0.4)(x)
    
    output_layer = Conv2DTranspose(1, kernel_size, activation='sigmoid', padding='same')(x)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model


def create_gpt_autoencoder_none_regularization(input_shape, activation_func='relu', filter_count=32, kernel_size=(3, 3), strides=(1, 1)):  #favori
    
    # Encoder
    input_layer = Input(shape=input_shape)
    
    x = Conv2D(filter_count, kernel_size, activation='elu', padding='same')(input_layer)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(0.4)(x)
    
    x = Conv2D(filter_count*2, kernel_size, activation='elu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(0.4)(x)
    
    # Decoder
    x = Conv2DTranspose(filter_count*2, kernel_size, activation='elu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Dropout(0.4)(x)
    
    x = Conv2DTranspose(filter_count, kernel_size, activation='elu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Dropout(0.4)(x)
    
    output_layer = Conv2DTranspose(1, kernel_size, activation='sigmoid', padding='same')(x)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model



input_shape = (544, 544, 1)
activation_func = 'relu'
filtre_adet = 32
kernel_size = (4, 4)
strds = (2, 2)
batch_size = 8

model = create_gpt_autoencoder_none_regularization(input_shape, activation_func, filtre_adet, kernel_size, strds)


#model = load_model("son_model.h5")

adam = tf.keras.optimizers.Adam(lr=0.0005)
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])


# Modelin özetini yazdırın
model.summary()

# Model adını oluşturun
model_adi = '_'+str(zaman)+'_'+str(filtre_adet)+'_'+str(kernel_size[0])+'_'+str(strds)+'_.h5'

# Özel callback sınıfını oluşturun
periodic_save = PeriodicSave(save_every=500)

# # Veri yollarını yükleyin (örnek)
# train_image_paths = ["C:\d_surucusu\satnap\output_ps"]
# val_image_paths = ["path/to/val/image1", "path/to/val/image2", ...]

# tf.data.Dataset kullanarak veri ön yükleme
train_dataset = tf.data.Dataset.from_tensor_slices(train_image_paths)
train_dataset = train_dataset.map(load_and_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices(val_image_paths)
val_dataset = val_dataset.map(load_and_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


checkpoint = ModelCheckpoint('_'+str(zaman)+"_"'_model_f'+str(filtre_adet)+"_k"+str(
    kernel_size[0])+'_epoch_{epoch:05d}_'+activation_func+'_'+str(strds)+'_.h5', period=1, save_best_only=False)



hist = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=21,
    callbacks=[checkpoint]  # Veya diğer callback'ler
)


# Modeli kaydet
model.save("son_model.h5")
    