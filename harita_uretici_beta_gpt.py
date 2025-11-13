import os
import cv2
import numpy as np
import glob
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from matplotlib import pyplot, cm
from natsort import natsorted

dirname = os.path.dirname(os.path.abspath(__file__))

def load_image(filename, size=(544, 544)):
    pixels = load_img(filename, target_size=size, color_mode="grayscale")
    pixels = img_to_array(pixels)
    pixels = (pixels - 127.5) / 127.5
    pixels = np.expand_dims(pixels, 0)
    return pixels

def process_image(model, yol2, olusturulacak_dosya, filename):
    src_image = load_image(yol2 + filename)
    print('Loaded', src_image.shape)

    try:
        gen_image = model.predict(src_image,verbose=0)
    except:
        return

    filename_parcalar = olusturulacak_dosya + '/goruntu_{}.jpg'.format(filename[:-4])

    goruntu = gen_image[0].reshape(544, 544)
    pyplot.imsave(filename_parcalar, goruntu, cmap=cm.gray)

def merge_images(frame_adedi_x, frame_adedi_y, olusturulacak_dosya, harita, modelname):
    path = olusturulacak_dosya
    liste2 = natsorted(os.listdir(path))

    img_array = []
    for img in liste2:
        genisleme = 8
        img_ary = cv2.imread(os.path.join(path, img))
        img_array.append(img_ary[genisleme:544 - genisleme, genisleme:544 - genisleme])

    res = []
    t = 0
    for i in range(frame_adedi_x):
        hor = img_array[t]
        t += 1
        for j in range(1, frame_adedi_y):
            hor = np.hstack((hor, img_array[t]))
            t += 1
        res.append(hor)

    merged_image = res[0]
    for i in range(1, frame_adedi_x):
        merged_image = np.vstack((merged_image, res[i]))

    cv2.imwrite("ana_haritalar/ana_harita_{}_{}.jpg".format(harita, modelname), merged_image)

if __name__ == "__main__":
    yol1 = dirname + '/bolunmus/'
    liste1 = natsorted(os.listdir(yol1))
    
    for bolunmus in liste1:
        yol_model = dirname + '/modeller/'
        yol2 = yol1 + "/" + bolunmus + "/"
    
        for modelname in os.listdir(yol_model):
            model = load_model(yol_model + modelname)
            liste2 = natsorted(os.listdir(yol2))
            harita = liste2[0][0:12]
            olusturulacak_dosya = 'c:/d_surucusu/parcalar/' + harita + "_" + modelname
            os.makedirs(olusturulacak_dosya, exist_ok=True)
    
            process_image_partial = partial(process_image, model, yol2, olusturulacak_dosya)
    
            with ThreadPoolExecutor() as executor:
                executor.map(process_image_partial, liste2)
                
                
    
            #ürgüp için
            frame_adedi_x=52
            frame_adedi_y=71  
            
            
            #köy için
            # frame_adedi_x=71
            # frame_adedi_y=41
    
            karekok = np.sqrt(len(liste2))
    
            if len(liste2) % karekok == 0:
                frame_adedi_x = int(karekok)
                frame_adedi_y = int(karekok)
    
            merge_images(frame_adedi_x, frame_adedi_y, olusturulacak_dosya, harita, modelname)
    
    print("FINISHED!!!")

