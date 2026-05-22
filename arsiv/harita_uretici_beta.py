# load, split and scale the maps dataset ready for training
from os import listdir
from numpy import asarray
from numpy import vstack
from numpy import savez_compressed
# example of loading a pix2pix model and using it for one-off image translation
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from numpy import load
from numpy import expand_dims
from matplotlib import pyplot
from matplotlib.pyplot import cm
import pandas as pd

import os
import cv2
import numpy as np 
import matplotlib.pyplot as plt
from natsort import natsorted   #dosyaları doğru sıralamak için eklendi

dirname = os.path.dirname(os.path.abspath(__file__))

#%%
# load an image
def load_image(filename, size=(544,544)): #görüntününn boyutuna göre size yazılır
	# load image with the preferred size
	pixels = load_img(filename, target_size=size,color_mode = "grayscale")
	# convert to numpy array
	pixels = img_to_array(pixels)
	# scale from [0,255] to [-1,1]
    
   #pixels =pixels.astype('float32')/255.0
	pixels = (pixels - 127.5)/127.5 
    
    
	# reshape to 1 sample
	pixels = expand_dims(pixels, 0)
	return pixels





#%%
from matplotlib import pylab
from natsort import natsorted   #dosyaları doğru sıralamak için eklendi

yol1 = dirname+'/bolunmus/'
liste1 = natsorted(os.listdir(yol1))

for bolunmus in liste1:
    
    yol_model = dirname+'/modeller/'
    yol2 = yol1+"/"+bolunmus+"/"
    
    
    
    for modelname in listdir(yol_model):
        liste2 = natsorted(os.listdir(yol2))  #liste1 bolunmus klasörünü okur
        harita =liste2[0][0:12]
        t=yol_model+modelname  
        print(t)
          
        model = load_model(yol_model+modelname)
        
        
        #modelin filtre sayısı vs görüntüleme. summery methodunu bir tabloya aktarma özetle
        ############################################################################################################
        #table=pd.DataFrame(columns=["Name","Type","Shape"])
        #for layer in model.layers:
        #    table = table.append({"Name":layer.name, "Type": layer.__class__.__name__,"Shape":layer.output_shape}, ignore_index=True)
    
        #kernel_adedi= (table.Shape[0][3])  #modelin flitre-kerne sayısı elde edilir
        ############################################################################################################
       
        olusturulacak_dosya='c:/d_surucusu/parcalar/'+harita+"_"+modelname
       
        #os.mkdir(olusturulacak_dosya)
        os.makedirs(olusturulacak_dosya, exist_ok=True) #dosya varsa da yine de oluştur
       
        dosya=[]
        i=0
        for filename in liste2:
            dosya.append(yol2 + filename) 
            
            img = pyplot.imread(dosya[i])
        # load source image
            src_image = load_image(dosya[i]) 
            
            print('Loaded {}__{}'.format(modelname,i), src_image.shape)
            # load model
            # generate image from source
            
            #src_image = src_image[0].reshape(512,512,1)
            try:
                gen_image = model.predict(src_image)
            # scale from [-1,1] to [0,1]
            #gen_image = (gen_image + 1) / 2.0
            except:
                continue
            
            
            
            filename_parcalar = olusturulacak_dosya+'/goruntu_{}_{}.jpg'.format(i,modelname)   
            
            # pyplot.imshow(gen_image[0],cmap=cm.gray)
            # input("pause")
            goruntu = gen_image[0].reshape(544,544)  #bw photo siyah beyaz görüntüler için
            #input("pause")
            
            pyplot.imsave(filename_parcalar, goruntu,cmap=cm.gray) #for color image
            #cv2.imwrite(filename_parcalar, gen_image[0])   
            
            # fig = pylab.gcf()
        
            # #pyplot grafik adı ayarlama
            # fig.canvas.set_window_title(filename)
            # fig=pyplot.figure()
            # fig.suptitle(filename, fontsize=20)
            # pyplot.subplot(121), pyplot.imshow(img)
            # pyplot.title("Orijinal Görüntü"), pyplot.axis("on")
            # pyplot.subplot(122), pyplot.imshow(gen_image[0])
            # pyplot.title("Üretilen HArita"), pyplot.axis("on")
            # filename=filename+'_.jpg'
            # pyplot.savefig(dirname+'/sonuclar/'+filename+modelname+".jpg")
        
            
            i=i+1
            
            
#%%
    
    #_______________________________________________________________________________________________________________#
            
        #oluşturulan parçaları birleştir
        
        
        #goruntu_bolme_dosyasindan elde edilir.
        #ürgüp için
        # frame_adedi_x=52
        # frame_adedi_y=71  
        
        
        #köy için
        frame_adedi_x=71
        frame_adedi_y=41
        
        
        karekok=np.sqrt(len(dosya))
        
        if len(dosya)%karekok==0:
            #♦klasörde kaç tane frame var sayısını getirir oluşturalacak görüntünün bir kenarındaki frame sayısı gibi
            frame_adedi_x=int(karekok)
            frame_adedi_y=int(karekok)
            
        path = olusturulacak_dosya
        liste2 = natsorted(os.listdir(path))  #liste2 parçaların yazıldığı klasörü okur
        img_array=[]
        for img in liste2:
                
                genisleme=8
                #img_array.append(cv2.imread(os.path.join(path,img)))
                img_ary=cv2.imread(os.path.join(path,img))
                img_array.append(img_ary[genisleme:544-genisleme,genisleme:544-genisleme])                   
                
        
                
        
        
        # hor = img_array[0]
        # ver = []
        # for i in range(4*4-1):
        #     print(i)    
        #     hor = np.hstack((hor,img_array[i+1]))
        #     if i%4==0:
        #         ver.append(hor)
        #         hor=img_array[i]
        #     cv2.imshow("Yatay",hor)
            
        # for i in range(len(ver)):
        #     son = np.vstack((ver[i],ver[i+1]))
        
        res=[]
        ver = []
        hor = []
        
        t=0
        for i in range(frame_adedi_x):
            
            for j in range(frame_adedi_y):
                
                res.append(img_array[t])
                print(t)
            
                t+=1
                
        k=1
        hor =  res[0]   
        while k<frame_adedi_x*frame_adedi_y:
            
            if k%frame_adedi_y==0:
                ver.append(hor)
                hor=res[k]
                k+=1
                
                
            hor = np.hstack((hor,res[k]))
            
            k+=1
        ver.append(hor)
            
            
            
        
            
        
          
        k=1
        son =  ver[0]   
        while k<frame_adedi_x:
            son = np.vstack((son,ver[k]))
           
            k+=1
        
        #cv2.imshow("son",son)
        
        cv2.imwrite("ana_haritalar/ana_harita_{}_{}.jpg".format(harita,modelname),son)        
        
        
print("FINISHED!!!")

