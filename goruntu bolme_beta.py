import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import cv2

import matplotlib.pyplot as plt

#path="oku_harita2_geo.tif"
#path="swistopo_big.tif"
#path="oku_harita.tif"
#path = "bern_swistopo_small.tif"
#path = "karlik_30_cm_bingmap.tif"
#path = "karlik_30_cm_bingmap_utm.tif"
path = "urgup_bingmap_30cm_utm.tif"
#path = "urgup_30_cm_yeni_gmaps.tif"
#path= "urgup_genel_genis_kendi_uretimim_30cm.tif"


img = cv2.imread(path)


from osgeo import gdal
raster = gdal.Open(path)
gt =raster.GetGeoTransform()
print (gt)
pixelSizeX = gt[1]
pixelSizeY = -gt[5]
print ("x = ",pixelSizeX)
print ("y = ",pixelSizeY)


#%%






"""

##gdal.Warp('outputRaster.tif', path, xRes=1.2, yRes=1.2) # xRes=1.2, yRes=1.2)  #piksel çözünürlüğü 1.2 metreye ayarlanır
gdal.Warp('outputRaster_level_19.tif', path, xRes=0.299, yRes=0.299)  #piksel çözünürlüğü 0.596 metreye ayarlanır

img = cv2.imread('outputRaster_level_19.tif')
raster = gdal.Open('outputRaster_level_19.tif')

##############
img = cv2.imread('karlik_bing_30cm_georef.tif')
#############


#img = cv2.imread('outputRaster.tif')
#raster = gdal.Open('outputRaster.tif')
gt =raster.GetGeoTransform()
print (gt)
pixelSizeX = gt[1]
pixelSizeY = -gt[5]
print ("x = ",pixelSizeX)
print ("y = ",pixelSizeY)
"""

cx,cy,cz=img.shape #bir kenarının uzunluğu bulunur cx'e atanır
print(cx,cy)

#%%



print("Resim boyutu: ", img.shape)
#cv2.imshow("Orijinal", img)

satir_x=cx
satir_y=cy

# # resized
# imgResized = cv2.resize(img, (kenar,kenar))
# print("Resized Img Shape: ", imgResized.shape)
# #cv2.imshow("Img Resized",imgResized)


frame_size=512    ##512
genisletme=32
frame_adedi_x=int(satir_x/frame_size)
frame_adedi_y=int(satir_y/frame_size)
imgCropped =[]

#print("frame adeti tek kenar = {}, toplam = {}".format(frame_adedi,frame_adedi*frame_adedi))
t=0
for i in range(frame_adedi_x):
    for j in range(frame_adedi_y):      
        
        
        
        imgCropped.append(img[((frame_size)*i):(frame_size)*(i+1)+genisletme,((frame_size)*j):(frame_size)*(j+1)+genisletme]) # width height -> height width
        filename = 'bolunmus/bolunmus/{}_goruntu{}_g.jpg'.format(path,t)
        cv2.imwrite(filename, imgCropped[t])
        
        
        
        print(i)
        t+=1

t=0      



    # imgCropped.append(img[:frame_size,frame_size*i:frame_size*(i+1)]) # width height -> height width 
    # cv2.imshow("Kirpik Resim",imgCropped[i])
fig,ax = plt.subplots(frame_adedi_x,frame_adedi_y, figsize = (10,10)) 
for i in range(frame_adedi_x):
    for j in range(frame_adedi_y):
        imgCropped[t] = cv2.cvtColor(imgCropped[t], cv2.COLOR_BGR2RGB)
        ax[i,j].imshow(imgCropped[t])
        ax[i,j].axis('off')
        t+=1
        #fig.savefig("goruntu_{}.png".format(frame_adedi*i+j))
        #plt.figure(),plt.imshow(imgCropped[i]),plt.axis("off"),plt.title("orijinal"),plt.show()