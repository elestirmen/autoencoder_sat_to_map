import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import cv2

import matplotlib.pyplot as plt

#path="oku_harita2_geo.tif"
#path="swistopo_big.tif"
#path="oku_harita.tif"
#path = "bern_swistopo_small.tif"
path = "urgup_kare_level_18.tif"


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

#spatial çözünürlük elde etme
camera_sensor_genislik=6 #mavic2zoom için 6  milimetre sensör genişliği
camera_focal_lenght=4 #mavic2zoom için 4 milimetre
ucus_yuksekligi=200  #metre olarak uçuş yüksekliği
goruntu_piksel_genisligi = 4000 #pipksel olarak resmin genişliği
goruntu_piksel_yuksekligi = 3000 #pipksel olarak resmin genişliği


mekansal_cozunurluk = (camera_sensor_genislik*ucus_yuksekligi*100)/(camera_focal_lenght*goruntu_piksel_genisligi)  #mekansal çözünürlük cantimeter/pixel olarak



#piksel çözünürlüğünü metre olarak ayarlar

##gdal.Warp('outputRaster.tif', path, xRes=1.2, yRes=1.2) # xRes=1.2, yRes=1.2)  #piksel çözünürlüğü 1.2 metreye ayarlanır
gdal.Warp('outputRaster_level_18.tif', path, xRes=0.596, yRes=0.596)  #piksel çözünürlüğü 0.596 metreye ayarlanır

img = cv2.imread('outputRaster_level_18.tif')
raster = gdal.Open('outputRaster_level_18.tif')

#img = cv2.imread('outputRaster.tif')
#raster = gdal.Open('outputRaster.tif')
gt =raster.GetGeoTransform()
print (gt)
pixelSizeX = gt[1]
pixelSizeY = -gt[5]
print ("x = ",pixelSizeX)
print ("y = ",pixelSizeY)


cx,cy,cz=img.shape #bir kenarının uzunluğu bulunur cx'e atanır
print(cx)

#%%



print("Resim boyutu: ", img.shape)
#cv2.imshow("Orijinal", img)

kenar=cx

# resized
imgResized = cv2.resize(img, (kenar,kenar))
print("Resized Img Shape: ", imgResized.shape)
#cv2.imshow("Img Resized",imgResized)


frame_size = 544
genisletme = 32
frame_adedi = int(kenar/frame_size)
imgCropped = []

print("frame adeti tek kenar = {}, toplam = {}".format(frame_adedi,frame_adedi*frame_adedi))

for i in range(frame_adedi):
    for j in range(frame_adedi):             
        
        
        imgCropped.append(imgResized[((frame_size)*i):(frame_size)*(i+1)+genisletme,((frame_size)*j):(frame_size)*(j+1)+genisletme]) # width height -> height width
        filename = 'bolunmus/{}_goruntu{}_g.jpg'.format(path,frame_adedi*i+j)
        cv2.imwrite(filename, imgCropped[frame_adedi*i+j])       
        
      
        
    # imgCropped.append(img[:frame_size,frame_size*i:frame_size*(i+1)]) # width height -> height width 
    # cv2.imshow("Kirpik Resim",imgCropped[i])
fig,ax = plt.subplots(frame_adedi,frame_adedi, figsize = (10,10)) 
for i in range(frame_adedi):
    for j in range(frame_adedi):
        imgCropped[frame_adedi*i+j] = cv2.cvtColor(imgCropped[frame_adedi*i+j], cv2.COLOR_BGR2RGB)
        ax[i,j].imshow(imgCropped[frame_adedi*i+j])
        ax[i,j].axis('off')
       
        #fig.savefig("goruntu_{}.png".format(frame_adedi*i+j))
        #plt.figure(),plt.imshow(imgCropped[i]),plt.axis("off"),plt.title("orijinal"),plt.show()