#import shutil
from osgeo import gdal, osr
import os
import cv2
import rasterio as rio
from rasterio.transform import from_gcps
from rasterio.control import GroundControlPoint
import os
import rasterio
import numpy as np
 



DATADIR_haritalar = r"ana_haritalar"


haritalar_list =os.listdir(DATADIR_haritalar)


for harita in haritalar_list:
    
    print(harita)


    
    
    orig_fn = harita
    
    
   
        
        
        # Filepath
    dataset = r"ana_haritalar/"+harita
    
    
    #hedef dizin ve isim
    output = "georefli/harita/"+orig_fn+"_geo.tif"
    
    # Outfile path
    outpath = output
    
    # Open multiband raster
    raster = rasterio.open(dataset)
    
    #ornek referanslı tiff dosdyası seçilir
    #georasterref=rasterio.open("urgup_32_k2_s1_georef.tif")
    #georasterref=rasterio.open("urgup_gmap_30_cm_georeferans.tif")
    #georasterref=rasterio.open("ana_harita_karlik_30_cm_bingmap_Georeferans.tif")
    georasterref=rasterio.open("ana_harita_urgup_30_cm__Georefference.tif")
    
    dosya = raster.read(1)
   
    
    
  
    
    out_meta = georasterref.meta.copy()

    out_meta.update({'driver':'GTiff',
                 'width':georasterref.shape[1],
                 'height':georasterref.shape[0],
                 'count':1,
                 'dtype':'uint8',
                 'crs':georasterref.crs, 
                 'transform':georasterref.transform,
                 'compress': 'LZW',
                 'nodata':0})
    
    with rasterio.open(output, 'w', **out_meta) as dst:
        # dst.crs = crs
        # dst.transform = transform
        dst.write_band(1, dosya.astype(rasterio.uint8))
        
#%% 
###########################################################################################

DATADIR_haritalar = r"georefli/harita/"


haritalar_list =os.listdir(DATADIR_haritalar)


for harita in haritalar_list:
    
    print(harita)


    
    
    orig_fn = harita
    
    
    #hedef dizin ve isim
    output = "georefli/"+orig_fn+"_geo.tif"
        
        
        # Filepath
    
    dataset= "georefli/harita/"+harita
    
    
    raster = gdal.Open(dataset)
    gt =raster.GetGeoTransform()
    print (gt)
    pixelSizeX = gt[1]
    pixelSizeY = -gt[5]
    print ("x = ",pixelSizeX)
    print ("y = ",pixelSizeY)
    
    #img=cv2.imread(dataset)
    #gdal.Warp(harita+"_r.tif", raster, xRes=0.299, yRes=0.299)  #piksel çözünürlüğü 0.596 metreye ayarlanır
    
    translate_options = gdal.TranslateOptions(format = 'GTiff',
                                          creationOptions = ['TFW=NO', 'COMPRESS=jpeg']
                                          )
    
    
    out=gdal.Translate(output+"_r.tif", raster,options=translate_options)
    
    #img_resized=cv2.resize(img, (38592,28234))
    #cv2.imwrite(harita+"_resized.jpg", img_resized,[cv2.IMWRITE_JPEG_QUALITY, 75])  


    

