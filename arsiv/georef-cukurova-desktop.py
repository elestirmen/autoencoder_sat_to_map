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
    
    
    #hedef dizin ve isim
    output = "georefli/"+orig_fn+"_geo.tif"
        
        
        # Filepath
    dataset = r"ana_haritalar/"+harita
    
    # Outfile path
    outpath = output
    
    # Open multiband raster
    raster = rasterio.open(dataset)
    
    #ornek referanslı tiff dosdyası seçilir
    #georasterref=rasterio.open("urgup_32_k2_s1_georef.tif")
    georasterref=rasterio.open("urgup_georefli_6336.tif")
    
    dosya = raster.read(1)
   
    
    
  
    
    out_meta = georasterref.meta.copy()

    out_meta.update({'driver':'GTiff',
                 'width':georasterref.shape[1],
                 'height':georasterref.shape[0],
                 'count':1,
                 'dtype':'uint8',
                 'crs':georasterref.crs, 
                 'transform':georasterref.transform,
                 'nodata':0})
    
    with rasterio.open(output, 'w', **out_meta) as dst:
        # dst.crs = crs
        # dst.transform = transform
        dst.write_band(1, dosya.astype(rasterio.uint8))
        

    

