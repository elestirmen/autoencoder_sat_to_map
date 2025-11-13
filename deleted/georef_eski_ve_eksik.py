#import shutil
from osgeo import gdal, osr
import os
import cv2



DATADIR_haritalar = r"ana_haritalar"


haritalar_list =os.listdir(DATADIR_haritalar)


for harita in haritalar_list:
    
    print(harita)


    
    
    orig_fn = harita
    
    # Create a copy of the original file and save it as the output filename:
    #shutil.copy(orig_fn, output_fn)
    
    # Open the output file for writing for writing:
    #ds = gdal.Open(output_fn, gdal.GA_Update)
    
    
    
    output = "georefli/"+orig_fn+"_geo.tif"
    
    #ds = gdal.Translate(output, "ana_haritalar/"+harita)
    ds = gdal.Translate(output, "ana_haritalar/"+harita, width=6541, height=6541,outputSRS="EPSG:4326")

    
    # Set spatial reference:
    sr = osr.SpatialReference()
    sr.ImportFromEPSG(4326) #My projection system
    
    # Enter the GCPs
    #   Format: [map x-coordinate(longitude)], [map y-coordinate (latitude)], [elevation],
    #   [image column index(x)], [image row index (y)]
    
    
    gcps = [gdal.GCP(34.9013, 38.6499, 0, 0, 0),
            gdal.GCP(34.9363, 38.6499, 0, 6539, 0),
            gdal.GCP(34.9013, 38.6226, 0, 0, 6539),
            gdal.GCP(34.9363, 38.6226, 0, 6539, 6539)]
    
    
    
    # Apply the GCPs to the open output file:
    ds.SetGCPs(gcps, sr.ExportToWkt())
    
    # Close the output file in order to be able to work with it in other programs:
    ds = None
    
