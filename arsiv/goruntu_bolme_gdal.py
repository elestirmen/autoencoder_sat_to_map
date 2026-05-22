import os, gdal




path = "bern_google_maps_cropped.tif"

out_path = 'test/'
output_filename = 'tile_'

tile_size_x = 560
tile_size_y = 560

ds = gdal.Open(path)
band = ds.GetRasterBand(1)
xsize = band.XSize
ysize = band.YSize

for i in range(0, xsize, tile_size_x):
    for j in range(0, ysize, tile_size_y):
        com_string = "gdal_translate -of GTIFF -srcwin " + str(i)+ ", " + str(j) + ", " + str(tile_size_x) + ", " + str(tile_size_y) + " " +  str(path) + " " + str(out_path) + str(output_filename) + str(i) + "_" + str(j) + ".tif"
        os.system(com_string)
        print(j)
        input("pause")
        j-=64
    i-=64
    
    
    
# cmd = 'gdal_retile.py -v -r bilinear -levels 1 -ps 560 560 -co "TILED=YES" -co "COMPRESS=JPEG" -targetDir C:/d_surucusu/out bern_google_maps_cropped.tif'
# os.system(cmd)

i=0
while i<xsize:
    j=0
    while j<ysize: 
        com_string = "gdal_translate -of GTIFF -srcwin " + str(i)+ ", " + str(j) + ", " + str(tile_size_x) + ", " + str(tile_size_y) + " " +  str(path) + " " + str(out_path) + str(output_filename) + str(i) + "_" + str(j) + ".tif"
        os.system(com_string)
        j+=560        
        j-=64
    i+=560
    i-=64
    