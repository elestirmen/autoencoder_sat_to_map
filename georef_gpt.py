import os
import rasterio
from osgeo import gdal


def get_geo_transform_and_pixel_sizes(dataset):
    """
    Verilen veri kümesinden coğrafi dönüşüm ve piksel boyutlarını alır.

    :param dataset: GDAL dataset
    :return: tuple (GeoTransform, pixel_size_x, pixel_size_y)
    """
    gt = dataset.GetGeoTransform()
    pixel_size_x = gt[1]
    pixel_size_y = -gt[5]
    return gt, pixel_size_x, pixel_size_y


def main():
    # Ana haritaların bulunduğu dizin
    data_dir_haritalar = "ana_haritalar"
    haritalar_list = os.listdir(data_dir_haritalar)

    # Ana haritalar dizinindeki tüm haritalar üzerinde işlem yap
    for harita in haritalar_list:
        dataset_path = os.path.join(data_dir_haritalar, harita)
        raster = rasterio.open(dataset_path)

        # Referanslı tiff dosyasını aç
        #georasterref = rasterio.open("ana_harita_karlik_30_cm_bingmap_Georeferans.tif")
        #georasterref = rasterio.open("urgup_gmap_30_cm_georeferans.tif")
        #georasterref = rasterio.open("ana_harita_urgup_30_cm__Georefference.tif")
        #georasterref = rasterio.open("urgup_genel_genis_kendi_uretimim_georefference.tif")
        georasterref = rasterio.open("ana_harita_urgup_30_cm__Georefference_utm.tif")
        #georasterref = rasterio.open("karlik_30_cm_bingmap_utm_georefference.tif")

        # Raster dosyasını oku
        dosya = raster.read(1)

        # Meta verileri güncelle
        out_meta = georasterref.meta.copy()
        out_meta.update({'driver': 'GTiff',
                         'width': georasterref.shape[1],
                         'height': georasterref.shape[0],
                         'count': 1,
                         'dtype': 'uint8',
                         'crs': georasterref.crs,
                         'transform': georasterref.transform,
                         'compress': 'LZW',
                         'nodata': 0})

        # Çıktı dosyasının yolunu oluştur
        output_path = os.path.join("georefli", "harita_temp", f"{harita}_geo.tif")

        # Yeni raster dosyasını yaz
        with rasterio.open(output_path, 'w', **out_meta) as dst:
            dst.write_band(1, dosya.astype(rasterio.uint8))

    # Georeferanslı haritaların bulunduğu dizin
    data_dir_haritalar = "georefli/harita_temp"
    haritalar_list = os.listdir(data_dir_haritalar)

    # Georeferanslı haritalar dizinindeki tüm haritalar üzerinde işlem yap
    for harita in haritalar_list:
        dataset_path = os.path.join(data_dir_haritalar, harita)
        raster = gdal.Open(dataset_path)

        # Coğrafi dönüşüm ve piksel boyutlarını al
        gt, pixel_size_x, pixel_size_y = get_geo_transform_and_pixel_sizes(raster)
        print(gt)
        print("x = ", pixel_size_x)
        print("y = ", pixel_size_y)

        # Çıktı dosyasının yolunu oluştur
        output_path = os.path.join("georefli/harita", f"{harita}_UTM_geo_r.tif")

        # GDAL Translate seçeneklerini belirle
        translate_options = gdal.TranslateOptions(format='GTiff',
                                                  creationOptions=['TFW=NO', 'COMPRESS=jpeg'])
        
        # translate_options = gdal.TranslateOptions(format='GTiff',
        #                                           creationOptions=['TFW=NO', 'COMPRESS=DEFLATE'])

        # Haritayı yeni formatta kaydet
        out = gdal.Translate(output_path, raster, options=translate_options)


if __name__ == "__main__":
    main()
