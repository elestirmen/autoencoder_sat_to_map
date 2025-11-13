"""
Görüntü İşlemleri Modülü
=========================

Bu modül görüntü bölme, birleştirme ve jeoreferanslama işlemlerini içerir.
Tüm fonksiyonlar robust hata kontrolü ve esnek parametrelerle donatılmıştır.

Kullanım:
    python goruntu_islemleri.py split --input image.tif --output_dir parcalar
    python goruntu_islemleri.py merge --input_dir parcalar --output merged.jpg
    python goruntu_islemleri.py georef --input image.jpg --reference ref.tif --output geo.tif
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import re

# Progress bar için tqdm
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # Basit progress bar fallback
    class tqdm:
        def __init__(self, iterable=None, total=None, desc=None, unit=None, ncols=None):
            self.iterable = iterable
            self.total = total
            self.desc = desc
            self.unit = unit
            self.n = 0
        
        def __iter__(self):
            if self.desc:
                print(f"{self.desc}...", end="", flush=True)
            return iter(self.iterable) if self.iterable else range(self.total)
        
        def __enter__(self):
            return self
        
        def __exit__(self, *args):
            if self.desc:
                print(" ✓")
        
        def update(self, n=1):
            self.n += n
            if self.desc and self.total:
                print(f"\r{self.desc}... {self.n}/{self.total}", end="", flush=True)
        
        def set_description(self, desc):
            self.desc = desc

# OpenCV için maksimum görüntü piksel limitini ayarla
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2, 40))

import cv2
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal
from natsort import natsorted
import rasterio
from rasterio.transform import from_bounds

# Logging yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# TensorFlow/Keras için (opsiyonel - sadece model varsa yüklenecek)
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.image import img_to_array, load_img
    from matplotlib import pyplot, cm
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("TensorFlow bulunamadı. Model inference özelliği kullanılamayacak.")


class ImageProcessor:
    """Görüntü işleme sınıfı - bölme, birleştirme ve jeoreferanslama işlemleri."""
    
    def __init__(self, reference_dir: Optional[str] = None):
        """
        ImageProcessor sınıfını başlatır.
        
        Args:
            reference_dir: Referans raster dosyalarının bulunduğu dizin
        """
        self.supported_formats = ['.tif', '.tiff', '.jpg', '.jpeg', '.png']
        self.reference_dir = reference_dir or "."
    
    def find_reference_raster(self, image_filename: str, reference_dir: Optional[str] = None) -> Optional[str]:
        """
        Görüntü dosya adına göre uygun referans raster'ı bulur.
        
        Args:
            image_filename: Görüntü dosya adı (örn: "karlik_30_cm_bingmap_utm.tif")
            reference_dir: Referans dosyalarının bulunduğu dizin (varsayılan: "georeferans_sample")
            
        Returns:
            Bulunan referans raster yolu veya None
        """
        if reference_dir is None:
            # Varsayılan olarak georeferans_sample klasörünü kullan
            reference_dir = "georeferans_sample"
            # Eğer yoksa mevcut dizini dene
            if not os.path.exists(reference_dir):
                reference_dir = self.reference_dir
        
        if not os.path.exists(reference_dir):
            logger.warning(f"Referans dizini bulunamadı: {reference_dir}")
            return None
        
        # Görüntü adından anahtar kelimeleri çıkar
        base_name = os.path.splitext(os.path.basename(image_filename))[0].lower()
        
        # Anahtar kelimeleri bul (urgup, karlik, vb.)
        keywords = []
        common_names = ['urgup', 'karlik', 'kapadokya', 'bern']
        for name in common_names:
            if name in base_name:
                keywords.append(name)
        
        # Referans dosyalarını ara
        reference_files = []
        if os.path.isdir(reference_dir):
            for file in os.listdir(reference_dir):
                if file.lower().endswith(('.tif', '.tiff')):
                    reference_files.append(file)
        else:
            # Tek dosya olabilir
            if os.path.exists(reference_dir) and reference_dir.lower().endswith(('.tif', '.tiff')):
                return reference_dir
        
        if len(reference_files) == 0:
            logger.warning(f"Referans dizininde dosya bulunamadı: {reference_dir}")
            return None
        
        logger.info(f"Referans dizininde {len(reference_files)} dosya bulundu: {reference_dir}")
        
        # Anahtar kelimelere göre eşleşme bul
        best_match = None
        best_score = 0
        
        for ref_file in reference_files:
            ref_lower = ref_file.lower()
            score = 0
            
            # Anahtar kelimeler için puan ver (daha yüksek öncelik)
            for keyword in keywords:
                if keyword in ref_lower:
                    score += 20  # Daha yüksek puan
            
            # "ana_harita" ile başlayan dosyalar için ekstra puan (standart format)
            if ref_lower.startswith('ana_harita'):
                score += 10
            
            # "georef", "georeference", "reference" gibi kelimeler için ekstra puan
            if any(word in ref_lower for word in ['georef', 'reference']):
                score += 5
            
            # "utm" kelimesi için ekstra puan
            if 'utm' in ref_lower:
                score += 3
            
            if score > best_score:
                best_score = score
                best_match = ref_file
        
        if best_match:
            ref_path = os.path.join(reference_dir, best_match)
            logger.info(f"✓ Referans raster bulundu: {best_match} (eşleşme puanı: {best_score})")
            logger.info(f"  Görüntü: {os.path.basename(image_filename)}")
            logger.info(f"  Referans: {best_match}")
            return ref_path
        
        # Eşleşme yoksa ilk referans dosyasını kullan
        if len(reference_files) > 0:
            ref_path = os.path.join(reference_dir, reference_files[0])
            logger.warning(f"Tam eşleşme bulunamadı, varsayılan referans kullanılıyor: {reference_files[0]}")
            return ref_path
        
        return None
    
    def load_image(self, path: str) -> np.ndarray:
        """
        Görüntüyü yükler ve kontrol eder.
        
        Args:
            path: Görüntü dosyasının yolu
            
        Returns:
            Görüntü array'i (numpy.ndarray)
            
        Raises:
            ValueError: Görüntü yüklenemezse
            FileNotFoundError: Dosya bulunamazsa
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dosya bulunamadı: {path}")
        
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Görüntü yüklenemedi: {path}. Desteklenen formatlar: {self.supported_formats}")
        
        logger.info(f"Görüntü yüklendi: {path}, Boyut: {img.shape}")
        return img
    
    def get_geotransform(self, path: str) -> Tuple[Any, float, float]:
        """
        Coğrafi transformasyon bilgilerini alır.
        
        Args:
            path: Raster dosyasının yolu
            
        Returns:
            Tuple (GeoTransform, pixelSizeX, pixelSizeY)
            
        Raises:
            ValueError: Raster açılamazsa
        """
        raster = gdal.Open(path)
        if raster is None:
            raise ValueError(f"Raster açılamadı: {path}")
        
        gt = raster.GetGeoTransform()
        pixelSizeX = gt[1]
        pixelSizeY = abs(gt[5])  # Negatif değeri pozitif yap
        
        logger.info(f"GeoTransform: {gt}")
        logger.info(f"Pixel Size X: {pixelSizeX}, Y: {pixelSizeY}")
        
        return gt, pixelSizeX, pixelSizeY
    
    def create_output_directory(self, output_path: str) -> None:
        """
        Çıktı dizinini oluşturur.
        
        Args:
            output_path: Oluşturulacak dizin yolu
        """
        Path(output_path).mkdir(parents=True, exist_ok=True)
        logger.debug(f"Çıktı dizini oluşturuldu/kontrol edildi: {output_path}")
    
    def split_image(
        self,
        img: np.ndarray,
        frame_size: int = 512,
        overlap: int = 32,
        output_dir: str = 'bolunmus/bolunmus',
        prefix: str = 'goruntu',
        format: str = 'jpg',
        save_metadata: bool = False,
        original_path: Optional[str] = None,
        show_progress: bool = True
    ) -> Tuple[List[np.ndarray], List[str], Dict[str, Any]]:
        """
        Görüntüyü küçük parçalara böler.
        
        Args:
            img: Görüntü array'i
            frame_size: Her parçanın boyutu (piksel)
            overlap: Parçalar arası örtüşme (piksel)
            output_dir: Çıktı dizini
            prefix: Dosya adı öneki
            format: Kayıt formatı ('jpg', 'png', 'tif')
            save_metadata: Metadata kaydedilsin mi (jeoreferans için)
            original_path: Orijinal görüntü yolu (metadata için)
            
        Returns:
            Tuple (parçalar listesi, dosya isimleri listesi, metadata dict)
            
        Raises:
            ValueError: Geçersiz parametreler verilirse
        """
        if frame_size <= 0:
            raise ValueError("frame_size 0'dan büyük olmalıdır")
        if overlap < 0:
            raise ValueError("overlap negatif olamaz")
        if overlap >= frame_size:
            raise ValueError("overlap frame_size'dan küçük olmalıdır")
        
        height, width = img.shape[:2]
        
        # Çıktı dizinini oluştur
        self.create_output_directory(output_dir)
        
        # Kaç parça oluşturulacağını hesapla
        num_frames_x = int(height / frame_size)
        num_frames_y = int(width / frame_size)
        
        logger.info(f"Görüntü boyutu: {height}x{width}")
        logger.info(f"Parça sayısı: {num_frames_x}x{num_frames_y} = {num_frames_x * num_frames_y}")
        logger.info(f"Parça boyutu: {frame_size}x{frame_size}, Örtüşme: {overlap}px")
        
        img_cropped = []
        filenames = []
        metadata = {
            'num_frames_x': num_frames_x,
            'num_frames_y': num_frames_y,
            'frame_size': frame_size,
            'overlap': overlap,
            'original_size': (height, width),
            'original_path': original_path
        }
        
        # GeoTransform bilgilerini al (eğer varsa)
        geotransform = None
        if original_path and save_metadata:
            try:
                geotransform, _, _ = self.get_geotransform(original_path)
                metadata['geotransform'] = geotransform
            except Exception as e:
                logger.warning(f"GeoTransform bilgisi alınamadı: {e}")
        
        total_pieces = num_frames_x * num_frames_y
        
        # Progress bar ile işle
        if show_progress:
            pbar = tqdm(total=total_pieces, desc="Parçalar bölünüyor", unit="parça", ncols=100)
        
        for i in range(num_frames_x):
            for j in range(num_frames_y):
                # Başlangıç ve bitiş koordinatlarını hesapla
                start_y = frame_size * i
                end_y = min(frame_size * (i + 1) + overlap, height)
                start_x = frame_size * j
                end_x = min(frame_size * (j + 1) + overlap, width)
                
                # Parçayı kes
                crop = img[start_y:end_y, start_x:end_x]
                
                # Dosya adını oluştur
                filename = os.path.join(output_dir, f'{prefix}_{i}_{j}.{format}')
                
                # Kaydet
                if format.lower() in ['tif', 'tiff']:
                    cv2.imwrite(filename, crop)
                else:
                    cv2.imwrite(filename, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                
                img_cropped.append(crop)
                filenames.append(filename)
                
                if show_progress:
                    pbar.update(1)
                    pbar.set_description(f"Parça [{i+1}/{num_frames_x}, {j+1}/{num_frames_y}]")
        
        if show_progress:
            pbar.close()
        
        logger.info(f"Toplam {len(filenames)} parça oluşturuldu.")
        
        return img_cropped, filenames, metadata
    
    def merge_images(
        self,
        input_dir: str,
        output_path: str,
        num_frames_x: Optional[int] = None,
        num_frames_y: Optional[int] = None,
        crop_overlap: int = 0,
        frame_size: Optional[int] = None,
        sort_files: bool = True
    ) -> np.ndarray:
        """
        Parçalara bölünmüş görüntüleri birleştirir.
        
        Args:
            input_dir: Parçaların bulunduğu dizin
            output_path: Birleştirilmiş görüntünün kaydedileceği yol
            num_frames_x: X eksenindeki parça sayısı (None ise otomatik hesaplanır)
            num_frames_y: Y eksenindeki parça sayısı (None ise otomatik hesaplanır)
            crop_overlap: Örtüşme miktarını kesmek için piksel sayısı
            frame_size: Her parçanın boyutu (None ise otomatik algılanır)
            sort_files: Dosyaları sırala mı (natsort kullanarak)
            
        Returns:
            Birleştirilmiş görüntü array'i
            
        Raises:
            ValueError: Geçersiz parametreler veya dosya bulunamazsa
        """
        if not os.path.exists(input_dir):
            raise FileNotFoundError(f"Dizin bulunamadı: {input_dir}")
        
        # Dosyaları listele ve sırala
        files = [f for f in os.listdir(input_dir) 
                 if any(f.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff'])]
        
        if len(files) == 0:
            raise ValueError(f"Dizinde görüntü dosyası bulunamadı: {input_dir}")
        
        if sort_files:
            files = natsorted(files)
        
        logger.info(f"{len(files)} dosya bulundu: {input_dir}")
        
        # İlk görüntüyü yükle ve boyutları al
        first_img_path = os.path.join(input_dir, files[0])
        first_img = cv2.imread(first_img_path)
        if first_img is None:
            raise ValueError(f"Görüntü yüklenemedi: {first_img_path}")
        
        img_height, img_width = first_img.shape[:2]
        
        # Frame size'ı belirle
        if frame_size is None:
            # Dosya adından çıkarmaya çalış (örn: goruntu_0_0.jpg)
            # Veya ilk görüntünün boyutunu kullan
            frame_size = min(img_height, img_width)
            logger.info(f"Frame size otomatik algılandı: {frame_size}")
        
        # Parça sayılarını belirle
        if num_frames_x is None or num_frames_y is None:
            # Dosya sayısından kare kök alarak tahmin et
            total_files = len(files)
            sqrt_val = int(np.sqrt(total_files))
            
            if num_frames_x is None:
                num_frames_x = sqrt_val
            if num_frames_y is None:
                num_frames_y = sqrt_val
            
            # Eğer kare değilse, dosya adlarından çıkarmaya çalış
            if num_frames_x * num_frames_y != total_files:
                logger.warning(f"Parça sayısı ({num_frames_x}x{num_frames_y}={num_frames_x*num_frames_y}) "
                             f"dosya sayısına ({total_files}) eşit değil. Dosya adlarından çıkarılmaya çalışılıyor...")
                # Dosya adlarından maksimum indeksleri bul
                max_i, max_j = 0, 0
                for f in files:
                    try:
                        # goruntu_i_j.jpg formatından i ve j'yi çıkar
                        parts = f.split('_')
                        if len(parts) >= 3:
                            i = int(parts[-2])
                            j = int(parts[-1].split('.')[0])
                            max_i = max(max_i, i)
                            max_j = max(max_j, j)
                    except:
                        continue
                
                if max_i > 0 or max_j > 0:
                    num_frames_x = max_i + 1
                    num_frames_y = max_j + 1
                    logger.info(f"Dosya adlarından parça sayısı çıkarıldı: {num_frames_x}x{num_frames_y}")
        
        logger.info(f"Birleştirme parametreleri: {num_frames_x}x{num_frames_y}, crop_overlap={crop_overlap}")
        
        # Tüm görüntüleri yükle (progress bar ile)
        img_array = []
        pbar = tqdm(files, desc="Görüntüler yükleniyor", unit="dosya", ncols=100) if TQDM_AVAILABLE else files
        
        for img_file in pbar:
            img_path = os.path.join(input_dir, img_file)
            img = cv2.imread(img_path)
            if img is None:
                logger.warning(f"Görüntü yüklenemedi, atlanıyor: {img_path}")
                continue
            
            # Örtüşmeyi kes
            if crop_overlap > 0:
                h, w = img.shape[:2]
                img = img[crop_overlap:h-crop_overlap, crop_overlap:w-crop_overlap]
            
            img_array.append(img)
        
        if TQDM_AVAILABLE and isinstance(pbar, tqdm):
            pbar.close()
        
        if len(img_array) != num_frames_x * num_frames_y:
            logger.warning(f"Yüklenen görüntü sayısı ({len(img_array)}) "
                         f"beklenen sayıdan ({num_frames_x * num_frames_y}) farklı!")
        
        # Görüntüleri birleştir
        rows = []
        idx = 0
        
        for i in range(num_frames_x):
            if idx >= len(img_array):
                break
            
            # Yatay satır oluştur
            row = img_array[idx]
            idx += 1
            
            for j in range(1, num_frames_y):
                if idx >= len(img_array):
                    break
                row = np.hstack((row, img_array[idx]))
                idx += 1
            
            rows.append(row)
        
        # Satırları dikey olarak birleştir
        if len(rows) == 0:
            raise ValueError("Birleştirilecek görüntü bulunamadı!")
        
        merged_image = rows[0]
        for i in range(1, len(rows)):
            merged_image = np.vstack((merged_image, rows[i]))
        
        # Çıktı dizinini oluştur
        output_dir = os.path.dirname(output_path)
        if output_dir:
            self.create_output_directory(output_dir)
        
        # Kaydet
        if output_path.lower().endswith(('.tif', '.tiff')):
            cv2.imwrite(output_path, merged_image)
        else:
            cv2.imwrite(output_path, merged_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        logger.info(f"Birleştirilmiş görüntü kaydedildi: {output_path}, Boyut: {merged_image.shape}")
        
        return merged_image
    
    def georeference_image(
        self,
        input_path: str,
        reference_path: str,
        output_path: str,
        band: int = 1,
        compress: str = 'LZW',
        nodata: Optional[float] = None
    ) -> None:
        """
        Görüntüyü referans raster'ın coğrafi bilgileriyle jeoreferanslar.
        
        Args:
            input_path: Jeoreferanslanacak görüntü yolu
            reference_path: Referans GeoTIFF dosyası yolu
            output_path: Çıktı dosyası yolu
            band: Okunacak band numarası (1-indexed)
            compress: Sıkıştırma tipi ('LZW', 'DEFLATE', 'JPEG', 'NONE')
            nodata: NoData değeri
            
        Raises:
            FileNotFoundError: Dosya bulunamazsa
            ValueError: Raster açılamazsa
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Giriş dosyası bulunamadı: {input_path}")
        if not os.path.exists(reference_path):
            raise FileNotFoundError(f"Referans dosyası bulunamadı: {reference_path}")
        
        logger.info(f"Jeoreferanslama başlatılıyor: {input_path}")
        logger.info(f"Referans: {reference_path}")
        
        # Giriş raster'ı aç
        try:
            input_raster = rasterio.open(input_path)
        except Exception as e:
            raise ValueError(f"Giriş raster açılamadı: {e}")
        
        # Referans raster'ı aç
        try:
            reference_raster = rasterio.open(reference_path)
        except Exception as e:
            raise ValueError(f"Referans raster açılamadı: {e}")
        
        # Giriş görüntüsünü oku
        try:
            data = input_raster.read(band)
        except Exception as e:
            raise ValueError(f"Band {band} okunamadı: {e}")
        
        # Meta verileri referans raster'dan kopyala
        out_meta = reference_raster.meta.copy()
        out_meta.update({
            'driver': 'GTiff',
            'width': reference_raster.shape[1],
            'height': reference_raster.shape[0],
            'count': 1,
            'dtype': 'uint8',
            'crs': reference_raster.crs,
            'transform': reference_raster.transform,
            'compress': compress
        })
        
        if nodata is not None:
            out_meta['nodata'] = nodata
        
        # Çıktı dizinini oluştur
        output_dir = os.path.dirname(output_path)
        if output_dir:
            self.create_output_directory(output_dir)
        
        # Veriyi uint8'e dönüştür
        if data.dtype != np.uint8:
            # Normalize et ve uint8'e dönüştür
            data_min = data.min()
            data_max = data.max()
            if data_max > data_min:
                data = ((data - data_min) / (data_max - data_min) * 255).astype(np.uint8)
            else:
                data = data.astype(np.uint8)
        
        # Yeni raster dosyasını yaz
        with rasterio.open(output_path, 'w', **out_meta) as dst:
            dst.write_band(1, data)
        
        logger.info(f"Jeoreferanslı görüntü kaydedildi: {output_path}")
        
        # İkinci aşama: GDAL Translate ile optimize et
        if compress == 'JPEG' or compress == 'jpeg':
            output_optimized = output_path.replace('.tif', '_optimized.tif')
            logger.info(f"GDAL Translate ile optimize ediliyor: {output_optimized}")
            
            raster_gdal = gdal.Open(output_path)
            if raster_gdal:
                translate_options = gdal.TranslateOptions(
                    format='GTiff',
                    creationOptions=['TFW=NO', f'COMPRESS={compress.lower()}']
                )
                gdal.Translate(output_optimized, raster_gdal, options=translate_options)
                logger.info(f"Optimize edilmiş dosya kaydedildi: {output_optimized}")
        
        input_raster.close()
        reference_raster.close()
    
    def visualize_crops(
        self,
        img_cropped: List[np.ndarray],
        num_frames_x: int,
        num_frames_y: int,
        figsize: Tuple[int, int] = (10, 10),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Parçaları görselleştirir.
        
        Args:
            img_cropped: Parça görüntülerinin listesi
            num_frames_x: X eksenindeki parça sayısı
            num_frames_y: Y eksenindeki parça sayısı
            figsize: Figür boyutu
            save_path: Kaydetme yolu (None ise kaydetmez)
            
        Returns:
            Matplotlib figure objesi
        """
        fig, ax = plt.subplots(num_frames_x, num_frames_y, figsize=figsize)
        
        # Tek boyutlu array için düzeltme
        if num_frames_x == 1:
            ax = ax.reshape(1, -1)
        if num_frames_y == 1:
            ax = ax.reshape(-1, 1)
        
        idx = 0
        for i in range(num_frames_x):
            for j in range(num_frames_y):
                if idx < len(img_cropped):
                    # BGR'den RGB'ye dönüştür
                    img_rgb = cv2.cvtColor(img_cropped[idx], cv2.COLOR_BGR2RGB)
                    ax[i, j].imshow(img_rgb)
                    ax[i, j].axis('off')
                    idx += 1
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Görselleştirme kaydedildi: {save_path}")
        
        return fig
    
    def process_images_with_model(
        self,
        input_dir: str,
        output_dir: str,
        model_path: str,
        image_size: Tuple[int, int] = (544, 544),
        color_mode: str = "grayscale",
        crop_overlap: int = 16,
        use_threading: bool = True
    ) -> List[str]:
        """
        Parçalara bölünmüş görüntüleri sinir ağı modelinden geçirir.
        
        Args:
            input_dir: Giriş parçalarının bulunduğu dizin
            output_dir: İşlenmiş parçaların kaydedileceği dizin
            model_path: Model dosyasının yolu
            image_size: Görüntü boyutu (height, width)
            color_mode: Renk modu ("grayscale" veya "rgb")
            crop_overlap: Örtüşme kesme miktarı
            use_threading: Threading kullanılsın mı
            
        Returns:
            İşlenmiş dosya isimlerinin listesi
            
        Raises:
            ImportError: TensorFlow yüklü değilse
            FileNotFoundError: Model dosyası bulunamazsa
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow yüklü değil. Model inference kullanılamaz.")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")
        
        if not os.path.exists(input_dir):
            raise FileNotFoundError(f"Giriş dizini bulunamadı: {input_dir}")
        
        # Çıktı dizinini oluştur
        self.create_output_directory(output_dir)
        
        # Modeli yükle
        logger.info(f"Model yükleniyor: {model_path}")
        try:
            model = load_model(model_path)
        except Exception as e:
            # Custom loss fonksiyonları için tekrar dene
            try:
                def ssim_loss(y_true, y_pred):
                    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))
                model = load_model(model_path, custom_objects={'ssim_loss': ssim_loss})
            except:
                raise ValueError(f"Model yüklenemedi: {e}")
        
        logger.info("Model yüklendi.")
        
        # Dosyaları listele ve sırala
        files = [f for f in os.listdir(input_dir) 
                 if any(f.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff'])]
        files = natsorted(files)
        
        if len(files) == 0:
            raise ValueError(f"Dizinde görüntü dosyası bulunamadı: {input_dir}")
        
        logger.info(f"{len(files)} dosya bulundu, model inference başlatılıyor...")
        
        def load_image_for_model(filename: str) -> np.ndarray:
            """Model için görüntü yükleme fonksiyonu."""
            filepath = os.path.join(input_dir, filename)
            if color_mode == "grayscale":
                pixels = load_img(filepath, target_size=image_size, color_mode="grayscale")
                pixels = img_to_array(pixels)
                # Histogram eşitleme (opsiyonel)
                pixels = cv2.equalizeHist(pixels.astype(np.uint8))
                pixels = (pixels - 127.5) / 127.5
            else:
                pixels = load_img(filepath, target_size=image_size)
                pixels = img_to_array(pixels)
                pixels = (pixels - 127.5) / 127.5
            
            pixels = np.expand_dims(pixels, 0)
            return pixels
        
        def process_single_image(filename: str) -> Optional[str]:
            """Tek bir görüntüyü işle."""
            try:
                # Görüntüyü yükle
                src_image = load_image_for_model(filename)
                
                # Model inference
                gen_image = model.predict(src_image, verbose=0)
                
                # Çıktı dosya adı
                base_name = os.path.splitext(filename)[0]
                output_filename = os.path.join(output_dir, f'goruntu_{base_name}.jpg')
                
                # Görüntüyü kaydet
                if len(gen_image[0].shape) == 2:
                    # Grayscale
                    goruntu = gen_image[0].reshape(image_size[0], image_size[1])
                    pyplot.imsave(output_filename, goruntu, cmap=cm.gray)
                elif len(gen_image[0].shape) == 3:
                    # RGB
                    goruntu = gen_image[0]
                    # Normalize et [0, 1] -> [0, 255]
                    if goruntu.max() <= 1.0:
                        goruntu = (goruntu * 255).astype(np.uint8)
                    cv2.imwrite(output_filename, cv2.cvtColor(goruntu, cv2.COLOR_RGB2BGR))
                
                return output_filename
            except Exception as e:
                logger.error(f"Görüntü işlenirken hata ({filename}): {e}")
                return None
        
        # Threading ile işle (progress bar ile)
        if use_threading:
            with ThreadPoolExecutor() as executor:
                if TQDM_AVAILABLE:
                    results = list(tqdm(
                        executor.map(process_single_image, files),
                        total=len(files),
                        desc="Model inference",
                        unit="görüntü",
                        ncols=100
                    ))
                else:
                    results = list(executor.map(process_single_image, files))
        else:
            if TQDM_AVAILABLE:
                results = [process_single_image(f) for f in tqdm(files, desc="Model inference", unit="görüntü", ncols=100)]
            else:
                results = [process_single_image(f) for f in files]
        
        # Başarılı sonuçları filtrele
        output_files = [f for f in results if f is not None]
        
        logger.info(f"Toplam {len(output_files)}/{len(files)} görüntü başarıyla işlendi.")
        
        return output_files
    
    def run_full_pipeline(
        self,
        input_image: str,
        model_path: Optional[str] = None,
        model_dir: Optional[str] = None,
        split_frame_size: int = 512,
        split_overlap: int = 32,
        split_output_dir: str = "bolunmus/bolunmus",
        processed_output_dir: str = "parcalar",
        merge_output_dir: str = "ana_haritalar",
        reference_raster: Optional[str] = None,
        georef_output_dir: str = "georefli/harita",
        crop_overlap: int = 16,
        image_size: Tuple[int, int] = (544, 544),
        color_mode: str = "grayscale"
    ) -> Dict[str, Any]:
        """
        Tüm işlemleri sırayla çalıştırır: Böl -> Model Inference -> Birleştir -> Jeoreferansla
        
        Args:
            input_image: Giriş görüntü dosyası
            model_path: Tek model dosyası yolu (veya model_dir kullanılır)
            model_dir: Model dosyalarının bulunduğu dizin (tüm modeller işlenir)
            split_frame_size: Bölme işlemi için frame boyutu
            split_overlap: Bölme işlemi için örtüşme
            split_output_dir: Bölünmüş parçaların dizini
            processed_output_dir: Model'den geçmiş parçaların dizini
            merge_output_dir: Birleştirilmiş görüntülerin dizini
            reference_raster: Jeoreferans için referans raster
            georef_output_dir: Jeoreferanslı görüntülerin dizini
            crop_overlap: Birleştirme sırasında kesilecek örtüşme
            image_size: Model için görüntü boyutu
            color_mode: Model için renk modu
            
        Returns:
            İşlem sonuçlarını içeren dict
        """
        results = {
            'split': None,
            'inference': [],
            'merge': [],
            'georef': []
        }
        
        try:
            # Görüntü adından klasör adını oluştur
            base_name = os.path.splitext(os.path.basename(input_image))[0]
            image_folder_name = base_name
            
            # Bölünmüş parçalar için klasör yolu (görüntü adıyla)
            split_output_dir_with_name = os.path.join(split_output_dir, image_folder_name)
            
            # Metadata dosyası yolu
            metadata_path = os.path.join(split_output_dir_with_name, 'metadata.json')
            
            # 1. BÖLME İŞLEMİ (eğer klasör yoksa veya boşsa)
            skip_split = False
            if os.path.exists(split_output_dir_with_name):
                # Klasör var, içinde dosya var mı kontrol et
                files_in_dir = [f for f in os.listdir(split_output_dir_with_name) 
                               if any(f.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff'])]
                
                if len(files_in_dir) > 0:
                    # Metadata varsa yükle
                    if os.path.exists(metadata_path):
                        try:
                            import json
                            with open(metadata_path, 'r') as f:
                                metadata = json.load(f)
                            logger.info("=" * 60)
                            logger.info("1. ADIM: Görüntü Bölme (ATLANDI - Klasör zaten mevcut)")
                            logger.info("=" * 60)
                            logger.info(f"✓ Mevcut klasör bulundu: {split_output_dir_with_name}")
                            logger.info(f"✓ {len(files_in_dir)} parça mevcut, bölme işlemi atlanıyor...")
                            skip_split = True
                            
                            results['split'] = {
                                'num_pieces': len(files_in_dir),
                                'metadata': metadata,
                                'output_dir': split_output_dir_with_name,
                                'skipped': True
                            }
                        except Exception as e:
                            logger.warning(f"Metadata yüklenemedi, bölme işlemi tekrar yapılacak: {e}")
                    else:
                        logger.info("=" * 60)
                        logger.info("1. ADIM: Görüntü Bölme (ATLANDI - Klasör mevcut ama metadata yok)")
                        logger.info("=" * 60)
                        logger.info(f"✓ Mevcut klasör bulundu: {split_output_dir_with_name}")
                        logger.info(f"✓ {len(files_in_dir)} parça mevcut, bölme işlemi atlanıyor...")
                        skip_split = True
                        
                        # Metadata olmadan devam et, parça sayılarını tahmin et
                        # Dosya adlarından çıkarmaya çalış (goruntu_i_j.jpg formatından)
                        max_i, max_j = 0, 0
                        for f in files_in_dir:
                            try:
                                parts = f.split('_')
                                if len(parts) >= 3:
                                    i = int(parts[-2])
                                    j = int(parts[-1].split('.')[0])
                                    max_i = max(max_i, i)
                                    max_j = max(max_j, j)
                            except:
                                continue
                        
                        if max_i > 0 or max_j > 0:
                            num_frames_x = max_i + 1
                            num_frames_y = max_j + 1
                        else:
                            # Kare kök tahmini
                            sqrt_val = int(np.sqrt(len(files_in_dir)))
                            num_frames_x = sqrt_val
                            num_frames_y = sqrt_val
                        
                        metadata = {
                            'num_frames_x': num_frames_x,
                            'num_frames_y': num_frames_y,
                            'frame_size': split_frame_size,
                            'overlap': split_overlap,
                            'original_size': None,
                            'original_path': input_image
                        }
                        
                        results['split'] = {
                            'num_pieces': len(files_in_dir),
                            'metadata': metadata,
                            'output_dir': split_output_dir_with_name,
                            'skipped': True
                        }
            
            if not skip_split:
                logger.info("=" * 60)
                logger.info("1. ADIM: Görüntü Bölme")
                logger.info("=" * 60)
                
                img = self.load_image(input_image)
                
                # Coğrafi bilgileri al (eğer mümkünse)
                try:
                    gt, px, py = self.get_geotransform(input_image)
                except:
                    logger.warning("Coğrafi bilgiler alınamadı, devam ediliyor...")
                
                img_cropped, filenames, metadata = self.split_image(
                    img,
                    frame_size=split_frame_size,
                    overlap=split_overlap,
                    output_dir=split_output_dir_with_name,  # Görüntü adıyla klasör
                    prefix='goruntu',
                    format='jpg',
                    save_metadata=True,
                    original_path=input_image,
                    show_progress=True
                )
                
                results['split'] = {
                    'num_pieces': len(filenames),
                    'metadata': metadata,
                    'output_dir': split_output_dir_with_name,
                    'skipped': False
                }
                
                logger.info(f"✓ Bölme tamamlandı: {len(filenames)} parça")
            
            # Metadata'yı kullan (skip edilmişse bile)
            metadata = results['split']['metadata']
            
            # 2. MODEL INFERENCE (eğer model belirtilmişse)
            if model_path or model_dir:
                logger.info("=" * 60)
                logger.info("2. ADIM: Model Inference")
                logger.info("=" * 60)
                
                if model_dir and os.path.exists(model_dir):
                    # Tüm modelleri işle
                    model_files = [f for f in os.listdir(model_dir) 
                                  if f.endswith(('.h5', '.keras', '.pb'))]
                    
                    if len(model_files) == 0:
                        logger.warning(f"Model dizininde model dosyası bulunamadı: {model_dir}")
                    else:
                        logger.info(f"{len(model_files)} model bulundu, işleniyor...")
                        
                        for model_file in model_files:
                            model_path_full = os.path.join(model_dir, model_file)
                            model_name = os.path.splitext(model_file)[0]
                            
                            logger.info(f"\nModel işleniyor: {model_name}")
                            
                            try:
                                # Her model için ayrı çıktı dizini (görüntü adıyla)
                                model_output_dir = os.path.join(processed_output_dir, image_folder_name, model_name)
                                
                                processed_files = self.process_images_with_model(
                                    input_dir=split_output_dir_with_name,  # Görüntü adıyla klasör
                                    output_dir=model_output_dir,
                                    model_path=model_path_full,
                                    image_size=image_size,
                                    color_mode=color_mode,
                                    crop_overlap=crop_overlap
                                )
                                
                                results['inference'].append({
                                    'model': model_name,
                                    'output_dir': model_output_dir,
                                    'num_processed': len(processed_files)
                                })
                                
                                logger.info(f"✓ Model {model_name} tamamlandı")
                                
                            except Exception as e:
                                logger.error(f"✗ Model {model_name} işlenirken hata: {e}")
                                continue
                
                elif model_path and os.path.exists(model_path):
                    # Tek model işle
                    logger.info(f"Model işleniyor: {model_path}")
                    
                    model_name = os.path.splitext(os.path.basename(model_path))[0]
                    model_output_dir = os.path.join(processed_output_dir, image_folder_name, model_name)
                    
                    processed_files = self.process_images_with_model(
                        input_dir=split_output_dir_with_name,  # Görüntü adıyla klasör
                        output_dir=model_output_dir,
                        model_path=model_path,
                        image_size=image_size,
                        color_mode=color_mode,
                        crop_overlap=crop_overlap
                    )
                    
                    results['inference'].append({
                        'model': model_name,
                        'output_dir': model_output_dir,
                        'num_processed': len(processed_files)
                    })
                    
                    logger.info(f"✓ Model inference tamamlandı")
                
                else:
                    logger.warning("Model dosyası/dizini bulunamadı, inference atlanıyor...")
                    # Inference olmadan direkt birleştirme yap
                    processed_output_dir = split_output_dir_with_name
            
            else:
                logger.info("Model belirtilmedi, inference atlanıyor...")
                processed_output_dir = split_output_dir_with_name
            
            # 3. BİRLEŞTİRME İŞLEMİ
            logger.info("=" * 60)
            logger.info("3. ADIM: Görüntü Birleştirme")
            logger.info("=" * 60)
            
            # Inference yapıldıysa her model için ayrı birleştirme yap
            if results['inference']:
                for inference_result in results['inference']:
                    model_name = inference_result['model']
                    model_output_dir = inference_result['output_dir']
                    
                    logger.info(f"\nBirleştirme yapılıyor: {model_name}")
                    
                    # Parça sayılarını metadata'dan al
                    num_frames_x = metadata['num_frames_x']
                    num_frames_y = metadata['num_frames_y']
                    
                    # Çıktı dosya adı
                    base_name = os.path.splitext(os.path.basename(input_image))[0]
                    merge_output_file = os.path.join(merge_output_dir, f"ana_harita_{base_name}_{model_name}.jpg")
                    
                    merged = self.merge_images(
                        input_dir=model_output_dir,
                        output_path=merge_output_file,
                        num_frames_x=num_frames_x,
                        num_frames_y=num_frames_y,
                        crop_overlap=crop_overlap,
                        frame_size=split_frame_size
                    )
                    
                    results['merge'].append({
                        'model': model_name,
                        'output_file': merge_output_file
                    })
                    
                    logger.info(f"✓ Birleştirme tamamlandı: {merge_output_file}")
            else:
                # Inference yapılmadıysa sadece bölünmüş parçaları birleştir
                logger.info("Bölünmüş parçalar birleştiriliyor...")
                
                num_frames_x = metadata['num_frames_x']
                num_frames_y = metadata['num_frames_y']
                
                base_name = os.path.splitext(os.path.basename(input_image))[0]
                merge_output_file = os.path.join(merge_output_dir, f"ana_harita_{base_name}.jpg")
                
                merged = self.merge_images(
                    input_dir=split_output_dir_with_name,  # Görüntü adıyla klasör
                    output_path=merge_output_file,
                    num_frames_x=num_frames_x,
                    num_frames_y=num_frames_y,
                    crop_overlap=crop_overlap,
                    frame_size=split_frame_size
                )
                
                results['merge'].append({
                    'model': None,
                    'output_file': merge_output_file
                })
                
                logger.info(f"✓ Birleştirme tamamlandı: {merge_output_file}")
            
            # 4. JEOREFERANSLAMA
            logger.info("=" * 60)
            logger.info("4. ADIM: Jeoreferanslama")
            logger.info("=" * 60)
            
            # Referans raster'ı bul (görüntü adına göre)
            selected_reference = None
            if reference_raster and os.path.exists(reference_raster):
                # Manuel olarak belirtilmiş referans kullan
                selected_reference = reference_raster
                logger.info(f"Manuel referans kullanılıyor: {os.path.basename(selected_reference)}")
            else:
                # Görüntü adına göre referans bul (georeferans_sample klasöründen)
                selected_reference = self.find_reference_raster(input_image, reference_dir="georeferans_sample")
                
                if not selected_reference:
                    logger.warning("Referans raster bulunamadı, jeoreferanslama atlanıyor...")
                    logger.info("İpucu: Referans raster dosyalarını 'georeferans_sample' klasörüne koyun veya reference_raster parametresi ile belirtin.")
                else:
                    logger.info(f"✓ Otomatik referans bulundu: {os.path.basename(selected_reference)}")
            
            if selected_reference and os.path.exists(selected_reference):
                # Birleştirilmiş görüntüleri jeoreferansla
                merge_files = [r['output_file'] for r in results['merge']]
                
                if TQDM_AVAILABLE:
                    pbar = tqdm(merge_files, desc="Jeoreferanslama", unit="dosya", ncols=100)
                else:
                    pbar = merge_files
                
                for merge_file in pbar:
                    if not os.path.exists(merge_file):
                        logger.warning(f"Birleştirilmiş dosya bulunamadı: {merge_file}")
                        continue
                    
                    if TQDM_AVAILABLE:
                        pbar.set_description(f"Jeoreferanslama: {os.path.basename(merge_file)}")
                    
                    # Çıktı dosya adı
                    base_name = os.path.splitext(os.path.basename(merge_file))[0]
                    georef_output_file = os.path.join(georef_output_dir, f"{base_name}_geo.tif")
                    
                    try:
                        self.georeference_image(
                            input_path=merge_file,
                            reference_path=selected_reference,
                            output_path=georef_output_file,
                            band=1,
                            compress='LZW'
                        )
                        
                        results['georef'].append({
                            'input': merge_file,
                            'output': georef_output_file,
                            'reference': selected_reference
                        })
                        
                        if not TQDM_AVAILABLE:
                            logger.info(f"✓ Jeoreferanslama tamamlandı: {os.path.basename(georef_output_file)}")
                    except Exception as e:
                        logger.error(f"✗ Jeoreferanslama hatası ({merge_file}): {e}")
                        continue
                
                if TQDM_AVAILABLE:
                    pbar.close()
            else:
                logger.info("Referans raster bulunamadı, jeoreferanslama atlanıyor...")
            
            logger.info("=" * 60)
            logger.info("TÜM İŞLEMLER TAMAMLANDI!")
            logger.info("=" * 60)
            
            return results
            
        except Exception as e:
            logger.error(f"Pipeline hatası: {e}", exc_info=True)
            raise


def main():
    """Ana fonksiyon - CLI arayüzü ve varsayılan değerlerle çalışma."""
    
    # Varsayılan dosya isimleri ve parametreler
    DEFAULT_INPUT_IMAGE = "urgup_bingmap_30cm_utm.tif"
    DEFAULT_OUTPUT_DIR = "bolunmus/bolunmus"
    DEFAULT_MERGE_INPUT_DIR = "parcalar"
    DEFAULT_MERGE_OUTPUT = "birlestirilmis.jpg"
    DEFAULT_GEOREF_INPUT_DIR = "ana_haritalar"
    DEFAULT_GEOREF_REFERENCE = "ana_harita_urgup_30_cm__Georefference_utm.tif"
    DEFAULT_GEOREF_OUTPUT_DIR = "georefli/harita"
    
    parser = argparse.ArgumentParser(
        description='Görüntü bölme, birleştirme ve jeoreferanslama işlemleri',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Örnekler:
  # Görüntü bölme (varsayılan: {DEFAULT_INPUT_IMAGE})
  python goruntu_islemleri.py split
  python goruntu_islemleri.py split -i image.tif -o parcalar --frame_size 512 --overlap 32
  
  # Görüntü birleştirme (varsayılan: {DEFAULT_MERGE_INPUT_DIR} -> {DEFAULT_MERGE_OUTPUT})
  python goruntu_islemleri.py merge
  python goruntu_islemleri.py merge -i parcalar -o merged.jpg --num_frames_x 10 --num_frames_y 10
  
  # Jeoreferanslama (varsayılan: {DEFAULT_GEOREF_INPUT_DIR} dizinindeki tüm dosyalar)
  python goruntu_islemleri.py georef
  python goruntu_islemleri.py georef -i image.jpg -r reference.tif -o geo.tif
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='İşlem tipi')
    
    # Split komutu
    split_parser = subparsers.add_parser('split', help='Görüntüyü parçalara böl')
    split_parser.add_argument('-i', '--input', default=DEFAULT_INPUT_IMAGE, 
                             help=f'Giriş görüntü dosyası (varsayılan: {DEFAULT_INPUT_IMAGE})')
    split_parser.add_argument('-o', '--output_dir', default=DEFAULT_OUTPUT_DIR, 
                             help=f'Çıktı dizini (varsayılan: {DEFAULT_OUTPUT_DIR})')
    split_parser.add_argument('--frame_size', type=int, default=512, help='Parça boyutu (piksel)')
    split_parser.add_argument('--overlap', type=int, default=32, help='Örtüşme miktarı (piksel)')
    split_parser.add_argument('--prefix', default='goruntu', help='Dosya adı öneki')
    split_parser.add_argument('--format', default='jpg', choices=['jpg', 'png', 'tif'], help='Çıktı formatı')
    split_parser.add_argument('--save_metadata', action='store_true', help='Metadata kaydet')
    split_parser.add_argument('--visualize', action='store_true', help='Görselleştirme oluştur')
    
    # Merge komutu
    merge_parser = subparsers.add_parser('merge', help='Parçaları birleştir')
    merge_parser.add_argument('-i', '--input_dir', default=DEFAULT_MERGE_INPUT_DIR, 
                             help=f'Parçaların bulunduğu dizin (varsayılan: {DEFAULT_MERGE_INPUT_DIR})')
    merge_parser.add_argument('-o', '--output', default=DEFAULT_MERGE_OUTPUT, 
                             help=f'Çıktı dosyası (varsayılan: {DEFAULT_MERGE_OUTPUT})')
    merge_parser.add_argument('--num_frames_x', type=int, help='X eksenindeki parça sayısı')
    merge_parser.add_argument('--num_frames_y', type=int, help='Y eksenindeki parça sayısı')
    merge_parser.add_argument('--crop_overlap', type=int, default=0, help='Örtüşmeyi kesme miktarı')
    merge_parser.add_argument('--frame_size', type=int, help='Parça boyutu (otomatik algılanır)')
    
    # Georef komutu
    georef_parser = subparsers.add_parser('georef', help='Görüntüyü jeoreferansla')
    georef_parser.add_argument('-i', '--input', default=None, 
                              help=f'Giriş görüntü dosyası veya dizin (varsayılan: {DEFAULT_GEOREF_INPUT_DIR} dizinindeki tüm dosyalar)')
    georef_parser.add_argument('-r', '--reference', default=DEFAULT_GEOREF_REFERENCE, 
                             help=f'Referans GeoTIFF dosyası (varsayılan: {DEFAULT_GEOREF_REFERENCE})')
    georef_parser.add_argument('-o', '--output', default=None, 
                              help='Çıktı dosyası veya dizin (varsayılan: otomatik)')
    georef_parser.add_argument('--band', type=int, default=1, help='Okunacak band numarası')
    georef_parser.add_argument('--compress', default='LZW', choices=['LZW', 'DEFLATE', 'JPEG', 'NONE'], 
                              help='Sıkıştırma tipi')
    georef_parser.add_argument('--nodata', type=float, help='NoData değeri')
    
    args = parser.parse_args()
    
    processor = ImageProcessor()
    
    # Varsayılan değerleri tekrar tanımla (main fonksiyonu içinde erişilebilir olması için)
    DEFAULT_INPUT_IMAGE = "urgup_bingmap_30cm_utm.tif"
    DEFAULT_OUTPUT_DIR = "bolunmus/bolunmus"
    DEFAULT_MERGE_INPUT_DIR = "parcalar"
    DEFAULT_MERGE_OUTPUT = "birlestirilmis.jpg"
    DEFAULT_GEOREF_INPUT_DIR = "ana_haritalar"
    DEFAULT_GEOREF_REFERENCE = "ana_harita_urgup_30_cm__Georefference_utm.tif"
    DEFAULT_GEOREF_OUTPUT_DIR = "georefli/harita"
    
    # Eğer komut belirtilmemişse varsayılan olarak split yap
    if not args.command:
        args.command = 'split'
        # Split için gerekli argümanları varsayılan değerlerle ekle
        if not hasattr(args, 'input'):
            args.input = DEFAULT_INPUT_IMAGE
        if not hasattr(args, 'output_dir'):
            args.output_dir = DEFAULT_OUTPUT_DIR
        if not hasattr(args, 'frame_size'):
            args.frame_size = 512
        if not hasattr(args, 'overlap'):
            args.overlap = 32
        if not hasattr(args, 'prefix'):
            args.prefix = 'goruntu'
        if not hasattr(args, 'format'):
            args.format = 'jpg'
        if not hasattr(args, 'save_metadata'):
            args.save_metadata = False
        if not hasattr(args, 'visualize'):
            args.visualize = False
        logger.info("Komut belirtilmedi, varsayılan olarak 'split' işlemi yapılıyor...")
        logger.info(f"Varsayılan dosya: {args.input}")
    
    try:
        if args.command == 'split':
            # Görüntüyü yükle
            input_file = getattr(args, 'input', DEFAULT_INPUT_IMAGE)
            img = processor.load_image(input_file)
            
            # Coğrafi bilgileri al (eğer mümkünse)
            try:
                gt, px, py = processor.get_geotransform(input_file)
            except:
                logger.warning("Coğrafi bilgiler alınamadı, devam ediliyor...")
            
            # Böl
            img_cropped, filenames, metadata = processor.split_image(
                img,
                frame_size=getattr(args, 'frame_size', 512),
                overlap=getattr(args, 'overlap', 32),
                output_dir=getattr(args, 'output_dir', DEFAULT_OUTPUT_DIR),
                prefix=getattr(args, 'prefix', 'goruntu'),
                format=getattr(args, 'format', 'jpg'),
                save_metadata=getattr(args, 'save_metadata', False),
                original_path=input_file
            )
            
            # Metadata kaydet
            if getattr(args, 'save_metadata', False):
                import json
                output_dir = getattr(args, 'output_dir', DEFAULT_OUTPUT_DIR)
                metadata_path = os.path.join(output_dir, 'metadata.json')
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)
                logger.info(f"Metadata kaydedildi: {metadata_path}")
            
            # Görselleştir
            if getattr(args, 'visualize', False):
                num_frames_x = metadata['num_frames_x']
                num_frames_y = metadata['num_frames_y']
                fig = processor.visualize_crops(img_cropped, num_frames_x, num_frames_y)
                plt.show()
            
            logger.info("Bölme işlemi tamamlandı!")
        
        elif args.command == 'merge':
            # Birleştir
            merged = processor.merge_images(
                input_dir=args.input_dir,
                output_path=args.output,
                num_frames_x=args.num_frames_x,
                num_frames_y=args.num_frames_y,
                crop_overlap=args.crop_overlap,
                frame_size=args.frame_size
            )
            
            logger.info("Birleştirme işlemi tamamlandı!")
        
        elif args.command == 'georef':
            # Jeoreferansla
            # Eğer input belirtilmemişse, varsayılan dizindeki tüm dosyaları işle
            if args.input is None:
                input_dir = DEFAULT_GEOREF_INPUT_DIR
                if not os.path.exists(input_dir):
                    raise FileNotFoundError(f"Varsayılan dizin bulunamadı: {input_dir}")
                
                # Dizindeki tüm görüntü dosyalarını bul
                image_files = [f for f in os.listdir(input_dir) 
                              if any(f.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff'])]
                
                if len(image_files) == 0:
                    raise ValueError(f"Dizinde görüntü dosyası bulunamadı: {input_dir}")
                
                logger.info(f"{len(image_files)} dosya bulundu, işleniyor...")
                
                # Her dosyayı işle
                for img_file in image_files:
                    input_path = os.path.join(input_dir, img_file)
                    
                    # Çıktı yolunu belirle
                    if args.output:
                        if os.path.isdir(args.output):
                            output_path = os.path.join(args.output, f"{img_file}_geo.tif")
                        else:
                            output_path = args.output
                    else:
                        output_path = os.path.join(DEFAULT_GEOREF_OUTPUT_DIR, f"{img_file}_geo.tif")
                    
                    try:
                        processor.georeference_image(
                            input_path=input_path,
                            reference_path=args.reference,
                            output_path=output_path,
                            band=args.band,
                            compress=args.compress,
                            nodata=args.nodata
                        )
                        logger.info(f"✓ {img_file} işlendi")
                    except Exception as e:
                        logger.error(f"✗ {img_file} işlenirken hata: {e}")
                        continue
                
                logger.info(f"Toplam {len(image_files)} dosya işlendi!")
            else:
                # Tek dosya işle
                if args.output is None:
                    # Çıktı yolunu otomatik oluştur
                    base_name = os.path.splitext(os.path.basename(args.input))[0]
                    output_path = os.path.join(DEFAULT_GEOREF_OUTPUT_DIR, f"{base_name}_geo.tif")
                else:
                    output_path = args.output
                
                processor.georeference_image(
                    input_path=args.input,
                    reference_path=args.reference,
                    output_path=output_path,
                    band=args.band,
                    compress=args.compress,
                    nodata=args.nodata
                )
            
            logger.info("Jeoreferanslama işlemi tamamlandı!")
    
    except Exception as e:
        logger.error(f"Hata oluştu: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    # Varsayılan değerler
    #DEFAULT_INPUT_IMAGE = "urgup_bingmap_30cm_utm.tif"
    DEFAULT_INPUT_IMAGE = "karlik_30_cm_bingmap_utm.tif"
    DEFAULT_MODEL_DIR = "modeller"
    DEFAULT_REFERENCE_DIR = "georeferans_sample"  # Referans dosyalarının bulunduğu klasör
    DEFAULT_REFERENCE_RASTER = None  # None ise otomatik bulunur
    
    # Eğer hiç argüman verilmemişse, varsayılan değerlerle tam pipeline çalıştır
    if len(sys.argv) == 1:
        logger.info("=" * 60)
        logger.info("PARAMETRE VERİLMEDİ, VARSAYILAN DEĞERLERLE TAM PİPELİNE ÇALIŞTIRILIYOR")
        logger.info("=" * 60)
        
        processor = ImageProcessor(reference_dir=DEFAULT_REFERENCE_DIR)
        
        # Georeferans_sample klasörünü oluştur (eğer yoksa)
        if not os.path.exists(DEFAULT_REFERENCE_DIR):
            logger.info(f"'{DEFAULT_REFERENCE_DIR}' klasörü oluşturuluyor...")
            os.makedirs(DEFAULT_REFERENCE_DIR, exist_ok=True)
            logger.info(f"✓ '{DEFAULT_REFERENCE_DIR}' klasörü oluşturuldu.")
            logger.info(f"  Lütfen referans raster dosyalarını bu klasöre koyun.")
            logger.info(f"  Örnek: ana_harita_urgup_30_cm__Georefference_utm.tif")
            logger.info(f"         ana_harita_karlik_30_cm_bingmap_Georeferans.tif")
        
        try:
            # Referans raster'ı belirle
            reference_raster = None
            if DEFAULT_REFERENCE_RASTER and os.path.exists(DEFAULT_REFERENCE_RASTER):
                reference_raster = DEFAULT_REFERENCE_RASTER
            else:
                # Otomatik bul (georeferans_sample klasöründen)
                reference_raster = processor.find_reference_raster(DEFAULT_INPUT_IMAGE, DEFAULT_REFERENCE_DIR)
            
            # Tam pipeline'ı çalıştır
            results = processor.run_full_pipeline(
                input_image=DEFAULT_INPUT_IMAGE,
                model_dir=DEFAULT_MODEL_DIR if os.path.exists(DEFAULT_MODEL_DIR) else None,
                split_frame_size=512,
                split_overlap=32,
                split_output_dir="bolunmus/bolunmus",
                processed_output_dir="parcalar",
                merge_output_dir="ana_haritalar",
                reference_raster=reference_raster,
                georef_output_dir="georefli/harita",
                crop_overlap=16,
                image_size=(544, 544),
                color_mode="grayscale"
            )
            
            # Sonuçları özetle
            logger.info("\n" + "=" * 60)
            logger.info("İŞLEM ÖZETİ")
            logger.info("=" * 60)
            logger.info(f"Bölme: {results['split']['num_pieces']} parça oluşturuldu")
            if results['inference']:
                logger.info(f"Inference: {len(results['inference'])} model işlendi")
            logger.info(f"Birleştirme: {len(results['merge'])} görüntü birleştirildi")
            if results['georef']:
                logger.info(f"Jeoreferanslama: {len(results['georef'])} görüntü jeoreferanslandı")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"Hata oluştu: {e}", exc_info=True)
            sys.exit(1)
    else:
        # Normal argparse ile çalıştır
        main()

