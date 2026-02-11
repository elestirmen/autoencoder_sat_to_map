"""
Görüntü İşlemleri Modülü
=========================

Bu modül görüntü bölme, birleştirme ve jeoreferanslama işlemlerini içerir.
Tüm fonksiyonlar robust hata kontrolü ve esnek parametrelerle donatılmıştır.

Parametre İlişkileri.:
    tile_size: Sabit karo boyutu (piksel) - sinir ağı girdisi (varsayılan: 544)
    overlap: Komşu karolar arası örtüşme (piksel) (varsayılan: 128)
    frame_size: Adım boyutu - DİNAMİK olarak hesaplanır: frame_size = tile_size - overlap
    crop_overlap: Birleştirmede her kenardan kırpılacak piksel = overlap / 2
    
    Örnek: tile_size=544, overlap=128 → frame_size=416 (hesaplanır)

Kullanım:
    # Yeni kullanım (tile_size ile - ÖNERİLEN):
    python goruntu_islemleri.py split --input image.tif --output_dir parcalar --tile_size 544 --overlap 128
    python goruntu_islemleri.py pipeline --input image.tif --tile_size 544 --overlap 128
    
    # Eski kullanım (frame_size ile - DEPRECATED):
    python goruntu_islemleri.py split --input image.tif --output_dir parcalar --frame_size 512 --overlap 32
    
    # Birleştirme (metadata'dan otomatik okur):
    python goruntu_islemleri.py merge --input_dir parcalar --output merged.jpg
    
    # Jeoreferanslama:
    python goruntu_islemleri.py georef --input image.jpg --reference ref.tif --output geo.tif
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import re

# ============================================================================
# MERKEZI YAPILANDIRMA (CONFIG)
# ============================================================================
# Tüm varsayılan parametreler bu sözlükten okunur. Bir parametreyi değiştirmek
# için buradaki değeri güncellemek yeterlidir; kodun geri kalanı otomatik olarak
# bu değerleri kullanır.
#
# ÖNEMLİ PARAMETRE İLİŞKİLERİ:
#   tile_size = frame_size + overlap                 (karo boyutu sabit!)
#   frame_size = tile_size - overlap                 (adım boyutu hesaplanır)
#   crop_overlap = overlap / 2                       (eşleşmeli!)
#
#   Örnek (varsayılan):
#     tile_size=544 (sabit)       →  model girdisi = 544  ✓ eşleşiyor
#     overlap=128                 →  örtüşme miktarı
#     frame_size=416 (hesaplanan) →  416 = 544 - 128      ✓ doğru
#     crop_overlap=64             →  64*2=128 = overlap   ✓ eşleşiyor
#     Birleştirme sonrası net karo = 544 - 128 = 416 = frame_size  ✓
#
#   DİKKAT: tile_size sabit tutulur (sinir ağı girdisi), frame_size dinamik
#   olarak hesaplanır. Overlap artırıldığında tile_size değişmez, frame_size
#   küçülür (daha fazla karo oluşur).
# ============================================================================
CONFIG = {

    # ── OpenCV Ayarları ─────────────────────────────────────────────────────
    "opencv": {
        # OpenCV'nin okuyabileceği maksimum piksel sayısı. Büyük uydu
        # görüntüleri (onbinlerce piksel) için varsayılan limiti aşmak gerekir.
        # 2^40 ≈ 1 trilyon piksel; pratikte herhangi bir görüntüyü okuyabilir.
        "max_image_pixels": 2 ** 40,
    },

    # ── Referans Raster Eşleştirme Ayarları ─────────────────────────────────
    "reference": {
        # Jeoreferanslama için kullanılacak referans GeoTIFF dosyalarının
        # bulunduğu klasör. Pipeline bu klasördeki dosyalar arasından
        # görüntü adına göre en uygun referansı otomatik seçer.
        "default_dir": "georeferans_sample",

        # Otomatik referans eşleştirmede kullanılan bölge anahtar kelimeleri.
        # Görüntü dosya adında bu kelimelerden biri varsa, aynı kelimeyi
        # içeren referans dosyasına +20 puan verilir.
        # Yeni bölge eklemek için listeye ekleme yapın.
        "auto_match_keywords": ["urgup", "karlik", "kapadokya", "bern"],
    },

    # ── Görüntü Bölme (Split) Ayarları ──────────────────────────────────────
    "split": {
        # Varsayılan giriş görüntüsü. Parametresiz çalıştırmada veya
        # CLI'da -i belirtilmediğinde bu dosya kullanılır.
        "input_image": "urgup_bingmap_30cm_utm.tif",

        # Bölünmüş karoların kaydedileceği klasör.
        # Pipeline modunda alt klasör olarak görüntü adı eklenir:
        #   bolunmus/bolunmus/<görüntü_adı>/goruntu_0_0.jpg
        "output_dir": "bolunmus/bolunmus",

        # Sabit karo boyutu (piksel). Tüm karolar bu boyutta çıkarılır.
        # Bu değer sinir ağının beklediği girdi boyutuyla eşleşmelidir.
        #
        # DİKKAT: Bu değer SABİTTİR ve image_size ile eşleşmelidir:
        #   tile_size == image_size[0] == image_size[1]
        #
        # Overlap artırıldığında tile_size değişmez, frame_size küçülür.
        "tile_size": 544,

        # Grid adım boyutu (piksel). Bu değer ARTIK DİNAMİK OLARAK HESAPLANIR:
        #   frame_size = tile_size - overlap
        #
        # Eski davranış (geriye uyumluluk): Eğer frame_size parametresi
        # verilirse, tile_size = frame_size + overlap olarak hesaplanır.
        # Ancak bu kullanım önerilmez (deprecated).
        #
        # Örnek: tile_size=544, overlap=128 → frame_size=416 (hesaplanır)
        # "frame_size": 512,  # REMOVED - artık dinamik hesaplanıyor

        # Her karonun kenarına eklenen örtüşme pikseli. Komşu karolarla
        # örtüşme sağlayarak birleştirme sonrası dikiş izlerini azaltır.
        #
        # İlişki: 
        #   frame_size = tile_size - overlap  (adım boyutu hesaplanır)
        #   crop_overlap = overlap / 2        (birleştirmede her kenardan kırpılır)
        "overlap": 128,

        # Dosya adı öneki. Karolar "{prefix}_{satır}_{sütun}.{format}"
        # şeklinde adlandırılır. Örn: goruntu_0_0.jpg, goruntu_3_12.jpg
        "prefix": "goruntu",

        # Karo kayıt formatı. "jpg" lossy ama küçük boyut; "png" kayıpsız
        # ama büyük; "tif" coğrafi veri için.
        "format": "jpg",

        # True ise bölme sonrası metadata.json dosyası oluşturulur.
        # Metadata grid boyutları, tile_size, frame_size, overlap gibi bilgileri içerir
        # ve birleştirme aşamasında otomatik okunur.
        "save_metadata": False,

        # True ise bölme sonrası matplotlib ile tüm karolar görselleştirilir.
        # Büyük görüntülerde (binlerce karo) çok yavaş olabilir.
        "visualize": False,

        # True ise bölme sırasında tqdm progress bar gösterilir.
        "show_progress": True,

        # True ise tüm karolar RAM'de tutulur (görselleştirme için gerekli).
        # False ise sadece diske yazılır, RAM tasarrufu sağlar.
        # Pipeline modunda otomatik olarak False kullanılır.
        "keep_in_memory": True,
    },

    # ── Görüntü Birleştirme (Merge) Ayarları ────────────────────────────────
    "merge": {
        # Birleştirilecek karoların bulunduğu varsayılan dizin.
        # CLI'da -i ile değiştirilebilir.
        "input_dir": "parcalar",

        # Birleştirilmiş görüntünün kaydedileceği varsayılan dosya adı.
        # CLI'da -o ile değiştirilebilir.
        "output": "birlestirilmis.jpg",

        # Her karonun HER KENARINDAN kırpılacak piksel sayısı.
        # Bölme sırasında eklenen örtüşme (overlap) burada kırpılır.
        # Formül: crop_overlap = overlap / 2
        #
        #   overlap=128 → crop_overlap=64
        #   Karo boyutu 544 → 544 - 64*2 = 416 = frame_size  ✓
        #
        # Bu değer yanlışsa parçalar arası boşluk veya üst üste binme oluşur.
        "crop_overlap": 64,

        # Adım boyutu (piksel). Bu değer metadata.json'dan okunur.
        # None ise metadata'dan veya karo boyutundan otomatik hesaplanır.
        #
        # DİKKAT: Bu değer artık metadata'dan okunmalıdır. Manuel belirtmek
        # yerine split işlemi sırasında oluşturulan metadata.json kullanılmalıdır.
        "frame_size": None,

        # True ise dosyalar natsort ile doğal sıralama ile sıralanır.
        # goruntu_0_0, goruntu_0_1, ..., goruntu_1_0 şeklinde doğru sıra.
        # False ise işletim sisteminin varsayılan sıralaması kullanılır.
        "sort_files": True,
    },

    # ── Tam Pipeline Ayarları ────────────────────────────────────────────────
    "pipeline": {
        # Eğitilmiş model dosyalarının (.h5) bulunduğu klasör.
        # Pipeline bu klasördeki TÜM .h5 dosyalarını otomatik bulur ve
        # her biri ile ayrı ayrı inference yapar.
        "model_dir": "modeller",

        # Tek bir model dosyası yolu. Belirtilirse model_dir yerine
        # sadece bu model kullanılır. None ise model_dir kullanılır.
        "model_path": None,

        # Bölünmüş karoların kaydedileceği üst klasör.
        # Pipeline görüntü adıyla alt klasör oluşturur:
        #   bolunmus/bolunmus/urgup_bingmap_30cm_utm/
        "split_output_dir": "bolunmus/bolunmus",

        # Model inference çıktılarının kaydedileceği üst klasör.
        # Pipeline görüntü adı ve model adıyla alt klasörler oluşturur:
        #   parcalar/urgup_bingmap_30cm_utm/model_v1/
        "processed_output_dir": "parcalar",

        # Birleştirilmiş mozaik haritaların kaydedileceği klasör.
        # Çıktı: ana_haritalar/ana_harita_<görüntü>_<model>.jpg
        "merge_output_dir": "ana_haritalar",

        # Jeoreferanslanmış GeoTIFF çıktılarının kaydedileceği klasör.
        # Çıktı: georefli/harita/<dosya>_geo.tif
        "georef_output_dir": "georefli/harita",

        # Sabit karo boyutu (piksel). Tüm karolar bu boyutta çıkarılır.
        # Bu değer sinir ağının beklediği girdi boyutuyla eşleşmelidir.
        #
        # DİKKAT: tile_size SABİTTİR ve image_size ile eşleşmelidir:
        #   tile_size == image_size[0] == image_size[1]
        #
        # Overlap artırıldığında tile_size değişmez, frame_size küçülür.
        "tile_size": 544,

        # Model girdi boyutu (yükseklik, genişlik). Tüm karolar bu boyuta
        # yeniden boyutlandırılarak modele verilir.
        #
        # DİKKAT: tile_size ile eşleşmelidir!
        #   tile_size=544 → image_size=(544,544)  ✓
        #   Eşleşmezse karolar sıkıştırılır/genişletilir (kalite kaybı!).
        "image_size": (544, 544),

        # Modelin renk modu.
        #   "auto":      Model kanal sayısından otomatik algıla (ÖNERİLEN)
        #                1 kanal → grayscale, 3 kanal → rgb
        #   "grayscale": 1 kanal gri tonlama (otomatik algılamayı EZER)
        #   "rgb":       3 kanal renkli (otomatik algılamayı EZER)
        "color_mode": "auto",

        # Batch inference boyutu. Aynı anda kaç karonun GPU'ya verileceği.
        # Büyük değer → hızlı ama çok GPU belleği gerektirir.
        # Küçük değer → yavaş ama az bellek.
        #
        # Öneriler:  4 GB VRAM → 2-4  |  8 GB → 8-16  |  12+ GB → 16-32
        # OutOfMemoryError alırsanız bu değeri düşürün.
        "batch_size": 16,

        # Referans raster dosyalarının bulunduğu klasör.
        # reference["default_dir"] ile aynı değer; pipeline içinden erişim
        # kolaylığı için tekrarlanmıştır.
        "reference_dir": "georeferans_sample",

        # Manuel referans raster dosyası yolu. None ise otomatik eşleştirme
        # kullanılır (görüntü adına göre georeferans_sample/ klasöründen).
        # Belirtilirse otomatik eşleştirme atlanır, bu dosya kullanılır.
        "reference_raster": None,
    },

    # ── Jeoreferanslama (Georef) Ayarları ────────────────────────────────────
    "georef": {
        # CLI'da georef alt komutunun varsayılan girişi.
        # None ise input_dir klasöründeki tüm görüntüler işlenir.
        # Bir dosya yolu verilirse sadece o dosya jeoreferanslanır.
        "input": None,

        # Tek dosya belirtilmediğinde taranacak varsayılan dizin.
        # Bu dizindeki tüm .jpg/.png/.tif dosyaları otomatik işlenir.
        "input_dir": "ana_haritalar",

        # Varsayılan referans GeoTIFF dosyası. CLI'da -r ile değiştirilebilir.
        # Bu dosyanın CRS, transform ve boyut bilgileri çıktıya kopyalanır.
        # Pipeline modunda bu değer kullanılmaz (otomatik eşleştirme tercih edilir).
        "reference": "ana_harita_urgup_30_cm__Georefference_utm.tif",

        # Varsayılan çıktı dosya yolu. None ise otomatik oluşturulur:
        #   georefli/harita/<dosya_adı>_geo.tif
        "output": None,

        # Çıktı dosyalarının kaydedileceği varsayılan dizin.
        "output_dir": "georefli/harita",

        # Raster'dan okunacak band numarası (1-indexed).
        # Gri tonlamalı görüntüler için 1, renkli için 1-2-3 ayrı okunabilir.
        # Mevcut kodda sadece tek band destekleniyor.
        "band": 1,

        # GeoTIFF sıkıştırma algoritması.
        #   "LZW":     Kayıpsız, orta boyut. Varsayılan ve önerilen.
        #   "DEFLATE":  Kayıpsız, LZW'den biraz daha küçük, biraz daha yavaş.
        #   "JPEG":     Kayıplı, en küçük boyut. İkinci aşama optimize eder.
        #   "NONE":     Sıkıştırma yok, en büyük boyut, en hızlı yazma.
        "compress": "LZW",

        # NoData değeri. Bu piksel değeri "veri yok" olarak işaretlenir.
        # None ise NoData tanımlanmaz. Genellikle 0 veya -9999 kullanılır.
        "nodata": None,
    },
}

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
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(CONFIG["opencv"]["max_image_pixels"])

import cv2
import numpy as np
import matplotlib.pyplot as plt
from natsort import natsorted
import rasterio
from affine import Affine

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
            reference_dir = CONFIG["reference"]["default_dir"]
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
        common_names = CONFIG["reference"]["auto_match_keywords"]
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
        try:
            with rasterio.open(path) as raster:
                transform = raster.transform
        except Exception as e:
            raise ValueError(f"Raster açılamadı: {path} ({e})")

        # rasterio Affine -> GDAL geotransform tuple:
        # (c, a, b, f, d, e)
        gt = transform.to_gdal()
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
        tile_size: int = CONFIG["split"]["tile_size"],
        overlap: int = CONFIG["split"]["overlap"],
        frame_size: Optional[int] = None,
        output_dir: str = CONFIG["split"]["output_dir"],
        prefix: str = CONFIG["split"]["prefix"],
        format: str = CONFIG["split"]["format"],
        save_metadata: bool = CONFIG["split"]["save_metadata"],
        original_path: Optional[str] = None,
        show_progress: bool = CONFIG["split"]["show_progress"],
        keep_in_memory: bool = CONFIG["split"]["keep_in_memory"]
    ) -> Tuple[List[np.ndarray], List[str], Dict[str, Any]]:
        """
        Görüntüyü küçük parçalara böler.
        
        Bu metod görüntüyü sabit boyutlu karolara (tiles) böler. Karo boyutu (tile_size)
        sabittir ve sinir ağının beklediği girdi boyutuyla eşleşmelidir. Adım boyutu
        (frame_size) dinamik olarak tile_size - overlap formülüyle hesaplanır.
        
        Args:
            img: Görüntü array'i (numpy array)
            tile_size: Sabit karo boyutu (piksel). Tüm karolar bu boyutta çıkarılır.
                      Bu değer sinir ağının girdi boyutuyla eşleşmelidir. (varsayılan: 544)
            overlap: Komşu karolar arası örtüşme (piksel). Daha fazla örtüşme daha fazla
                    karo oluşturur ancak tile_size sabit kalır. (varsayılan: 128)
            frame_size: (DEPRECATED) Geriye uyumluluk için. Eğer belirtilirse,
                       tile_size = frame_size + overlap olarak hesaplanır.
                       Yeni kodda tile_size kullanılmalıdır. (varsayılan: None)
            output_dir: Karoların kaydedileceği dizin
            prefix: Dosya adı öneki (örn: "goruntu" → goruntu_0_0.jpg)
            format: Kayıt formatı ('jpg', 'png', 'tif')
            save_metadata: True ise metadata.json dosyası oluşturulur
            original_path: Orijinal görüntü yolu (metadata için)
            show_progress: True ise progress bar gösterilir
            keep_in_memory: True ise karolar RAM'de tutulur
            
        Returns:
            Tuple[List[np.ndarray], List[str], Dict[str, Any]]:
                - Karo listesi (keep_in_memory=True ise dolu, False ise boş)
                - Dosya yolları listesi
                - Metadata sözlüğü (tile_size, frame_size, overlap, grid boyutları vb.)
            
        Raises:
            ValueError: Geçersiz parametreler verilirse (tile_size <= 0, overlap < 0,
                       overlap >= tile_size, vb.)
                       
        Notlar:
            - Karo boyutu (tile_size) SABİTTİR ve değişmez
            - Adım boyutu (frame_size) DİNAMİK olarak hesaplanır: frame_size = tile_size - overlap
            - Overlap artırıldığında tile_size sabit kalır, frame_size küçülür (daha fazla karo)
            - Sınır karoları tile_size'dan küçük olabilir (görüntü kenarlarında)
            
        Örnek:
            >>> processor = ImageProcessor()
            >>> img = cv2.imread("image.tif")
            >>> tiles, files, meta = processor.split_image(
            ...     img, tile_size=544, overlap=128, output_dir="tiles/"
            ... )
            >>> print(f"Frame size: {meta['frame_size']}")  # 544 - 128 = 416
            >>> print(f"Tile count: {len(tiles)}")
        """
        # 1. BACKWARD COMPATIBILITY CHECK
        if frame_size is not None and tile_size == CONFIG["split"]["tile_size"]:
            # Legacy mode: user provided frame_size, calculate tile_size
            logger.warning(
                "Parameter 'frame_size' is deprecated. "
                "Use 'tile_size' instead. "
                "Calculating tile_size = frame_size + overlap for compatibility."
            )
            tile_size = frame_size + overlap
        elif frame_size is not None and tile_size != CONFIG["split"]["tile_size"]:
            # Both provided: tile_size takes precedence
            logger.warning(
                "Both 'tile_size' and 'frame_size' provided. "
                "Using 'tile_size' and ignoring 'frame_size'."
            )
        
        # 2. PARAMETER VALIDATION
        if tile_size <= 0:
            raise ValueError("tile_size must be greater than 0")
        if overlap < 0:
            raise ValueError("overlap cannot be negative")
        if overlap >= tile_size:
            raise ValueError(
                f"overlap ({overlap}) must be less than tile_size ({tile_size}). "
                f"Current configuration would result in frame_size <= 0."
            )
        
        # 3. CALCULATE FRAME SIZE
        calculated_frame_size = tile_size - overlap
        
        if calculated_frame_size <= 0:
            raise ValueError(
                f"Calculated frame_size ({calculated_frame_size}) must be positive. "
                f"Reduce overlap or increase tile_size."
            )
        
        # 4. LOG CONFIGURATION
        logger.info(f"Tiling configuration:")
        logger.info(f"  Tile size: {tile_size}x{tile_size} (fixed)")
        logger.info(f"  Overlap: {overlap}px")
        logger.info(f"  Frame size (step): {calculated_frame_size}px (calculated)")
        
        height, width = img.shape[:2]
        
        # Çıktı dizinini oluştur
        self.create_output_directory(output_dir)
        
        # Kaç parça oluşturulacağını hesapla (kenarlari tam kapsayacak sekilde)
        def _compute_starts(length: int, tile: int, step: int) -> List[int]:
            if length <= tile:
                return [0]

            starts = list(range(0, max(length - tile + 1, 1), step))
            last_start = max(0, length - tile)
            if len(starts) == 0 or starts[-1] != last_start:
                starts.append(last_start)
            return starts

        y_starts = _compute_starts(height, tile_size, calculated_frame_size)
        x_starts = _compute_starts(width, tile_size, calculated_frame_size)
        num_frames_x = len(y_starts)
        num_frames_y = len(x_starts)
        
        logger.info(f"Image size: {height}x{width}")
        logger.info(f"Grid dimensions: {num_frames_x}x{num_frames_y} = {num_frames_x * num_frames_y} tiles")
        
        img_cropped = []
        filenames = []
        metadata = {
            'tile_size': tile_size,
            'frame_size': calculated_frame_size,
            'num_frames_x': num_frames_x,
            'num_frames_y': num_frames_y,
            'overlap': overlap,
            'y_starts': y_starts,
            'x_starts': x_starts,
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
        
        for i, start_y in enumerate(y_starts):
            for j, start_x in enumerate(x_starts):
                # Başlangıç ve bitiş koordinatlarını hesapla
                end_y = min(start_y + tile_size, height)
                end_x = min(start_x + tile_size, width)
                
                # Parçayı kes
                crop = img[start_y:end_y, start_x:end_x]
                
                # Dosya adını oluştur
                filename = os.path.join(output_dir, f'{prefix}_{i}_{j}.{format}')
                
                # Kaydet
                if format.lower() in ['tif', 'tiff']:
                    cv2.imwrite(filename, crop)
                else:
                    cv2.imwrite(filename, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                
                if keep_in_memory:
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
        crop_overlap: int = CONFIG["merge"]["crop_overlap"],
        frame_size: Optional[int] = None,
        tile_size: Optional[int] = None,
        sort_files: bool = CONFIG["merge"]["sort_files"]
    ) -> np.ndarray:
        """
        Parçalara bölünmüş görüntüleri birleştirir.
        
        Bu metod, split_image() tarafından oluşturulan parçaları birleştirir.
        tile_size ve frame_size parametreleri metadata.json dosyasından otomatik
        olarak yüklenir. Manuel olarak sağlanmazlarsa, metadata'dan okunur.
        
        Parametre İlişkileri:
        - tile_size: Orijinal parça boyutu (ör. 544x544)
        - frame_size: Adım boyutu = tile_size - overlap (ör. 512)
        - crop_overlap: Her kenardan kırpılacak piksel = overlap / 2 (ör. 16)
        - Kırpılmış parça boyutu = tile_size - 2*crop_overlap = frame_size
        
        Args:
            input_dir: Parçaların bulunduğu dizin
            output_path: Birleştirilmiş görüntünün kaydedileceği yol
            num_frames_x: X eksenindeki parça sayısı (None ise metadata'dan veya otomatik hesaplanır)
            num_frames_y: Y eksenindeki parça sayısı (None ise metadata'dan veya otomatik hesaplanır)
            crop_overlap: Her kenardan kırpılacak piksel sayısı (overlap/2 olmalı, ör. overlap=32 → crop_overlap=16)
            frame_size: Adım boyutu (None ise metadata'dan okunur veya hesaplanır)
            tile_size: Orijinal parça boyutu (None ise metadata'dan okunur veya ilk parçadan algılanır)
            sort_files: Dosyaları sırala mı (natsort kullanarak)
            
        Returns:
            Birleştirilmiş görüntü array'i
            
        Raises:
            ValueError: Geçersiz parametreler veya dosya bulunamazsa
            FileNotFoundError: input_dir bulunamazsa
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
        
        # Metadata.json dosyasını yüklemeye çalış
        metadata_path = os.path.join(input_dir, 'metadata.json')
        metadata = None
        
        if os.path.exists(metadata_path):
            try:
                import json
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                logger.info(f"Metadata yüklendi: {metadata_path}")
                
                # Metadata'dan parametreleri çıkar
                if tile_size is None and 'tile_size' in metadata:
                    tile_size = metadata['tile_size']
                    logger.info(f"  tile_size metadata'dan yüklendi: {tile_size}")
                
                if frame_size is None and 'frame_size' in metadata:
                    frame_size = metadata['frame_size']
                    logger.info(f"  frame_size metadata'dan yüklendi: {frame_size}")
                
                if num_frames_x is None and 'num_frames_x' in metadata:
                    num_frames_x = metadata['num_frames_x']
                    logger.info(f"  num_frames_x metadata'dan yüklendi: {num_frames_x}")
                
                if num_frames_y is None and 'num_frames_y' in metadata:
                    num_frames_y = metadata['num_frames_y']
                    logger.info(f"  num_frames_y metadata'dan yüklendi: {num_frames_y}")
                    
            except Exception as e:
                logger.warning(f"Metadata yüklenemedi: {e}")
        else:
            logger.info(f"Metadata dosyası bulunamadı: {metadata_path}")
        
        # İlk görüntüyü yükle ve boyutları al
        first_img_path = os.path.join(input_dir, files[0])
        first_img = cv2.imread(first_img_path)
        if first_img is None:
            raise ValueError(f"Görüntü yüklenemedi: {first_img_path}")
        
        img_height, img_width = first_img.shape[:2]
        
        # tile_size'ı belirle (metadata'dan yüklenmediyse)
        if tile_size is None:
            # İlk görüntünün boyutunu kullan
            tile_size = min(img_height, img_width)
            logger.warning(f"tile_size metadata'da bulunamadı, ilk görüntüden algılandı: {tile_size}")
        
        # Frame size'ı belirle (metadata'dan yüklenmediyse)
        if frame_size is None:
            # tile_size - (crop_overlap * 2) olarak hesapla
            frame_size = tile_size - (crop_overlap * 2)
            logger.warning(f"frame_size metadata'da bulunamadı, hesaplandı: tile_size - (crop_overlap * 2) = {frame_size}")
        
        # Parametrelerin geçerliliğini kontrol et
        if tile_size <= 0:
            raise ValueError(f"tile_size ({tile_size}) pozitif olmalıdır")
        if frame_size <= 0:
            raise ValueError(f"frame_size ({frame_size}) pozitif olmalıdır")
        if crop_overlap < 0:
            raise ValueError(f"crop_overlap ({crop_overlap}) negatif olamaz")
        
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
        logger.info(f"Birleştirme yapılandırması:")
        logger.info(f"  Karo boyutu: {tile_size}x{tile_size}")
        logger.info(f"  Frame boyutu: {frame_size}px")
        logger.info(f"  Kırpma örtüşmesi: {crop_overlap}px (her kenardan)")
        logger.info(f"  Grid: {num_frames_x}x{num_frames_y}")

        # Tüm görüntüleri yükle ve sadece iç kenarlardaki overlap'i kırp.
        # Dış sınır korunur; böylece jeoreferanslama sırasında global ofset oluşmaz.
        # Metadata'da varsa kesin yerlesim koordinatlarini kullan.
        y_starts = None
        x_starts = None
        expected_original_size = None
        if isinstance(metadata, dict):
            raw_y_starts = metadata.get('y_starts')
            raw_x_starts = metadata.get('x_starts')
            raw_original_size = metadata.get('original_size')

            if isinstance(raw_y_starts, list) and len(raw_y_starts) > 0:
                try:
                    y_starts = [int(v) for v in raw_y_starts]
                except Exception:
                    y_starts = None

            if isinstance(raw_x_starts, list) and len(raw_x_starts) > 0:
                try:
                    x_starts = [int(v) for v in raw_x_starts]
                except Exception:
                    x_starts = None

            if isinstance(raw_original_size, (list, tuple)) and len(raw_original_size) == 2:
                try:
                    expected_original_size = (int(raw_original_size[0]), int(raw_original_size[1]))
                except Exception:
                    expected_original_size = None

        if y_starts is not None and len(y_starts) != num_frames_x:
            logger.warning(
                f"Metadata y_starts uzunlugu ({len(y_starts)}) "
                f"num_frames_x ({num_frames_x}) ile uyusmuyor. Grid tabanli yerlestirme kullanilacak."
            )
            y_starts = None
        if x_starts is not None and len(x_starts) != num_frames_y:
            logger.warning(
                f"Metadata x_starts uzunlugu ({len(x_starts)}) "
                f"num_frames_y ({num_frames_y}) ile uyusmuyor. Grid tabanli yerlestirme kullanilacak."
            )
            x_starts = None
        if (y_starts is None) != (x_starts is None):
            logger.warning("Metadata start koordinatlari eksik. Grid tabanli yerlestirme kullanilacak.")
            y_starts = None
            x_starts = None
        if y_starts is not None and x_starts is not None:
            logger.info("Konumsal yerlestirme: metadata x_starts/y_starts kullanilacak")

        # Dosyalari i/j indeksine gore esle. B?ylece stale dosyalar sirayi bozmaz.
        expected_tiles = num_frames_x * num_frames_y
        indexed_files: Dict[Tuple[int, int], str] = {}
        unindexed_files = []
        index_pattern = re.compile(r'_(\d+)_(\d+)\.[^.]+$')

        for img_file in files:
            m = index_pattern.search(img_file)
            if not m:
                unindexed_files.append(img_file)
                continue
            i = int(m.group(1))
            j = int(m.group(2))
            if i >= num_frames_x or j >= num_frames_y:
                continue
            key = (i, j)
            if key in indexed_files:
                logger.warning(
                    f"Ayni karo indeksi icin birden fazla dosya bulundu ({i},{j}): "
                    f"{indexed_files[key]} ve {img_file}. Son dosya kullanilacak."
                )
            indexed_files[key] = img_file

        tile_plan = []
        if len(indexed_files) > 0:
            for i in range(num_frames_x):
                for j in range(num_frames_y):
                    img_file = indexed_files.get((i, j))
                    if img_file is not None:
                        tile_plan.append((i, j, img_file))
            missing = expected_tiles - len(tile_plan)
            if missing > 0:
                logger.warning(
                    f"Gridde beklenen {expected_tiles} karodan {missing} tanesi eksik. "
                    f"Bulunan karo: {len(tile_plan)}"
                )
        else:
            if len(unindexed_files) > 0:
                logger.warning("Dosya adlarindan i/j indeksi okunamadi, sirali grid varsayimi kullanilacak.")
            for tile_idx, img_file in enumerate(files[:expected_tiles]):
                i = tile_idx // num_frames_y
                j = tile_idx % num_frames_y
                tile_plan.append((i, j, img_file))

        if len(tile_plan) == 0:
            raise ValueError("Birle?tirilecek g?r?nt? bulunamad?!")

        # T?m g?r?nt?leri y?kle ve sadece i? kenarlardaki overlap'i k?rp.
        # D?? s?n?r korunur; b?ylece jeoreferanslama s?ras?nda global ofset olu?maz.
        tile_records = []
        pbar = tqdm(tile_plan, desc="G?r?nt?ler y?kleniyor", unit="dosya", ncols=100) if TQDM_AVAILABLE else tile_plan

        for i, j, img_file in pbar:
            img_path = os.path.join(input_dir, img_file)
            img = cv2.imread(img_path)
            if img is None:
                logger.warning(f"G?r?nt? y?klenemedi, atlan?yor: {img_path}")
                continue

            # Y?klenen karonun boyutlar?n? do?rula
            h, w = img.shape[:2]
            if h != tile_size or w != tile_size:
                logger.warning(
                    f"Karo {img_file} beklenmeyen boyuta sahip: {h}x{w}, "
                    f"beklenen: {tile_size}x{tile_size}"
                )

            # Sadece i? kenarlar k?rp?l?r, d?? s?n?r korunur.
            top_crop = crop_overlap if i > 0 else 0
            bottom_crop = crop_overlap if i < (num_frames_x - 1) else 0
            left_crop = crop_overlap if j > 0 else 0
            right_crop = crop_overlap if j < (num_frames_y - 1) else 0

            y2 = h - bottom_crop if bottom_crop > 0 else h
            x2 = w - right_crop if right_crop > 0 else w

            if y2 <= top_crop or x2 <= left_crop:
                logger.warning(
                    f"Karo {img_file} i?in k?rpma ge?ersiz: "
                    f"({top_crop}, {bottom_crop}, {left_crop}, {right_crop}), "
                    f"boyut={h}x{w}. Karo atland?."
                )
                continue

            img = img[top_crop:y2, left_crop:x2]

            # Hedef konum: metadata starts varsa onlari, yoksa frame_size grid'ini kullan.
            if y_starts is not None and x_starts is not None:
                dest_y = y_starts[i] + top_crop
                dest_x = x_starts[j] + left_crop
            else:
                dest_y = (i * frame_size) + top_crop
                dest_x = (j * frame_size) + left_crop

            tile_records.append((img, dest_y, dest_x, img_file))

        if TQDM_AVAILABLE and isinstance(pbar, tqdm):
            pbar.close()

        if len(tile_records) == 0:
            raise ValueError("Birleştirilecek görüntü bulunamadı!")

        if len(tile_records) != expected_tiles:
            logger.warning(
                f"Yuklenen goruntu sayisi ({len(tile_records)}) beklenen sayidan ({expected_tiles}) farkli!"
            )

        # Çıktı boyutunu yerleştirme koordinatlarından hesapla.
        max_height = max(dest_y + tile.shape[0] for tile, dest_y, dest_x, _ in tile_records)
        max_width = max(dest_x + tile.shape[1] for tile, dest_y, dest_x, _ in tile_records)

        sample_tile = tile_records[0][0]
        if sample_tile.ndim == 2:
            merged_image = np.zeros((max_height, max_width), dtype=sample_tile.dtype)
        else:
            merged_image = np.zeros((max_height, max_width, sample_tile.shape[2]), dtype=sample_tile.dtype)

        for tile, dest_y, dest_x, img_file in tile_records:
            th, tw = tile.shape[:2]
            merged_image[dest_y:dest_y + th, dest_x:dest_x + tw] = tile

        actual_height, actual_width = merged_image.shape[:2]
        if expected_original_size is not None:
            exp_h, exp_w = expected_original_size
            if actual_height != exp_h or actual_width != exp_w:
                logger.warning(
                    f"Birlestirilmis goruntu boyutu ({actual_height}x{actual_width}) "
                    f"orijinal boyuttan ({exp_h}x{exp_w}) farkli."
                )
        logger.info(f"Birleştirilmiş görüntü boyutu: {actual_height}x{actual_width}")
        
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
        band: int = CONFIG["georef"]["band"],
        compress: str = CONFIG["georef"]["compress"],
        nodata: Optional[float] = CONFIG["georef"]["nodata"],
        target_transform: Optional[Any] = None,
        target_crs: Optional[Any] = None,
        force_reference_shape: bool = False
    ) -> None:
        """
        Görüntüyü referans raster'ın coğrafi bilgileriyle jeoreferanslar.
        
        Giriş görüntüsünün kanal sayısını otomatik algılar:
        - 1 kanal (grayscale) → 1-band GeoTIFF
        - 3 kanal (RGB) → 3-band GeoTIFF
        
        Args:
            input_path: Jeoreferanslanacak görüntü yolu
            reference_path: Referans GeoTIFF dosyası yolu
            output_path: Çıktı dosyası yolu
            band: Okunacak band numarası (1-indexed, sadece tek band için)
            compress: Sıkıştırma tipi ('LZW', 'DEFLATE', 'JPEG', 'NONE')
            nodata: NoData değeri
            target_transform: Çıktıya uygulanacak transform (None ise referans transformu)
            target_crs: Çıktıya uygulanacak CRS (None ise referans CRS'i)
            force_reference_shape: True ise çıktı boyutu referans raster ile aynı yapılır
            
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
        
        # Giriş görüntüsünün kanal sayısını algıla
        input_band_count = input_raster.count
        logger.info(f"Giriş görüntüsü kanal sayısı: {input_band_count}")
        
        # Tüm bandları oku
        try:
            if input_band_count == 1:
                # Grayscale: tek band
                data = input_raster.read(1)
                data = np.expand_dims(data, axis=0)  # (H, W) -> (1, H, W)
            elif input_band_count >= 3:
                # RGB veya RGBA: ilk 3 bandı al
                data = input_raster.read([1, 2, 3])  # (3, H, W)
                if input_band_count > 3:
                    logger.info(f"  Alpha kanalı atlandı (toplam {input_band_count} band)")
            else:
                # 2 kanallı nadir durum
                data = input_raster.read()
                logger.warning(f"  Beklenmeyen kanal sayısı: {input_band_count}")
        except Exception as e:
            raise ValueError(f"Bandlar okunamadı: {e}")
        
        # Çıktı kanal sayısını belirle
        output_band_count = 3 if input_band_count >= 3 else 1

        input_height = data.shape[1]
        input_width = data.shape[2]
        out_height = reference_raster.shape[0] if force_reference_shape else input_height
        out_width = reference_raster.shape[1] if force_reference_shape else input_width

        if not force_reference_shape and (input_height != reference_raster.shape[0] or input_width != reference_raster.shape[1]):
            logger.warning(
                f"Giriş boyutu ({input_height}x{input_width}) referans boyutundan "
                f"({reference_raster.shape[0]}x{reference_raster.shape[1]}) farklı. "
                f"Çıkış grid'i giriş boyutunda korunacak."
            )

        out_transform = target_transform if target_transform is not None else reference_raster.transform
        out_crs = target_crs if target_crs is not None else reference_raster.crs
        
        # Meta verileri referans raster'dan kopyala
        out_meta = reference_raster.meta.copy()
        out_meta.update({
            'driver': 'GTiff',
            'width': out_width,
            'height': out_height,
            'count': output_band_count,
            'dtype': 'uint8',
            'crs': out_crs,
            'transform': out_transform,
            'compress': compress
        })
        
        if nodata is not None:
            out_meta['nodata'] = nodata
        
        # Çıktı dizinini oluştur
        output_dir = os.path.dirname(output_path)
        if output_dir:
            self.create_output_directory(output_dir)
        
        # Veriyi uint8'e dönüştür (her band için ayrı ayrı)
        if data.dtype != np.uint8:
            data_out = np.zeros_like(data, dtype=np.uint8)
            for i in range(data.shape[0]):
                band_data = data[i]
                data_min = band_data.min()
                data_max = band_data.max()
                if data_max > data_min:
                    data_out[i] = ((band_data - data_min) / (data_max - data_min) * 255).astype(np.uint8)
                else:
                    data_out[i] = band_data.astype(np.uint8)
            data = data_out

        # İstenirse referans grid boyutuna yeniden örnekle.
        if (data.shape[1] != out_height) or (data.shape[2] != out_width):
            logger.warning(
                f"Çıktı boyutu için yeniden örnekleme yapılıyor: "
                f"{data.shape[1]}x{data.shape[2]} -> {out_height}x{out_width}"
            )
            resized = np.zeros((data.shape[0], out_height, out_width), dtype=data.dtype)
            for i in range(data.shape[0]):
                resized[i] = cv2.resize(data[i], (out_width, out_height), interpolation=cv2.INTER_LINEAR)
            data = resized
        
        # Yeni raster dosyasını yaz
        with rasterio.open(output_path, 'w', **out_meta) as dst:
            if output_band_count == 1:
                dst.write_band(1, data[0])
            else:
                for i in range(output_band_count):
                    dst.write_band(i + 1, data[i])
        
        logger.info(f"Jeoreferanslı görüntü kaydedildi: {output_path} ({output_band_count} band)")
        
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
        image_size: Tuple[int, int] = CONFIG["pipeline"]["image_size"],
        color_mode: str = CONFIG["pipeline"]["color_mode"],
        batch_size: int = CONFIG["pipeline"]["batch_size"]
    ) -> List[str]:
        """
        Parçalara bölünmüş görüntüleri sinir ağı modelinden batch inference ile geçirir.
        
        Eski ThreadPoolExecutor yaklaşımı yerine toplu (batch) tahmin kullanır.
        Bu yöntem GPU'yu daha verimli kullanır ve TensorFlow thread-safety
        sorunlarını ortadan kaldırır.
        
        Args:
            input_dir: Giriş parçalarının bulunduğu dizin
            output_dir: İşlenmiş parçaların kaydedileceği dizin
            model_path: Model dosyasının yolu
            image_size: Görüntü boyutu (height, width)
            color_mode: Renk modu ("grayscale" veya "rgb")
            batch_size: Batch boyutu (GPU VRAM'a göre ayarlayın, varsayılan: 16)
            
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
        # Eski inference dosyalari yeni merge sirasini bozmasin.
        for old_name in os.listdir(output_dir):
            old_path = os.path.join(output_dir, old_name)
            if os.path.isfile(old_path) and old_name.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
                try:
                    os.remove(old_path)
                except Exception as e:
                    logger.warning(f"Eski inference dosyasi silinemedi ({old_path}): {e}")

        
        # Modeli yukle (farkli Keras surumleri icin uyumluluk fallback'leri ile)
        logger.info(f"Model yukleniyor: {model_path}")

        def ssim_loss(y_true, y_pred):
            return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))

        class CompatibleConv2DTranspose(tf.keras.layers.Conv2DTranspose):
            """
            Eski modellerde serialize edilen "groups" argumanini yoksayarak
            yeni surumlerde modelin acilmasini saglar.
            """
            def __init__(self, *args, groups=1, **kwargs):
                super().__init__(*args, **kwargs)

        load_attempts = [
            ("standart", {"compile": False}),
            ("ssim_loss", {"custom_objects": {"ssim_loss": ssim_loss}, "compile": False}),
            (
                "ssim_loss + Conv2DTranspose uyumluluk",
                {
                    "custom_objects": {
                        "ssim_loss": ssim_loss,
                        "Conv2DTranspose": CompatibleConv2DTranspose,
                    },
                    "compile": False,
                },
            ),
        ]

        model = None
        load_errors = []

        for attempt_name, attempt_kwargs in load_attempts:
            try:
                model = load_model(model_path, **attempt_kwargs)
                logger.info(f"Model yuklendi ({attempt_name}).")
                break
            except Exception as e:
                load_errors.append(f"{attempt_name}: {e}")

        if model is None:
            detailed_error = "\n  - ".join(load_errors) if load_errors else "Bilinmeyen hata"
            raise ValueError(f"Model yuklenemedi. Denenen yontemler:\n  - {detailed_error}")
        
        # ═══════════════════════════════════════════════════════════════════════
        # RENK MODU BELİRLEME
        # "auto" → model kanal sayısından otomatik algıla
        # "grayscale" veya "rgb" → kullanıcının belirlediği değer (ezme)
        # ═══════════════════════════════════════════════════════════════════════
        try:
            input_shape = model.input_shape
            output_shape = model.output_shape
            
            # Kanal sayısını al (son boyut)
            input_channels = input_shape[-1] if input_shape[-1] is not None else None
            output_channels = output_shape[-1] if output_shape[-1] is not None else None
            
            logger.info(f"Model input shape: {input_shape}")
            logger.info(f"Model output shape: {output_shape}")
            
            # Kanal sayısına göre algılanan modu belirle
            if input_channels == 1 or output_channels == 1:
                detected_mode = "grayscale"
            elif input_channels == 3 or output_channels == 3:
                detected_mode = "rgb"
            else:
                detected_mode = None
            
            # color_mode belirleme mantığı
            if color_mode == "auto":
                # Otomatik algılama modu
                if detected_mode:
                    logger.info(f"📌 Color mode otomatik algılandı: '{detected_mode}' (model kanal: {input_channels})")
                    color_mode = detected_mode
                else:
                    logger.warning("Color mode algılanamadı, varsayılan 'rgb' kullanılıyor.")
                    color_mode = "rgb"
            else:
                # Kullanıcı belirli bir mod seçmiş (ezme)
                if detected_mode and detected_mode != color_mode:
                    logger.warning(f"⚠️  Model kanal sayısı ({input_channels}) ile color_mode '{color_mode}' uyumsuz!")
                    logger.warning(f"   Model '{detected_mode}' bekliyor. Yine de '{color_mode}' kullanılacak.")
                else:
                    logger.info(f"✓ Color mode: '{color_mode}' (kullanıcı tarafından belirlendi)")
        except Exception as e:
            if color_mode == "auto":
                logger.warning(f"Color mode otomatik algılanamadı, varsayılan 'rgb' kullanılıyor.")
                color_mode = "rgb"
            else:
                logger.info(f"Color mode: '{color_mode}' (kullanıcı tarafından belirlendi)")
            logger.debug(f"Algılama hatası: {e}")
        
        # Dosyaları listele ve sırala
        files = [f for f in os.listdir(input_dir) 
                 if any(f.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff'])]
        files = natsorted(files)
        
        if len(files) == 0:
            raise ValueError(f"Dizinde görüntü dosyası bulunamadı: {input_dir}")
        
        logger.info(f"{len(files)} dosya bulundu, batch inference başlatılıyor (batch_size={batch_size})...")
        
        model_input_channels = None
        model_output_channels = None
        try:
            model_input_channels = model.input_shape[-1]
            model_output_channels = model.output_shape[-1]
        except Exception:
            pass

        if model_input_channels in (1, 3):
            channels = int(model_input_channels)
            if color_mode != "auto":
                requested_channels = 1 if color_mode == "grayscale" else 3
                if requested_channels != channels:
                    logger.warning(
                        f"İstenen color_mode ('{color_mode}') model girişi ile uyumsuz. "
                        f"Model beklediği kanal sayısı kullanılacak: {channels}"
                    )
        else:
            channels = 1 if color_mode == "grayscale" else 3

        logger.info(f"Model giriş kanalı: {channels}")
        output_files = []
        
        pbar = tqdm(total=len(files), desc="Model inference", unit="görüntü", ncols=100)
        
        # Batch inference ile işle
        for batch_start in range(0, len(files), batch_size):
            batch_files = files[batch_start:batch_start + batch_size]
            batch_images = []
            valid_indices = []
            
            # Batch için görüntüleri yükle
            for idx, filename in enumerate(batch_files):
                filepath = os.path.join(input_dir, filename)
                try:
                    pixels_raw = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
                    if pixels_raw is None:
                        raise ValueError("Dosya okunamadı veya format desteklenmiyor")

                    if pixels_raw.ndim == 2:
                        src_channels = 1
                        pixels = pixels_raw
                    elif pixels_raw.ndim == 3:
                        src_channels = pixels_raw.shape[2]
                        if src_channels == 1:
                            src_channels = 1
                            pixels = pixels_raw[:, :, 0]
                        else:
                            # OpenCV BGR/BGRA okur; alfa varsa at.
                            pixels = cv2.cvtColor(pixels_raw[:, :, :3], cv2.COLOR_BGR2RGB)
                            src_channels = 3
                    else:
                        raise ValueError(f"Beklenmeyen görüntü boyutu: {pixels_raw.shape}")

                    # Modelin beklediği kanal sayısına otomatik dönüştür.
                    if channels == 1 and src_channels == 3:
                        pixels = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
                    elif channels == 3 and src_channels == 1:
                        pixels = cv2.cvtColor(pixels, cv2.COLOR_GRAY2RGB)

                    pixels = cv2.resize(pixels, (image_size[1], image_size[0]), interpolation=cv2.INTER_NEAREST)
                    pixels = pixels.astype(np.float32)
                    if channels == 1 and pixels.ndim == 2:
                        pixels = np.expand_dims(pixels, axis=-1)
                    pixels = (pixels - 127.5) / 127.5
                    batch_images.append(pixels)
                    valid_indices.append(idx)
                except Exception as e:
                    logger.error(f"Görüntü yüklenemedi ({filename}): {e}")
            
            if len(batch_images) == 0:
                pbar.update(len(batch_files))
                continue
            
            batch_array = np.array(batch_images)
            
            # Toplu tahmin (batch prediction)
            try:
                predictions = model.predict(batch_array, verbose=0)
            except Exception as e:
                logger.error(f"Batch prediction hatası: {e}")
                pbar.update(len(batch_files))
                continue
            
            # Sonuçları kaydet
            for pred_idx, file_idx in enumerate(valid_indices):
                filename = batch_files[file_idx]
                try:
                    base_name = os.path.splitext(filename)[0]
                    # Ara ciktilarda kayipli sikistirma kaynakli artefakti azaltmak icin PNG kullan.
                    output_filename = os.path.join(output_dir, f'goruntu_{base_name}.png')

                    pred = predictions[pred_idx].astype(np.float32)
                    if pred.ndim == 3 and pred.shape[-1] == 1:
                        pred = pred[:, :, 0]

                    pmin = float(np.min(pred))
                    pmax = float(np.max(pred))

                    # Parca bazli dinamik normalize etme dikis artefaktlarini artirir.
                    # Bu nedenle global sabit olcek kullanilir.
                    if -1.1 <= pmin and pmax <= 1.1:
                        pred_uint8 = np.clip((pred + 1.0) * 127.5, 0, 255).astype(np.uint8)
                    elif -0.1 <= pmin and pmax <= 1.1:
                        pred_uint8 = np.clip(pred * 255.0, 0, 255).astype(np.uint8)
                    else:
                        pred_uint8 = np.clip(pred, 0, 255).astype(np.uint8)

                    output_is_grayscale = (model_output_channels == 1)
                    if output_is_grayscale or pred_uint8.ndim == 2:
                        if pred_uint8.ndim == 3:
                            pred_uint8 = pred_uint8[:, :, 0]
                        cv2.imwrite(output_filename, pred_uint8)
                    else:
                        if pred_uint8.ndim == 2:
                            pred_uint8 = cv2.cvtColor(pred_uint8, cv2.COLOR_GRAY2RGB)
                        cv2.imwrite(output_filename, cv2.cvtColor(pred_uint8, cv2.COLOR_RGB2BGR))

                    output_files.append(output_filename)
                except Exception as e:
                    logger.error(f"Görüntü kaydedilemedi ({filename}): {e}")
            
            pbar.update(len(batch_files))
        
        pbar.close()
        
        logger.info(f"Toplam {len(output_files)}/{len(files)} görüntü başarıyla işlendi.")
        
        return output_files
    
    def run_full_pipeline(
        self,
        input_image: str,
        model_path: Optional[str] = CONFIG["pipeline"]["model_path"],
        model_dir: Optional[str] = CONFIG["pipeline"]["model_dir"],
        split_tile_size: int = CONFIG["pipeline"]["tile_size"],
        split_frame_size: Optional[int] = None,
        split_overlap: int = CONFIG["split"]["overlap"],
        split_output_dir: str = CONFIG["pipeline"]["split_output_dir"],
        processed_output_dir: str = CONFIG["pipeline"]["processed_output_dir"],
        merge_output_dir: str = CONFIG["pipeline"]["merge_output_dir"],
        reference_raster: Optional[str] = CONFIG["pipeline"]["reference_raster"],
        georef_output_dir: str = CONFIG["pipeline"]["georef_output_dir"],
        crop_overlap: int = CONFIG["merge"]["crop_overlap"],
        image_size: Tuple[int, int] = CONFIG["pipeline"]["image_size"],
        color_mode: str = CONFIG["pipeline"]["color_mode"],
        batch_size: int = CONFIG["pipeline"]["batch_size"]
    ) -> Dict[str, Any]:
        """
        Tüm işlemleri sırayla çalıştırır: Böl -> Model Inference -> Birleştir -> Jeoreferansla
        
        Args:
            input_image: Giriş görüntü dosyası
            model_path: Tek model dosyası yolu (veya model_dir kullanılır)
            model_dir: Model dosyalarının bulunduğu dizin (tüm modeller işlenir)
            split_tile_size: Sabit karo boyutu (piksel) - sinir ağı girdisi (varsayılan: 544)
            split_frame_size: (DEPRECATED) Adım boyutu - tile_size kullanın
            split_overlap: Bölme işlemi için örtüşme
            split_output_dir: Bölünmüş parçaların dizini
            processed_output_dir: Model'den geçmiş parçaların dizini
            merge_output_dir: Birleştirilmiş görüntülerin dizini
            reference_raster: Jeoreferans için referans raster
            georef_output_dir: Jeoreferanslı görüntülerin dizini
            crop_overlap: Birleştirme sırasında kesilecek örtüşme
            image_size: Model için görüntü boyutu
            color_mode: Model için renk modu ("grayscale" veya "rgb")
            batch_size: Batch boyutu (GPU VRAM'a göre ayarlayın)
            
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
                            logger.info("1. ADIM: G?r?nt? B?lme (ATLANDI - Klas?r zaten mevcut)")
                            logger.info("=" * 60)
                            has_new_starts = ('x_starts' in metadata and 'y_starts' in metadata)
                            if has_new_starts:
                                logger.info(f"? Mevcut klas?r bulundu: {split_output_dir_with_name}")
                                logger.info(f"? {len(files_in_dir)} par?a mevcut, b?lme i?lemi atlan?yor...")
                                skip_split = True

                                results['split'] = {
                                    'num_pieces': len(files_in_dir),
                                    'metadata': metadata,
                                    'output_dir': split_output_dir_with_name,
                                    'skipped': True
                                }
                            else:
                                logger.warning(
                                    "Mevcut metadata eski formatta (x_starts/y_starts yok). "
                                    "Tam kapsama i?in b?lme ad?m? yeniden ?al??t?r?lacak."
                                )
                        except Exception as e:
                            logger.warning(f"Metadata yüklenemedi, bölme işlemi tekrar yapılacak: {e}")
                    else:
                        logger.warning(
                            "Metadata dosyasi bulunamadi. Tam kapsama icin bolme adimi yeniden calistirilacak."
                        )
                        
            
            if not skip_split:
                logger.info("=" * 60)
                logger.info("1. ADIM: Görüntü Bölme")
                logger.info("=" * 60)
                
                img = self.load_image(input_image)
                # Eski/yarim kalan parcalarin yeni split sonucunu bozmamasi icin temizle.
                if os.path.exists(split_output_dir_with_name):
                    for old_name in os.listdir(split_output_dir_with_name):
                        old_path = os.path.join(split_output_dir_with_name, old_name)
                        if os.path.isfile(old_path) and (
                            old_name.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))
                            or old_name == 'metadata.json'
                        ):
                            try:
                                os.remove(old_path)
                            except Exception as e:
                                logger.warning(f"Eski parca dosyasi silinemedi ({old_path}): {e}")

                
                # Coğrafi bilgileri al (eğer mümkünse)
                try:
                    gt, px, py = self.get_geotransform(input_image)
                except:
                    logger.warning("Coğrafi bilgiler alınamadı, devam ediliyor...")
                
                img_cropped, filenames, metadata = self.split_image(
                    img,
                    tile_size=split_tile_size,
                    overlap=split_overlap,
                    frame_size=split_frame_size,
                    output_dir=split_output_dir_with_name,  # Görüntü adıyla klasör
                    prefix='goruntu',
                    format='jpg',
                    save_metadata=True,
                    original_path=input_image,
                    show_progress=True,
                    keep_in_memory=False  # Pipeline'da RAM tasarrufu
                )
                
                # Metadata'yı JSON olarak kaydet
                import json
                meta_path = os.path.join(split_output_dir_with_name, 'metadata.json')
                with open(meta_path, 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)
                logger.info(f"Metadata kaydedildi: {meta_path}")
                
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
                                    batch_size=batch_size
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
                        batch_size=batch_size
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
                        tile_size=metadata.get('tile_size'),
                        frame_size=metadata.get('frame_size')
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
                    tile_size=metadata.get('tile_size'),
                    frame_size=metadata.get('frame_size')
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

            # Mümkünse georeferans için split metadata'daki orijinal geotransform'u kullan.
            pipeline_transform = None
            pipeline_crs = None
            metadata_geotransform = metadata.get('geotransform') if isinstance(metadata, dict) else None
            if metadata_geotransform and len(metadata_geotransform) == 6:
                try:
                    gt_vals = tuple(float(v) for v in metadata_geotransform)
                    pipeline_transform = Affine.from_gdal(*gt_vals)
                    logger.info("Jeoreferanslama için metadata geotransform'u kullanılacak.")
                except Exception as e:
                    logger.warning(f"Metadata geotransform'u parse edilemedi, referans transform kullanılacak: {e}")

            try:
                with rasterio.open(input_image) as src_raster:
                    if pipeline_transform is None:
                        pipeline_transform = src_raster.transform
                        logger.info("Jeoreferanslama icin kaynak transform kullanilacak.")
                    pipeline_crs = src_raster.crs
                    if pipeline_crs:
                        logger.info(f"Jeoreferanslama icin kaynak CRS kullanilacak: {pipeline_crs}")
            except Exception as e:
                logger.warning(f"Kaynak transform/CRS okunamadi, referans bilgileri kullanilacak: {e}")
            
            # Referans raster'ı bul (görüntü adına göre)
            selected_reference = None
            if reference_raster and os.path.exists(reference_raster):
                # Manuel olarak belirtilmiş referans kullan
                selected_reference = reference_raster
                logger.info(f"Manuel referans kullanılıyor: {os.path.basename(selected_reference)}")
            else:
                # Görüntü adına göre referans bul (georeferans_sample klasöründen)
                selected_reference = self.find_reference_raster(
                    input_image, reference_dir=CONFIG["pipeline"]["reference_dir"]
                )
                
                if not selected_reference:
                    logger.warning("Referans raster bulunamadı, jeoreferanslama atlanıyor...")
                    logger.info(
                        f"Ipucu: Referans raster dosyalarini '{CONFIG['pipeline']['reference_dir']}' klasorune koyun veya reference_raster parametresi ile belirtin."
                    )
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
                            band=CONFIG["georef"]["band"],
                            compress=CONFIG["georef"]["compress"],
                            nodata=CONFIG["georef"]["nodata"],
                            target_transform=pipeline_transform,
                            target_crs=pipeline_crs,
                            force_reference_shape=False,
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
    """CLI entry point."""

    split_cfg = CONFIG["split"]
    merge_cfg = CONFIG["merge"]
    pipeline_cfg = CONFIG["pipeline"]
    georef_cfg = CONFIG["georef"]

    parser = argparse.ArgumentParser(
        description='Goruntu bolme, birlestirme ve jeoreferanslama islemleri',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Ornekler:
  # Tam pipeline (Bol -> Inference -> Birlestir -> Jeoreferansla)
  python goruntu_islemleri.py pipeline -i image.tif
  python goruntu_islemleri.py pipeline -i image.tif --color_mode rgb --batch_size 8
  python goruntu_islemleri.py pipeline -i image.tif --model_path model.h5 --reference ref.tif

  # Goruntu bolme (varsayilan: {split_cfg['input_image']})
  python goruntu_islemleri.py split
  python goruntu_islemleri.py split -i image.tif -o parcalar --frame_size 512 --overlap 32

  # Goruntu birlestirme (varsayilan: {merge_cfg['input_dir']} -> {merge_cfg['output']})
  python goruntu_islemleri.py merge
  python goruntu_islemleri.py merge -i parcalar -o merged.jpg --num_frames_x 10 --num_frames_y 10

  # Jeoreferanslama (varsayilan: {georef_cfg['input_dir']} dizinindeki tum dosyalar)
  python goruntu_islemleri.py georef
  python goruntu_islemleri.py georef -i image.jpg -r reference.tif -o geo.tif
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Islem tipi')

    # Split command
    split_parser = subparsers.add_parser('split', help='Goruntuyu parcalara bol')
    split_parser.add_argument(
        '-i',
        '--input',
        default=split_cfg["input_image"],
        help=f'Giris goruntu dosyasi (varsayilan: {split_cfg["input_image"]})'
    )
    split_parser.add_argument(
        '-o',
        '--output_dir',
        default=split_cfg["output_dir"],
        help=f'Cikti dizini (varsayilan: {split_cfg["output_dir"]})'
    )
    split_parser.add_argument(
        '--tile_size',
        type=int,
        default=split_cfg["tile_size"],
        help='Sabit karo boyutu (piksel) - sinir agi girdisi (varsayilan: 544)'
    )
    split_parser.add_argument(
        '--frame_size',
        type=int,
        default=None,
        help='(DEPRECATED) Adim boyutu - tile_size kullanin'
    )
    split_parser.add_argument('--overlap', type=int, default=split_cfg["overlap"], help='Ortusme miktari (piksel)')
    split_parser.add_argument('--prefix', default=split_cfg["prefix"], help='Dosya adi oneki')
    split_parser.add_argument('--format', default=split_cfg["format"], choices=['jpg', 'png', 'tif'], help='Cikti formati')
    split_parser.add_argument('--save_metadata', action='store_true', default=split_cfg["save_metadata"], help='Metadata kaydet')
    split_parser.add_argument('--visualize', action='store_true', default=split_cfg["visualize"], help='Gorsellestirme olustur')

    # Merge command
    merge_parser = subparsers.add_parser('merge', help='Parcalari birlestir')
    merge_parser.add_argument(
        '-i',
        '--input_dir',
        default=merge_cfg["input_dir"],
        help=f'Parcalarin bulundugu dizin (varsayilan: {merge_cfg["input_dir"]})'
    )
    merge_parser.add_argument(
        '-o',
        '--output',
        default=merge_cfg["output"],
        help=f'Cikti dosyasi (varsayilan: {merge_cfg["output"]})'
    )
    merge_parser.add_argument('--num_frames_x', type=int, help='X eksenindeki parca sayisi')
    merge_parser.add_argument('--num_frames_y', type=int, help='Y eksenindeki parca sayisi')
    merge_parser.add_argument('--crop_overlap', type=int, default=merge_cfg["crop_overlap"], help='Birlestirmede kullanilacak ortusme genisligi')
    merge_parser.add_argument('--tile_size', type=int, default=None, help='Karo boyutu (metadata\'dan okunur)')
    merge_parser.add_argument('--frame_size', type=int, default=None, help='Parca boyutu (metadata\'dan okunur)')

    # Pipeline command
    pipeline_parser = subparsers.add_parser('pipeline', help='Tam pipeline: Bol -> Inference -> Birlestir -> Jeoreferansla')
    pipeline_parser.add_argument(
        '-i',
        '--input',
        default=split_cfg["input_image"],
        help=f'Giris goruntu dosyasi (varsayilan: {split_cfg["input_image"]})'
    )
    pipeline_parser.add_argument(
        '--model_dir',
        default=pipeline_cfg["model_dir"],
        help=f'Model dosyalarinin bulundugu dizin (varsayilan: {pipeline_cfg["model_dir"]})'
    )
    pipeline_parser.add_argument('--model_path', default=pipeline_cfg["model_path"], help='Tek model dosyasi yolu (model_dir yerine)')
    pipeline_parser.add_argument(
        '--tile_size',
        type=int,
        default=split_cfg["tile_size"],
        help='Sabit karo boyutu (piksel) - sinir agi girdisi (varsayilan: 544)'
    )
    pipeline_parser.add_argument(
        '--frame_size',
        type=int,
        default=None,
        help='(DEPRECATED) Adim boyutu - tile_size kullanin'
    )
    pipeline_parser.add_argument('--overlap', type=int, default=split_cfg["overlap"], help='Bolme ortusme miktari (piksel)')
    pipeline_parser.add_argument('--crop_overlap', type=int, default=merge_cfg["crop_overlap"], help='Birlestirmede kullanilacak ortusme genisligi')
    pipeline_parser.add_argument('--color_mode', default=pipeline_cfg["color_mode"], choices=['grayscale', 'rgb'], help='Renk modu')
    pipeline_parser.add_argument('--batch_size', type=int, default=pipeline_cfg["batch_size"], help='Batch boyutu')
    pipeline_parser.add_argument('--reference', default=pipeline_cfg["reference_raster"], help='Referans raster dosyasi')
    pipeline_parser.add_argument('--reference_dir', default=pipeline_cfg["reference_dir"], help='Referans raster dizini')

    # Georef command
    georef_parser = subparsers.add_parser('georef', help='Goruntuyu jeoreferansla')
    georef_parser.add_argument(
        '-i',
        '--input',
        default=georef_cfg["input"],
        help=f'Giris goruntu dosyasi veya dizin (varsayilan: {georef_cfg["input_dir"]} dizinindeki tum dosyalar)'
    )
    georef_parser.add_argument(
        '-r',
        '--reference',
        default=georef_cfg["reference"],
        help=f'Referans GeoTIFF dosyasi (varsayilan: {georef_cfg["reference"]})'
    )
    georef_parser.add_argument('-o', '--output', default=georef_cfg["output"], help='Cikti dosyasi veya dizin (varsayilan: otomatik)')
    georef_parser.add_argument('--band', type=int, default=georef_cfg["band"], help='Okunacak band numarasi')
    georef_parser.add_argument(
        '--compress',
        default=georef_cfg["compress"],
        choices=['LZW', 'DEFLATE', 'JPEG', 'NONE'],
        help='Sikistirma tipi'
    )
    georef_parser.add_argument('--nodata', type=float, default=georef_cfg["nodata"], help='NoData degeri')

    args = parser.parse_args()
    processor = ImageProcessor(reference_dir=pipeline_cfg["reference_dir"])

    if not args.command:
        args.command = 'split'
        args.input = split_cfg["input_image"]
        args.output_dir = split_cfg["output_dir"]
        args.tile_size = split_cfg["tile_size"]
        args.frame_size = None
        args.overlap = split_cfg["overlap"]
        args.prefix = split_cfg["prefix"]
        args.format = split_cfg["format"]
        args.save_metadata = split_cfg["save_metadata"]
        args.visualize = split_cfg["visualize"]
        logger.info("Komut belirtilmedi, varsayilan olarak 'split' islemi yapiliyor...")
        logger.info(f"Varsayilan dosya: {args.input}")

    try:
        if args.command == 'split':
            img = processor.load_image(args.input)

            try:
                processor.get_geotransform(args.input)
            except Exception:
                logger.warning("Cografi bilgiler alinamadi, devam ediliyor...")

            img_cropped, filenames, metadata = processor.split_image(
                img,
                tile_size=args.tile_size,
                overlap=args.overlap,
                frame_size=args.frame_size,
                output_dir=args.output_dir,
                prefix=args.prefix,
                format=args.format,
                save_metadata=args.save_metadata,
                original_path=args.input
            )

            if args.save_metadata:
                import json
                metadata_path = os.path.join(args.output_dir, 'metadata.json')
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, default=str)
                logger.info(f"Metadata kaydedildi: {metadata_path}")

            if args.visualize:
                processor.visualize_crops(
                    img_cropped,
                    metadata['num_frames_x'],
                    metadata['num_frames_y']
                )
                plt.show()

            logger.info("Bolme islemi tamamlandi!")

        elif args.command == 'merge':
            processor.merge_images(
                input_dir=args.input_dir,
                output_path=args.output,
                num_frames_x=args.num_frames_x,
                num_frames_y=args.num_frames_y,
                crop_overlap=args.crop_overlap,
                tile_size=args.tile_size,
                frame_size=args.frame_size
            )
            logger.info("Birlestirme islemi tamamlandi!")

        elif args.command == 'pipeline':
            pipeline_processor = ImageProcessor(reference_dir=args.reference_dir)

            ref_raster = args.reference
            if not ref_raster:
                ref_raster = pipeline_processor.find_reference_raster(args.input, args.reference_dir)

            m_dir = args.model_dir
            m_path = args.model_path

            results = pipeline_processor.run_full_pipeline(
                input_image=args.input,
                model_path=m_path,
                model_dir=m_dir if os.path.exists(m_dir) else None,
                split_tile_size=args.tile_size,
                split_frame_size=args.frame_size,
                split_overlap=args.overlap,
                split_output_dir=pipeline_cfg["split_output_dir"],
                processed_output_dir=pipeline_cfg["processed_output_dir"],
                merge_output_dir=pipeline_cfg["merge_output_dir"],
                reference_raster=ref_raster,
                georef_output_dir=pipeline_cfg["georef_output_dir"],
                crop_overlap=args.crop_overlap,
                image_size=pipeline_cfg["image_size"],
                color_mode=args.color_mode,
                batch_size=args.batch_size
            )

            logger.info("\n" + "=" * 60)
            logger.info("ISLEM OZETI")
            logger.info("=" * 60)
            if results['split']:
                logger.info(f"Bolme: {results['split']['num_pieces']} parca")
            if results['inference']:
                logger.info(f"Inference: {len(results['inference'])} model islendi")
            if results['merge']:
                logger.info(f"Birlestirme: {len(results['merge'])} goruntu")
            if results['georef']:
                logger.info(f"Jeoreferanslama: {len(results['georef'])} dosya")
            logger.info("=" * 60)
            logger.info("Pipeline tamamlandi!")

        elif args.command == 'georef':
            if args.input is None:
                input_dir = georef_cfg["input_dir"]
                if not os.path.exists(input_dir):
                    raise FileNotFoundError(f"Varsayilan dizin bulunamadi: {input_dir}")

                image_files = [
                    f for f in os.listdir(input_dir)
                    if any(f.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff'])
                ]

                if len(image_files) == 0:
                    raise ValueError(f"Dizinde goruntu dosyasi bulunamadi: {input_dir}")

                logger.info(f"{len(image_files)} dosya bulundu, isleniyor...")

                for img_file in image_files:
                    input_path = os.path.join(input_dir, img_file)

                    if args.output:
                        if os.path.isdir(args.output):
                            output_path = os.path.join(args.output, f"{img_file}_geo.tif")
                        else:
                            output_path = args.output
                    else:
                        output_path = os.path.join(georef_cfg["output_dir"], f"{img_file}_geo.tif")

                    try:
                        processor.georeference_image(
                            input_path=input_path,
                            reference_path=args.reference,
                            output_path=output_path,
                            band=args.band,
                            compress=args.compress,
                            nodata=args.nodata
                        )
                        logger.info(f"OK {img_file} islendi")
                    except Exception as e:
                        logger.error(f"HATA {img_file} islenirken hata: {e}")
                        continue

                logger.info(f"Toplam {len(image_files)} dosya islendi!")
            else:
                if args.output is None:
                    base_name = os.path.splitext(os.path.basename(args.input))[0]
                    output_path = os.path.join(georef_cfg["output_dir"], f"{base_name}_geo.tif")
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

            logger.info("Jeoreferanslama islemi tamamlandi!")

    except Exception as e:
        logger.error(f"Hata olustu: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    split_cfg = CONFIG["split"]
    merge_cfg = CONFIG["merge"]
    pipeline_cfg = CONFIG["pipeline"]

    if len(sys.argv) == 1:
        logger.info("=" * 60)
        logger.info("PARAMETRE VERILMEDI, VARSAYILAN DEGERLERLE TAM PIPELINE CALISTIRILIYOR")
        logger.info("=" * 60)

        processor = ImageProcessor(reference_dir=pipeline_cfg["reference_dir"])

        if not os.path.exists(pipeline_cfg["reference_dir"]):
            logger.info(f"'{pipeline_cfg['reference_dir']}' klasoru olusturuluyor...")
            os.makedirs(pipeline_cfg["reference_dir"], exist_ok=True)
            logger.info(f"OK '{pipeline_cfg['reference_dir']}' klasoru olusturuldu.")
            logger.info("  Lutfen referans raster dosyalarini bu klasore koyun.")
            logger.info("  Ornek: ana_harita_urgup_30_cm__Georefference_utm.tif")
            logger.info("         ana_harita_karlik_30_cm_bingmap_Georeferans.tif")

        try:
            reference_raster = None
            configured_reference = pipeline_cfg["reference_raster"]
            if configured_reference and os.path.exists(configured_reference):
                reference_raster = configured_reference
            else:
                reference_raster = processor.find_reference_raster(
                    split_cfg["input_image"],
                    pipeline_cfg["reference_dir"]
                )

            results = processor.run_full_pipeline(
                input_image=split_cfg["input_image"],
                model_path=pipeline_cfg["model_path"],
                model_dir=pipeline_cfg["model_dir"] if os.path.exists(pipeline_cfg["model_dir"]) else None,
                split_frame_size=None,
                split_overlap=split_cfg["overlap"],
                split_output_dir=pipeline_cfg["split_output_dir"],
                processed_output_dir=pipeline_cfg["processed_output_dir"],
                merge_output_dir=pipeline_cfg["merge_output_dir"],
                reference_raster=reference_raster,
                georef_output_dir=pipeline_cfg["georef_output_dir"],
                crop_overlap=merge_cfg["crop_overlap"],
                image_size=pipeline_cfg["image_size"],
                color_mode=pipeline_cfg["color_mode"],
                batch_size=pipeline_cfg["batch_size"]
            )

            logger.info("\n" + "=" * 60)
            logger.info("ISLEM OZETI")
            logger.info("=" * 60)
            logger.info(f"Bolme: {results['split']['num_pieces']} parca olusturuldu")
            if results['inference']:
                logger.info(f"Inference: {len(results['inference'])} model islendi")
            logger.info(f"Birlestirme: {len(results['merge'])} goruntu birlestirildi")
            if results['georef']:
                logger.info(f"Jeoreferanslama: {len(results['georef'])} goruntu jeoreferanslandi")
            logger.info("=" * 60)

        except Exception as e:
            logger.error(f"Hata olustu: {e}", exc_info=True)
            sys.exit(1)
    else:
        main()
