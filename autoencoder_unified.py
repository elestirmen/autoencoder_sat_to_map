# -*- coding: utf-8 -*-
"""
Birleşik AutoEncoder Eğitim Scripti
===================================
Bu script, tüm autoencoder varyasyonlarını tek bir dosyada birleştirir.
Tüm parametreler CONFIG bölümünden yönetilebilir.

Kullanım:
    python autoencoder_unified.py

Desteklenen Modlar:
    - grayscale: Gri tonlamalı görüntüler (1 kanal)
    - rgb: Renkli görüntüler (3 kanal)
    - grayscale_with_equalize: Gri + Histogram eşitleme
"""

import os
import cv2
import numpy as np
import datetime
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Conv2D, Conv2DTranspose, Input, UpSampling2D,
    MaxPooling2D, Dropout, BatchNormalization, Concatenate, Add, Activation, Lambda
)
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K

# tensorflow_addons sadece histogram eşitleme için gerekli
try:
    import tensorflow_addons as tfa
    TFA_AVAILABLE = True
except ImportError:
    TFA_AVAILABLE = False
    print("UYARI: tensorflow_addons yüklü değil. Histogram eşitleme devre dışı.")


# ==============================================================================
#                              CONFIG BÖLÜMÜ
# ==============================================================================

class Config:
    """Tüm eğitim parametrelerini buradan yönetin."""
    
    # ------------------------------ VERİ AYARLARI -----------------------------
    # Veri dizini yolu
    DATA_DIR = "maps_zoom_level_19/sonuclar_19/"
    
    # Giris renk modu: "grayscale", "rgb", "grayscale_with_equalize"
    COLOR_MODE = "grayscale"

    # Cikis renk modu: "auto" (giris ile ayni), "grayscale", "rgb"
    OUTPUT_COLOR_MODE = "grayscale"
    
    # Eğitim/Doğrulama oranı (0.0 - 1.0 arası, eğitim için kullanılacak oran)
    TRAIN_SPLIT = 0.9
    
    # Random seed (tekrarlanabilirlik için)
    RANDOM_SEED = 42
    
    # ------------------------------ MODEL AYARLARI ----------------------------
    # Görüntü boyutu (genişlik ve yükseklik eşit olmalı)
    IMAGE_SIZE = 544
    
    # Dinamik giriş boyutu (inference için)
    # True: Model herhangi bir boyutta görüntü kabul eder (inference'da)
    # False: Sadece IMAGE_SIZE boyutunda görüntü kabul eder
    DYNAMIC_INPUT_FOR_INFERENCE = True
    
    # Multi-scale eğitim (data augmentation)
    # True: Her batch farklı boyutlarda olabilir (daha robust model)
    # False: Tüm görüntüler IMAGE_SIZE boyutuna resize edilir
    MULTI_SCALE_TRAINING = False
    
    # Multi-scale için boyut listesi (MULTI_SCALE_TRAINING=True ise kullanılır)
    MULTI_SCALE_SIZES = [256, 384, 448, 512, 544]
    
    # Model mimarisi secimi:
    # "unet_residual", "autoencoder", "autoencoder_backup", "upsampled",
    # "advanced", "gpt", "gpt_no_reg", "deneysel", "classic"
    MODEL_TYPE = "classic"
    
    # Önceden eğitilmiş model yükle (None = sıfırdan başla)
    PRETRAINED_MODEL = None  # Örn: "son_model.h5" veya None
    
    # Aktivasyon fonksiyonu: "sigmoid", "relu", "elu", "tanh"
    ACTIVATION_FUNC = "elu"
    
    # Filtre sayısı (başlangıç filtre sayısı)
    FILTER_COUNT = 32
    
    # Kernel boyutu
    KERNEL_SIZE = (3, 3)
    
    # Stride değeri
    STRIDES = (1, 1)
    
    # ------------------------------ EĞİTİM AYARLARI ---------------------------
    # Batch size
    BATCH_SIZE = 8
    
    # Epoch sayısı
    EPOCHS = 20
    
    # Learning rate
    LEARNING_RATE = 0.001
    
    # Optimizer: "adam", "sgd", "rmsprop"
    OPTIMIZER = "adam"
    
    # Loss fonksiyonu: "weighted_hybrid", "hybrid", "mse", "mae", "binary_crossentropy", "ssim"
    LOSS_FUNCTION = "weighted_hybrid"

    # weighted_hybrid loss ayarlari (etikette beyaz agirlik baskinsa cokusmeyi azaltmak icin)
    # -1.0 siyah, +1.0 beyaz oldugu varsayilir.
    WEIGHTED_HYBRID_L1 = 0.8
    WEIGHTED_HYBRID_SSIM = 0.2
    WEIGHTED_NON_WHITE_THRESHOLD = 0.85
    WEIGHTED_DARK_THRESHOLD = -0.6
    WEIGHTED_NON_WHITE_WEIGHT = 2.0
    WEIGHTED_DARK_WEIGHT = 8.0
    
    # ------------------------------ KAYIT AYARLARI ----------------------------
    # Model kayıt dizini (None = mevcut dizin)
    SAVE_DIR = None
    
    # Her epoch'ta checkpoint kaydet
    SAVE_CHECKPOINTS = True
    
    # Kaç batch'te bir ara kayıt yapılsın (0 = devre dışı)
    PERIODIC_SAVE_STEPS = 0
    
    # Final model adı
    FINAL_MODEL_NAME = "son_model.h5"
    
    # ------------------------------ VERİ PIPELINE AYARLARI --------------------
    # Paralel işlem sayısı (None = AUTOTUNE)
    NUM_PARALLEL_CALLS = None
    
    # Prefetch buffer boyutu (None = AUTOTUNE)
    PREFETCH_BUFFER = None
    
    # Dosya doğrulama modu:
    # "none": Doğrulama yapma (en hızlı, ignore_errors ile hatalar atlanır)
    # "quick": Sadece dosya boyutu kontrolü (hızlı)
    # "cached": İlk sefer doğrula, sonucu cache'le (400k dosya için önerilen)
    # "full": Her seferinde tam doğrulama (yavaş)
    VALIDATE_FILES = "cached"
    
    # Cache dosyası adı (VALIDATE_FILES="cached" için)
    VALIDATION_CACHE_FILE = "valid_files_cache.json"

    # Bos/neredeyse bos hedef karolari filtreleme (label tarafi)
    FILTER_EMPTY_TILES = True
    FILTER_EMPTY_TILES_ON_VAL = False
    EMPTY_TILE_WHITE_THRESHOLD = 245.0
    EMPTY_TILE_DARK_THRESHOLD = 10.0
    EMPTY_TILE_MAX_WHITE_RATIO = 0.99
    EMPTY_TILE_MIN_DARK_RATIO = 0.005

    # ------------------------------ GELİŞMİŞ AYARLAR --------------------------
    # Early stopping (0 = devre dışı)
    EARLY_STOPPING_PATIENCE = 8
    
    # Learning rate azaltma (0 = devre dışı)
    REDUCE_LR_PATIENCE = 4
    REDUCE_LR_FACTOR = 0.5
    
    # Dropout oranı (bazı modeller için)
    DROPOUT_RATE = 0.3
    
    # L1 Regularization (gpt modeli için)
    L1_REGULARIZATION = 1e-5


# ==============================================================================
#                              YARDIMCI FONKSİYONLAR
# ==============================================================================

def get_timestamp():
    """Zaman damgası döndürür."""
    return datetime.datetime.strftime(datetime.datetime.now(), '%d_%m_%Y__%H_%M')


def get_input_channels():
    """Giris renk moduna gore kanal sayisini dondurur."""
    if Config.COLOR_MODE == "rgb":
        return 3
    return 1


def get_output_channels():
    """Cikis renk moduna gore kanal sayisini dondurur."""
    mode = str(Config.OUTPUT_COLOR_MODE).lower()
    if mode in ("auto", "same", "input", "none"):
        return get_input_channels()
    if mode == "rgb":
        return 3
    if mode in ("grayscale", "gray", "bw", "blackwhite", "black_white"):
        return 1
    raise ValueError(
        f"Geçersiz OUTPUT_COLOR_MODE: {Config.OUTPUT_COLOR_MODE}. "
        f"Geçerli değerler: auto, grayscale, rgb"
    )


def get_channels():
    """Geriye donuk uyumluluk icin giris kanal sayisi."""
    return get_input_channels()


def get_input_shape(for_inference=False):
    """Model giriş boyutlarını döndürür.
    
    Args:
        for_inference: True ise dinamik boyut döndürür (None, None, channels)
    """
    if for_inference and Config.DYNAMIC_INPUT_FOR_INFERENCE:
        return (None, None, get_input_channels())
    return (Config.IMAGE_SIZE, Config.IMAGE_SIZE, get_input_channels())


# ==============================================================================
#                              LOSS FONKSİYONLARI
# ==============================================================================

def ssim_loss(y_true, y_pred):
    """SSIM tabanli kayip fonksiyonu."""
    y_true = tf.clip_by_value(y_true, -1.0, 1.0)
    y_pred = tf.clip_by_value(y_pred, -1.0, 1.0)
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 2.0))


def hybrid_loss(y_true, y_pred):
    """L1 + SSIM karisik kayip (daha net cizgiler icin)."""
    y_true = tf.clip_by_value(y_true, -1.0, 1.0)
    y_pred = tf.clip_by_value(y_pred, -1.0, 1.0)
    l1 = tf.reduce_mean(tf.abs(y_true - y_pred))
    ssim = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 2.0))
    return 0.8 * l1 + 0.2 * ssim


def weighted_hybrid_loss(y_true, y_pred):
    """L1 + SSIM (etikette koyu detaylara agirlik vererek)."""
    y_true = tf.clip_by_value(y_true, -1.0, 1.0)
    y_pred = tf.clip_by_value(y_pred, -1.0, 1.0)

    # Etikette beyaz agirlik baskin oldugunda, koyu/icerik piksellerine daha fazla agirlik ver.
    non_white_mask = tf.cast(y_true <= Config.WEIGHTED_NON_WHITE_THRESHOLD, tf.float32)
    dark_mask = tf.cast(y_true <= Config.WEIGHTED_DARK_THRESHOLD, tf.float32)

    weights = tf.ones_like(y_true)
    weights += non_white_mask * (Config.WEIGHTED_NON_WHITE_WEIGHT - 1.0)
    weights += dark_mask * (Config.WEIGHTED_DARK_WEIGHT - Config.WEIGHTED_NON_WHITE_WEIGHT)

    abs_err = tf.abs(y_true - y_pred)
    weighted_l1 = tf.reduce_sum(abs_err * weights) / (tf.reduce_sum(weights) + K.epsilon())

    ssim = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 2.0))
    return Config.WEIGHTED_HYBRID_L1 * weighted_l1 + Config.WEIGHTED_HYBRID_SSIM * ssim


def ssim_metric(y_true, y_pred):
    """Takip metrigi: SSIM (1'e yaklastikca daha iyi)."""
    y_true = tf.clip_by_value(y_true, -1.0, 1.0)
    y_pred = tf.clip_by_value(y_pred, -1.0, 1.0)
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, 2.0))


def get_metrics():
    """Egitim metrikleri."""
    return [tf.keras.metrics.MeanAbsoluteError(name='mae'), ssim_metric]

def get_loss_function():
    """Yapılandırmaya göre kayıp fonksiyonunu döndürür."""
    if Config.LOSS_FUNCTION == "ssim":
        return ssim_loss
    elif Config.LOSS_FUNCTION == "weighted_hybrid":
        return weighted_hybrid_loss
    elif Config.LOSS_FUNCTION == "hybrid":
        return hybrid_loss
    elif Config.LOSS_FUNCTION == "mse":
        return "mse"
    elif Config.LOSS_FUNCTION == "mae":
        return "mae"
    elif Config.LOSS_FUNCTION == "binary_crossentropy":
        return "binary_crossentropy"
    else:
        return Config.LOSS_FUNCTION


# ==============================================================================
#                              CALLBACK SINIFI
# ==============================================================================

class PeriodicSave(tf.keras.callbacks.Callback):
    """Belirli aralıklarla model kaydeden callback."""
    
    def __init__(self, save_every=1000, model_name_prefix="model"):
        self.save_every = save_every
        self.model_name_prefix = model_name_prefix
        super(PeriodicSave, self).__init__()

    def on_train_batch_end(self, batch, logs=None):
        if self.save_every > 0 and batch % self.save_every == 0 and batch > 0:
            save_path = f'{self.model_name_prefix}_step_{batch}.h5'
            self.model.save(save_path)
            print(f"\n[PeriodicSave] Model kaydedildi: {save_path}")

    def on_epoch_end(self, epoch, logs=None):
        save_path = f'{self.model_name_prefix}_epoch_{epoch}.h5'
        self.model.save(save_path)


# ==============================================================================
#                              VERİ YÜKLEME
# ==============================================================================

def get_random_size():
    """Multi-scale eğitim için rastgele boyut seçer."""
    sizes = Config.MULTI_SCALE_SIZES
    idx = tf.random.uniform([], 0, len(sizes), dtype=tf.int32)
    return tf.gather(sizes, idx)


def convert_channels(image, source_channels, target_channels):
    """Goruntuyu kaynak kanaldan hedef kanal sayisina cevirir."""
    if source_channels == target_channels:
        return image
    if source_channels == 3 and target_channels == 1:
        return tf.image.rgb_to_grayscale(image)
    if source_channels == 1 and target_channels == 3:
        return tf.image.grayscale_to_rgb(image)
    raise ValueError(
        f"Desteklenmeyen kanal donusumu: {source_channels} -> {target_channels}"
    )


def load_and_preprocess(image_path):
    """Görüntüyü yükler ve ön işler."""
    input_channels = get_input_channels()
    output_channels = get_output_channels()
    decode_channels = 3 if (input_channels == 3 or output_channels == 3) else 1
    use_equalize = Config.COLOR_MODE == "grayscale_with_equalize"

    # Dosyayı oku ve decode et
    img_raw = tf.io.read_file(image_path)
    img = tf.image.decode_image(img_raw, channels=decode_channels)

    # Veri tipini float32'ye çevir
    img = tf.cast(img, tf.float32)

    # Boyutları al
    shape = tf.shape(img)
    height = shape[0]
    width = shape[1] // 2  # Girdi ve etiket yan yana

    # Girdi ve etiketi ayır
    input_img = tf.slice(img, [0, 0, 0], [height, width, decode_channels])
    label_img = tf.slice(img, [0, width, 0], [height, width, decode_channels])

    # Girdi ve hedef kanal sayilarini ayri ayri hazirla
    input_img = convert_channels(input_img, decode_channels, input_channels)
    label_img = convert_channels(label_img, decode_channels, output_channels)

    # Histogram eşitleme (sadece grayscale_with_equalize modunda)
    if use_equalize and TFA_AVAILABLE:
        input_img = tfa.image.equalize(input_img)

    # Boyutlandırma
    target_size = Config.IMAGE_SIZE
    input_img = tf.image.resize(input_img, [target_size, target_size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    label_img = tf.image.resize(label_img, [target_size, target_size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    # Normalizasyon (-1 ile 1 arası)
    input_img = (input_img - 127.5) / 127.5
    label_img = (label_img - 127.5) / 127.5

    return (input_img, label_img)


def load_and_preprocess_multiscale(image_path):
    """Multi-scale eğitim için görüntüyü rastgele boyutta yükler."""
    input_channels = get_input_channels()
    output_channels = get_output_channels()
    decode_channels = 3 if (input_channels == 3 or output_channels == 3) else 1
    use_equalize = Config.COLOR_MODE == "grayscale_with_equalize"

    # Dosyayı oku ve decode et
    img_raw = tf.io.read_file(image_path)
    img = tf.image.decode_image(img_raw, channels=decode_channels)
    img = tf.cast(img, tf.float32)

    # Boyutları al
    shape = tf.shape(img)
    height = shape[0]
    width = shape[1] // 2

    # Girdi ve etiketi ayır
    input_img = tf.slice(img, [0, 0, 0], [height, width, decode_channels])
    label_img = tf.slice(img, [0, width, 0], [height, width, decode_channels])

    # Girdi ve hedef kanal sayilarini ayri ayri hazirla
    input_img = convert_channels(input_img, decode_channels, input_channels)
    label_img = convert_channels(label_img, decode_channels, output_channels)

    # Histogram eşitleme
    if use_equalize and TFA_AVAILABLE:
        input_img = tfa.image.equalize(input_img)

    # Rastgele boyut seç
    target_size = get_random_size()
    input_img = tf.image.resize(input_img, [target_size, target_size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    label_img = tf.image.resize(label_img, [target_size, target_size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    # Normalizasyon
    input_img = (input_img - 127.5) / 127.5
    label_img = (label_img - 127.5) / 127.5

    return (input_img, label_img)


def keep_informative_tile(input_img, label_img):
    """Hedefte tamamen/neredeyse tamamen bos (beyaz) karolari eler."""
    del input_img  # Sadece hedef etiket dagilimi kullanilir.

    label_img = tf.clip_by_value(label_img, -1.0, 1.0)
    label_u8 = (label_img + 1.0) * 127.5
    label_gray = tf.reduce_mean(label_u8, axis=-1)

    white_ratio = tf.reduce_mean(
        tf.cast(label_gray >= Config.EMPTY_TILE_WHITE_THRESHOLD, tf.float32)
    )
    dark_ratio = tf.reduce_mean(
        tf.cast(label_gray <= Config.EMPTY_TILE_DARK_THRESHOLD, tf.float32)
    )

    keep_by_white_ratio = white_ratio <= Config.EMPTY_TILE_MAX_WHITE_RATIO
    keep_by_dark_ratio = dark_ratio >= Config.EMPTY_TILE_MIN_DARK_RATIO
    return tf.logical_or(keep_by_white_ratio, keep_by_dark_ratio)


import json
import hashlib


def get_cache_path():
    """Cache dosyasının tam yolunu döndürür."""
    cache_dir = Config.DATA_DIR
    return os.path.join(cache_dir, Config.VALIDATION_CACHE_FILE)


def compute_dataset_hash(image_paths):
    """Veri setinin hash'ini hesaplar (dosya listesi + toplam sayı)."""
    # Sadece dosya adlarını ve toplam sayısını kullan (hızlı)
    content = f"{len(image_paths)}:{sorted([os.path.basename(p) for p in image_paths[:100]])}"
    return hashlib.md5(content.encode()).hexdigest()


def load_validation_cache():
    """Cache dosyasını yükler."""
    cache_path = get_cache_path()
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            pass
    return None


def save_validation_cache(valid_files, dataset_hash):
    """Doğrulama sonuçlarını cache'e kaydeder."""
    cache_path = get_cache_path()
    cache_data = {
        "hash": dataset_hash,
        "valid_files": valid_files,
        "count": len(valid_files),
        "timestamp": datetime.datetime.now().isoformat()
    }
    try:
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f)
        print(f"✅ Doğrulama sonuçları cache'lendi: {cache_path}")
    except Exception as e:
        print(f"⚠️ Cache kaydedilemedi: {e}")


def validate_quick(image_paths):
    """Hızlı doğrulama - sadece dosya boyutu kontrolü."""
    valid = []
    invalid_count = 0
    
    for path in image_paths:
        try:
            if os.path.getsize(path) > 0:
                valid.append(path)
            else:
                invalid_count += 1
        except:
            invalid_count += 1
    
    if invalid_count > 0:
        print(f"⚠️ {invalid_count} boş/erişilemeyen dosya atlandı")
    
    return valid


def validate_full(image_paths):
    """Tam doğrulama - header kontrolü dahil."""
    valid_paths = []
    invalid_count = 0
    total = len(image_paths)
    
    print("Tam doğrulama yapılıyor...")
    
    for i, path in enumerate(image_paths):
        # Progress her 10000 dosyada bir göster
        if i > 0 and i % 10000 == 0:
            print(f"  İşleniyor: {i}/{total} ({i*100//total}%)")
        
        try:
            file_size = os.path.getsize(path)
            if file_size < 100:  # 100 byte'tan küçük dosyalar şüpheli
                invalid_count += 1
                continue
            valid_paths.append(path)
        except:
            invalid_count += 1
    
    if invalid_count > 0:
        print(f"⚠️ {invalid_count} bozuk/boş dosya atlandı")
    
    return valid_paths


def get_validated_paths(all_image_paths):
    """Doğrulama moduna göre geçerli dosyaları döndürür."""
    mode = Config.VALIDATE_FILES.lower()
    
    if mode == "none":
        print("📌 Dosya doğrulama: DEVRE DIŞI (ignore_errors aktif)")
        return all_image_paths
    
    elif mode == "quick":
        print("📌 Dosya doğrulama: HIZLI (sadece boyut kontrolü)")
        return validate_quick(all_image_paths)
    
    elif mode == "cached":
        print("📌 Dosya doğrulama: CACHE MODUNDA")
        cache = load_validation_cache()
        dataset_hash = compute_dataset_hash(all_image_paths)
        
        if cache and cache.get("hash") == dataset_hash:
            print(f"✅ Cache'den yüklendi: {cache['count']} geçerli dosya")
            # Cache'deki dosyaların hala var olduğunu kontrol et
            valid_from_cache = [p for p in cache["valid_files"] if os.path.exists(p)]
            if len(valid_from_cache) == cache['count']:
                return valid_from_cache
            print("⚠️ Bazı dosyalar silinmiş, yeniden doğrulanıyor...")
        
        # Cache yoksa veya geçersizse, hızlı doğrulama yap ve cache'le
        print("🔄 İlk kez doğrulama yapılıyor (bu sadece bir kez olacak)...")
        valid_paths = validate_quick(all_image_paths)
        save_validation_cache(valid_paths, dataset_hash)
        return valid_paths
    
    elif mode == "full":
        print("📌 Dosya doğrulama: TAM (her dosya kontrol ediliyor)")
        return validate_full(all_image_paths)
    
    else:
        print(f"⚠️ Bilinmeyen doğrulama modu: {mode}, 'none' kullanılıyor")
        return all_image_paths


def create_datasets():
    """Eğitim ve doğrulama veri setlerini oluşturur."""
    # Görüntü yollarını al
    all_image_paths = [
        os.path.join(Config.DATA_DIR, fname) 
        for fname in os.listdir(Config.DATA_DIR)
        if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
    ]
    
    if len(all_image_paths) == 0:
        raise ValueError(f"Veri dizininde görüntü bulunamadı: {Config.DATA_DIR}")
    
    print(f"Toplam dosya sayısı: {len(all_image_paths)}")
    
    # Doğrulama moduna göre filtreleme
    all_image_paths = get_validated_paths(all_image_paths)
    
    if len(all_image_paths) == 0:
        raise ValueError("Hiç geçerli görüntü dosyası bulunamadı!")
    
    # Random seed ayarla
    np.random.seed(Config.RANDOM_SEED)
    
    # Görüntü yollarını karıştır
    np.random.shuffle(all_image_paths)
    
    # Eğitim ve doğrulama setlerini ayır
    split_at = int(len(all_image_paths) * Config.TRAIN_SPLIT)
    train_paths = all_image_paths[:split_at]
    val_paths = all_image_paths[split_at:]
    
    print(f"Eğitim seti: {len(train_paths)} görüntü")
    print(f"Doğrulama seti: {len(val_paths)} görüntü")
    if Config.FILTER_EMPTY_TILES:
        print(
            "Bos karo filtresi aktif: "
            f"white>={Config.EMPTY_TILE_WHITE_THRESHOLD}, "
            f"max_white_ratio={Config.EMPTY_TILE_MAX_WHITE_RATIO}, "
            f"min_dark_ratio={Config.EMPTY_TILE_MIN_DARK_RATIO}"
        )
    
    # Multi-scale eğitim bilgisi
    if Config.MULTI_SCALE_TRAINING:
        print(f"Multi-scale eğitim aktif. Boyutlar: {Config.MULTI_SCALE_SIZES}")
    
    # Paralel işlem ayarları
    parallel_calls = Config.NUM_PARALLEL_CALLS
    if parallel_calls is None:
        parallel_calls = tf.data.experimental.AUTOTUNE
    
    prefetch_buffer = Config.PREFETCH_BUFFER
    if prefetch_buffer is None:
        prefetch_buffer = tf.data.experimental.AUTOTUNE
    
    # Preprocessing fonksiyonunu seç
    if Config.MULTI_SCALE_TRAINING:
        # Multi-scale: her batch farklı boyutta, batch_size=1 olmalı
        preprocess_fn = load_and_preprocess_multiscale
        batch_size = 1  # Multi-scale'de batch=1 zorunlu (farklı boyutlar)
        print("UYARI: Multi-scale modunda batch_size=1 kullanılıyor.")
    else:
        preprocess_fn = load_and_preprocess
        batch_size = Config.BATCH_SIZE
    
    # Dataset oluştur - ignore_errors ile bozuk dosyaları atla
    train_dataset = tf.data.Dataset.from_tensor_slices(train_paths)
    train_dataset = train_dataset.map(preprocess_fn, num_parallel_calls=parallel_calls)
    train_dataset = train_dataset.apply(tf.data.experimental.ignore_errors())  # Hataları atla
    if Config.FILTER_EMPTY_TILES:
        train_dataset = train_dataset.filter(keep_informative_tile)
    train_dataset = train_dataset.batch(batch_size).prefetch(buffer_size=prefetch_buffer)
    
    val_dataset = tf.data.Dataset.from_tensor_slices(val_paths)
    val_dataset = val_dataset.map(load_and_preprocess, num_parallel_calls=parallel_calls)
    val_dataset = val_dataset.apply(tf.data.experimental.ignore_errors())  # Hataları atla
    if Config.FILTER_EMPTY_TILES and Config.FILTER_EMPTY_TILES_ON_VAL:
        val_dataset = val_dataset.filter(keep_informative_tile)
    val_dataset = val_dataset.batch(Config.BATCH_SIZE).prefetch(buffer_size=prefetch_buffer)
    
    return train_dataset, val_dataset


# ==============================================================================
#                              MODEL MİMARİLERİ
# ==============================================================================

def create_autoencoder_model(input_shape):
    """Standart autoencoder modeli."""
    channels = get_output_channels()
    filter_count = Config.FILTER_COUNT
    kernel_size = Config.KERNEL_SIZE
    strides = Config.STRIDES
    activation_func = Config.ACTIVATION_FUNC
    
    input_layer = Input(shape=input_shape)
    
    # Encoder
    x = Conv2D(filter_count, kernel_size=kernel_size, strides=strides,
               activation='elu', padding='same', kernel_initializer='he_normal')(input_layer)
    x = Conv2D(filter_count * 2, kernel_size=kernel_size, strides=strides,
               activation='elu', padding='same', kernel_initializer='he_normal')(x)
    x = Conv2D(filter_count * 4, kernel_size=kernel_size, strides=strides,
               activation='elu', padding='same', kernel_initializer='he_normal')(x)
    
    # Decoder
    x = Conv2DTranspose(filter_count * 4, kernel_size=kernel_size, strides=strides,
                        activation='elu', padding='same', kernel_initializer='he_normal')(x)
    x = Conv2DTranspose(filter_count * 2, kernel_size=kernel_size, strides=strides,
                        activation='elu', padding='same', kernel_initializer='he_normal')(x)
    x = Conv2DTranspose(filter_count, kernel_size=kernel_size, strides=strides,
                        activation='elu', padding='same', kernel_initializer='he_normal')(x)
    
    output_layer = Conv2D(channels, kernel_size=kernel_size, strides=(1, 1),
                          activation=activation_func, padding='same')(x)
    
    return Model(inputs=input_layer, outputs=output_layer)


def create_autoencoder_model_backup(input_shape):
    """Backup autoencoder modeli (alternatif mimari)."""
    channels = get_output_channels()
    filter_count = Config.FILTER_COUNT
    kernel_size = Config.KERNEL_SIZE
    strides = Config.STRIDES
    activation_func = Config.ACTIVATION_FUNC
    
    input_layer = Input(shape=input_shape)
    
    x = Conv2D(filter_count, kernel_size=kernel_size, strides=strides,
               activation='elu', padding='same', kernel_initializer='he_normal')(input_layer)
    x = Conv2D(filter_count * 2, kernel_size=kernel_size, strides=strides,
               activation='elu', padding='same', kernel_initializer='he_normal')(x)
    x = Conv2D(filter_count * 4, kernel_size=(3, 3), strides=(1, 1),
               activation='elu', padding='same', kernel_initializer='he_normal')(x)
    x = Conv2D(filter_count * 8, kernel_size=(3, 3), strides=(2, 2),
               activation='elu', padding='same', kernel_initializer='he_normal')(x)
    
    x = Conv2DTranspose(filter_count * 4, kernel_size=(3, 3), strides=(2, 2),
                        activation='elu', padding='same', kernel_initializer='he_normal')(x)
    x = Conv2DTranspose(filter_count * 2, kernel_size=kernel_size, strides=strides,
                        activation='elu', padding='same', kernel_initializer='he_normal')(x)
    output_layer = Conv2DTranspose(channels, kernel_size=kernel_size, strides=strides,
                                   activation=activation_func, padding='same', kernel_initializer='he_normal')(x)
    
    return Model(inputs=input_layer, outputs=output_layer)


def create_upsampled_autoencoder(input_shape):
    """UpSampling tabanli simetrik autoencoder (giris/cikis boyutu esit)."""
    channels = get_output_channels()
    filter_count = Config.FILTER_COUNT
    kernel_size = Config.KERNEL_SIZE
    activation_func = Config.ACTIVATION_FUNC

    input_layer = Input(shape=input_shape)

    # Encoder
    x = Conv2D(filter_count, kernel_size=kernel_size, activation='elu',
               padding='same', kernel_initializer='he_normal')(input_layer)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(filter_count * 2, kernel_size=kernel_size, activation='elu',
               padding='same', kernel_initializer='he_normal')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(filter_count * 4, kernel_size=(3, 3), activation='elu',
               padding='same', kernel_initializer='he_normal')(x)

    # Decoder
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(filter_count * 2, kernel_size=kernel_size, activation='elu',
               padding='same', kernel_initializer='he_normal')(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(filter_count, kernel_size=kernel_size, activation='elu',
               padding='same', kernel_initializer='he_normal')(x)

    output_layer = Conv2D(channels, kernel_size=(1, 1), activation=activation_func, padding='same')(x)
    return Model(inputs=input_layer, outputs=output_layer)

def create_advanced_autoencoder(input_shape):
    """Gelismis autoencoder (MaxPooling + Dropout)."""
    channels = get_output_channels()
    filter_count = Config.FILTER_COUNT
    kernel_size = Config.KERNEL_SIZE
    strides = Config.STRIDES
    activation_func = Config.ACTIVATION_FUNC
    dropout_rate = Config.DROPOUT_RATE

    input_layer = Input(shape=input_shape)

    # Encoder
    x = Conv2D(filter_count // 2, kernel_size=kernel_size, strides=strides,
               activation='elu', padding='same')(input_layer)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(dropout_rate)(x)

    x = Conv2D(filter_count, kernel_size=kernel_size, strides=strides,
               activation='elu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(filter_count * 2, kernel_size=kernel_size, strides=strides,
               activation='elu', padding='same')(x)

    # Decoder
    x = Conv2DTranspose(filter_count * 2, kernel_size=kernel_size, strides=strides,
                        activation='elu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Dropout(dropout_rate)(x)

    x = Conv2DTranspose(filter_count, kernel_size=kernel_size, strides=strides,
                        activation='elu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Dropout(dropout_rate)(x)

    x = Conv2DTranspose(filter_count // 2, kernel_size=kernel_size, strides=(1, 1),
                        activation='elu', padding='same')(x)

    output_layer = Conv2DTranspose(channels, kernel_size=kernel_size, strides=strides,
                                   activation=activation_func, padding='same')(x)

    return Model(inputs=input_layer, outputs=output_layer)


def residual_block(x, filters, dropout_rate=0.0):
    """Residual block: gradient akis guclendirme + detay koruma."""
    shortcut = x
    if x.shape[-1] != filters:
        shortcut = Conv2D(filters, (1, 1), padding='same', kernel_initializer='he_normal')(shortcut)

    y = Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    y = BatchNormalization()(y)
    y = Activation('elu')(y)

    y = Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal')(y)
    y = BatchNormalization()(y)

    if dropout_rate > 0:
        y = Dropout(dropout_rate)(y)

    y = Add()([shortcut, y])
    y = Activation('elu')(y)
    return y


def conv_bn_elu(x, filters, kernel_size=(3, 3), dropout_rate=0.0):
    """Conv + BN + ELU blogu."""
    y = Conv2D(
        filters,
        kernel_size,
        padding='same',
        use_bias=False,
        kernel_initializer='he_normal'
    )(x)
    y = BatchNormalization()(y)
    y = Activation('elu')(y)
    if dropout_rate > 0:
        y = Dropout(dropout_rate)(y)
    return y


def align_spatial_to(source_tensor, reference_tensor, name):
    """Kaynak tensörü referansın uzamsal boyutlarına hizalar."""
    return Lambda(
        lambda tensors: tf.image.resize_with_crop_or_pad(
            tensors[0],
            tf.shape(tensors[1])[1],
            tf.shape(tensors[1])[2]
        ),
        name=name
    )([source_tensor, reference_tensor])


def create_unet_residual_model(input_shape):
    """Uydu -> harita stil cevirimi icin oncelikli model (U-Net + residual skip)."""
    channels = get_output_channels()
    base_filters = max(16, int(Config.FILTER_COUNT))
    dropout_rate = Config.DROPOUT_RATE
    activation_func = Config.ACTIVATION_FUNC

    input_layer = Input(shape=input_shape)

    # Encoder path
    e1 = residual_block(input_layer, base_filters)
    d1 = Conv2D(base_filters, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal')(e1)

    e2 = residual_block(d1, base_filters * 2)
    d2 = Conv2D(base_filters * 2, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal')(e2)

    e3 = residual_block(d2, base_filters * 4, dropout_rate=dropout_rate * 0.5)
    d3 = Conv2D(base_filters * 4, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal')(e3)

    bottleneck = residual_block(d3, base_filters * 8, dropout_rate=dropout_rate)

    # Decoder path with skip connections
    u3 = Conv2DTranspose(base_filters * 4, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal')(bottleneck)
    u3 = Concatenate()([u3, e3])
    u3 = residual_block(u3, base_filters * 4, dropout_rate=dropout_rate * 0.5)

    u2 = Conv2DTranspose(base_filters * 2, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal')(u3)
    u2 = Concatenate()([u2, e2])
    u2 = residual_block(u2, base_filters * 2)

    u1 = Conv2DTranspose(base_filters, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal')(u2)
    u1 = Concatenate()([u1, e1])
    u1 = residual_block(u1, base_filters)

    output_layer = Conv2D(channels, (1, 1), activation=activation_func, padding='same')(u1)
    return Model(inputs=input_layer, outputs=output_layer)

def create_gpt_autoencoder(input_shape):
    """GPT tarzi autoencoder (L1 regularization ile)."""
    channels = get_output_channels()
    filter_count = Config.FILTER_COUNT
    kernel_size = Config.KERNEL_SIZE
    l1_reg = Config.L1_REGULARIZATION
    activation_func = Config.ACTIVATION_FUNC
    half_filters = max(1, filter_count // 2)

    input_layer = Input(shape=input_shape)

    # Encoder
    x = Conv2D(half_filters, kernel_size, activation='elu', padding='same',
               activity_regularizer=regularizers.l1(l1_reg))(input_layer)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(0.4)(x)

    x = Conv2D(filter_count, kernel_size, activation='elu', padding='same',
               activity_regularizer=regularizers.l1(l1_reg))(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(0.4)(x)

    x = Conv2D(filter_count * 2, kernel_size, activation='elu', padding='same',
               activity_regularizer=regularizers.l1(l1_reg))(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(0.4)(x)

    # Decoder
    x = Conv2DTranspose(filter_count * 2, kernel_size, activation='elu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Dropout(0.4)(x)

    x = Conv2DTranspose(filter_count, kernel_size, activation='elu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Dropout(0.4)(x)

    x = Conv2DTranspose(half_filters, kernel_size, activation='elu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Dropout(0.4)(x)

    output_layer = Conv2DTranspose(channels, kernel_size, activation=activation_func, padding='same')(x)

    return Model(inputs=input_layer, outputs=output_layer)

def create_gpt_autoencoder_no_reg(input_shape):
    """GPT tarzi autoencoder (regularization olmadan)."""
    channels = get_output_channels()
    filter_count = Config.FILTER_COUNT
    kernel_size = Config.KERNEL_SIZE
    activation_func = Config.ACTIVATION_FUNC

    input_layer = Input(shape=input_shape)

    # Encoder
    x = Conv2D(filter_count, kernel_size, activation='elu', padding='same')(input_layer)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(0.4)(x)

    x = Conv2D(filter_count * 2, kernel_size, activation='elu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(0.4)(x)

    # Decoder
    x = Conv2DTranspose(filter_count * 2, kernel_size, activation='elu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Dropout(0.4)(x)

    x = Conv2DTranspose(filter_count, kernel_size, activation='elu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Dropout(0.4)(x)

    output_layer = Conv2DTranspose(channels, kernel_size, activation=activation_func, padding='same')(x)

    return Model(inputs=input_layer, outputs=output_layer)

def create_deneysel_model(input_shape):
    """Deneysel model (basit ve hızlı)."""
    channels = get_output_channels()
    filter_count = Config.FILTER_COUNT
    kernel_size = Config.KERNEL_SIZE
    activation_func = Config.ACTIVATION_FUNC
    
    input_layer = Input(shape=input_shape)
    
    x = Conv2D(filter_count * 16, kernel_size=(5, 5), strides=(2, 2),
               activation='relu', padding='same', kernel_initializer='he_normal')(input_layer)
    x = Conv2D(filter_count * 8, kernel_size=kernel_size, strides=(1, 1),
               activation='relu', padding='same', kernel_initializer='he_normal')(x)
    
    x = Conv2DTranspose(filter_count * 4, kernel_size=kernel_size, strides=(1, 1),
                        activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = Conv2DTranspose(filter_count * 2, kernel_size=(5, 5), strides=(2, 2),
                        activation='relu', padding='same', kernel_initializer='he_normal')(x)
    
    output_layer = Conv2D(channels, kernel_size=kernel_size, strides=(1, 1),
                          activation=activation_func, padding='same')(x)
    
    return Model(inputs=input_layer, outputs=output_layer)


def create_classic_model(input_shape):
    """Iyilestirilmis klasik autoencoder (bottleneck + skip connection + BN)."""
    channels = get_output_channels()
    base_filters = max(16, int(Config.FILTER_COUNT))
    kernel_size = Config.KERNEL_SIZE
    activation_func = Config.ACTIVATION_FUNC
    dropout_rate = float(np.clip(Config.DROPOUT_RATE, 0.0, 0.6))

    input_layer = Input(shape=input_shape)

    # Encoder
    x = conv_bn_elu(input_layer, base_filters, kernel_size=kernel_size)
    skip1 = conv_bn_elu(
        x,
        base_filters,
        kernel_size=kernel_size,
        dropout_rate=dropout_rate * 0.25
    )

    x = Conv2D(
        base_filters * 2,
        kernel_size=kernel_size,
        strides=(2, 2),
        padding='same',
        use_bias=False,
        kernel_initializer='he_normal'
    )(skip1)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    skip2 = conv_bn_elu(
        x,
        base_filters * 2,
        kernel_size=kernel_size,
        dropout_rate=dropout_rate * 0.5
    )

    x = Conv2D(
        base_filters * 4,
        kernel_size=kernel_size,
        strides=(2, 2),
        padding='same',
        use_bias=False,
        kernel_initializer='he_normal'
    )(skip2)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = conv_bn_elu(
        x,
        base_filters * 4,
        kernel_size=kernel_size,
        dropout_rate=dropout_rate
    )
    x = conv_bn_elu(
        x,
        base_filters * 4,
        kernel_size=kernel_size,
        dropout_rate=dropout_rate * 0.5
    )

    # Decoder
    x = Conv2DTranspose(
        base_filters * 2,
        kernel_size=kernel_size,
        strides=(2, 2),
        padding='same',
        use_bias=False,
        kernel_initializer='he_normal'
    )(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = align_spatial_to(x, skip2, name='classic_align_skip2')
    x = Concatenate()([x, skip2])
    x = conv_bn_elu(
        x,
        base_filters * 2,
        kernel_size=kernel_size,
        dropout_rate=dropout_rate * 0.5
    )

    x = Conv2DTranspose(
        base_filters,
        kernel_size=kernel_size,
        strides=(2, 2),
        padding='same',
        use_bias=False,
        kernel_initializer='he_normal'
    )(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = align_spatial_to(x, skip1, name='classic_align_skip1')
    x = Concatenate()([x, skip1])
    x = conv_bn_elu(
        x,
        base_filters,
        kernel_size=kernel_size,
        dropout_rate=dropout_rate * 0.25
    )
    x = conv_bn_elu(x, base_filters, kernel_size=kernel_size)

    logits = Conv2D(
        channels,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='same',
        kernel_initializer='he_normal'
    )(x)

    # Dinamik giriste tek/çift boyut farklarinda cikisi girisle hizala.
    logits = align_spatial_to(logits, input_layer, name='classic_output_align')

    output_layer = Activation(activation_func)(logits)
    return Model(inputs=input_layer, outputs=output_layer)


def get_model(input_shape):
    """Yapılandırmaya göre model oluşturur veya yükler."""
    
    # Önceden eğitilmiş model varsa yükle
    if Config.PRETRAINED_MODEL is not None:
        if os.path.exists(Config.PRETRAINED_MODEL):
            print(f"Önceden eğitilmiş model yükleniyor: {Config.PRETRAINED_MODEL}")
            return load_model(
                Config.PRETRAINED_MODEL,
                custom_objects={
                    'ssim_loss': ssim_loss,
                    'hybrid_loss': hybrid_loss,
                    'weighted_hybrid_loss': weighted_hybrid_loss,
                    'ssim_metric': ssim_metric
                }
            )
        else:
            print(f"UYARI: Model dosyası bulunamadı: {Config.PRETRAINED_MODEL}")
            print("Sıfırdan model oluşturuluyor...")
    
    # Model tipine göre oluştur
    model_builders = {
        "unet_residual": create_unet_residual_model,
        "autoencoder": create_autoencoder_model,
        "autoencoder_backup": create_autoencoder_model_backup,
        "upsampled": create_upsampled_autoencoder,
        "advanced": create_advanced_autoencoder,
        "gpt": create_gpt_autoencoder,
        "gpt_no_reg": create_gpt_autoencoder_no_reg,
        "deneysel": create_deneysel_model,
        "classic": create_classic_model,
    }
    
    model_type = Config.MODEL_TYPE.lower()
    if model_type not in model_builders:
        raise ValueError(f"Bilinmeyen model tipi: {Config.MODEL_TYPE}. "
                        f"Geçerli tipler: {list(model_builders.keys())}")
    
    print(f"Model oluşturuluyor: {Config.MODEL_TYPE}")
    return model_builders[model_type](input_shape)


def validate_model_output_shape(model, input_shape):
    """Modelin cikis seklinin giris ile uyumlu oldugunu dogrular."""
    output_shape = model.output_shape
    expected_channels = get_output_channels()

    if output_shape[-1] != expected_channels:
        raise ValueError(
            f"Model cikis kanali uyusmuyor. Beklenen={expected_channels}, bulunan={output_shape[-1]}"
        )

    if input_shape[0] is not None and output_shape[1] is not None and input_shape[0] != output_shape[1]:
        raise ValueError(
            f"Model cikis yuksekligi uyusmuyor. Beklenen={input_shape[0]}, bulunan={output_shape[1]}"
        )

    if input_shape[1] is not None and output_shape[2] is not None and input_shape[1] != output_shape[2]:
        raise ValueError(
            f"Model cikis genisligi uyusmuyor. Beklenen={input_shape[1]}, bulunan={output_shape[2]}"
        )


# ==============================================================================
#                              OPTIMIZER
# ==============================================================================

def get_optimizer():
    """Yapılandırmaya göre optimizer döndürür."""
    if Config.OPTIMIZER.lower() == "adam":
        return tf.keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE)
    elif Config.OPTIMIZER.lower() == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=Config.LEARNING_RATE)
    elif Config.OPTIMIZER.lower() == "rmsprop":
        return tf.keras.optimizers.RMSprop(learning_rate=Config.LEARNING_RATE)
    else:
        return Config.OPTIMIZER


# ==============================================================================
#                              CALLBACKS
# ==============================================================================

def get_callbacks():
    """Tüm callback'leri oluşturur."""
    callbacks = []
    timestamp = get_timestamp()
    
    # Checkpoint callback
    if Config.SAVE_CHECKPOINTS:
        checkpoint_path = f'_{timestamp}_model_f{Config.FILTER_COUNT}_k{Config.KERNEL_SIZE[0]}_epoch_{{epoch:05d}}_{Config.ACTIVATION_FUNC}_{Config.STRIDES}_.h5'
        if Config.SAVE_DIR:
            checkpoint_path = os.path.join(Config.SAVE_DIR, checkpoint_path)
        
        checkpoint = ModelCheckpoint(
            checkpoint_path,
            save_freq='epoch',
            save_best_only=False,
            verbose=1
        )
        callbacks.append(checkpoint)
    
    # Periodic save callback
    if Config.PERIODIC_SAVE_STEPS > 0:
        model_prefix = f'_{timestamp}_{Config.FILTER_COUNT}_{Config.KERNEL_SIZE[0]}_{Config.STRIDES}'
        if Config.SAVE_DIR:
            model_prefix = os.path.join(Config.SAVE_DIR, model_prefix)
        
        periodic_save = PeriodicSave(
            save_every=Config.PERIODIC_SAVE_STEPS,
            model_name_prefix=model_prefix
        )
        callbacks.append(periodic_save)
    
    # Early stopping
    if Config.EARLY_STOPPING_PATIENCE > 0:
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=Config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
    
    # Learning rate reduction
    if Config.REDUCE_LR_PATIENCE > 0:
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=Config.REDUCE_LR_FACTOR,
            patience=Config.REDUCE_LR_PATIENCE,
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(reduce_lr)
    
    return callbacks


# ==============================================================================
#                              ANA FONKSİYON
# ==============================================================================

def print_config():
    """Mevcut yapılandırmayı yazdırır."""
    input_channels = get_input_channels()
    output_channels = get_output_channels()
    print("\n" + "=" * 60)
    print("                    YAPILANDIRMA")
    print("=" * 60)
    print(f"  Veri Dizini:        {Config.DATA_DIR}")
    print(f"  Giris Renk Modu:    {Config.COLOR_MODE} ({input_channels} kanal)")
    print(f"  Cikis Renk Modu:    {Config.OUTPUT_COLOR_MODE} ({output_channels} kanal)")
    print(f"  Görüntü Boyutu:     {Config.IMAGE_SIZE}x{Config.IMAGE_SIZE}")
    print(f"  Model Tipi:         {Config.MODEL_TYPE}")
    print(f"  Önceden Eğitilmiş:  {Config.PRETRAINED_MODEL or 'Yok'}")
    print(f"  Aktivasyon:         {Config.ACTIVATION_FUNC}")
    print(f"  Filtre Sayısı:      {Config.FILTER_COUNT}")
    print(f"  Kernel Boyutu:      {Config.KERNEL_SIZE}")
    print(f"  Stride:             {Config.STRIDES}")
    print(f"  Batch Size:         {Config.BATCH_SIZE}")
    print(f"  Epochs:             {Config.EPOCHS}")
    print(f"  Learning Rate:      {Config.LEARNING_RATE}")
    print(f"  Optimizer:          {Config.OPTIMIZER}")
    print(f"  Loss:               {Config.LOSS_FUNCTION}")
    print(f"  Train/Val Split:    {Config.TRAIN_SPLIT}/{1-Config.TRAIN_SPLIT}")
    print(f"  Random Seed:        {Config.RANDOM_SEED}")
    print("=" * 60 + "\n")


def main():
    """Ana eğitim fonksiyonu."""
    
    # Yapılandırmayı yazdır
    print_config()
    
    # Kayıt dizinini oluştur
    if Config.SAVE_DIR and not os.path.exists(Config.SAVE_DIR):
        os.makedirs(Config.SAVE_DIR)
        print(f"Kayıt dizini oluşturuldu: {Config.SAVE_DIR}")
    
    # Veri setlerini oluştur
    print("Veri setleri oluşturuluyor...")
    train_dataset, val_dataset = create_datasets()
    
    # Model oluştur
    input_shape = get_input_shape()
    print(f"Giriş boyutu: {input_shape}")
    
    model = get_model(input_shape)
    validate_model_output_shape(model, input_shape)

    # Model derle
    optimizer = get_optimizer()
    loss_fn = get_loss_function()
    
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=get_metrics())
    
    # Modelin özetini yazdır
    model.summary()
    
    # Callback'leri al
    callbacks = get_callbacks()
    
    # Eğitimi başlat
    print("\n" + "=" * 60)
    print("                    EĞİTİM BAŞLIYOR")
    print("=" * 60 + "\n")
    
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=Config.EPOCHS,
        callbacks=callbacks
    )
    
    # Final modeli kaydet
    final_model_path = Config.FINAL_MODEL_NAME
    if Config.SAVE_DIR:
        final_model_path = os.path.join(Config.SAVE_DIR, final_model_path)
    
    model.save(final_model_path)
    print(f"\nFinal model kaydedildi: {final_model_path}")
    
    print("\n" + "=" * 60)
    print("                    EĞİTİM TAMAMLANDI")
    print("=" * 60 + "\n")
    
    return model, history


if __name__ == "__main__":
    main()


