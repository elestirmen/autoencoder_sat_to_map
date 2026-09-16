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
import json
import time
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
    
    # ------------------------------ VE AYARLARI -----------------------------
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
    # "advanced", "gpt", "gpt_no_reg", "deneysel", "classic",
    # "flat_bottleneck"  <- benchmark'ta 0.70 success_25m alan mimari sinifi
    MODEL_TYPE = "classic"
    
    # Önceden eğitilmiş model yükle (None = sıfırdan başla)
    PRETRAINED_MODEL = r"C:\d_surucusu\visual_navigation\autoencoder_sat_to_map\ciktilar\_27_08_2026__16_59_model_f32_k3_epoch_00011_elu_(1, 1)_.h5"  # Örn: "son_model.h5" veya None
    
    # Gizli katman aktivasyonu: "sigmoid", "relu", "elu", "tanh"
    HIDDEN_ACTIVATION = "elu"

    # Cikis aktivasyonu: "tanh" veya "sigmoid".
    # Etiket normalizasyonu buna gore secilir ve tutarlilik
    # validate_activation_contract() tarafindan fail-fast dogrulanir:
    #   tanh    -> etiketler [-1, 1]
    #   sigmoid -> etiketler [ 0, 1]
    # Benchmark'ta 0.70 success_25m alan 2022 modelleri sigmoid ciktiliydi;
    # bu yuzden sigmoid desteklenir ve yasaklanmaz.
    OUTPUT_ACTIVATION = "tanh"

    # Giris normalizasyonu: "minus1_1" veya "zero_1".
    # Benchmark cikarimi varsayilan olarak minus1_1 besler
    # (run_urgup_cevresi_models.py), bu yuzden egitim de varsayilan olarak
    # minus1_1 kullanir; boylece egitim ve cikarim girisi ayni olur.
    INPUT_NORMALIZATION = "minus1_1"
    
    # Filtre sayısı (başlangıç filtre sayısı)
    FILTER_COUNT = 32
    
    # Kernel boyutu
    KERNEL_SIZE = (3, 3)
    
    # Stride değeri
    STRIDES = (1, 1)

    # flat_bottleneck mimarisinin daralma derinligi.
    # 3 -> f, f/2, f/4 (2022'de kazanan modellerin yapisi).
    FLAT_DEPTH = 3
    
    # ------------------------------ EM AYARLARI ---------------------------
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
    
    # ------------------------------ VE PIPELINE AYARLARI --------------------
    # Paralel işlem sayısı (None = AUTOTUNE)
    NUM_PARALLEL_CALLS = None
    
    # Prefetch buffer boyutu (None = AUTOTUNE)
    PREFETCH_BUFFER = None
    
    # Dosya doğrulama modu:
    # "none": Doğrulama yapma (en hızlı, ignore_errors ile hatalar atlanır)
    # "quick": Sadece dosya boyutu kontrolü (hızlı)
    # "cached":lk sefer doğrula, sonucu cache'le (400k dosya için önerilen)
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

    # ------------------------------ GELŞ AYARLAR --------------------------
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
        f"Gecersiz OUTPUT_COLOR_MODE: {Config.OUTPUT_COLOR_MODE}. "
        "Gecerli degerler: auto, grayscale, rgb"
    )


# Cikis aktivasyonu -> (etiket alt siniri, etiket ust siniri).
# Kayip fonksiyonlari ve etiket normalizasyonu bu tablodan turetilir; boylece
# aktivasyon ile etiket araligi hicbir zaman birbirinden ayrisamaz.
OUTPUT_RANGES = {
    "tanh": (-1.0, 1.0),
    "sigmoid": (0.0, 1.0),
}

INPUT_NORMALIZATIONS = ("minus1_1", "zero_1")


def validate_activation_contract():
    """Gizli/cikis aktivasyonlarini veri normalizasyonu ile uyumlu dogrula."""
    supported_hidden = {"sigmoid", "relu", "elu", "tanh"}
    hidden = str(Config.HIDDEN_ACTIVATION).lower()
    output = str(Config.OUTPUT_ACTIVATION).lower()
    if hidden not in supported_hidden:
        raise ValueError(
            f"Desteklenmeyen gizli katman aktivasyonu: {Config.HIDDEN_ACTIVATION}. "
            f"Gecerli degerler: {sorted(supported_hidden)}"
        )
    if output not in OUTPUT_RANGES:
        raise ValueError(
            f"Desteklenmeyen cikis aktivasyonu: {Config.OUTPUT_ACTIVATION!r}. "
            f"Gecerli degerler: {sorted(OUTPUT_RANGES)}"
        )
    input_norm = str(getattr(Config, "INPUT_NORMALIZATION", "minus1_1")).lower()
    if input_norm not in INPUT_NORMALIZATIONS:
        raise ValueError(
            f"Desteklenmeyen giris normalizasyonu: {input_norm!r}. "
            f"Gecerli degerler: {list(INPUT_NORMALIZATIONS)}"
        )
    return hidden, output


def get_target_range():
    """Cikis aktivasyonunun gerektirdigi etiket araligini dondur."""
    return OUTPUT_RANGES[validate_activation_contract()[1]]


def get_input_normalization():
    validate_activation_contract()
    return str(getattr(Config, "INPUT_NORMALIZATION", "minus1_1")).lower()


def normalize_to_range(pixels, low, high):
    """[0, 255] float goruntuyu [low, high] araligina tasi."""
    return pixels * ((high - low) / 255.0) + low


def preprocessing_contract():
    """Modelin eğitildiği girdi ön işlemesini sözlük olarak döndürür.

    Bu sözlük her model dosyasının yanına "<model>.pre.json" olarak yazılır.
    Çıkarım tarafı (src/goruntu_islemleri.py ve template-matching) bu dosyayı
    okuyarak aynı ön işlemeyi uygular. Model dosyasının kendisinden giriş
    normalizasyonu OKUNAMAZ: çıkış aktivasyonu yalnızca çıkış aralığını söyler.
    """
    return {
        "input_normalization": get_input_normalization(),
        # Eğitimde histogram eşitleme yalnızca bu renk modunda uygulanır
        # (bkz. load_and_preprocess). tfa.image.equalize, PIL ImageOps.equalize'in
        # portudur; OpenCV'nin equalizeHist'i DEĞİLDİR. Çıkarım tarafındaki
        # karşılığı bu yüzden "hist_eq" değil "hist_eq_pil"dir.
        "enhancement": "hist_eq_pil" if Config.COLOR_MODE == "grayscale_with_equalize" else "none",
        "color_mode": "rgb" if get_input_channels() == 3 else "grayscale",
        "image_size": int(Config.IMAGE_SIZE),
        "output_activation": Config.OUTPUT_ACTIVATION,
        "output_color_mode": "rgb" if get_output_channels() == 3 else "grayscale",
        "trained_by": "autoencoder_unified.py",
    }


def write_preprocessing_sidecar(model_path, quiet=False):
    """Model dosyasının yanına ön işleme sözleşmesini yazar; yolunu döndürür."""
    sidecar_path = os.path.splitext(str(model_path))[0] + ".pre.json"
    try:
        with open(sidecar_path, "w", encoding="utf-8") as f:
            json.dump(preprocessing_contract(), f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[UYARI] Ön işleme sözleşmesi yazılamadı ({sidecar_path}): {e}")
        return None
    if not quiet:
        print(f"[Sözleşme] {sidecar_path}")
    return sidecar_path


def normalize_input(pixels):
    """Giris goruntusunu INPUT_NORMALIZATION ayarina gore olcekle.

    Cikarim tarafinda benchmark ayni islemi goruntu_islemleri.
    apply_normalization() ile yapar; iki tarafin ayni kalmasi sarttir.
    """
    if get_input_normalization() == "zero_1":
        return normalize_to_range(pixels, 0.0, 1.0)
    return normalize_to_range(pixels, -1.0, 1.0)


def get_hidden_activation():
    return validate_activation_contract()[0]


def get_output_activation():
    return validate_activation_contract()[1]


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
#                              LOSS FONKYONLARI
# ==============================================================================

def clip_to_target_range(tensor):
    low, high = get_target_range()
    return tf.clip_by_value(tensor, low, high)


def target_dynamic_range():
    """tf.image.ssim'in max_val parametresi: etiket araliginin genisligi."""
    low, high = get_target_range()
    return high - low


def rescaled_threshold(value):
    """[-1, 1] icin yazilmis bir esigi gecerli etiket araligina tasi.

    WEIGHTED_* esikleri tarihsel olarak [-1, 1] varsayimiyla yazildi. Cikis
    sigmoid oldugunda ayni goreli parlaklik seviyesine karsilik gelmeleri icin
    yeniden olceklenirler; aksi halde -0.6 gibi bir esik [0, 1] araliginda
    hicbir pikseli secmez ve agirliklandirma sessizce devre disi kalir.
    """
    low, high = get_target_range()
    if (low, high) == (-1.0, 1.0):
        return value
    return low + (value + 1.0) * (high - low) / 2.0


def ssim_loss(y_true, y_pred):
    """SSIM tabanli kayip fonksiyonu."""
    y_true = clip_to_target_range(y_true)
    y_pred = clip_to_target_range(y_pred)
    return 1 - tf.reduce_mean(
        tf.image.ssim(y_true, y_pred, target_dynamic_range())
    )


def hybrid_loss(y_true, y_pred):
    """L1 + SSIM karisik kayip (daha net cizgiler icin)."""
    y_true = clip_to_target_range(y_true)
    y_pred = clip_to_target_range(y_pred)
    l1 = tf.reduce_mean(tf.abs(y_true - y_pred))
    ssim = 1 - tf.reduce_mean(
        tf.image.ssim(y_true, y_pred, target_dynamic_range())
    )
    return 0.8 * l1 + 0.2 * ssim


def weighted_hybrid_loss(y_true, y_pred):
    """L1 + SSIM (etikette koyu detaylara agirlik vererek)."""
    y_true = clip_to_target_range(y_true)
    y_pred = clip_to_target_range(y_pred)

    # Etikette beyaz agirlik baskin oldugunda, koyu/icerik piksellerine daha fazla agirlik ver.
    non_white_mask = tf.cast(
        y_true <= rescaled_threshold(Config.WEIGHTED_NON_WHITE_THRESHOLD), tf.float32
    )
    dark_mask = tf.cast(
        y_true <= rescaled_threshold(Config.WEIGHTED_DARK_THRESHOLD), tf.float32
    )

    weights = tf.ones_like(y_true)
    weights += non_white_mask * (Config.WEIGHTED_NON_WHITE_WEIGHT - 1.0)
    weights += dark_mask * (Config.WEIGHTED_DARK_WEIGHT - Config.WEIGHTED_NON_WHITE_WEIGHT)

    abs_err = tf.abs(y_true - y_pred)
    weighted_l1 = tf.reduce_sum(abs_err * weights) / (tf.reduce_sum(weights) + K.epsilon())

    ssim = 1 - tf.reduce_mean(
        tf.image.ssim(y_true, y_pred, target_dynamic_range())
    )
    return Config.WEIGHTED_HYBRID_L1 * weighted_l1 + Config.WEIGHTED_HYBRID_SSIM * ssim


def ssim_metric(y_true, y_pred):
    """Takip metrigi: SSIM (1'e yaklastikca daha iyi)."""
    y_true = clip_to_target_range(y_true)
    y_pred = clip_to_target_range(y_pred)
    return tf.reduce_mean(
        tf.image.ssim(y_true, y_pred, target_dynamic_range())
    )


def get_metrics():
    """Egitim metrikleri."""
    return [
        tf.keras.metrics.MeanAbsoluteError(name='mae'),
        tf.keras.metrics.MeanSquaredError(name='mse'),
        ssim_metric,
    ]

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
            write_preprocessing_sidecar(save_path, quiet=True)
            print(f"\n[PeriodicSave] Model kaydedildi: {save_path}")

    def on_epoch_end(self, epoch, logs=None):
        save_path = f'{self.model_name_prefix}_epoch_{epoch}.h5'
        self.model.save(save_path)
        write_preprocessing_sidecar(save_path, quiet=True)


class PreprocessingSidecarWriter(tf.keras.callbacks.Callback):
    """Eğitim sırasında oluşan her model dosyasının yanına sözleşmeyi yazar.

    ModelCheckpoint dosya adını kendi içinde ({epoch:05d} gibi) çözdüğü için
    yol kalıbını dışarıdan bilmek güvenilir değildir; bunun yerine kayıt klasörü
    taranır ve eğitim başladıktan SONRA oluşmuş, sözleşmesi olmayan model
    dosyaları için sözleşme yazılır. Eski modellere dokunulmaz.
    """

    def __init__(self, save_dir=None):
        super(PreprocessingSidecarWriter, self).__init__()
        self.save_dir = save_dir or "."
        self.started_at = time.time()

    def _sweep(self):
        try:
            names = os.listdir(self.save_dir)
        except Exception:
            return
        for name in names:
            if not name.endswith((".h5", ".keras")):
                continue
            path = os.path.join(self.save_dir, name)
            sidecar = os.path.splitext(path)[0] + ".pre.json"
            if os.path.exists(sidecar):
                continue
            try:
                # Eğitimden önce var olan modellere sözleşme yazma: onların
                # hangi ön işlemeyle eğitildiğini bilmiyoruz.
                if os.path.getmtime(path) < self.started_at - 1:
                    continue
            except OSError:
                continue
            write_preprocessing_sidecar(path, quiet=True)

    def on_epoch_end(self, epoch, logs=None):
        self._sweep()

    def on_train_end(self, logs=None):
        self._sweep()


# ==============================================================================
#                              VERİ KLEME
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

    # Normalizasyon: giris INPUT_NORMALIZATION'a, etiket cikis
    # aktivasyonunun gerektirdigi aralia gore ayri ayri olceklenir.
    input_img = normalize_input(input_img)
    label_low, label_high = get_target_range()
    label_img = normalize_to_range(label_img, label_low, label_high)

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

    # Normalizasyon (bkz. load_and_preprocess)
    input_img = normalize_input(input_img)
    label_low, label_high = get_target_range()
    label_img = normalize_to_range(label_img, label_low, label_high)

    return (input_img, label_img)


def keep_informative_tile(input_img, label_img):
    """Hedefte tamamen/neredeyse tamamen bos (beyaz) karolari eler."""
    del input_img  # Sadece hedef etiket dagilimi kullanilir.

    low, high = get_target_range()
    label_img = tf.clip_by_value(label_img, low, high)
    label_u8 = (label_img - low) * (255.0 / (high - low))
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
        print(f"? Doğrulama sonuçları cache'lendi: {cache_path}")
    except Exception as e:
        print(f"?? Cache kaydedilemedi: {e}")


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
        print(f"?? {invalid_count} boş/erişilemeyen dosya atlandı")
    
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
            print(f" şleniyor: {i}/{total} ({i*100//total}%)")
        
        try:
            file_size = os.path.getsize(path)
            if file_size < 100:  # 100 byte'tan küçük dosyalar şüpheli
                invalid_count += 1
                continue
            valid_paths.append(path)
        except:
            invalid_count += 1
    
    if invalid_count > 0:
        print(f"?? {invalid_count} bozuk/boş dosya atlandı")
    
    return valid_paths


def get_validated_paths(all_image_paths):
    """Doğrulama moduna göre geçerli dosyaları döndürür."""
    mode = Config.VALIDATE_FILES.lower()
    
    if mode == "none":
        print("?? Dosya doğrulama: DEVRE DI (ignore_errors aktif)")
        return all_image_paths
    
    elif mode == "quick":
        print("?? Dosya doğrulama: HIZLI (sadece boyut kontrolü)")
        return validate_quick(all_image_paths)
    
    elif mode == "cached":
        print("?? Dosya doğrulama: CACHE MODUNDA")
        cache = load_validation_cache()
        dataset_hash = compute_dataset_hash(all_image_paths)
        
        if cache and cache.get("hash") == dataset_hash:
            print(f"? Cache'den yüklendi: {cache['count']} geçerli dosya")
            # Cache'deki dosyaların hala var olduğunu kontrol et
            valid_from_cache = [p for p in cache["valid_files"] if os.path.exists(p)]
            if len(valid_from_cache) == cache['count']:
                return valid_from_cache
            print("?? Bazı dosyalar silinmiş, yeniden doğrulanıyor...")
        
        # Cache yoksa veya geçersizse, hızlı doğrulama yap ve cache'le
        print("?? İlk kez doğrulama yapılıyor (bu sadece bir kez olacak)...")
        valid_paths = validate_quick(all_image_paths)
        save_validation_cache(valid_paths, dataset_hash)
        return valid_paths
    
    elif mode == "full":
        print("?? Dosya doğrulama: TAM (her dosya kontrol ediliyor)")
        return validate_full(all_image_paths)
    
    else:
        print(f"?? Bilinmeyen doğrulama modu: {mode}, 'none' kullanılıyor")
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
#                              MODEL MALE
# ==============================================================================

def create_autoencoder_model(input_shape):
    """Standart autoencoder modeli."""
    channels = get_output_channels()
    filter_count = Config.FILTER_COUNT
    kernel_size = Config.KERNEL_SIZE
    strides = Config.STRIDES
    hidden_activation = get_hidden_activation()
    output_activation = get_output_activation()
    
    input_layer = Input(shape=input_shape)
    
    # Encoder
    x = Conv2D(filter_count, kernel_size=kernel_size, strides=strides,
               activation=hidden_activation, padding='same', kernel_initializer='he_normal')(input_layer)
    x = Conv2D(filter_count * 2, kernel_size=kernel_size, strides=strides,
               activation=hidden_activation, padding='same', kernel_initializer='he_normal')(x)
    x = Conv2D(filter_count * 4, kernel_size=kernel_size, strides=strides,
               activation=hidden_activation, padding='same', kernel_initializer='he_normal')(x)
    
    # Decoder
    x = Conv2DTranspose(filter_count * 4, kernel_size=kernel_size, strides=strides,
                        activation=hidden_activation, padding='same', kernel_initializer='he_normal')(x)
    x = Conv2DTranspose(filter_count * 2, kernel_size=kernel_size, strides=strides,
                        activation=hidden_activation, padding='same', kernel_initializer='he_normal')(x)
    x = Conv2DTranspose(filter_count, kernel_size=kernel_size, strides=strides,
                        activation=hidden_activation, padding='same', kernel_initializer='he_normal')(x)
    
    output_layer = Conv2D(channels, kernel_size=kernel_size, strides=(1, 1),
                          activation=output_activation, padding='same')(x)
    
    return Model(inputs=input_layer, outputs=output_layer)


def create_autoencoder_model_backup(input_shape):
    """Backup autoencoder modeli (alternatif mimari)."""
    channels = get_output_channels()
    filter_count = Config.FILTER_COUNT
    kernel_size = Config.KERNEL_SIZE
    strides = Config.STRIDES
    hidden_activation = get_hidden_activation()
    output_activation = get_output_activation()
    
    input_layer = Input(shape=input_shape)
    
    x = Conv2D(filter_count, kernel_size=kernel_size, strides=strides,
               activation=hidden_activation, padding='same', kernel_initializer='he_normal')(input_layer)
    x = Conv2D(filter_count * 2, kernel_size=kernel_size, strides=strides,
               activation=hidden_activation, padding='same', kernel_initializer='he_normal')(x)
    x = Conv2D(filter_count * 4, kernel_size=(3, 3), strides=(1, 1),
               activation=hidden_activation, padding='same', kernel_initializer='he_normal')(x)
    x = Conv2D(filter_count * 8, kernel_size=(3, 3), strides=(2, 2),
               activation=hidden_activation, padding='same', kernel_initializer='he_normal')(x)
    
    x = Conv2DTranspose(filter_count * 4, kernel_size=(3, 3), strides=(2, 2),
                        activation=hidden_activation, padding='same', kernel_initializer='he_normal')(x)
    x = Conv2DTranspose(filter_count * 2, kernel_size=kernel_size, strides=strides,
                        activation=hidden_activation, padding='same', kernel_initializer='he_normal')(x)
    output_layer = Conv2DTranspose(channels, kernel_size=kernel_size, strides=strides,
                                   activation=output_activation, padding='same', kernel_initializer='he_normal')(x)
    
    return Model(inputs=input_layer, outputs=output_layer)


def create_upsampled_autoencoder(input_shape):
    """UpSampling tabanli simetrik autoencoder (giris/cikis boyutu esit)."""
    channels = get_output_channels()
    filter_count = Config.FILTER_COUNT
    kernel_size = Config.KERNEL_SIZE
    hidden_activation = get_hidden_activation()
    output_activation = get_output_activation()

    input_layer = Input(shape=input_shape)

    # Encoder
    x = Conv2D(filter_count, kernel_size=kernel_size, activation=hidden_activation,
               padding='same', kernel_initializer='he_normal')(input_layer)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(filter_count * 2, kernel_size=kernel_size, activation=hidden_activation,
               padding='same', kernel_initializer='he_normal')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(filter_count * 4, kernel_size=(3, 3), activation=hidden_activation,
               padding='same', kernel_initializer='he_normal')(x)

    # Decoder
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(filter_count * 2, kernel_size=kernel_size, activation=hidden_activation,
               padding='same', kernel_initializer='he_normal')(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(filter_count, kernel_size=kernel_size, activation=hidden_activation,
               padding='same', kernel_initializer='he_normal')(x)

    output_layer = Conv2D(channels, kernel_size=(1, 1), activation=output_activation, padding='same')(x)
    return Model(inputs=input_layer, outputs=output_layer)

def create_advanced_autoencoder(input_shape):
    """Gelismis autoencoder (MaxPooling + Dropout)."""
    channels = get_output_channels()
    filter_count = Config.FILTER_COUNT
    kernel_size = Config.KERNEL_SIZE
    strides = Config.STRIDES
    hidden_activation = get_hidden_activation()
    output_activation = get_output_activation()
    dropout_rate = Config.DROPOUT_RATE

    input_layer = Input(shape=input_shape)

    # Encoder
    x = Conv2D(filter_count // 2, kernel_size=kernel_size, strides=strides,
               activation=hidden_activation, padding='same')(input_layer)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(dropout_rate)(x)

    x = Conv2D(filter_count, kernel_size=kernel_size, strides=strides,
               activation=hidden_activation, padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(filter_count * 2, kernel_size=kernel_size, strides=strides,
               activation=hidden_activation, padding='same')(x)

    # Decoder
    x = Conv2DTranspose(filter_count * 2, kernel_size=kernel_size, strides=strides,
                        activation=hidden_activation, padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Dropout(dropout_rate)(x)

    x = Conv2DTranspose(filter_count, kernel_size=kernel_size, strides=strides,
                        activation=hidden_activation, padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Dropout(dropout_rate)(x)

    x = Conv2DTranspose(filter_count // 2, kernel_size=kernel_size, strides=(1, 1),
                        activation=hidden_activation, padding='same')(x)

    output_layer = Conv2DTranspose(channels, kernel_size=kernel_size, strides=strides,
                                   activation=output_activation, padding='same')(x)

    return Model(inputs=input_layer, outputs=output_layer)


def residual_block(x, filters, dropout_rate=0.0):
    """Residual block: gradient akis guclendirme + detay koruma."""
    shortcut = x
    if x.shape[-1] != filters:
        shortcut = Conv2D(filters, (1, 1), padding='same', kernel_initializer='he_normal')(shortcut)

    y = Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    y = BatchNormalization()(y)
    y = Activation(get_hidden_activation())(y)

    y = Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal')(y)
    y = BatchNormalization()(y)

    if dropout_rate > 0:
        y = Dropout(dropout_rate)(y)

    y = Add()([shortcut, y])
    y = Activation(get_hidden_activation())(y)
    return y


def conv_bn_elu(x, filters, kernel_size=(3, 3), dropout_rate=0.0):
    """Conv + BN + yapilandirilabilir gizli aktivasyon blogu."""
    y = Conv2D(
        filters,
        kernel_size,
        padding='same',
        use_bias=False,
        kernel_initializer='he_normal'
    )(x)
    y = BatchNormalization()(y)
    y = Activation(get_hidden_activation())(y)
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
    output_activation = get_output_activation()

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

    output_layer = Conv2D(channels, (1, 1), activation=output_activation, padding='same')(u1)
    return Model(inputs=input_layer, outputs=output_layer)

def create_gpt_autoencoder(input_shape):
    """GPT tarzi autoencoder (L1 regularization ile)."""
    channels = get_output_channels()
    filter_count = Config.FILTER_COUNT
    kernel_size = Config.KERNEL_SIZE
    l1_reg = Config.L1_REGULARIZATION
    hidden_activation = get_hidden_activation()
    output_activation = get_output_activation()
    half_filters = max(1, filter_count // 2)

    input_layer = Input(shape=input_shape)

    # Encoder
    x = Conv2D(half_filters, kernel_size, activation=hidden_activation, padding='same',
               activity_regularizer=regularizers.l1(l1_reg))(input_layer)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(0.4)(x)

    x = Conv2D(filter_count, kernel_size, activation=hidden_activation, padding='same',
               activity_regularizer=regularizers.l1(l1_reg))(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(0.4)(x)

    x = Conv2D(filter_count * 2, kernel_size, activation=hidden_activation, padding='same',
               activity_regularizer=regularizers.l1(l1_reg))(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(0.4)(x)

    # Decoder
    x = Conv2DTranspose(filter_count * 2, kernel_size, activation=hidden_activation, padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Dropout(0.4)(x)

    x = Conv2DTranspose(filter_count, kernel_size, activation=hidden_activation, padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Dropout(0.4)(x)

    x = Conv2DTranspose(half_filters, kernel_size, activation=hidden_activation, padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Dropout(0.4)(x)

    output_layer = Conv2DTranspose(channels, kernel_size, activation=output_activation, padding='same')(x)

    return Model(inputs=input_layer, outputs=output_layer)

def create_gpt_autoencoder_no_reg(input_shape):
    """GPT tarzi autoencoder (regularization olmadan)."""
    channels = get_output_channels()
    filter_count = Config.FILTER_COUNT
    kernel_size = Config.KERNEL_SIZE
    hidden_activation = get_hidden_activation()
    output_activation = get_output_activation()

    input_layer = Input(shape=input_shape)

    # Encoder
    x = Conv2D(filter_count, kernel_size, activation=hidden_activation, padding='same')(input_layer)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(0.4)(x)

    x = Conv2D(filter_count * 2, kernel_size, activation=hidden_activation, padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(0.4)(x)

    # Decoder
    x = Conv2DTranspose(filter_count * 2, kernel_size, activation=hidden_activation, padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Dropout(0.4)(x)

    x = Conv2DTranspose(filter_count, kernel_size, activation=hidden_activation, padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Dropout(0.4)(x)

    output_layer = Conv2DTranspose(channels, kernel_size, activation=output_activation, padding='same')(x)

    return Model(inputs=input_layer, outputs=output_layer)

def create_deneysel_model(input_shape):
    """Deneysel model (basit ve hızlı)."""
    channels = get_output_channels()
    filter_count = Config.FILTER_COUNT
    kernel_size = Config.KERNEL_SIZE
    hidden_activation = get_hidden_activation()
    output_activation = get_output_activation()
    
    input_layer = Input(shape=input_shape)
    
    x = Conv2D(filter_count * 16, kernel_size=(5, 5), strides=(2, 2),
               activation=hidden_activation, padding='same', kernel_initializer='he_normal')(input_layer)
    x = Conv2D(filter_count * 8, kernel_size=kernel_size, strides=(1, 1),
               activation=hidden_activation, padding='same', kernel_initializer='he_normal')(x)
    
    x = Conv2DTranspose(filter_count * 4, kernel_size=kernel_size, strides=(1, 1),
                        activation=hidden_activation, padding='same', kernel_initializer='he_normal')(x)
    x = Conv2DTranspose(filter_count * 2, kernel_size=(5, 5), strides=(2, 2),
                        activation=hidden_activation, padding='same', kernel_initializer='he_normal')(x)
    
    output_layer = Conv2D(channels, kernel_size=kernel_size, strides=(1, 1),
                          activation=output_activation, padding='same')(x)
    
    return Model(inputs=input_layer, outputs=output_layer)


def create_flat_bottleneck_model(input_shape):
    """Tam cozunurlukte, daralan, sig conv yigini. Benchmark'ta kazanan mimari.

    arsiv/autoencoder_gpu_froom_kaggle.py:163-171 mimarisinin parametrize
    edilmis hali. top_modeller/ altindaki 0.68-0.70 success_25m alan modeller
    bu yapiyla uretildi (dogrulama: parametre sayilari birebir uyusuyor,
    ornegin f48/k4 icin 816 / 18456 / 4620 ...).

    Kritik ozellikler ve NEDEN oyle olmalari gerektigi:

    - Uzamsal indirme YOK. Her katman girisle ayni cozunurlukte kalir
      (STRIDES=(1,1), pooling yok). Sablon eslesmesi piksel duzeyinde
      karsiliklilik ister; indirme + geri yukseltme bu karsilikliligi bozar.
    - Kanal sayisi DARALIR (f -> f/2 -> f/4), genislemez. Darboga bilgi
      miktarini sinirlar ve agin semantik harita cizmesini engeller.
    - Skip connection ve BatchNorm YOK. Alici alan kucuk kalir
      (kernel k, 6 katman -> ~1 + 6*(k-1) piksel), boylece ag ancak yerel
      doku/kenar donusumu ogrenebilir.

    Olculen sonuc: bu sinif kaynaklar-arasi yanlis-tepe oranini 0.62'den
    0.32'ye indirir. Buna karsilik indirme yapan derin create_classic_model
    ayni orani 2.9'a cikarir, yani ham goruntuden kotu hale gelir.
    """
    channels = get_output_channels()
    filter_count = max(4, int(Config.FILTER_COUNT))
    kernel_size = Config.KERNEL_SIZE
    strides = Config.STRIDES
    hidden_activation = get_hidden_activation()
    output_activation = get_output_activation()

    if tuple(strides) != (1, 1):
        raise ValueError(
            "flat_bottleneck mimarisi tam cozunurluk gerektirir; "
            f"STRIDES=(1, 1) olmalidir, bulunan={tuple(strides)}."
        )

    # Daralma zinciri: f, f/2, f/4, ... Config.FLAT_DEPTH kadar.
    depth = max(1, int(getattr(Config, "FLAT_DEPTH", 3)))
    widths = []
    for level in range(depth):
        width = filter_count // (2 ** level)
        if width < 1:
            raise ValueError(
                f"FILTER_COUNT={filter_count} bu derinlik icin fazla kucuk "
                f"(FLAT_DEPTH={depth}); daralma zinciri sifira dusuyor."
            )
        widths.append(width)

    input_layer = Input(shape=input_shape)
    x = input_layer

    # Encoder: daralan conv yigini.
    for width in widths:
        x = Conv2D(
            width,
            kernel_size=kernel_size,
            strides=(1, 1),
            activation=hidden_activation,
            padding='same',
            kernel_initializer='he_normal',
        )(x)

    # Decoder: ayni genislikler ters sirada, transpose conv ile.
    for width in reversed(widths):
        x = Conv2DTranspose(
            width,
            kernel_size=kernel_size,
            strides=(1, 1),
            activation=hidden_activation,
            padding='same',
            kernel_initializer='he_normal',
        )(x)

    output_layer = Conv2D(
        channels,
        kernel_size=kernel_size,
        strides=(1, 1),
        activation=output_activation,
        padding='same',
        name='model_output',
    )(x)

    return Model(inputs=input_layer, outputs=output_layer)


def create_classic_model(input_shape):
    """Iyilestirilmis klasik autoencoder (bottleneck + skip connection + BN)."""
    channels = get_output_channels()
    base_filters = max(16, int(Config.FILTER_COUNT))
    kernel_size = Config.KERNEL_SIZE
    hidden_activation = get_hidden_activation()
    output_activation = get_output_activation()
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
    x = Activation(hidden_activation)(x)
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
    x = Activation(hidden_activation)(x)
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
    x = Activation(hidden_activation)(x)
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
    x = Activation(hidden_activation)(x)
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

    output_layer = Activation(output_activation, name='model_output_tanh')(logits)
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
        "flat_bottleneck": create_flat_bottleneck_model,
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
        checkpoint_path = (
            f'_{timestamp}_model_f{Config.FILTER_COUNT}_k{Config.KERNEL_SIZE[0]}'
            f'_epoch_{{epoch:05d}}_h{Config.HIDDEN_ACTIVATION}'
            f'_o{Config.OUTPUT_ACTIVATION}_{Config.STRIDES}_.h5'
        )
        if Config.SAVE_DIR:
            checkpoint_path = os.path.join(Config.SAVE_DIR, checkpoint_path)
        
        checkpoint = ModelCheckpoint(
            checkpoint_path,
            save_freq='epoch',
            save_best_only=False,
            verbose=1
        )
        callbacks.append(checkpoint)

    # Best validation loss Callback
    loss_checkpoint_path = f'_{timestamp}_best_val_loss_model_f{Config.FILTER_COUNT}_k{Config.KERNEL_SIZE[0]}.h5'
    if Config.SAVE_DIR:
        loss_checkpoint_path = os.path.join(Config.SAVE_DIR, loss_checkpoint_path)
    loss_checkpoint = ModelCheckpoint(
        loss_checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1,
    )
    callbacks.append(loss_checkpoint)

    # Best MAE Callback
    mae_checkpoint_path = f'_{timestamp}_best_mae_model_f{Config.FILTER_COUNT}_k{Config.KERNEL_SIZE[0]}.h5'
    if Config.SAVE_DIR:
        mae_checkpoint_path = os.path.join(Config.SAVE_DIR, mae_checkpoint_path)
    mae_checkpoint = ModelCheckpoint(
        mae_checkpoint_path,
        monitor='val_mae',
        save_best_only=True,
        mode='min',
        verbose=1,
    )
    callbacks.append(mae_checkpoint)

    # Best SSIM Callback
    ssim_checkpoint_path = f'_{timestamp}_best_ssim_model_f{Config.FILTER_COUNT}_k{Config.KERNEL_SIZE[0]}.h5'
    if Config.SAVE_DIR:
        ssim_checkpoint_path = os.path.join(Config.SAVE_DIR, ssim_checkpoint_path)
    ssim_checkpoint = ModelCheckpoint(
        ssim_checkpoint_path,
        monitor='val_ssim_metric',
        save_best_only=True,
        mode='max',
        verbose=1,
    )
    callbacks.append(ssim_checkpoint)
    
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
    
    # Ön işleme sözleşmesi: checkpoint'lerin yanına "<model>.pre.json" yazar.
    callbacks.append(PreprocessingSidecarWriter(save_dir=Config.SAVE_DIR or "."))

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
    print(f"  Gizli Aktivasyon:   {Config.HIDDEN_ACTIVATION}")
    print(f"  Cikis Aktivasyonu:  {Config.OUTPUT_ACTIVATION}")
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
    validate_activation_contract()
    
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
    print("                    EĞİTİM BLIYOR")
    print("=" * 60 + "\n")
    
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=Config.EPOCHS,
        callbacks=callbacks
    )
    
    # Final modeli kaydet
    final_model_path = f"_{get_timestamp()}_{Config.FINAL_MODEL_NAME}"
    if Config.SAVE_DIR:
        final_model_path = os.path.join(Config.SAVE_DIR, final_model_path)
    
    model.save(final_model_path)
    write_preprocessing_sidecar(final_model_path)
    print(f"\nFinal model kaydedildi: {final_model_path}")
    
    print("\n" + "=" * 60)
    print("                    EM TAMAMLANDI")
    print("=" * 60 + "\n")
    
    return model, history


if __name__ == "__main__":
    main()


