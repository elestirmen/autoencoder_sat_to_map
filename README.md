# AutoEncoder Map Generation Pipeline (tf.data + Keras)

Bu repo; büyük ortofoto/uydu görüntülerini karolara bölerek, eğitilmiş otoenkoder tabanlı derin öğrenme modelleriyle karolardan harita/stil tahmini yapan, sonrasında karoları mozaikleyip GeoTIFF olarak jeoreferanslayan bir uçtan uca iş hattı sunar.

Önceki Pix2Pix/GAN denemeleri arşivlenmiştir; güncel ve sadeleştirilmiş akış tf.data ile beslenen otoenkoder(ler) etrafında şekillenmiştir.

## İçindekiler
- Özellikler ve Mimari
- Dizin Yapısı
- Akış Şeması (E2E)
- Kurulum ve Bağımlılıklar
- Veri Hazırlama (Karo Üretimi)
- Eğitim
  - Renkli (3→3)
  - Gri/tek kanal (3→1 veya 1→1)
- Çıkarım (Toplu Karo Tahmini) ve Birleştirme
- Jeoreferans (GeoTIFF)
- Yapılandırma ve Parametreler
- Performans İpuçları
- Sorun Giderme (FAQ)

---

## Özellikler ve Mimari
- tf.data ile diskten akışkan okuma: Yan yana (sol: giriş, sağ: hedef) tutulan eğitim görsellerini runtime’da ikiye bölerek RAM kullanımını sınırlar.
- Hafif U‑Net benzeri otoenkoderler: Encoder’de Conv+Pool, decoder’de UpSampling/TransposeConv; ELU + Dropout ile stabil ve hızlı eğitim.
- Büyük görüntüler için karo tabanlı üretim: 512–544 kare boyutları, bindirme payı ile dikiş izlerini azaltma.
- Çoklu model desteği: modeller klasöründeki tüm .h5 dosyalarıyla aynı parça seti üstünde çıkarım ve karşılaştırma.
- Jeoreferans/GeoTIFF: Referans bir raster’ın CRS ve transform’u kopyalanarak çıktı mozaikler koordinatlandırılır.

## Dizin Yapısı
Önemli dosya/klasörler:
- `goruntu bolme.py`, `goruntu bolme_beta.py`: Kaynak TIF’ten karolar üretir (bindirme ile). Çıktı: `bolunmus/<harita>/...jpg`.
- `autoencoder_dinamik_bellek_dosyadan_okuma_tf.data_renkli.py`: Renkli (3 kanal giriş → 3 kanal hedef) eğitim hattı.
- `autoencoder_dinamik_bellek_dosyadan_okuma_tf.data_3_kanal_to_1_kanal.py`: Tek kanal hedef (gri) eğitim hattı; opsiyonel histogram eşitleme.
- `harita_uretici_beta_gpt_hizli.py`: Gri/tek-kanal çıkarım + mozaik birleştirme.
- `harita_uretici_beta_gpt_hizli_renkli.py`: Renkli çıkarım + mozaik birleştirme.
- `harita_uretici_beta_gpt_hizli_3_kanal_to_1_kanal.py`: RGB giriş → 1 kanal çıktı çıkarım varyantı.
- `georef_gpt.py`, `georef_gpt-ertugrul.py`: Mozaikleri referans raster’a göre GeoTIFF olarak jeoreferanslar.
- `modeller/`: Çıkarımda kullanılacak Keras `.h5` modelleri.
- `ana_haritalar/`: Birleştirilmiş mozaik (çıkarım) çıktıları.
- `georefli/`: Jeoreferanslı GeoTIFF çıktıları.
- `deleted/`: Eski/Arşiv script’ler (Pix2Pix vb.).

## Akış Şeması (E2E)

```mermaid
flowchart TD
    A[Kaynak TIF (Büyük Ortofoto/Uydu)] --> B[Kırpma/Karo Üretimi\ngoruntu bolme*.py]
    B --> C[Parça Klasörü: bolunmus/<harita>/]
    subgraph Eğitim (Opsiyonel, tf.data)
      D1[Yan Yana Veri\n(sol:girdi | sağ:hedef)] --> D2[tf.data ile yükle\nve ikiye böl]
      D2 --> D3[Otoenkoder Eğitim\n(Keras/ELU/Dropout)]
      D3 --> D4[Model Kaydı .h5]
    end
    C --> E[Çıkarım (Parça Tahmini)\nharita_uretici_* .py]
    D4 --> E
    E --> F[Mozaik Birleştirme\n(bindirme kırpma + h/v stack)]
    F --> G[Çıktı: ana_haritalar/]
    G --> H[Jeoreferanslama\n(georef_gpt*.py)]
    H --> I[GeoTIFF (UTM/CRS)\n georefli/]
```

---

## Kurulum ve Bağımlılıklar
- Python 3.8–3.10 önerilir
- Paketler:
  - Derin öğrenme: `tensorflow` (veya `tensorflow-gpu`), `keras` (TF içinden)
  - Görüntü: `opencv-python`, `Pillow`, `numpy`, `matplotlib`, `natsort`
  - Coğrafi: `rasterio`, `GDAL`
  - Opsiyonel: `tensorflow-addons` (histogram eşitleme)

Kurulum (Windows, PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install tensorflow opencv-python Pillow numpy matplotlib natsort rasterio tensorflow-addons
# GDAL: Windows’ta hazır wheel kullanın (örn. Gohlke veya conda).
# pip ile denerken GDAL_VERSION ve include/library yollarını ayarlamanız gerekebilir.
```

Conda ile (öneri):
```powershell
conda create -n mapa python=3.10 -y
conda activate mapa
conda install -c conda-forge tensorflow rasterio gdal opencv pillow matplotlib natsort tensorflow-addons -y
```

GDAL/Rasterio Windows kurulumunda sık hata alınır; mümkünse conda-forge tercih edin.

---

## Veri Hazırlama (Karo Üretimi)
- Kaynak: Büyük `.tif` ortofoto/uydu görseli
- Karo üretimi için script’lerden birini kullanın:
  - Basit ve kare ölçekli: `goruntu bolme.py` (544×544 + bindirme)
  - Tam grid üzerinde: `goruntu bolme_beta.py` (512×512 + bindirme, X×Y)

Örnek kullanım (PowerShell):
```powershell
# goruntu bolme.py içinde path değişkenini kaynak TIF’e ayarlayın
python "goruntu bolme.py"
# Çıktılar: bolunmus/<harita>/...jpg
```

Bindirme (genişleme) pikselleri birleştirme aşamasında içerden kırpılır (dikiş izini azaltır).

---

## Eğitim
Eğitim verisi tek görüntü içinde “yan yana ikili” formatta olmalı:
- Sol yarı: giriş (ör. uydu/ortofoto)
- Sağ yarı: hedef (istenen stil/harita)

Scriptler bu görüntüyü runtime’da ikiye böler, 544×544’e yeniden boyutlandırır ve [-1, 1] aralığına normalleştirir.

### Renkli (3→3)
Dosya: `autoencoder_dinamik_bellek_dosyadan_okuma_tf.data_renkli.py`
- Veri kökünü değiştirin:
  - `all_image_paths = "C:\\d_surucusu\\satnap\\output_ps_renkli\\" + ...`
- Model: `create_gpt_autoencoder_none_regularization(...)` (ELU + Dropout, 3 kanal çıktı)
- Not: Script, modeli oluşturduktan sonra `son_model.h5` yükleyerek devam eğitim kurgusuna uygun çalışır. Sıfırdan eğitim istiyorsanız `model = load_model("son_model.h5")` satırını yoruma alın.

Çalıştırma:
```powershell
python "autoencoder_dinamik_bellek_dosyadan_okuma_tf.data_renkli.py"
```

### Gri/Tek Kanal (3→1 veya 1→1)
Dosya: `autoencoder_dinamik_bellek_dosyadan_okuma_tf.data_3_kanal_to_1_kanal.py`
- Veri kökünü değiştirin:
  - `all_image_paths = "C:\\d_surucusu\\satmap\\output_full\\" + ...`
- Opsiyonel histogram eşitleme: `tensorflow-addons` ile `tfa.image.equalize`
- Varsayılan model: `create_advanced_autoencoder(...)` (1 kanal çıktı)
- Aynı şekilde, `model = load_model("son_model.h5")` satırı devam eğitim içindir.

Çalıştırma:
```powershell
python "autoencoder_dinamik_bellek_dosyadan_okuma_tf.data_3_kanal_to_1_kanal.py"
```

Eğitim sonunda `son_model.h5` ve epoch bazlı checkpoint’ler üretilir.

---

## Çıkarım (Toplu Karo Tahmini) ve Birleştirme
- Modelleri `modeller/` klasörüne koyun (birden fazla .h5 desteklenir).
- Karo klasörünüz `bolunmus/<harita>/...` şeklinde olmalı.

Gri/tek kanal çıkarım:
```powershell
python "harita_uretici_beta_gpt_hizli.py"
```

Renkli çıkarım:
```powershell
python "harita_uretici_beta_gpt_hizli_renkli.py"
```

3→1 varyantı:
```powershell
python "harita_uretici_beta_gpt_hizli_3_kanal_to_1_kanal.py"
```

Script, parçaları model(ler) ile tahmin eder, bindirme kenarlarını içeriden kırpar ve satır-sütun halinde birleştirerek `ana_haritalar/ana_harita_<harita>_<model>.jpg` dosyasını üretir.

Notlar:
- Grid ölçüleri: Bazı script’lerde sabit başlangıç değeri ve karekök tabanlı otomatik kare grid modu bulunur. Eğer parça sayınız kare sayı değilse sabit frame_adedi_x/y değerlerini doğru ayarlayın.
- Renkli akışta OpenCV BGR sırası ile RGB karışabilir; gerekli dönüşümler script’te yapılmıştır.

---

## Jeoreferans (GeoTIFF)
Mozaiklenmiş çıktı `.jpg` dosyalarını bir referans GeoTIFF’in CRS ve transform’u ile jeoreferanslayın.

- Referans raster yolunu ayarlayın (örnek: `ana_harita_urgup_30_cm__Georefference_utm.tif`).
- Çalıştırma:
```powershell
python "georef_gpt.py"
# veya
python "georef_gpt-ertugrul.py"
```
- Çıktılar `georefli/` altında `.tif` olarak yazılır (LZW ya da JPEG sıkıştırma seçenekleri script’te).

---

## Yapılandırma ve Parametreler
Önemli parametreler ve nerede ayarlanacağı:
- Karo boyutu ve bindirme: `goruntu bolme*.py` içinde `frame_size`, `genisletme`
- Eğitim verisi kökü: eğitim script’lerinde `all_image_paths`
- Giriş/çıkış kanal sayısı: kullanılan model fonksiyonuna göre (3→3, 3→1, 1→1)
- Batch size, optimizer, loss: eğitim script’lerinin alt bölümünde
- Model yükleme/başlangıç: `model = load_model("son_model.h5")` (devam eğitim) satırını ihtiyaca göre kullanın/yorumlayın
- Çıkarım model klasörü: `modeller/`
- Çıktı klasörleri: `c:/d_surucusu/parcalar/` (parçalar), `ana_haritalar/` (mozaik), `georefli/` (GeoTIFF)

Öneri: Yolları ve parametreleri merkezi bir `config.yaml` dosyasına almak taşınabilirliği artırır (isteğe bağlı).

---

## Performans İpuçları
- tf.data ayarları: `num_parallel_calls`, `batch`, `prefetch` değerlerini donanıma göre yükseltin. `AUTOTUNE` çoğu ortamda iyi çalışır.
- GPU kullanımı: TensorFlow GPU kurulumunu doğrulayın; batch size’ı VRAM’a göre ayarlayın.
- I/O: Karo boyutu ve bindirme, disk I/O ve RAM kullanımını belirler. Daha az bindirme daha hızlı, fakat dikiş riskini artırır.
- Çıkarımda ThreadPool: `harita_uretici_beta_gpt_hizli*.py` paralel çıkarım yapar. CPU çekirdeklerine göre thread sayısını sınırlandırmak isteyebilirsiniz.

---

## Sorun Giderme (FAQ)
- GDAL/Rasterio kurulumu hata veriyor:
  - Windows’ta conda-forge ile kurulum yapın. pip ile derleme yol ayarları zahmetlidir.
- Renkler ters görünüyor (çıkarım):
  - OpenCV’nin BGR, matplotlib’in RGB kullandığını unutmayın. Script’te `cvtColor` dönüşümü var; görsel yolunuza göre düzenleyin.
- Eğitim yeniden başlamak yerine “devam ediyor”:
  - Script’teki `model = load_model("son_model.h5")` satırını yoruma alın (sıfırdan başlar).
- Karo sayısından grid hesaplanamıyor:
  - Script’teki `frame_adedi_x/y` değerlerini manuel ve doğru şekilde ayarlayın.
- Bellek hataları:
  - `batch_size` düşürün, karoları diskten akışkan okuyun (zaten tf.data yapıyor), görsel çözünürlüğünü azaltmayı düşünün.

---

## Hızlı Başlangıç (Özet)
1) Karo üretimi: `goruntu bolme*.py` içinde `path` → kaynak TIF
```powershell
python "goruntu bolme.py"
```
2) Eğitim (opsiyonel): veri kökü → eğitim script’inde `all_image_paths`
```powershell
python "autoencoder_dinamik_bellek_dosyadan_okuma_tf.data_renkli.py"
```
3) Çıkarım + birleştirme: modelleri `modeller/` içine koyun
```powershell
python "harita_uretici_beta_gpt_hizli.py"
```
4) Jeoreferans/GeoTIFF:
```powershell
python "georef_gpt.py"
```

Keyifli çalışmalar!
