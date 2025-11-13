import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2, 40))
import cv2
import matplotlib.pyplot as plt
from osgeo import gdal
from pathlib import Path
import numpy as np


def load_image(path):
    """Görüntüyü yükler ve kontrol eder."""
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Görüntü yüklenemedi: {path}")
    return img


def get_geotransform(path):
    """Coğrafi transformasyon bilgilerini alır."""
    raster = gdal.Open(path)
    if raster is None:
        raise ValueError(f"Raster açılamadı: {path}")
    
    gt = raster.GetGeoTransform()
    pixelSizeX = gt[1]
    pixelSizeY = abs(gt[5])  # Negatif değeri pozitif yap
    
    print(f"GeoTransform: {gt}")
    print(f"Pixel Size X: {pixelSizeX}")
    print(f"Pixel Size Y: {pixelSizeY}")
    
    return gt, pixelSizeX, pixelSizeY


def create_output_directory(output_path):
    """Çıktı dizinini oluşturur."""
    Path(output_path).mkdir(parents=True, exist_ok=True)


def split_image(img, frame_size=512, overlap=32, output_dir='bolunmus/bolunmus'):
    """
    Görüntüyü küçük parçalara böler.
    
    Args:
        img: Görüntü array'i
        frame_size: Her parçanın boyutu (piksel)
        overlap: Parçalar arası örtüşme (piksel)
        output_dir: Çıktı dizini
    
    Returns:
        Parçaların listesi ve dosya isimleri
    """
    height, width = img.shape[:2]
    
    # Çıktı dizinini oluştur
    create_output_directory(output_dir)
    
    # Kaç parça oluşturulacağını hesapla
    num_frames_x = int(height / frame_size)
    num_frames_y = int(width / frame_size)
    
    print(f"Görüntü boyutu: {height}x{width}")
    print(f"Parça sayısı: {num_frames_x}x{num_frames_y} = {num_frames_x * num_frames_y}")
    
    img_cropped = []
    filenames = []
    
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
            filename = os.path.join(output_dir, f'goruntu_{i}_{j}.jpg')
            
            # Kaydet
            cv2.imwrite(filename, crop)
            
            img_cropped.append(crop)
            filenames.append(filename)
            
            print(f"Parça kaydedildi: {i}, {j} -> {filename}")
    
    return img_cropped, filenames


def visualize_crops(img_cropped, num_frames_x, num_frames_y, figsize=(10, 10)):
    """Parçaları görselleştirir."""
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
    return fig


def main():
    # Parametreler
    path = "urgup_bingmap_30cm_utm.tif"
    frame_size = 512
    overlap = 32
    output_dir = 'bolunmus/bolunmus'
    
    try:
        # Görüntüyü yükle
        print("Görüntü yükleniyor...")
        img = load_image(path)
        
        # Coğrafi bilgileri al
        print("\nCoğrafi bilgiler alınıyor...")
        gt, pixelSizeX, pixelSizeY = get_geotransform(path)
        
        # Görüntüyü böl
        print("\nGörüntü bölünüyor...")
        img_cropped, filenames = split_image(
            img, 
            frame_size=frame_size, 
            overlap=overlap, 
            output_dir=output_dir
        )
        
        # Görselleştir
        print("\nGörselleştirme oluşturuluyor...")
        num_frames_x = int(img.shape[0] / frame_size)
        num_frames_y = int(img.shape[1] / frame_size)
        fig = visualize_crops(img_cropped, num_frames_x, num_frames_y)
        
        # Görselleştirmeyi kaydet (isteğe bağlı)
        # fig.savefig("goruntu_parcalari.png", dpi=150, bbox_inches='tight')
        
        plt.show()
        
        print(f"\nToplam {len(filenames)} parça oluşturuldu.")
        
    except Exception as e:
        print(f"Hata oluştu: {e}")
        raise


if __name__ == "__main__":
    main()