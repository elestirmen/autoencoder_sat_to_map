import numpy as np

# birleştirmek istediğiniz dosyaların isimlerini bir liste olarak belirtin
file_names = ['maps_0_544.npz', 'maps_1_544.npz', 'maps_2_544.npz', 'maps_3_544.npz', 'maps_4_544.npz']

# dosyaları birleştirmek için kullanacağımız boş bir sözlük oluşturun
merged_data = {}

# tüm dosyaları döngüye alın ve her bir dosyadaki verileri birleştirin
for file_name in file_names:
    # np.load() fonksiyonunu kullanarak dosyayı yükleyin
    data = np.load(file_name)
    # her bir anahtar kelime ve değer çiftini ana birleştirilmiş veri sözlüğüne ekleyin
    for key, value in data.items():
        if key in merged_data:
            # Anahtar kelime zaten birleştirilmiş veri sözlüğünde varsa, o zaman yeniden adlandırın.
            # Örneğin: "key" -> "key_1"
            i = 1
            while f"{key}_{i}" in merged_data:
                i += 1
            merged_data[f"{key}_{i}"] = value
        else:
            merged_data[key] = value

# Birleştirilmiş verileri tek bir .npz dosyasına kaydedin.
np.savez("merged_maps.npz", **merged_data)
