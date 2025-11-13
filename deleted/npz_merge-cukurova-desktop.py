import numpy as np

# birleştirmek istediğiniz dosyaların isimlerini bir liste olarak belirtin
file_names = ['maps_0_544.npz', 'maps_1_544.npz', 'maps_2_544.npz', 'maps_3_544.npz', 'maps_4_544.npz']


# Yeni bir dizi oluşturun ve tüm verileri bu diziye ekleyin
merged_arr_0 = np.empty(0)
merged_arr_1 = np.empty(0)

for file_name in file_names:
    # .npz dosyasını yükleyin
    data = np.load(file_name)
    # 'arr_0' ve 'arr_1' sütunlarını alın
    arr_0 = data['arr_0']
    arr_1 = data['arr_1']
    # Tüm verileri yeni bir diziye ekleyin
    if merged_arr_0.size == 0:
        merged_arr_0 = arr_0
        merged_arr_1 = arr_1
    else:
        merged_arr_0 = np.concatenate((merged_arr_0, arr_0), axis=0)
        merged_arr_1 = np.concatenate((merged_arr_1, arr_1), axis=0)

# Yeni bir .npz dosyası oluşturun
np.savez_compressed("merged_data.npz", arr_0=merged_arr_0, arr_1=merged_arr_1)



#%% büyük npy dosyasını sıkıştır npz olarak kaydet



