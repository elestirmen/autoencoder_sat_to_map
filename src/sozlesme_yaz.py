"""
Ön İşleme Sözleşmesi Yazıcı
============================

Eğitimden GEÇMİŞ (eski) model dosyaları için "<model>.pre.json" sözleşmesini
elle oluşturur. Yeni eğitimler bu dosyayı kendiliğinden yazar
(bkz. egitim/autoencoder_unified.py -> write_preprocessing_sidecar).

Sözleşme neden gerekli?
    Bir modelin hangi girdi ön işlemesiyle eğitildiği model dosyasından
    OKUNAMAZ. Çıkış aktivasyonu (sigmoid/tanh) yalnızca çıkış aralığını
    söyler; girişin nasıl ölçeklendiğini söylemez. Bu depodaki 2022 modelleri
    bunun kanıtıdır: çıkışları sigmoid olduğu halde girişleri [-1, 1] ve
    histogram eşitlemelidir.

    Sözleşme yazıldıktan sonra hem harita üretici (src/goruntu_islemleri.py)
    hem template-matching aynı dosyayı okur; aynı model iki boru hattında da
    aynı girdiyi görür.

Kullanım:
    # 2022 eğitim scriptiyle (tfa equalize + [-1,1]) eğitilmiş modeller için:
    python sozlesme_yaz.py --legacy-2022 ../model/*.h5

    # Elle:
    python sozlesme_yaz.py -n minus1_1 -e hist_eq_pil --image-size 544 model.h5

    # Mevcut sözleşmeleri göster:
    python sozlesme_yaz.py --show model.h5
"""

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from goruntu_islemleri import (  # noqa: E402
    NORMALIZATION_MODES,
    ENHANCEMENT_MODES,
    load_preprocessing_sidecar,
    preprocessing_sidecar_path,
    save_preprocessing_sidecar,
)

# 2022 eğitim scripti (egitim/autoencoder_dinamik_bellek_dosyadan_okuma_tf.data_
# 3_kanal_to_1_kanal.py) şu sırayı kullanıyordu:
#   decode(1 kanal) -> tfa.image.equalize -> resize NEAREST 544 -> (x-127.5)/127.5
#
# DIKKAT: tfa.image.equalize, PIL ImageOps.equalize'in portudur; OpenCV'nin
# equalizeHist'i DEGILDIR (ayni karoda ortalama ~6, uc durumda 150+ gri seviye
# ayrisirlar). Bu yuzden iyilestirme "hist_eq" degil "hist_eq_pil"dir.
LEGACY_2022 = {
    "input_normalization": "minus1_1",
    "enhancement": "hist_eq_pil",
    "color_mode": "grayscale",
    "image_size": 544,
    "output_activation": "sigmoid",
    "output_color_mode": "grayscale",
    "trained_by": "autoencoder_dinamik_bellek_dosyadan_okuma_tf.data_3_kanal_to_1_kanal.py",
    "note": (
        "Sozlesme model dosyasindan degil, egitim scriptinin kaynagindan elle "
        "cikarildi. Cikis sigmoid olsa da giris [-1, 1] ve PIL/tfa uyumlu "
        "histogram esitlemelidir (cv2.equalizeHist DEGIL)."
    ),
}


# Sözleşme hem MODEL hem de üretilmiş HARİTA dosyasının yanına yazılabilir:
#   - model yanında: modelin hangi ön işlemeyle eğitildiği,
#   - harita yanında: haritanın hangi ön işlemeyle üretildiği.
# Template-matching, anlık görüntüyü haritanınkiyle eşlemek zorunda olduğu için
# harita sözleşmesini model sözleşmesinden ÖNCE dikkate alır.
HEDEF_UZANTILAR = ('.h5', '.keras', '.tif', '.tiff', '.png', '.jpg', '.jpeg')


def expand(patterns):
    """Kabuk genişletmesi yapmayan ortamlar için joker karakterleri çözer."""
    paths = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        paths.extend(matches if matches else [pattern])
    return [p for p in paths if p.lower().endswith(HEDEF_UZANTILAR)]


def main():
    ap = argparse.ArgumentParser(
        description="Model veya harita dosyasının yanına ön işleme sözleşmesi yazar/gösterir.")
    ap.add_argument("models", nargs="+",
                    help="Model (.h5/.keras) veya harita (.tif/.png/.jpg) dosyaları; joker kabul edilir")
    ap.add_argument("--show", action="store_true", help="Yazma, mevcut sözleşmeyi göster.")
    ap.add_argument("--legacy-2022", action="store_true",
                    help="2022 eğitim scriptinin ön işlemesini yaz (minus1_1 + hist_eq).")
    ap.add_argument("-n", "--normalization", choices=NORMALIZATION_MODES,
                    help="Giriş normalizasyonu.")
    ap.add_argument("-e", "--enhancement", choices=ENHANCEMENT_MODES,
                    help="Kontrast/histogram iyileştirmesi.")
    ap.add_argument("--color-mode", choices=["grayscale", "rgb"], help="Giriş renk modu.")
    ap.add_argument("--image-size", type=int, help="Model girdi boyutu (karo kenarı).")
    ap.add_argument("--output-activation", help="Model çıkış aktivasyonu (bilgi amaçlı).")
    ap.add_argument("-f", "--force", action="store_true",
                    help="Var olan sözleşmenin üzerine yaz.")
    args = ap.parse_args()

    paths = expand(args.models)
    if not paths:
        ap.error(f"Dosya bulunamadı. Desteklenen uzantılar: {', '.join(HEDEF_UZANTILAR)}")

    if args.show:
        for path in paths:
            contract = load_preprocessing_sidecar(path)
            print(f"\n{os.path.basename(path)}")
            if contract is None:
                print("  sözleşme YOK -> çıkarımda varsayılan (minus1_1 + none) kullanılır")
            else:
                for key, value in contract.items():
                    print(f"  {key}: {value}")
        return 0

    contract = dict(LEGACY_2022) if args.legacy_2022 else {}
    for key, value in (
        ("input_normalization", args.normalization),
        ("enhancement", args.enhancement),
        ("color_mode", args.color_mode),
        ("image_size", args.image_size),
        ("output_activation", args.output_activation),
    ):
        if value is not None:
            contract[key] = value

    if "input_normalization" not in contract:
        ap.error("En az --normalization (veya --legacy-2022) verilmeli.")

    for path in paths:
        if not os.path.exists(path):
            print(f"[ATLANDI] Dosya yok: {path}")
            continue
        existing = preprocessing_sidecar_path(path)
        if os.path.exists(existing) and not args.force:
            print(f"[ATLANDI] Sözleşme zaten var (--force ile ezin): {existing}")
            continue
        written = save_preprocessing_sidecar(path, **contract)
        print(f"[YAZILDI] {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
