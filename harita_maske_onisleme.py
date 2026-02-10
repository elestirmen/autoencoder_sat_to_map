"""
Yan yana (sol: uydu, sag: harita) goruntulerden siyah-etiket maske uretir.

Varsayilan cikti:
  - Sol yarim: orijinal uydu
  - Sag yarim: beyaz zemin uzerinde siyah nesne maskesi
  - Istekle: sol yarim icin isik/renk/exposure simulasyonlu varyantlar

Kullanim ornekleri:
  python harita_maske_onisleme.py
  python harita_maske_onisleme.py --input_dir renkli_egitim_seti_ornek --output_dir egitim_maskli
  python harita_maske_onisleme.py --input_dir renkli_egitim_seti_ornek --output_dir maskeler --mode mask
  python harita_maske_onisleme.py --input_file ornek.jpg --output_dir out --save_preview 1
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# -----------------------------------------------------------------------------
# TEK NOKTADAN AYARLAR (argumansiz calisma icin)
# -----------------------------------------------------------------------------
# Script'i direkt su sekilde calistirabilirsin:
#   python harita_maske_onisleme.py
# Gerekirse asagidaki degerleri degistir.
CONFIG = {
    # Giris/cikis
    "input_dir": "ham_veri",
    "input_file": None,  # Tek dosya icin yol ver; vermezsen input_dir taranir
    "output_dir": "training_set",
    "mode": "paired",  # "paired": sol uydu + sag maske, "mask": sadece maske
    "output_ext": ".png",

    # Uydu tarafi augmentation (action stili isik kosulu simulasyonu)
    "sat_aug_enable": 1,  # 1: acik, 0: kapali
    "sat_aug_keep_original": 1,  # 1: orijinali de yaz
    "sat_aug_count": 2,  # Ek varyant sayisi
    "sat_aug_seed": 42,  # <0 verirsen her calismada farkli olur
    "sat_exp_min": -30.0,
    "sat_exp_max": 30.0,
    "sat_contrast_min": 0.82,
    "sat_contrast_max": 1.24,
    "sat_gamma_min": 0.78,
    "sat_gamma_max": 1.25,
    "sat_sat_min": 0.75,
    "sat_sat_max": 1.28,
    "sat_hue_shift_max": 6.0,
    "sat_temp_shift_max": 14.0,
    "sat_shadow_prob": 0.35,
    "sat_shadow_strength_min": 0.58,
    "sat_shadow_strength_max": 0.88,
    "sat_noise_sigma_min": 0.0,
    "sat_noise_sigma_max": 4.5,
    "sat_blur_prob": 0.18,

    # Onizleme
    "save_preview": 0,  # Ilk N dosya icin 4-panelli preview kaydet
    "preview_dir": "test/mask_preview",
    "mask_profile": "balanced",  # balanced | strict

    # Segmentasyon parametreleri
    # Cikti arka plan gri tonu (orneklerdeki gibi acik gri: 232)
    "output_bg_gray": 232,
    # Arka plan tespiti (Lab renk uzakligi + HSV kosulu)
    "bg_dist_threshold": 8.0,
    "bg_sat_max": 8,
    "bg_val_min": 236,
    "bg_neutral_max_diff": 24,
    "seed_dist_threshold": 5.8,
    # Cizgi/bina tespiti icin adaptif gri esik farklari
    "line_delta": 18,
    "dark_delta": 26,
    "gray_line_sat_max": 35,
    # Bina rengi (Google/OpenStreetMap benzeri acik turuncu/tan)
    "tan_h_min": 6,
    "tan_h_max": 36,
    "tan_s_min": 8,
    "tan_v_min": 110,
    # Ana yol sari tonlari (sadece cizgi bilgisi icin)
    "yellow_h_min": 15,
    "yellow_h_max": 45,
    "yellow_s_min": 22,
    "yellow_v_min": 120,
    # Mavi cizgiler (su yollari vb.)
    "blue_h_min": 90,
    "blue_h_max": 130,
    "blue_s_min": 8,
    "blue_v_min": 130,
    # Morfoloji
    "line_close_kernel": 3,
    "line_long_close_kernel": 9,
    "building_close_kernel": 2,
    "building_open_kernel": 2,
    "line_min_major_len": 24,
    "line_dark_keep_delta": 28,
    "enclosed_min_density": 0.18,
    "enclosed_min_side": 5,
    "enclosed_max_area_ratio": 0.48,
    "enclosed_dist_min": 8.5,
    "enclosed_dark_delta": 12,
    "enclosed_tan_ratio_keep": 0.015,
    "enclosed_green_ratio_max": 0.42,
    "enclosed_blue_ratio_max": 0.32,
    "enclosed_boundary_ratio_keep": 0.03,
    "enclosed_compact_area_ratio": 0.0,
    "enclosed_compact_min_density": 0.42,
    "fill_hole_max_area": 160,
    "fill_hole_max_thickness": 3,
    "min_area_ratio": 0.00008,
    "min_area_px": 12,
    "building_max_area_ratio": 0.40,
    "building_min_side": 4,
    "building_max_aspect": 5.8,
    "building_min_density": 0.26,
    "building_tan_ratio_keep": 0.02,
    "suppress_green_blue": 1,
}

MASK_PROFILE_OVERRIDES: Dict[str, Dict[str, float]] = {
    # Varsayilan: dengeli recall/precision.
    "balanced": {},
    # Daha az false-positive icin daha katı esikler.
    "strict": {
        "bg_dist_threshold": 10.5,
        "seed_dist_threshold": 7.5,
        "line_delta": 24,
        "dark_delta": 32,
        "line_long_close_kernel": 7,
        "line_min_major_len": 32,
        "line_dark_keep_delta": 34,
        "enclosed_min_density": 0.24,
        "enclosed_dist_min": 10.0,
        "enclosed_boundary_ratio_keep": 0.06,
        "enclosed_compact_area_ratio": 0.0,
        "building_min_density": 0.34,
        "building_tan_ratio_keep": 0.03,
        "fill_hole_max_area": 80,
        "min_area_ratio": 0.00012,
        "min_area_px": 16,
        "suppress_green_blue": 1,
    },
}


def list_images(input_dir: Path) -> List[Path]:
    return sorted(
        [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    )


def estimate_background_bgr(map_bgr: np.ndarray, neutral_max_diff: int) -> np.ndarray:
    # Tum haritada "duz/tekdüze" (dususuk gradyan) piksellerde baskin tonu bul.
    # Bu, sehir/yesil/su agirlikli stillerde arka plan secimini dengeler.
    gray = cv2.cvtColor(map_bgr, cv2.COLOR_BGR2GRAY)
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    flat = grad <= 4

    pixels = map_bgr[flat]
    if pixels.shape[0] < 500:
        pixels = map_bgr.reshape(-1, 3)

    quantized = (pixels // 4) * 4
    values, counts = np.unique(quantized, axis=0, return_counts=True)
    if len(values) == 0:
        return np.median(map_bgr.reshape(-1, 3), axis=0).astype(np.uint8)
    return values[int(np.argmax(counts))].astype(np.uint8)


def remove_small_components(binary: np.ndarray, min_area_px: int) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary)
    for idx in range(1, num_labels):
        area = stats[idx, cv2.CC_STAT_AREA]
        if area >= min_area_px:
            cleaned[labels == idx] = 255
    return cleaned


def estimate_background_gray(gray: np.ndarray, hsv: np.ndarray, map_bgr: np.ndarray, neutral_max_diff: int) -> int:
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    ch_max = map_bgr.max(axis=2)
    ch_min = map_bgr.min(axis=2)
    neutral = (ch_max - ch_min) <= int(neutral_max_diff)

    candidate = (sat <= 28) & (val >= 170) & neutral
    if int(candidate.sum()) > 200:
        # Mod (quantized) degeri, cok modlu dagilimda mediana gore daha dayanikli.
        q = ((gray[candidate] // 2) * 2).astype(np.uint8)
        vals, cnts = np.unique(q, return_counts=True)
        return int(vals[int(np.argmax(cnts))])

    # BazÄ± tile'larda sat neredeyse sabit olabilir; bu durumda sadece parlak
    # piksellerden medyan almak line/park yapÄ±larÄ±nÄ± arka plan saymayÄ± engeller.
    bright = val >= 245
    if int(bright.sum()) > 200:
        return int(np.median(gray[bright]))

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    hist[:140] = 0
    return int(np.argmax(hist))


def remove_large_components(binary_u8: np.ndarray, max_area_px: int) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_u8, connectivity=8)
    out = np.zeros_like(binary_u8)
    for idx in range(1, num_labels):
        area = stats[idx, cv2.CC_STAT_AREA]
        if area <= max_area_px:
            out[labels == idx] = 255
    return out


def filter_line_components(
    binary_u8: np.ndarray,
    gray: np.ndarray,
    bg_gray: int,
    min_area_px: int,
    min_major_len: int,
    dark_keep_delta: int,
) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_u8, connectivity=8)
    out = np.zeros_like(binary_u8)
    for idx in range(1, num_labels):
        area = stats[idx, cv2.CC_STAT_AREA]
        if area < min_area_px:
            continue

        w = stats[idx, cv2.CC_STAT_WIDTH]
        h = stats[idx, cv2.CC_STAT_HEIGHT]
        major = max(w, h)
        box = float(w * h) if (w > 0 and h > 0) else 1.0
        density = float(area) / box
        aspect = float(max(w, h)) / float(max(1, min(w, h)))

        comp = labels == idx
        mean_gray = float(gray[comp].mean())
        keep_dark = mean_gray <= float(bg_gray - int(dark_keep_delta))
        keep_long = major >= int(min_major_len)
        keep_large = area >= max(24, int(min_area_px * 2))
        keep_structured = (
            aspect >= 2.6
            and density <= 0.55
            and major >= max(8, int(min_major_len // 2))
        )

        # Parsel tipi kisa dashed cizgileri at, uzun/karanlik/guclu cizgileri koru.
        if keep_dark or keep_long or keep_large or keep_structured:
            out[labels == idx] = 255
    return out


def extract_enclosed_regions(boundary_u8: np.ndarray) -> np.ndarray:
    h, w = boundary_u8.shape
    inv = cv2.bitwise_not(boundary_u8)
    flood = inv.copy()
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)

    # Sinirdan ulasilabilen acik bolgeleri sifirla; kapali bolgeler 255 kalir.
    cv2.floodFill(flood, ff_mask, (0, 0), 0)
    enclosed = np.zeros_like(boundary_u8)
    enclosed[(inv > 0) & (flood > 0)] = 255
    return enclosed


def filter_enclosed_components(
    enclosed_u8: np.ndarray,
    boundary_u8: np.ndarray,
    gray: np.ndarray,
    hsv: np.ndarray,
    color_dist: np.ndarray,
    tan_mask_u8: np.ndarray,
    bg_gray: int,
    min_area_px: int,
    max_area_px: int,
    min_side: int,
    min_density: float,
    min_color_dist: float,
    dark_delta: int,
    tan_ratio_keep: float,
    green_ratio_max: float,
    blue_ratio_max: float,
    boundary_ratio_keep: float,
    compact_area_max_px: int,
    compact_min_density: float,
) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(enclosed_u8, connectivity=8)
    out = np.zeros_like(enclosed_u8)
    tan_bool = tan_mask_u8 > 0
    boundary_bool = boundary_u8 > 0
    green_bool = (
        (hsv[:, :, 0] >= 35)
        & (hsv[:, :, 0] <= 95)
        & (hsv[:, :, 1] >= 22)
        & (hsv[:, :, 2] >= 110)
    )
    blue_bool = (
        (hsv[:, :, 0] >= 90)
        & (hsv[:, :, 0] <= 135)
        & (hsv[:, :, 1] >= 18)
        & (hsv[:, :, 2] >= 95)
    )

    for idx in range(1, num_labels):
        area = stats[idx, cv2.CC_STAT_AREA]
        if area < min_area_px or area > max_area_px:
            continue

        w = stats[idx, cv2.CC_STAT_WIDTH]
        h = stats[idx, cv2.CC_STAT_HEIGHT]
        if min(w, h) < min_side:
            continue

        box_area = float(max(1, w * h))
        density = float(area) / box_area
        comp = labels == idx
        tan_ratio = float(tan_bool[comp].mean())
        green_ratio = float(green_bool[comp].mean())
        blue_ratio = float(blue_bool[comp].mean())
        mean_gray = float(gray[comp].mean())
        mean_dist = float(color_dist[comp].mean())

        # Buyuk yesil/mavi polygonlari (park/su) bina adayi olarak alma.
        if green_ratio > float(green_ratio_max) and tan_ratio < float(tan_ratio_keep):
            continue
        if blue_ratio > float(blue_ratio_max) and tan_ratio < float(tan_ratio_keep):
            continue

        # Kapali bolgelerde yogunluk tek basina yeterli degil:
        # Arka plandan renk uzakligi, koyuluk veya tan oraniyla birlikte karar ver.
        valid_by_tone = mean_dist >= float(min_color_dist)
        valid_by_dark = mean_gray <= float(bg_gray - int(dark_delta))
        valid_by_tan = tan_ratio >= float(tan_ratio_keep)
        comp_u8 = (comp.astype(np.uint8) * 255)
        edge = cv2.morphologyEx(comp_u8, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
        edge_bool = edge > 0
        edge_count = int(edge_bool.sum())
        boundary_ratio = 0.0
        if edge_count > 0:
            boundary_ratio = float(np.logical_and(edge_bool, boundary_bool).sum()) / float(edge_count)

        compact_ok = (
            compact_area_max_px > 0
            and area <= int(compact_area_max_px)
            and density >= float(compact_min_density)
            and boundary_ratio >= float(boundary_ratio_keep)
            and (valid_by_dark or valid_by_tan)
        )
        keep = (density >= float(min_density)) and (
            valid_by_tone or valid_by_dark or valid_by_tan or compact_ok
        )
        if keep:
            out[comp] = 255

    return out

def filter_building_components(
    binary_u8: np.ndarray,
    tan_mask_u8: np.ndarray,
    gray: np.ndarray,
    min_area_px: int,
    max_area_px: int,
    min_side: int,
    max_aspect: float,
    min_density: float,
    tan_ratio_keep: float,
) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_u8, connectivity=8)
    out = np.zeros_like(binary_u8)
    tan_bool = tan_mask_u8 > 0

    for idx in range(1, num_labels):
        area = stats[idx, cv2.CC_STAT_AREA]
        if area < min_area_px or area > max_area_px:
            continue

        w = stats[idx, cv2.CC_STAT_WIDTH]
        h = stats[idx, cv2.CC_STAT_HEIGHT]
        if min(w, h) < min_side:
            continue

        aspect = float(max(w, h)) / float(max(1, min(w, h)))
        if aspect > max_aspect:
            continue

        box_area = float(w * h)
        density = float(area) / max(1.0, box_area)
        comp = labels == idx
        tan_ratio = float(tan_bool[comp].mean())
        mean_gray = float(gray[comp].mean())

        keep = (tan_ratio >= tan_ratio_keep) or (
            density >= min_density and mean_gray <= 185.0
        )
        if keep:
            out[comp] = 255

    return out


def fill_small_holes(
    binary_u8: np.ndarray,
    max_hole_area: int,
    max_hole_thickness: int,
) -> np.ndarray:
    inv = cv2.bitwise_not(binary_u8)
    flood = inv.copy()
    h, w = binary_u8.shape
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, ff_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(holes, connectivity=8)
    filled = binary_u8.copy()
    for idx in range(1, num_labels):
        area = stats[idx, cv2.CC_STAT_AREA]
        hw = stats[idx, cv2.CC_STAT_WIDTH]
        hh = stats[idx, cv2.CC_STAT_HEIGHT]
        if area <= max_hole_area or min(hw, hh) <= max_hole_thickness:
            filled[labels == idx] = 255
    return filled


def build_black_object_mask(
    map_bgr: np.ndarray,
    output_bg_gray: int,
    bg_dist_threshold: float,
    bg_sat_max: int,
    bg_val_min: int,
    bg_neutral_max_diff: int,
    seed_dist_threshold: float,
    line_delta: int,
    dark_delta: int,
    gray_line_sat_max: int,
    tan_h_min: int,
    tan_h_max: int,
    tan_s_min: int,
    tan_v_min: int,
    yellow_h_min: int,
    yellow_h_max: int,
    yellow_s_min: int,
    yellow_v_min: int,
    blue_h_min: int,
    blue_h_max: int,
    blue_s_min: int,
    blue_v_min: int,
    line_close_kernel: int,
    line_long_close_kernel: int,
    building_close_kernel: int,
    building_open_kernel: int,
    line_min_major_len: int,
    line_dark_keep_delta: int,
    enclosed_min_density: float,
    enclosed_min_side: int,
    enclosed_max_area_ratio: float,
    enclosed_dist_min: float,
    enclosed_dark_delta: int,
    enclosed_tan_ratio_keep: float,
    enclosed_green_ratio_max: float,
    enclosed_blue_ratio_max: float,
    enclosed_boundary_ratio_keep: float,
    enclosed_compact_area_ratio: float,
    enclosed_compact_min_density: float,
    fill_hole_max_area: int,
    fill_hole_max_thickness: int,
    min_area_ratio: float,
    min_area_px: int,
    building_max_area_ratio: float,
    building_min_side: int,
    building_max_aspect: float,
    building_min_density: float,
    building_tan_ratio_keep: float,
    suppress_green_blue: int,
) -> np.ndarray:
    h, w = map_bgr.shape[:2]

    hsv = cv2.cvtColor(map_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    lab = cv2.cvtColor(map_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    gray = cv2.cvtColor(map_bgr, cv2.COLOR_BGR2GRAY)
    gray_med = cv2.medianBlur(gray, 3)

    bg_bgr = estimate_background_bgr(map_bgr=map_bgr, neutral_max_diff=int(bg_neutral_max_diff))
    bg_gray = int(cv2.cvtColor(bg_bgr.reshape(1, 1, 3), cv2.COLOR_BGR2GRAY)[0, 0])
    bg_lab = cv2.cvtColor(bg_bgr.reshape(1, 1, 3), cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0, :]
    color_dist = np.linalg.norm(lab - bg_lab.reshape(1, 1, 3), axis=2)

    # Arka plan: arka plan rengine yakin + yeterince acik veya klasik acik-gri kriteri.
    bg_by_color = color_dist <= float(bg_dist_threshold)
    bg_by_hsv = (sat <= int(bg_sat_max)) & (val >= int(bg_val_min))
    background = bg_by_color
    bg_ratio = float(background.mean())
    if bg_ratio < 0.08:
        # Arka plan tahmini anormal ise daha guvenli birlesik kosula don.
        background = bg_by_hsv | (bg_by_color & (sat <= max(24, int(bg_sat_max) + 12)))

    # Adaptif esikler
    line_threshold = max(28, bg_gray - int(line_delta))
    dark_threshold = max(12, bg_gray - int(dark_delta))

    # Cizgi katmani: koyu gri/siyah cizgiler + sari/mavi cizgiler
    line_gray = (gray_med <= line_threshold) & (sat <= int(gray_line_sat_max))
    yellow_mask = (
        (hsv[:, :, 0] >= int(yellow_h_min))
        & (hsv[:, :, 0] <= int(yellow_h_max))
        & (sat >= int(yellow_s_min))
        & (val >= int(yellow_v_min))
    )
    blue_mask = (
        (hsv[:, :, 0] >= int(blue_h_min))
        & (hsv[:, :, 0] <= int(blue_h_max))
        & (sat >= int(blue_s_min))
        & (val >= int(blue_v_min))
        & (gray_med <= (bg_gray - 8))
    )
    line_seed = (line_gray | yellow_mask | blue_mask) & (~background)
    line_seed_u8 = (line_seed.astype(np.uint8) * 255)
    # Dusuk doygunluklu arka-plan-disi pikseller, acik gri bina tonlarini yakalamada yardimci olur.
    low_sat_fg = sat <= max(30, int(gray_line_sat_max))
    color_seed = (color_dist >= float(seed_dist_threshold)) & (~background) & low_sat_fg
    color_seed_u8 = (color_seed.astype(np.uint8) * 255)
    if building_open_kernel > 1:
        k_seed = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (int(building_open_kernel), int(building_open_kernel))
        )
        color_seed_u8 = cv2.morphologyEx(color_seed_u8, cv2.MORPH_OPEN, k_seed, iterations=1)

    # Yerel kararma (black-hat) ile dusuk kontrast cizgileri yakala.
    blackhat = cv2.morphologyEx(
        gray_med, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    )
    line_blackhat = (
        (blackhat >= 4)
        & (sat <= max(48, int(gray_line_sat_max) + 14))
        & (~background)
    )
    line_seed_u8 = cv2.bitwise_or(line_seed_u8, (line_blackhat.astype(np.uint8) * 255))

    # Ince gri cizgiler (demiryolu/servis) icin yerel kontrast tabanli seed.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray_med)
    thin_adapt_u8 = cv2.adaptiveThreshold(
        gray_eq, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 3
    )
    thin_mask = (
        (thin_adapt_u8 > 0)
        & (sat <= max(48, int(gray_line_sat_max) + 14))
        & (~background)
    )
    thin_u8 = (thin_mask.astype(np.uint8) * 255)
    line_seed_u8 = cv2.bitwise_or(line_seed_u8, thin_u8)

    # Canny sinirlari, acik gri/duz alan kenarlarini yakalamada yardimci olur.
    gray_median = float(np.median(gray_med))
    canny_low = int(max(0.0, 0.66 * gray_median))
    canny_high = int(min(255.0, 1.33 * gray_median))
    canny_u8 = cv2.Canny(gray_med, canny_low, canny_high)
    canny_mask = (canny_u8 > 0) & (sat <= max(80, int(gray_line_sat_max) + 35)) & (~background)
    line_for_enclosed = cv2.bitwise_or(line_seed_u8, (canny_mask.astype(np.uint8) * 255))

    line_u8 = line_seed_u8.copy()

    if line_close_kernel > 1:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (int(line_close_kernel), int(line_close_kernel))
        )
        line_u8 = cv2.morphologyEx(line_u8, cv2.MORPH_CLOSE, k, iterations=1)
        line_for_enclosed = cv2.morphologyEx(line_for_enclosed, cv2.MORPH_CLOSE, k, iterations=1)

    # Uzun-yonlu kapanma: dashed/kopuk sinirlari birlestirir.
    if line_long_close_kernel > 1:
        k_h = cv2.getStructuringElement(cv2.MORPH_RECT, (int(line_long_close_kernel), 1))
        k_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(line_long_close_kernel)))
        line_h = cv2.morphologyEx(line_u8, cv2.MORPH_CLOSE, k_h, iterations=1)
        line_v = cv2.morphologyEx(line_u8, cv2.MORPH_CLOSE, k_v, iterations=1)
        line_u8 = cv2.bitwise_or(line_u8, line_h)
        line_u8 = cv2.bitwise_or(line_u8, line_v)

        line_encl_h = cv2.morphologyEx(line_for_enclosed, cv2.MORPH_CLOSE, k_h, iterations=1)
        line_encl_v = cv2.morphologyEx(line_for_enclosed, cv2.MORPH_CLOSE, k_v, iterations=1)
        line_for_enclosed = cv2.bitwise_or(line_for_enclosed, line_encl_h)
        line_for_enclosed = cv2.bitwise_or(line_for_enclosed, line_encl_v)

    auto_min_area = int(h * w * float(min_area_ratio))
    final_min_area = max(int(min_area_px), auto_min_area)

    line_u8 = filter_line_components(
        binary_u8=line_u8,
        gray=gray_med,
        bg_gray=bg_gray,
        min_area_px=max(2, int(final_min_area // 2)),
        min_major_len=max(8, int(line_min_major_len // 2)),
        dark_keep_delta=int(line_dark_keep_delta),
    )

    # Bina rengi (Google/OpenStreetMap benzeri acik turuncu/tan)
    tan_mask = (
        (hsv[:, :, 0] >= int(tan_h_min))
        & (hsv[:, :, 0] <= int(tan_h_max))
        & (sat >= int(tan_s_min))
        & (val >= int(tan_v_min))
    )
    tan_u8 = (tan_mask.astype(np.uint8) * 255)

    # Kapanmis cizgi yapilarindan dogrudan kapali bolge cikart.
    enclosed_u8 = extract_enclosed_regions(line_for_enclosed)
    if building_close_kernel > 1:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (int(building_close_kernel), int(building_close_kernel))
        )
        enclosed_u8 = cv2.morphologyEx(enclosed_u8, cv2.MORPH_CLOSE, k, iterations=1)

    enclosed_max_area = int(h * w * float(enclosed_max_area_ratio))
    enclosed_u8 = filter_enclosed_components(
        enclosed_u8=enclosed_u8,
        boundary_u8=line_for_enclosed,
        gray=gray_med,
        hsv=hsv,
        color_dist=color_dist,
        tan_mask_u8=tan_u8,
        bg_gray=bg_gray,
        min_area_px=max(10, int(final_min_area)),
        max_area_px=max(12, enclosed_max_area),
        min_side=int(enclosed_min_side),
        min_density=float(enclosed_min_density),
        min_color_dist=float(enclosed_dist_min),
        dark_delta=int(enclosed_dark_delta),
        tan_ratio_keep=float(enclosed_tan_ratio_keep),
        green_ratio_max=float(enclosed_green_ratio_max),
        blue_ratio_max=float(enclosed_blue_ratio_max),
        boundary_ratio_keep=float(enclosed_boundary_ratio_keep),
        compact_area_max_px=max(18, int(h * w * float(enclosed_compact_area_ratio))),
        compact_min_density=float(enclosed_compact_min_density),
    )

    # Bina tohumlari: koyu yapilar + tan renk + kapali bolge priori.
    dark_seed = (gray_med <= dark_threshold) & (sat <= max(42, int(gray_line_sat_max) + 6))
    enclosed_ratio = float((enclosed_u8 > 0).mean())
    use_color_seed = enclosed_ratio < 0.004
    building_seed = (dark_seed | tan_mask | (enclosed_u8 > 0)) & (~background)
    if use_color_seed:
        building_seed = building_seed | ((color_seed_u8 > 0) & (~background))
    building_u8 = (building_seed.astype(np.uint8) * 255)

    if building_close_kernel > 1:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (int(building_close_kernel), int(building_close_kernel))
        )
        building_u8 = cv2.morphologyEx(building_u8, cv2.MORPH_CLOSE, k, iterations=1)
    if building_open_kernel > 1:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (int(building_open_kernel), int(building_open_kernel))
        )
        building_u8 = cv2.morphologyEx(building_u8, cv2.MORPH_OPEN, k, iterations=1)

    max_area = int(h * w * float(building_max_area_ratio))
    building_u8 = remove_small_components(building_u8, min_area_px=final_min_area)
    building_u8 = remove_large_components(building_u8, max_area_px=max_area)
    building_u8 = filter_building_components(
        binary_u8=building_u8,
        tan_mask_u8=tan_u8,
        gray=gray_med,
        min_area_px=final_min_area,
        max_area_px=max_area,
        min_side=max(int(building_min_side), int(enclosed_min_side)),
        max_aspect=float(building_max_aspect),
        min_density=float(building_min_density),
        tan_ratio_keep=float(building_tan_ratio_keep),
    )

    # Kapali bolge bilgisini bina katmanina tekrar birlestir.
    building_u8 = cv2.bitwise_or(building_u8, enclosed_u8)

    # Cizgi katmaninda sadece guclu/uzun bile?enleri son kata al.
    line_strict = filter_line_components(
        binary_u8=line_u8,
        gray=gray_med,
        bg_gray=bg_gray,
        min_area_px=max(6, int(final_min_area // 2)),
        min_major_len=max(12, int(line_min_major_len) - 2),
        dark_keep_delta=int(line_dark_keep_delta),
    )

    # Son katman: binalar + guclu cizgiler
    final_fg = cv2.bitwise_or(building_u8, line_strict)
    final_fg = fill_small_holes(
        final_fg,
        max_hole_area=int(fill_hole_max_area),
        max_hole_thickness=int(fill_hole_max_thickness),
    )

    # Istek halinde yesil/mavi duz alan bastirma (varsayilan: kapali)
    if int(suppress_green_blue) != 0:
        green_or_blue = (
            (hsv[:, :, 0] >= 35)
            & (hsv[:, :, 0] <= 135)
            & (sat >= 25)
            & (gray_med >= (bg_gray - 45))
        )
        final_fg[green_or_blue & (building_u8 == 0)] = 0
    final_fg[background] = 0

    # Islenen nesneler siyah (0), arka plan acik gri
    black_object_mask = np.full((h, w), int(output_bg_gray), dtype=np.uint8)
    black_object_mask[final_fg > 0] = 0
    return black_object_mask

def split_side_by_side(img_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h, w = img_bgr.shape[:2]
    half = w // 2
    left = img_bgr[:, :half]
    right = img_bgr[:, half : half * 2]
    return left, right


def sample_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    lo = float(low)
    hi = float(high)
    if hi < lo:
        lo, hi = hi, lo
    if abs(hi - lo) < 1e-9:
        return lo
    return float(rng.uniform(lo, hi))


def apply_random_shadow(
    img_f32: np.ndarray,
    rng: np.random.Generator,
    prob: float,
    strength_min: float,
    strength_max: float,
) -> np.ndarray:
    if rng.random() >= float(prob):
        return img_f32

    h, w = img_f32.shape[:2]
    xs = np.linspace(-1.0, 1.0, w, dtype=np.float32)
    ys = np.linspace(-1.0, 1.0, h, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)

    angle = float(rng.uniform(0.0, np.pi))
    grad = (np.cos(angle) * xx) + (np.sin(angle) * yy)
    grad = (grad - grad.min()) / max(1e-6, float(grad.max() - grad.min()))

    pivot = float(rng.uniform(0.3, 0.7))
    softness = float(rng.uniform(0.06, 0.18))
    smooth = 1.0 / (1.0 + np.exp(-(grad - pivot) / max(1e-4, softness)))
    if rng.random() < 0.5:
        smooth = 1.0 - smooth

    shadow_strength = sample_uniform(rng, strength_min, strength_max)
    shade = 1.0 - ((1.0 - shadow_strength) * smooth)
    shaded = img_f32 * shade[:, :, None]
    return np.clip(shaded, 0.0, 1.0)


def augment_satellite_bgr(
    sat_bgr: np.ndarray,
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> np.ndarray:
    img = sat_bgr.astype(np.float32) / 255.0

    # Exposure + contrast
    contrast = sample_uniform(rng, args.sat_contrast_min, args.sat_contrast_max)
    exposure = sample_uniform(rng, args.sat_exp_min, args.sat_exp_max) / 255.0
    img = ((img - 0.5) * contrast) + 0.5 + exposure
    img = np.clip(img, 0.0, 1.0)

    # Saturation + hue
    hsv = cv2.cvtColor((img * 255.0).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    sat_gain = sample_uniform(rng, args.sat_sat_min, args.sat_sat_max)
    hue_shift = sample_uniform(rng, -args.sat_hue_shift_max, args.sat_hue_shift_max)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_gain, 0.0, 255.0)
    hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180.0
    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0

    # Gamma
    gamma = max(0.01, sample_uniform(rng, args.sat_gamma_min, args.sat_gamma_max))
    img = np.power(np.clip(img, 0.0, 1.0), gamma)

    # Sicaklik (warm/cool)
    temp_shift = sample_uniform(rng, -args.sat_temp_shift_max, args.sat_temp_shift_max) / 255.0
    img[:, :, 2] = np.clip(img[:, :, 2] + temp_shift, 0.0, 1.0)  # red
    img[:, :, 0] = np.clip(img[:, :, 0] - (temp_shift * 0.9), 0.0, 1.0)  # blue

    # Yerel golge simulasyonu
    img = apply_random_shadow(
        img_f32=img,
        rng=rng,
        prob=args.sat_shadow_prob,
        strength_min=args.sat_shadow_strength_min,
        strength_max=args.sat_shadow_strength_max,
    )

    # Sensor noise
    sigma = sample_uniform(rng, args.sat_noise_sigma_min, args.sat_noise_sigma_max) / 255.0
    if sigma > 0.0:
        noise = rng.normal(0.0, sigma, size=img.shape).astype(np.float32)
        img = np.clip(img + noise, 0.0, 1.0)

    # Hafif blur (hareket/optik)
    if rng.random() < float(args.sat_blur_prob):
        img = cv2.GaussianBlur(img, (3, 3), sigmaX=0.7, sigmaY=0.7)

    return np.clip((img * 255.0) + 0.5, 0.0, 255.0).astype(np.uint8)


def build_satellite_variants(
    sat_bgr: np.ndarray,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> List[Tuple[str, np.ndarray]]:
    if args.mode != "paired" or int(args.sat_aug_enable) == 0:
        return [("", sat_bgr)]

    variants: List[Tuple[str, np.ndarray]] = []
    keep_original = int(args.sat_aug_keep_original) != 0
    aug_count = max(0, int(args.sat_aug_count))

    if keep_original:
        variants.append(("", sat_bgr))

    for idx in range(aug_count):
        aug_sat = augment_satellite_bgr(sat_bgr, rng=rng, args=args)
        suffix = f"_aug{idx + 1:02d}" if (keep_original or aug_count > 1) else ""
        variants.append((suffix, aug_sat))

    if not variants:
        variants.append(("", sat_bgr))
    return variants


def save_preview(
    out_path: Path,
    satellite_bgr: np.ndarray,
    map_bgr: np.ndarray,
    mask_gray: np.ndarray,
) -> None:
    mask_bgr = cv2.cvtColor(mask_gray, cv2.COLOR_GRAY2BGR)
    overlay = map_bgr.copy()
    overlay[mask_gray == 0] = (0, 0, 255)
    preview = np.concatenate([satellite_bgr, map_bgr, mask_bgr, overlay], axis=1)
    cv2.imwrite(str(out_path), preview)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def strip_known_extensions(filename: str) -> str:
    base = filename
    lowered = base.lower()
    changed = True
    while changed:
        changed = False
        for ext in IMAGE_EXTENSIONS:
            if lowered.endswith(ext):
                base = base[: -len(ext)]
                lowered = base.lower()
                changed = True
                break
    return base


def iter_inputs(input_dir: Path, input_file: Path) -> Iterable[Path]:
    if input_file is not None:
        yield input_file
    else:
        for p in list_images(input_dir):
            yield p


def get_cli_override_keys(argv: List[str]) -> Set[str]:
    keys: Set[str] = set()
    for token in argv:
        if not token.startswith("--"):
            continue
        key = token[2:]
        if "=" in key:
            key = key.split("=", 1)[0]
        keys.add(key.replace("-", "_"))
    return keys


def apply_mask_profile(args: argparse.Namespace, cli_override_keys: Set[str]) -> None:
    profile = str(getattr(args, "mask_profile", "balanced")).lower().strip()
    overrides = MASK_PROFILE_OVERRIDES.get(profile)
    if overrides is None:
        valid = ", ".join(sorted(MASK_PROFILE_OVERRIDES.keys()))
        raise ValueError(f"Gecersiz mask_profile: {profile}. Gecerli: {valid}")

    for key, value in overrides.items():
        # Kullanici CLI'da acikca verdiyse profile degeriyle ezme.
        if key in cli_override_keys:
            continue
        if hasattr(args, key):
            setattr(args, key, value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sag harita yarimindan bina/yol/nesne maske uretir."
    )
    parser.add_argument("--input_dir", type=str, default=CONFIG["input_dir"])
    parser.add_argument("--input_file", type=str, default=CONFIG["input_file"])
    parser.add_argument("--output_dir", type=str, default=CONFIG["output_dir"])
    parser.add_argument(
        "--mode",
        type=str,
        default=CONFIG["mode"],
        choices=["paired", "mask"],
        help="paired: sol uydu + sag maske, mask: sadece maske.",
    )
    parser.add_argument(
        "--output_ext",
        type=str,
        default=CONFIG["output_ext"],
        help="Cikti uzantisi (.png onerilir).",
    )
    parser.add_argument(
        "--save_preview",
        type=int,
        default=CONFIG["save_preview"],
        help="Ilk N dosya icin onizleme kaydet (0: kapali).",
    )
    parser.add_argument("--preview_dir", type=str, default=CONFIG["preview_dir"])
    parser.add_argument(
        "--mask_profile",
        type=str,
        default=CONFIG["mask_profile"],
        choices=sorted(MASK_PROFILE_OVERRIDES.keys()),
        help="Segmentasyon profili: balanced veya strict.",
    )

    # Uydu tarafi augmentation
    parser.add_argument("--sat_aug_enable", type=int, default=CONFIG["sat_aug_enable"])
    parser.add_argument(
        "--sat_aug_keep_original", type=int, default=CONFIG["sat_aug_keep_original"]
    )
    parser.add_argument("--sat_aug_count", type=int, default=CONFIG["sat_aug_count"])
    parser.add_argument("--sat_aug_seed", type=int, default=CONFIG["sat_aug_seed"])
    parser.add_argument("--sat_exp_min", type=float, default=CONFIG["sat_exp_min"])
    parser.add_argument("--sat_exp_max", type=float, default=CONFIG["sat_exp_max"])
    parser.add_argument("--sat_contrast_min", type=float, default=CONFIG["sat_contrast_min"])
    parser.add_argument("--sat_contrast_max", type=float, default=CONFIG["sat_contrast_max"])
    parser.add_argument("--sat_gamma_min", type=float, default=CONFIG["sat_gamma_min"])
    parser.add_argument("--sat_gamma_max", type=float, default=CONFIG["sat_gamma_max"])
    parser.add_argument("--sat_sat_min", type=float, default=CONFIG["sat_sat_min"])
    parser.add_argument("--sat_sat_max", type=float, default=CONFIG["sat_sat_max"])
    parser.add_argument("--sat_hue_shift_max", type=float, default=CONFIG["sat_hue_shift_max"])
    parser.add_argument("--sat_temp_shift_max", type=float, default=CONFIG["sat_temp_shift_max"])
    parser.add_argument("--sat_shadow_prob", type=float, default=CONFIG["sat_shadow_prob"])
    parser.add_argument(
        "--sat_shadow_strength_min", type=float, default=CONFIG["sat_shadow_strength_min"]
    )
    parser.add_argument(
        "--sat_shadow_strength_max", type=float, default=CONFIG["sat_shadow_strength_max"]
    )
    parser.add_argument("--sat_noise_sigma_min", type=float, default=CONFIG["sat_noise_sigma_min"])
    parser.add_argument("--sat_noise_sigma_max", type=float, default=CONFIG["sat_noise_sigma_max"])
    parser.add_argument("--sat_blur_prob", type=float, default=CONFIG["sat_blur_prob"])

    # Segmentasyon parametreleri
    parser.add_argument("--output_bg_gray", type=int, default=CONFIG["output_bg_gray"])
    parser.add_argument("--bg_dist_threshold", type=float, default=CONFIG["bg_dist_threshold"])
    parser.add_argument("--bg_sat_max", type=int, default=CONFIG["bg_sat_max"])
    parser.add_argument("--bg_val_min", type=int, default=CONFIG["bg_val_min"])
    parser.add_argument("--bg_neutral_max_diff", type=int, default=CONFIG["bg_neutral_max_diff"])
    parser.add_argument("--seed_dist_threshold", type=float, default=CONFIG["seed_dist_threshold"])
    parser.add_argument("--line_delta", type=int, default=CONFIG["line_delta"])
    parser.add_argument("--dark_delta", type=int, default=CONFIG["dark_delta"])
    parser.add_argument("--gray_line_sat_max", type=int, default=CONFIG["gray_line_sat_max"])
    parser.add_argument("--tan_h_min", type=int, default=CONFIG["tan_h_min"])
    parser.add_argument("--tan_h_max", type=int, default=CONFIG["tan_h_max"])
    parser.add_argument("--tan_s_min", type=int, default=CONFIG["tan_s_min"])
    parser.add_argument("--tan_v_min", type=int, default=CONFIG["tan_v_min"])
    parser.add_argument("--yellow_h_min", type=int, default=CONFIG["yellow_h_min"])
    parser.add_argument("--yellow_h_max", type=int, default=CONFIG["yellow_h_max"])
    parser.add_argument("--yellow_s_min", type=int, default=CONFIG["yellow_s_min"])
    parser.add_argument("--yellow_v_min", type=int, default=CONFIG["yellow_v_min"])
    parser.add_argument("--blue_h_min", type=int, default=CONFIG["blue_h_min"])
    parser.add_argument("--blue_h_max", type=int, default=CONFIG["blue_h_max"])
    parser.add_argument("--blue_s_min", type=int, default=CONFIG["blue_s_min"])
    parser.add_argument("--blue_v_min", type=int, default=CONFIG["blue_v_min"])
    parser.add_argument("--line_close_kernel", type=int, default=CONFIG["line_close_kernel"])
    parser.add_argument(
        "--line_long_close_kernel", type=int, default=CONFIG["line_long_close_kernel"]
    )
    parser.add_argument(
        "--building_close_kernel", type=int, default=CONFIG["building_close_kernel"]
    )
    parser.add_argument("--building_open_kernel", type=int, default=CONFIG["building_open_kernel"])
    parser.add_argument("--line_min_major_len", type=int, default=CONFIG["line_min_major_len"])
    parser.add_argument("--line_dark_keep_delta", type=int, default=CONFIG["line_dark_keep_delta"])
    parser.add_argument("--enclosed_min_density", type=float, default=CONFIG["enclosed_min_density"])
    parser.add_argument("--enclosed_min_side", type=int, default=CONFIG["enclosed_min_side"])
    parser.add_argument(
        "--enclosed_max_area_ratio", type=float, default=CONFIG["enclosed_max_area_ratio"]
    )
    parser.add_argument("--enclosed_dist_min", type=float, default=CONFIG["enclosed_dist_min"])
    parser.add_argument("--enclosed_dark_delta", type=int, default=CONFIG["enclosed_dark_delta"])
    parser.add_argument(
        "--enclosed_tan_ratio_keep", type=float, default=CONFIG["enclosed_tan_ratio_keep"]
    )
    parser.add_argument(
        "--enclosed_green_ratio_max", type=float, default=CONFIG["enclosed_green_ratio_max"]
    )
    parser.add_argument(
        "--enclosed_blue_ratio_max", type=float, default=CONFIG["enclosed_blue_ratio_max"]
    )
    parser.add_argument(
        "--enclosed_boundary_ratio_keep",
        type=float,
        default=CONFIG["enclosed_boundary_ratio_keep"],
    )
    parser.add_argument(
        "--enclosed_compact_area_ratio",
        type=float,
        default=CONFIG["enclosed_compact_area_ratio"],
    )
    parser.add_argument(
        "--enclosed_compact_min_density",
        type=float,
        default=CONFIG["enclosed_compact_min_density"],
    )
    parser.add_argument("--fill_hole_max_area", type=int, default=CONFIG["fill_hole_max_area"])
    parser.add_argument(
        "--fill_hole_max_thickness", type=int, default=CONFIG["fill_hole_max_thickness"]
    )
    parser.add_argument("--min_area_ratio", type=float, default=CONFIG["min_area_ratio"])
    parser.add_argument("--min_area_px", type=int, default=CONFIG["min_area_px"])
    parser.add_argument(
        "--building_max_area_ratio", type=float, default=CONFIG["building_max_area_ratio"]
    )
    parser.add_argument("--building_min_side", type=int, default=CONFIG["building_min_side"])
    parser.add_argument("--building_max_aspect", type=float, default=CONFIG["building_max_aspect"])
    parser.add_argument("--building_min_density", type=float, default=CONFIG["building_min_density"])
    parser.add_argument(
        "--building_tan_ratio_keep", type=float, default=CONFIG["building_tan_ratio_keep"]
    )
    parser.add_argument("--suppress_green_blue", type=int, default=CONFIG["suppress_green_blue"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cli_keys = get_cli_override_keys(sys.argv[1:])
    apply_mask_profile(args, cli_override_keys=cli_keys)

    input_dir = Path(args.input_dir)
    input_file = Path(args.input_file) if args.input_file else None
    output_dir = Path(args.output_dir)
    preview_dir = Path(args.preview_dir)
    output_ext = args.output_ext if args.output_ext.startswith(".") else f".{args.output_ext}"

    ensure_dir(output_dir)
    if args.save_preview > 0:
        ensure_dir(preview_dir)

    files = list(iter_inputs(input_dir=input_dir, input_file=input_file))
    if not files:
        raise FileNotFoundError("Islenecek goruntu bulunamadi.")

    rng_seed = None if int(args.sat_aug_seed) < 0 else int(args.sat_aug_seed)
    rng = np.random.default_rng(rng_seed)

    print(f"[INFO] mask_profile: {args.mask_profile}")

    processed_inputs = 0
    written_outputs = 0
    failed = 0
    preview_written = 0

    for idx, path in enumerate(files, start=1):
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            failed += 1
            print(f"[HATA] Okunamadi: {path}")
            continue

        sat_bgr, map_bgr = split_side_by_side(img)
        mask = build_black_object_mask(
            map_bgr=map_bgr,
            output_bg_gray=args.output_bg_gray,
            bg_dist_threshold=args.bg_dist_threshold,
            bg_sat_max=args.bg_sat_max,
            bg_val_min=args.bg_val_min,
            bg_neutral_max_diff=args.bg_neutral_max_diff,
            seed_dist_threshold=args.seed_dist_threshold,
            line_delta=args.line_delta,
            dark_delta=args.dark_delta,
            gray_line_sat_max=args.gray_line_sat_max,
            tan_h_min=args.tan_h_min,
            tan_h_max=args.tan_h_max,
            tan_s_min=args.tan_s_min,
            tan_v_min=args.tan_v_min,
            yellow_h_min=args.yellow_h_min,
            yellow_h_max=args.yellow_h_max,
            yellow_s_min=args.yellow_s_min,
            yellow_v_min=args.yellow_v_min,
            blue_h_min=args.blue_h_min,
            blue_h_max=args.blue_h_max,
            blue_s_min=args.blue_s_min,
            blue_v_min=args.blue_v_min,
            line_close_kernel=args.line_close_kernel,
            line_long_close_kernel=args.line_long_close_kernel,
            building_close_kernel=args.building_close_kernel,
            building_open_kernel=args.building_open_kernel,
            line_min_major_len=args.line_min_major_len,
            line_dark_keep_delta=args.line_dark_keep_delta,
            enclosed_min_density=args.enclosed_min_density,
            enclosed_min_side=args.enclosed_min_side,
            enclosed_max_area_ratio=args.enclosed_max_area_ratio,
            enclosed_dist_min=args.enclosed_dist_min,
            enclosed_dark_delta=args.enclosed_dark_delta,
            enclosed_tan_ratio_keep=args.enclosed_tan_ratio_keep,
            enclosed_green_ratio_max=args.enclosed_green_ratio_max,
            enclosed_blue_ratio_max=args.enclosed_blue_ratio_max,
            enclosed_boundary_ratio_keep=args.enclosed_boundary_ratio_keep,
            enclosed_compact_area_ratio=args.enclosed_compact_area_ratio,
            enclosed_compact_min_density=args.enclosed_compact_min_density,
            fill_hole_max_area=args.fill_hole_max_area,
            fill_hole_max_thickness=args.fill_hole_max_thickness,
            min_area_ratio=args.min_area_ratio,
            min_area_px=args.min_area_px,
            building_max_area_ratio=args.building_max_area_ratio,
            building_min_side=args.building_min_side,
            building_max_aspect=args.building_max_aspect,
            building_min_density=args.building_min_density,
            building_tan_ratio_keep=args.building_tan_ratio_keep,
            suppress_green_blue=args.suppress_green_blue,
        )

        clean_base_name = strip_known_extensions(path.name)
        variants = build_satellite_variants(sat_bgr=sat_bgr, args=args, rng=rng)
        if args.mode == "mask":
            variants = [("", sat_bgr)]

        wrote_for_this_input = 0
        right_mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        for suffix, sat_variant in variants:
            if args.mode == "paired":
                out_img = np.concatenate([sat_variant, right_mask_bgr], axis=1)
            else:
                out_img = mask

            out_name = f"{clean_base_name}{suffix}{output_ext}"
            out_path = output_dir / out_name
            ok = cv2.imwrite(str(out_path), out_img)
            if not ok:
                failed += 1
                print(f"[HATA] Yazilamadi: {out_path}")
                continue

            if preview_written < args.save_preview:
                preview_tag = suffix if suffix else "_orig"
                preview_path = preview_dir / f"{clean_base_name}{preview_tag}_preview{output_ext}"
                preview_sat = sat_variant if args.mode == "paired" else sat_bgr
                save_preview(preview_path, preview_sat, map_bgr, mask)
                preview_written += 1

            wrote_for_this_input += 1
            written_outputs += 1

        if wrote_for_this_input > 0:
            processed_inputs += 1
        else:
            failed += 1

        if idx % 200 == 0 or idx == len(files):
            print(f"[INFO] {idx}/{len(files)} islenen dosya.")

    print("-" * 50)
    print(f"Toplam giris: {len(files)}")
    print(f"Basarili giris: {processed_inputs}")
    print(f"Uretilen cikti: {written_outputs}")
    print(f"Hata: {failed}")
    print(f"Cikti klasoru: {output_dir}")
    if args.save_preview > 0:
        print(f"Onizleme klasoru: {preview_dir}")


if __name__ == "__main__":
    main()
