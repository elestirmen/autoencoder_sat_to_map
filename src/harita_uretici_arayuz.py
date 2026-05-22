# -*- coding: utf-8 -*-
"""
Harita Üretici Arayüz - Uydu Görüntüsü → Harita Pipeline
=========================================================
goruntu_islemleri.py içindeki run_full_pipeline'ı saran web arayüzü.
Bir uydu görüntüsünü: böl → modelden geçir → birleştir → jeoreferansla.

Çalıştırma:
    pip install gradio
    python harita_uretici_arayuz.py

Tarayıcıda http://127.0.0.1:7860 açılır.
"""

import os
import sys
import html
import time
import logging
import threading
import traceback

# Tüm göreli yollar (modeller/, georeferans_sample/, bolunmus/ ...) proje
# köküne göre çözülsün diye çalışma dizinini proje köküne sabitle.
# Bu betik src/ altında olduğundan proje kökü bir üst dizindir.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_PROJECT_ROOT)

try:
    import gradio as gr
except ImportError:
    print("Gradio yüklü değil. Kurulum için:  pip install gradio")
    sys.exit(1)

import cv2

from goruntu_islemleri import ImageProcessor, CONFIG, TENSORFLOW_AVAILABLE


# --- Açılır liste sentinel (özel) değerleri ---
MODEL_ALL = "🗂  Tüm modeller"
MODEL_NONE = "🚫  Model yok (sadece böl + birleştir)"
REF_AUTO = "✨  Otomatik (dosya adına göre, yoksa girişin kendi referansı)"
REF_FROM_INPUT = "📍  Giriş dosyasının kendi coğrafi bilgisini kullan"

# Karo, sinir ağına verilmeden önce uygulanacak normalizasyon seçenekleri.
# (görünen etiket, goruntu_islemleri'ne gönderilen değer)
NORMALIZATION_CHOICES = [
    ("[-1, 1]  ·  tanh modelleri (varsayılan)", "minus1_1"),
    ("[0, 1]  ·  sigmoid modelleri", "zero_1"),
    ("[0, 255]  ·  ham (normalize yok)", "raw"),
    ("Z-skoru  ·  karo bazlı standardizasyon", "zscore"),
]

# Normalizasyondan önce karoya uygulanacak kontrast/histogram iyileştirmesi.
ENHANCEMENT_CHOICES = [
    ("Yok  ·  iyileştirme uygulanmaz", "none"),
    ("Histogram eşitleme  ·  global kontrast", "hist_eq"),
    ("CLAHE  ·  uyarlamalı yerel kontrast", "clahe"),
]


# ----------------------------------------------------------------------------
# Klasör tarama yardımcıları
# ----------------------------------------------------------------------------
def list_models():
    """modeller/ klasöründeki model dosyalarını döndürür."""
    model_dir = CONFIG["pipeline"]["model_dir"]
    if not os.path.isdir(model_dir):
        return []
    return sorted(f for f in os.listdir(model_dir) if f.lower().endswith(('.h5', '.keras')))


def list_references():
    """georeferans_sample/ klasöründeki referans raster'ları döndürür."""
    ref_dir = CONFIG["pipeline"]["reference_dir"]
    if not os.path.isdir(ref_dir):
        return []
    return sorted(f for f in os.listdir(ref_dir) if f.lower().endswith(('.tif', '.tiff')))


def model_choices():
    return list_models() + [MODEL_ALL, MODEL_NONE]


def default_model():
    files = list_models()
    if not TENSORFLOW_AVAILABLE or not files:
        return MODEL_NONE
    return files[0]


def reference_choices():
    return [REF_AUTO, REF_FROM_INPUT] + list_references()


# ----------------------------------------------------------------------------
# Durum rozeti (HTML) - renk kodlu görsel geri bildirim
# ----------------------------------------------------------------------------
_STATUS_STYLES = {
    "idle":    ("#475569", "#f1f5f9", "#cbd5e1"),
    "running": ("#1d4ed8", "#dbeafe", "#93c5fd"),
    "done":    ("#15803d", "#dcfce7", "#86efac"),
    "warn":    ("#b45309", "#fef3c7", "#fcd34d"),
    "error":   ("#b91c1c", "#fee2e2", "#fca5a5"),
}


def _status_html(kind, message):
    fg, bg, border = _STATUS_STYLES.get(kind, _STATUS_STYLES["idle"])
    icon = {"idle": "○", "running": "◔", "done": "✓",
            "warn": "!", "error": "✕"}.get(kind, "○")
    spin = ' class="spin"' if kind == "running" else ""
    message = html.escape(str(message))
    return (
        f'<div class="status-badge" style="background:{bg};border-color:{border};color:{fg};">'
        f'<span class="status-icon"{spin}>{icon}</span>'
        f'<span class="status-text">{message}</span>'
        f'</div>'
    )


# ----------------------------------------------------------------------------
# Canlı log: logger mesajları + stdout/stderr (tqdm ilerleme çubukları dahil)
# ----------------------------------------------------------------------------
class _LiveLog:
    """Pipeline'ın tüm çıktısını thread-güvenli biriktirir.

    tqdm ilerleme çubukları satır başı ('\\r') ile aynı satırı sürekli
    günceller; bu sınıf onu tek bir 'güncel satır' olarak tutar, böylece
    arayüzde ilerleme yüzdesi canlı akar.
    """

    _MAX_LINES = 800

    def __init__(self):
        self._lines = []        # kalıcı (tamamlanmış) satırlar
        self._current = ""      # son '\r'/'\n' sonrası işlenmekte olan satır
        self._lock = threading.Lock()

    def add_line(self, line):
        """logging handler'dan gelen tam bir satır ekler."""
        with self._lock:
            if self._current:
                self._lines.append(self._current)
                self._current = ""
            self._lines.append(line)
            self._trim()

    def feed(self, text):
        """stdout/stderr akışından gelen ham metni işler ('\\r' duyarlı)."""
        with self._lock:
            for ch in text:
                if ch == '\r':
                    self._current = ""
                elif ch == '\n':
                    self._lines.append(self._current)
                    self._current = ""
                else:
                    self._current += ch
            self._trim()

    def _trim(self):
        if len(self._lines) > self._MAX_LINES:
            self._lines = self._lines[-self._MAX_LINES:]

    def get_text(self):
        with self._lock:
            lines = list(self._lines)
            if self._current:
                lines.append(self._current)
            return "\n".join(lines)

    def snapshot(self):
        """(tam metin, son anlamlı satır) ikilisini döndürür."""
        with self._lock:
            lines = list(self._lines)
            if self._current:
                lines.append(self._current)
            last = self._current.strip()
            if not last:
                for ln in reversed(self._lines):
                    if ln.strip():
                        last = ln.strip()
                        break
            return "\n".join(lines), last


class _BufferLogHandler(logging.Handler):
    """goruntu_islemleri logger çıktısını _LiveLog'a yönlendirir."""

    def __init__(self, live):
        super().__init__()
        self._live = live

    def emit(self, record):
        try:
            msg = self.format(record)
        except Exception:
            return
        self._live.add_line(msg)


class _Tee:
    """sys.stdout/stderr'i hem orijinal akışa hem de _LiveLog'a yazan sarmalayıcı."""

    def __init__(self, original, live):
        self._original = original
        self._live = live

    def write(self, text):
        try:
            self._original.write(text)
        except Exception:
            pass
        try:
            self._live.feed(text)
        except Exception:
            pass
        return len(text)

    def flush(self):
        try:
            self._original.flush()
        except Exception:
            pass

    def __getattr__(self, name):
        # isatty(), encoding, fileno() vb. orijinal akışa devredilir.
        return getattr(self._original, name)


def _make_preview(merge_outputs, max_side=1600):
    """Birleştirilmiş mozaikten küçültülmüş bir önizleme (RGB) üretir."""
    for path in merge_outputs:
        if path and os.path.isfile(path):
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                continue
            h, w = img.shape[:2]
            scale = min(1.0, max_side / float(max(h, w)))
            if scale < 1.0:
                img = cv2.resize(
                    img,
                    (max(1, int(w * scale)), max(1, int(h * scale))),
                    interpolation=cv2.INTER_AREA,
                )
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return None


# ----------------------------------------------------------------------------
# Ana iş: pipeline'ı çalıştır (canlı log akıtan üreteç)
# ----------------------------------------------------------------------------
def run_pipeline(input_path, model_choice, model_file, reference_choice, reference_file,
                 tile_size, overlap, crop_overlap, batch_size, color_mode,
                 normalization, enhancement, clahe_clip):
    """Pipeline'ı ayrı bir thread'de çalıştırır; logu canlı akıtan generator.

    Çıktı sırası: log_box, status_html, preview, downloads, run_btn
    """
    run_free = gr.update(interactive=True, value="▶  Pipeline'ı Çalıştır")
    run_busy = gr.update(interactive=False, value="⏳  Çalışıyor…")

    input_path = (input_path or "").strip().strip('"')

    # --- Girdi doğrulama ---
    if not input_path:
        yield "", _status_html("error", "Giriş haritası seçilmedi."), None, None, run_free
        return
    if not os.path.isfile(input_path):
        yield (f"Giriş dosyası bulunamadı:\n{input_path}",
               _status_html("error", "Dosya bulunamadı."), None, None, run_free)
        return

    try:
        tile_size = int(tile_size)
        overlap = int(overlap)
        crop_overlap = int(crop_overlap)
        batch_size = int(batch_size)
    except (TypeError, ValueError):
        yield "", _status_html("error", "Sayısal ayarlar geçersiz."), None, None, run_free
        return

    if tile_size <= 0 or batch_size <= 0:
        yield ("", _status_html("error", "tile_size ve batch_size pozitif olmalı."),
               None, None, run_free)
        return
    if overlap < 0 or overlap >= tile_size:
        yield ("", _status_html("error",
               f"overlap ({overlap}) 0 ile tile_size ({tile_size}) arasında olmalı."),
               None, None, run_free)
        return

    # --- Model seçimi ---
    # Öncelik sırası: disk'ten yüklenen model dosyası > açılır listedeki seçim.
    model_path = None
    model_dir = None
    model_file = (model_file or "").strip().strip('"') if isinstance(model_file, str) else model_file
    if model_file:
        if not os.path.isfile(model_file):
            yield (f"Yüklenen model dosyası bulunamadı:\n{model_file}",
                   _status_html("error", "Model dosyası bulunamadı."), None, None, run_free)
            return
        if not model_file.lower().endswith(('.h5', '.keras')):
            yield ("", _status_html("error", "Model dosyası .h5 veya .keras olmalı."),
                   None, None, run_free)
            return
        model_path = model_file
    elif model_choice == MODEL_ALL:
        model_dir = CONFIG["pipeline"]["model_dir"]
    elif model_choice and model_choice != MODEL_NONE:
        model_path = os.path.join(CONFIG["pipeline"]["model_dir"], model_choice)
        if not os.path.isfile(model_path):
            yield (f"Model dosyası bulunamadı:\n{model_path}",
                   _status_html("error", "Model dosyası bulunamadı."), None, None, run_free)
            return

    # --- Referans raster seçimi ---
    # Öncelik: yüklenen dosya > açılır listedeki dosya > giriş/otomatik.
    reference_raster = None
    auto_reference = True
    reference_file = (reference_file or "").strip().strip('"') if isinstance(reference_file, str) else reference_file
    if reference_file:
        if not os.path.isfile(reference_file):
            yield (f"Yüklenen referans dosyası bulunamadı:\n{reference_file}",
                   _status_html("error", "Referans dosyası bulunamadı."), None, None, run_free)
            return
        if not reference_file.lower().endswith(('.tif', '.tiff')):
            yield ("", _status_html("error", "Referans dosyası .tif / .tiff (GeoTIFF) olmalı."),
                   None, None, run_free)
            return
        reference_raster = reference_file
    elif reference_choice == REF_FROM_INPUT:
        # Klasör taramasını atla; girişin kendi coğrafi bilgisi kullanılsın.
        auto_reference = False
    elif reference_choice and reference_choice != REF_AUTO:
        reference_raster = os.path.join(CONFIG["pipeline"]["reference_dir"], reference_choice)

    # --- Pipeline çıktısını yakala: logger mesajları + stdout/stderr (tqdm) ---
    live = _LiveLog()
    handler = _BufferLogHandler(live)
    handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', '%H:%M:%S'))
    gi_logger = logging.getLogger('goruntu_islemleri')
    gi_logger.addHandler(handler)

    holder = {}

    def worker():
        try:
            processor = ImageProcessor(reference_dir=CONFIG["pipeline"]["reference_dir"])
            holder['results'] = processor.run_full_pipeline(
                input_image=input_path,
                model_path=model_path,
                model_dir=model_dir,
                split_tile_size=tile_size,
                split_overlap=overlap,
                crop_overlap=crop_overlap,
                reference_raster=reference_raster,
                image_size=(tile_size, tile_size),
                color_mode=color_mode,
                batch_size=batch_size,
                normalization=normalization,
                auto_reference=auto_reference,
                enhancement=enhancement,
                clahe_clip=float(clahe_clip),
            )
        except Exception as exc:
            holder['error'] = exc
            holder['trace'] = traceback.format_exc()

    # tqdm ilerleme çubukları ve print() çıktısı sys.stdout/stderr'e yazar;
    # bunları _LiveLog'a yönlendirmek için akışları thread başlamadan önce sar.
    thread = threading.Thread(target=worker, daemon=True)
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = _Tee(old_out, live)
    sys.stderr = _Tee(old_err, live)

    start = time.time()
    try:
        thread.start()
        yield "", _status_html("running", "Pipeline başlatılıyor…"), None, None, run_busy
        # Thread çalışırken logu, geçen süreyi ve güncel adımı akıt.
        while thread.is_alive():
            elapsed = int(time.time() - start)
            text, last = live.snapshot()
            hint = f"  ·  {last[:90]}" if last else ""
            yield (text,
                   _status_html("running", f"Çalışıyor…  ·  {elapsed} sn{hint}"),
                   None, None, run_busy)
            time.sleep(0.4)
        thread.join()
    finally:
        sys.stdout, sys.stderr = old_out, old_err
        gi_logger.removeHandler(handler)

    log_text = live.get_text()
    elapsed = int(time.time() - start)

    if 'error' in holder:
        yield (
            log_text + "\n\n=== HATA ===\n" + holder.get('trace', str(holder['error'])),
            _status_html("error", f"İşlem tamamlanamadı  ·  {elapsed} sn"),
            None, None, run_free,
        )
        return

    results = holder.get('results', {}) or {}
    merge_outputs = [m.get('output_file') for m in results.get('merge', []) if m.get('output_file')]
    georef_outputs = [g.get('output') for g in results.get('georef', []) if g.get('output')]

    downloads = [p for p in (georef_outputs + merge_outputs) if p and os.path.isfile(p)]
    preview = _make_preview(merge_outputs)

    if georef_outputs:
        status = _status_html(
            "done",
            f"Tamamlandı  ·  {len(merge_outputs)} mozaik, "
            f"{len(georef_outputs)} GeoTIFF  ·  {elapsed} sn",
        )
    elif merge_outputs:
        status = _status_html(
            "warn",
            f"Mozaik üretildi; jeoreferanslama atlandı "
            f"(uygun referans bulunamadı)  ·  {elapsed} sn",
        )
    else:
        status = _status_html("warn", f"Bitti ama çıktı üretilmedi - günlüğü kontrol edin  ·  {elapsed} sn")

    yield log_text, status, preview, (downloads or None), run_free


def refresh_lists():
    """modeller/ ve georeferans_sample/ klasörlerini yeniden tarar."""
    return (
        gr.update(choices=model_choices(), value=default_model()),
        gr.update(choices=reference_choices(), value=REF_AUTO),
        _status_html("idle", "Listeler yenilendi."),
    )


def clear_all():
    """Arayüzü başlangıç durumuna döndürür."""
    return (
        None,                                  # input file
        None,                                  # model file
        None,                                  # reference file
        "",                                    # log
        _status_html("idle", "Hazır - bir giriş haritası seçin."),
        None,                                  # preview
        None,                                  # downloads
    )


# ----------------------------------------------------------------------------
# Arayüz teması ve stili
# ----------------------------------------------------------------------------
_THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.indigo,
    secondary_hue=gr.themes.colors.slate,
    neutral_hue=gr.themes.colors.slate,
    radius_size=gr.themes.sizes.radius_md,
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
)

_CSS = """
:root { --hm-grad: linear-gradient(135deg, #4f46e5 0%, #7c3aed 52%, #0ea5e9 100%); }

.gradio-container { max-width: 1280px !important; margin: 0 auto !important; }

/* --- Üst başlık --- */
#app-header {
    position: relative; overflow: hidden;
    background: linear-gradient(120deg, #3730a3 0%, #6d28d9 42%, #4f46e5 68%, #0ea5e9 100%);
    border-radius: 20px; padding: 30px 34px; margin-bottom: 12px;
    color: #fff; box-shadow: 0 16px 40px -14px rgba(79,70,229,.65);
}
#app-header::after {
    content:""; position:absolute; right:-70px; top:-90px;
    width:280px; height:280px; border-radius:50%;
    background: radial-gradient(circle, rgba(255,255,255,.16), transparent 72%);
    pointer-events:none;
}
#app-header h1 { margin:0; font-size:1.95rem; font-weight:800; letter-spacing:-.02em; }
#app-header p  { margin:9px 0 0; opacity:.93; font-size:.97rem; max-width:660px; }

/* --- Adım şeridi (stepper) --- */
.steps { display:flex; align-items:center; margin-top:22px; flex-wrap:wrap; row-gap:10px; }
.step { display:flex; align-items:center; gap:8px; font-size:.85rem; font-weight:600; }
.step + .step::before {
    content:""; width:30px; height:2px; margin:0 10px;
    background: rgba(255,255,255,.4); border-radius:2px;
}
.step-num {
    display:flex; align-items:center; justify-content:center;
    width:25px; height:25px; border-radius:50%;
    background: rgba(255,255,255,.20); border:1px solid rgba(255,255,255,.5);
    font-size:.8rem; font-weight:700;
}

/* --- Durum rozeti --- */
.status-badge {
    display:flex; align-items:center; gap:11px;
    border:1px solid; border-radius:14px; padding:14px 18px;
    font-weight:600; font-size:.95rem;
    box-shadow: 0 4px 14px -8px rgba(15,23,42,.4);
}
.status-icon { font-size:1.15rem; line-height:1; }
.spin { display:inline-block; animation: spin 1.1s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }

/* --- Kartlar --- */
.panel-card {
    border:1px solid var(--border-color-primary);
    border-radius:16px !important; padding:18px !important;
    background: var(--background-fill-primary);
    box-shadow: 0 2px 12px -6px rgba(15,23,42,.18);
    transition: box-shadow .2s ease;
}
.panel-card:hover { box-shadow: 0 10px 28px -12px rgba(79,70,229,.28); }

/* --- Bölüm başlığı (numara rozetli) --- */
.section-head { display:flex; align-items:center; gap:10px; margin:0 0 6px; }
.sec-num {
    display:flex; align-items:center; justify-content:center;
    width:27px; height:27px; border-radius:9px; flex:none;
    background: var(--hm-grad); color:#fff;
    font-weight:800; font-size:.88rem;
    box-shadow: 0 4px 10px -3px rgba(79,70,229,.55);
}
.sec-label { font-weight:700; font-size:1rem; letter-spacing:-.01em; }

/* --- Çalıştır butonu --- */
#run-btn {
    font-size:1.08rem !important; font-weight:800 !important;
    padding:15px !important; border:none !important;
    background: var(--hm-grad) !important; color:#fff !important;
    box-shadow: 0 10px 24px -8px rgba(79,70,229,.65) !important;
    transition: transform .15s ease, box-shadow .15s ease !important;
}
#run-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 14px 30px -8px rgba(79,70,229,.75) !important;
}

/* --- İşlem günlüğü (terminal görünümü) --- */
#log-box textarea {
    font-family: ui-monospace, 'Cascadia Code', 'JetBrains Mono', Consolas, monospace !important;
    font-size:.82rem !important; line-height:1.5 !important;
    background:#0f172a !important; color:#d1d5db !important;
    border-radius:12px !important; border-color:#1e293b !important;
}

/* --- TF uyarısı --- */
.tf-warning {
    background:#fef3c7; border:1px solid #fcd34d; color:#92400e;
    border-radius:12px; padding:12px 16px; font-size:.9rem; font-weight:500;
}

footer { display:none !important; }
#app-footer { text-align:center; opacity:.5; font-size:.82rem; margin-top:16px; }
"""

_HEADER_HTML = """
<div id="app-header">
  <h1>🛰️ Harita Üretici</h1>
  <p>Bir uydu görüntüsünü otomatik haritaya dönüştürün — bölme, model çıkarımı,
     birleştirme ve jeoreferanslama tek tıkla.</p>
  <div class="steps">
    <div class="step"><span class="step-num">1</span>Böl</div>
    <div class="step"><span class="step-num">2</span>Modelden Geçir</div>
    <div class="step"><span class="step-num">3</span>Birleştir</div>
    <div class="step"><span class="step-num">4</span>Jeoreferansla</div>
  </div>
</div>
"""


# ----------------------------------------------------------------------------
# Arayüzü kur
# ----------------------------------------------------------------------------
def _section_title(num, text):
    """Numara rozetli bölüm başlığı (kart üstlerinde kullanılır)."""
    return gr.HTML(
        f'<div class="section-head">'
        f'<span class="sec-num">{html.escape(str(num))}</span>'
        f'<span class="sec-label">{html.escape(text)}</span>'
        f'</div>'
    )


def build_ui():
    with gr.Blocks(title="Harita Üretici") as demo:
        gr.HTML(_HEADER_HTML)

        if not TENSORFLOW_AVAILABLE:
            gr.HTML(
                '<div class="tf-warning">⚠️ <b>TensorFlow bulunamadı.</b> '
                'Model çıkarımı yapılamaz; yalnızca <i>böl + birleştir</i> adımları çalışır.</div>'
            )

        with gr.Row(equal_height=False):
            # ---------------- SOL: Girdi & Ayarlar ----------------
            with gr.Column(scale=4):
                with gr.Group(elem_classes="panel-card"):
                    _section_title("1", "📥  Giriş Haritası")
                    input_file = gr.File(
                        label="Uydu görüntüsü dosyası",
                        file_types=["image", ".tif", ".tiff"],
                        file_count="single",
                        type="filepath",
                        height=130,
                    )

                with gr.Group(elem_classes="panel-card"):
                    _section_title("2", "🧠  Model & Referans")
                    model_dd = gr.Dropdown(
                        label="Model",
                        choices=model_choices(),
                        value=default_model(),
                        info="modeller/ klasöründeki .h5 / .keras dosyaları.",
                    )
                    with gr.Accordion("📁  Bunun yerine disk'ten model dosyası yükle",
                                      open=False):
                        model_file = gr.File(
                            label="Model dosyası (.h5 / .keras)",
                            file_types=[".h5", ".keras"],
                            file_count="single",
                            type="filepath",
                            height=110,
                        )
                        gr.Markdown(
                            "<span style='font-size:.8rem;opacity:.6'>"
                            "Bir dosya yüklenirse yukarıdaki açılır liste yok sayılır.</span>"
                        )
                    reference_dd = gr.Dropdown(
                        label="Referans raster",
                        choices=reference_choices(),
                        value=REF_AUTO,
                        info="Jeoreferanslama kaynağı. 'Otomatik' dosya adına göre eşler; "
                             "eşleşme yoksa giriş GeoTIFF ise kendi coğrafi bilgisini kullanır.",
                    )
                    gr.Markdown(
                        "<span style='font-size:.8rem;opacity:.6'>"
                        "Giriş görüntüsü coğrafi referanslı bir GeoTIFF ise, ayrı referans "
                        "vermeden de coğrafi bilgisi çıktıya taşınır — boyut biraz değişse "
                        "bile kapsam korunur.</span>"
                    )
                    with gr.Accordion("📁  Bunun yerine disk'ten referans raster yükle",
                                      open=False):
                        reference_file = gr.File(
                            label="Referans raster (.tif / .tiff)",
                            file_types=[".tif", ".tiff"],
                            file_count="single",
                            type="filepath",
                            height=110,
                        )
                        gr.Markdown(
                            "<span style='font-size:.8rem;opacity:.6'>"
                            "Bir dosya yüklenirse yukarıdaki açılır liste yok sayılır.</span>"
                        )
                    refresh_btn = gr.Button(
                        "🔄  Listeleri Yenile", variant="secondary", size="sm",
                    )

                with gr.Accordion("⚙️  Gelişmiş Ayarlar", open=False):
                    with gr.Row():
                        tile_size = gr.Number(
                            label="tile_size",
                            value=CONFIG["pipeline"]["tile_size"], precision=0,
                            info="Karo boyutu = model girdisi (piksel).",
                        )
                        overlap = gr.Number(
                            label="overlap",
                            value=CONFIG["split"]["overlap"], precision=0,
                            info="Karolar arası örtüşme.",
                        )
                    with gr.Row():
                        crop_overlap = gr.Number(
                            label="crop_overlap",
                            value=CONFIG["merge"]["crop_overlap"], precision=0,
                            info="Birleştirmede kırpma (~ overlap/2).",
                        )
                        batch_size = gr.Number(
                            label="batch_size",
                            value=CONFIG["pipeline"]["batch_size"], precision=0,
                            info="GPU belleğine göre ayarlayın.",
                        )
                    color_mode = gr.Dropdown(
                        label="color_mode",
                        choices=["auto", "grayscale", "rgb"], value="auto",
                        info="auto = model kanal sayısından otomatik algıla.",
                    )
                    normalization = gr.Dropdown(
                        label="Giriş normalizasyonu",
                        choices=NORMALIZATION_CHOICES, value="minus1_1",
                        info="Karo, sinir ağına verilmeden önce nasıl ölçeklensin? "
                             "Modelin eğitimiyle aynı olmalı; yanlış seçim bozuk çıktı verir.",
                    )
                    enhancement = gr.Dropdown(
                        label="Görüntü iyileştirme (histogram)",
                        choices=ENHANCEMENT_CHOICES, value="none",
                        info="Normalizasyondan ÖNCE karoya uygulanan kontrast iyileştirmesi. "
                             "Renkliyse yalnızca parlaklık kanalına uygulanır.",
                    )
                    clahe_clip = gr.Slider(
                        label="CLAHE kontrast sınırı (clipLimit)",
                        minimum=1.0, maximum=10.0, value=2.0, step=0.5,
                        info="Yalnızca CLAHE seçiliyken etkilidir. Yüksek değer = "
                             "daha güçlü kontrast (ve daha çok gürültü).",
                    )

                with gr.Row():
                    clear_btn = gr.Button("🧹  Temizle", variant="secondary", scale=1)
                    run_btn = gr.Button(
                        "▶  Pipeline'ı Çalıştır", variant="primary",
                        elem_id="run-btn", scale=2,
                    )

            # ---------------- SAĞ: Durum & Sonuçlar ----------------
            with gr.Column(scale=6):
                status_html = gr.HTML(
                    _status_html("idle", "Hazır - bir giriş haritası seçin.")
                )

                with gr.Tabs():
                    with gr.Tab("🖼️  Önizleme"):
                        preview = gr.Image(
                            label="Birleştirilmiş mozaik",
                            interactive=False, height=460,
                        )
                    with gr.Tab("📜  İşlem Günlüğü"):
                        log_box = gr.Textbox(
                            label="Günlük", lines=20, max_lines=20,
                            interactive=False, autoscroll=True, elem_id="log-box",
                            placeholder="Pipeline çalıştığında günlük burada canlı akar…",
                        )
                    with gr.Tab("💾  Çıktı Dosyaları"):
                        downloads = gr.Files(label="GeoTIFF + mozaik dosyaları")

                with gr.Accordion("ℹ️  Bu pipeline nasıl çalışır?", open=False):
                    gr.Markdown(
                        "1. **Böl** — giriş haritası `tile_size` boyutunda karolara ayrılır.\n"
                        "2. **Modelden geçir** — her karo seçilen autoencoder modelinden geçirilir.\n"
                        "3. **Birleştir** — işlenen karolar tek mozaik halinde birleştirilir.\n"
                        "4. **Jeoreferansla** — referans raster ile koordinatlandırılıp GeoTIFF üretilir.\n\n"
                        "*İpucu:* Referans raster bulunamazsa jeoreferanslama atlanır, "
                        "yine de mozaik çıktısı alırsınız."
                    )

        gr.HTML('<div id="app-footer">Harita Üretici · goruntu_islemleri.py pipeline arayüzü</div>')

        # ---------------- Olay bağlantıları ----------------
        run_btn.click(
            fn=run_pipeline,
            inputs=[input_file, model_dd, model_file, reference_dd, reference_file,
                    tile_size, overlap, crop_overlap, batch_size, color_mode,
                    normalization, enhancement, clahe_clip],
            outputs=[log_box, status_html, preview, downloads, run_btn],
        )
        refresh_btn.click(
            fn=refresh_lists, inputs=[],
            outputs=[model_dd, reference_dd, status_html],
        )
        clear_btn.click(
            fn=clear_all, inputs=[],
            outputs=[input_file, model_file, reference_file, log_box,
                     status_html, preview, downloads],
        )

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.queue()
    # Gradio 6.0: theme ve css artık launch()'a veriliyor.
    ui.launch(inbrowser=True, theme=_THEME, css=_CSS)
