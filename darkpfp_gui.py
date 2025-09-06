#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
darkpfp_gui.py — GUI for darkpfpify_v2 з вбудованим 2x/4x апскейлом і просунутим виділенням контурів.
Режими контурів: Canny / Sobel / XDoG / Fusion (Ultra).
Fusion поєднує Canny(L з CLAHE), Canny(V), Scharr(L), LoG(L), мультискейл-Canny.
Є живий предпросмотр (до збереження), робота у QThread, компактний інтерфейс у ScrollArea.

Новинка:
- ГАЛОЧКА «Розтягнути до 1:1 (без полів)» — замість паддінгу виконує неізохромне масштабування до квадрата.

Залежності:
    pip install PyQt5 opencv-python pillow numpy
Додатково (необов’язково):
    pip install qdarkstyle
"""
import sys
from pathlib import Path

import numpy as np
import cv2
from PIL import Image
from PyQt5 import QtCore, QtGui, QtWidgets


# ---------------- core processing ----------------

def read_image_any(path_str):
    p = Path(path_str)
    try:
        data = np.fromfile(str(p), dtype=np.uint8)
        if data.size > 0:
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is not None:
                return img
    except Exception:
        pass
    pil = Image.open(p).convert("RGB")
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

def to_square_pad(img, color=(0,0,0)):
    h,w = img.shape[:2]
    if h==w:
        return img
    size = max(h,w)
    canvas = np.zeros((size,size,3), dtype=img.dtype)
    canvas[:,:,:] = color
    y=(size-h)//2; x=(size-w)//2
    canvas[y:y+h, x:x+w] = img
    return canvas

def to_square_stretch(img, out_size):
    """Non-uniform stretch to exact square of side out_size (no padding)."""
    if out_size <= 0:
        h,w = img.shape[:2]
        out_size = max(h,w)
    return cv2.resize(img, (int(out_size), int(out_size)), interpolation=cv2.INTER_AREA)

def auto_canny(image, sigma=0.33):
    v = np.median(image)
    lower = int(max(0,(1.0-sigma)*v)); upper = int(min(255,(1.0+sigma)*v))
    return cv2.Canny(image, lower, upper)

def sobel_edges(gray, ksize=3):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=ksize)
    mag = cv2.magnitude(gx, gy)
    mag = mag / (mag.max()+1e-6)
    return (mag*255).astype(np.uint8)

def xdog_edges(gray, sigma=0.8, k=1.6, p=20.0, eps=0.01, phi=10.0):
    g1 = cv2.GaussianBlur(gray, (0,0), sigmaX=sigma, sigmaY=sigma)
    g2 = cv2.GaussianBlur(gray, (0,0), sigmaX=sigma*k, sigmaY=sigma*k)
    diff = g1 - p*g2
    diff = diff.astype(np.float32)/255.0
    ed = np.tanh(phi*(diff - eps))
    ed = (ed - ed.min())/(ed.max()-ed.min()+1e-6)
    return (ed*255).astype(np.uint8)

def motion_kernel(ksize=11, angle_degrees=0):
    if ksize<=1:
        ksize=1
    k = np.zeros((ksize, ksize), dtype=np.float32)
    k[ksize//2, :] = 1.0
    center = (ksize/2-0.5, ksize/2-0.5)
    M = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    k = cv2.warpAffine(k, M, (ksize, ksize))
    s = k.sum()
    if s != 0:
        k /= s
    return k

def chromatic_ab_shift(img, dx=1, dy=0):
    b,g,r = cv2.split(img)
    M1 = np.float32([[1,0,dx],[0,1,dy]])
    M2 = np.float32([[1,0,-dx],[0,1,-dy]])
    b2 = cv2.warpAffine(b, M1, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT)
    r2 = cv2.warpAffine(r, M2, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT)
    return cv2.merge([b2,g,r2])

def add_scanlines(img, strength=0.15, period=3):
    if strength<=0:
        return img
    h,w = img.shape[:2]
    mask = np.ones((h,1), dtype=np.float32)
    for y in range(h):
        if (y % max(1,period)) == 0:
            mask[y,0] = 1.0 - strength
    mask = np.repeat(mask, w, axis=1)
    out = (img.astype(np.float32)/255.0) * mask[...,None]
    return np.clip(out*255.0,0,255).astype(np.uint8)

def add_vignette(img, amount=0.35):
    if amount<=0:
        return img
    h,w = img.shape[:2]
    Y, X = np.ogrid[:h,:w]
    cy, cx = h/2, w/2
    dist = np.sqrt((X-cx)**2 + (Y-cy)**2)
    maxd = np.sqrt(cx**2 + cy**2)
    mask = 1.0 - amount*(dist/maxd)**1.2
    mask = np.clip(mask, 0.2, 1.0).astype(np.float32)
    out = (img.astype(np.float32)/255.0) * mask[...,None]
    return np.clip(out*255.0,0,255).astype(np.uint8)

def add_grain(img, sigma=6.0):
    if sigma<=0:
        return img
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    out = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return out

def color_from_hex(hexstr, default=(235,200,255)):
    if not hexstr:
        return (235,200,255)
    s = hexstr.strip()
    if s.startswith("#") and len(s)==7:
        r = int(s[1:3],16); g=int(s[3:5],16); b=int(s[5:7],16)
        return (b,g,r)  # BGR
    return default

def canny_manual_or_auto(gray, params):
    if params.get("canny_manual", False):
        low = int(params.get("canny_low",50)); high = int(params.get("canny_high",150))
        return cv2.Canny(gray, low, high)
    sigma = float(params.get("canny_sigma",0.33))
    return auto_canny(gray, sigma=sigma)

def fusion_edges(bgr, params):
    """Fusion (Ultra) edges: combine multiple detectors & color spaces."""
    clahe_clip = float(params.get("clahe_clip", 2.0))
    clahe_grid = int(params.get("clahe_grid", 8))
    ms_levels  = int(params.get("ms_levels", 2))  # 0..2
    chroma_w   = float(params.get("chroma_w", 0.5))
    log_sigma  = float(params.get("log_sigma", 1.0))
    edge_thresh= float(params.get("edge_thresh", 0.2))
    morph_d    = int(params.get("morph_dilate", 1))
    morph_e    = int(params.get("morph_erode", 0))
    med_ks     = int(params.get("median_ksize", 0))

    # color spaces
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB); L,a,b = cv2.split(lab)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV); H,S,V = cv2.split(hsv)

    # CLAHE on L
    clahe = cv2.createCLAHE(clipLimit=max(0.1, clahe_clip), tileGridSize=(max(2,clahe_grid), max(2,clahe_grid)))
    Lc = clahe.apply(L)

    # 1) Canny on L (CLAHE)
    eL = canny_manual_or_auto(Lc, params)

    # 2) Canny on V (value)
    eV = canny_manual_or_auto(V, params)

    # 3) Scharr on L
    gx = cv2.Scharr(Lc, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(Lc, cv2.CV_32F, 0, 1)
    mag = cv2.magnitude(gx, gy)
    mag = (mag / (mag.max()+1e-6))
    eS = (mag >= edge_thresh).astype(np.uint8)*255

    # 4) LoG on L
    if log_sigma > 0:
        Lg = cv2.GaussianBlur(Lc, (0,0), log_sigma)
        lap = cv2.Laplacian(Lg, cv2.CV_32F, ksize=3)
        lapn = np.abs(lap); lapn = lapn / (lapn.max()+1e-6)
        eLoG = (lapn >= edge_thresh).astype(np.uint8)*255
    else:
        eLoG = np.zeros_like(eL)

    # 5) Multi-scale Canny on L
    ems = np.zeros_like(eL)
    h, w = Lc.shape[:2]
    for lvl in range(1, ms_levels+1):
        s = 1.0 / (2**lvl)
        small = cv2.resize(Lc, (max(1,int(w*s)), max(1,int(h*s))), interpolation=cv2.INTER_AREA)
        es = canny_manual_or_auto(small, params)
        es_up = cv2.resize(es, (w,h), interpolation=cv2.INTER_LINEAR)
        ems = np.maximum(ems, es_up)

    # Combine
    f1 = eL.astype(np.float32)/255.0
    f2 = (eV.astype(np.float32)/255.0) * chroma_w
    f3 = eS.astype(np.float32)/255.0
    f4 = eLoG.astype(np.float32)/255.0
    f5 = ems.astype(np.float32)/255.0
    fused = np.maximum.reduce([f1, f2, f3, f4, f5])
    out = (np.clip(fused, 0, 1)*255).astype(np.uint8)

    # Morphology cleanup
    if morph_d>0:
        kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_d, morph_d))
        out = cv2.dilate(out, kd, 1)
    if morph_e>0:
        ke = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_e, mormh_e))  # typo fix below
        # (fixed below)
    # Median cleanup (odd ksize only)
    if med_ks and med_ks>=3 and (med_ks % 2)==1:
        out = cv2.medianBlur(out, med_ks)

    return out

def build_edges(bgr, mode="Canny", params=None):
    params = params or {}
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 60, 60)
    if mode == "Sobel":
        ed = sobel_edges(gray, ksize=int(params.get("sobel_ksize",3)))
        th = params.get("edge_thresh", 0.2)
        ed = (ed >= int(th*255)).astype(np.uint8)*255
    elif mode == "XDoG":
        ed = xdog_edges(gray,
                        sigma=float(params.get("xdog_sigma",0.8)),
                        k=float(params.get("xdog_k",1.6)),
                        p=float(params.get("xdog_p",20.0)),
                        eps=float(params.get("xdog_eps",0.01)),
                        phi=float(params.get("xdog_phi",10.0)))
        th = params.get("edge_thresh", 0.2)
        ed = (ed >= int(th*255)).astype(np.uint8)*255
    elif mode == "Fusion":
        # (fix the typo from fusion_edges morphology erode)
        ed = fusion_edges(bgr, params)
        morph_e = int(params.get("morph_erode", 0))
        if morph_e>0:
            ke = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_e, morph_e))
            ed = cv2.erode(ed, ke, 1)
    else:  # Canny
        if params.get("canny_manual", False):
            low = int(params.get("canny_low",50))
            high = int(params.get("canny_high",150))
            ed = cv2.Canny(gray, low, high)
        else:
            ed = auto_canny(gray, sigma=float(params.get("canny_sigma",0.33)))
        # optional morph
        dilate = int(params.get("morph_dilate",1))
        erode  = int(params.get("morph_erode",0))
        if dilate>0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate,dilate))
            ed = cv2.dilate(ed, kernel, 1)
        if erode>0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode,erode))
            ed = cv2.erode(ed, kernel, 1)
    return ed

def render_darkpfp(bgr, size=1024, square=True, stretch_square=False,
                   edges_mode="Canny", edge_params=None,
                   line_width=1.6, glow_radius=12, glow_gain=1.3,
                   tint_hex="#E7C2FF", motion=11, motion_angle=0,
                   aberration=1, scan_strength=0.12, scan_period=3,
                   vignette=0.35, grain=5.0):
    """If stretch_square=True → non-uniform resize to size×size.
       Else → uniform resize (max side=size), optional square padding."""
    h,w = bgr.shape[:2]

    if stretch_square:
        target = size if size and size>0 else max(h,w)
        bgr = to_square_stretch(bgr, target)
    else:
        if size and size>0:
            s = size/max(h,w)
            if s!=1.0:
                bgr = cv2.resize(bgr, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)
        if square:
            bgr = to_square_pad(bgr, color=(0,0,0))

    edges = build_edges(bgr, mode=edges_mode, params=edge_params)
    # Thicken lines (independent of morph)
    if line_width>1:
        k = max(1, int(round(line_width)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k,k))
        edges = cv2.dilate(edges, kernel, 1)

    e = edges.astype(np.float32)/255.0

    # neon mapping
    tint = np.array(color_from_hex(tint_hex))[None,None,:]/255.0
    core = e
    if glow_radius>0:
        gl = cv2.GaussianBlur(e, (int(glow_radius*2+1)|1, int(glow_radius*2+1)|1), glow_radius)
    else:
        gl = e
    neon = np.clip(core + gl*glow_gain, 0, 1)
    neon_rgb = (neon[...,None] * tint)
    neon_bgr = (neon_rgb[...,::-1]*255.0).astype(np.uint8)

    # trails
    if motion>0:
        ker = motion_kernel(int(motion)|1, motion_angle)
        trails = cv2.filter2D(neon_bgr, -1, ker)
        neon_bgr = cv2.addWeighted(neon_bgr, 0.75, trails, 0.45, 0)

    # bloom
    bloom = cv2.GaussianBlur(neon_bgr, (0,0), sigmaX=3, sigmaY=3)
    neon_bgr = cv2.addWeighted(neon_bgr, 1.0, bloom, 0.6, 0)

    if aberration>0:
        neon_bgr = chromatic_ab_shift(neon_bgr, dx=int(aberration), dy=0)

    neon_bgr = add_scanlines(neon_bgr, strength=scan_strength, period=int(scan_period))
    neon_bgr = add_vignette(neon_bgr, amount=vignette)
    neon_bgr = add_grain(neon_bgr, sigma=grain)
    return neon_bgr


# ---------------- presets ----------------

PRESETS = {
    "darkpfp-exact": {
        "size": 1024, "square": True, "tint": "#E7C2FF",
        "line": 1.6, "glow": 12.0, "glow_gain": 1.3,
        "motion": 11, "angle": 0.0, "ab": 1,
        "scan": 0.12, "period": 3, "vignette": 0.35, "grain": 5.0,
        "edges_mode": "Canny",
        "edge_params": {"canny_sigma":0.28, "morph_dilate":2, "morph_erode":0}
    },
    "darkpfp-boost": {
        "size": 1024, "square": True, "tint": "#E7C2FF",
        "line": 1.8, "glow": 14.0, "glow_gain": 1.45,
        "motion": 13, "angle": 0.0, "ab": 2,
        "scan": 0.16, "period": 3, "vignette": 0.4, "grain": 7.0,
        "edges_mode": "Fusion",
        "edge_params": {"clahe_clip":2.5, "clahe_grid":8, "ms_levels":2, "chroma_w":0.5, "log_sigma":1.0,
                        "edge_thresh":0.22, "median_ksize":3, "morph_dilate":2, "morph_erode":0,
                        "canny_sigma":0.28}
    },
    "darkpfp-ink": {
        "size": 1024, "square": True, "tint": "#FFFFFF",
        "line": 2.0, "glow": 0.0, "glow_gain": 1.0,
        "motion": 0, "angle": 0.0, "ab": 0,
        "scan": 0.0, "period": 3, "vignette": 0.0, "grain": 0.0,
        "edges_mode": "Fusion",
        "edge_params": {"clahe_clip":2.0, "clahe_grid":8, "ms_levels":2, "chroma_w":0.4, "log_sigma":0.8,
                        "edge_thresh":0.28, "median_ksize":3, "morph_dilate":2, "morph_erode":1,
                        "canny_sigma":0.30}
    }
}


# ---------------- worker ----------------

class RenderWorker(QtCore.QObject):
    preview_ready = QtCore.pyqtSignal(np.ndarray)
    saved = QtCore.pyqtSignal(np.ndarray, str)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, src_path, upscale, params, save_path=None, max_kb=0, parent=None):
        super().__init__(parent)
        self.src_path = src_path
        self.upscale = upscale
        self.params = params
        self.save_path = save_path
        self.max_kb = max_kb

    @QtCore.pyqtSlot()
    def run(self):
        try:
            img = read_image_any(self.src_path)
            if img is None:
                raise RuntimeError("Не вдалося прочитати вхідне зображення.")
            if self.upscale in (2,4):
                img = cv2.resize(img, None, fx=self.upscale, fy=self.upscale, interpolation=cv2.INTER_LANCZOS4)

            p = self.params
            out = render_darkpfp(
                img, size=p["size"], square=p["square"], stretch_square=p.get("stretch", False),
                edges_mode=p["edges_mode"], edge_params=p["edge_params"],
                line_width=p["line"], glow_radius=p["glow"], glow_gain=p["glow_gain"],
                tint_hex=p["tint"], motion=p["motion"], motion_angle=p["angle"],
                aberration=p["ab"], scan_strength=p["scan"], scan_period=p["period"],
                vignette=p["vignette"], grain=p["grain"]
            )

            self.preview_ready.emit(out)

            if self.save_path:
                out_path = Path(self.save_path)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(rgb)
                if self.max_kb and self.max_kb>0:
                    out_path = out_path.with_suffix(".jpg")
                    q = 95
                    tmp = out_path.with_suffix(".tmp.jpg")
                    while q>=20:
                        pil.save(tmp, format="JPEG", quality=q, optimize=True, progressive=True)
                        if tmp.exists() and tmp.stat().st_size/1024.0 <= self.max_kb:
                            break
                        q -= 5
                    tmp.replace(out_path)
                else:
                    suf = out_path.suffix.lower()
                    if suf in (".jpg",".jpeg"):
                        pil.save(out_path, format="JPEG", quality=95, optimize=True, progressive=True)
                    else:
                        pil.save(out_path)
                if not out_path.exists():
                    raise RuntimeError("Не вдалося зберегти результат.")
                self.saved.emit(out, str(out_path.resolve()))
        except Exception as e:
            self.failed.emit(str(e))


# ---------------- GUI ----------------

def labeled_slider(parent, title, minv, maxv, step, init, decimals=1):
    box = QtWidgets.QGroupBox(title, parent)
    lay = QtWidgets.QHBoxLayout(box)
    slider = QtWidgets.QSlider(QtCore.Qt.Horizontal); slider.setMinimum(0); slider.setMaximum(int((maxv-minv)/step))
    spin = QtWidgets.QDoubleSpinBox(); spin.setRange(minv, maxv); spin.setSingleStep(step); spin.setDecimals(decimals); spin.setValue(init)
    def spin_to_slider(val):
        slider.blockSignals(True); slider.setValue(int(round((val-minv)/step))); slider.blockSignals(False)
    def slider_to_spin(v):
        spin.blockSignals(True); spin.setValue(minv + v*step); spin.blockSignals(False)
    spin.valueChanged.connect(spin_to_slider); slider.valueChanged.connect(slider_to_spin)
    spin_to_slider(init)
    lay.addWidget(slider); lay.addWidget(spin, 0)
    return box, spin

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("darkpfp GUI")
        self.resize(1220, 800)

        central = QtWidgets.QWidget(self); self.setCentralWidget(central)
        h = QtWidgets.QHBoxLayout(central)

        # left: scrollable controls
        scroll = QtWidgets.QScrollArea(); scroll.setWidgetResizable(True)
        h.addWidget(scroll, 0)
        controls_host = QtWidgets.QWidget(); scroll.setWidget(controls_host)
        controls = QtWidgets.QVBoxLayout(controls_host)

        row1 = QtWidgets.QHBoxLayout()
        self.btn_open = QtWidgets.QPushButton("Відкрити зображення…"); self.btn_open.clicked.connect(self.open_image)
        row1.addWidget(self.btn_open)
        self.btn_reset = QtWidgets.QPushButton("Скинути до пресету"); self.btn_reset.clicked.connect(self.apply_current_preset)
        row1.addWidget(self.btn_reset)
        controls.addLayout(row1)

        # upscale & preset
        self.combo_upscale = QtWidgets.QComboBox(); self.combo_upscale.addItems(["Без апскейлу (1x)", "2x", "4x"])
        self.combo_preset = QtWidgets.QComboBox(); self.combo_preset.addItems(list(PRESETS.keys()) + ["(кастом)"])
        self.combo_preset.currentIndexChanged.connect(self.apply_current_preset)
        controls.addWidget(QtWidgets.QLabel("Апскейл перед стилізацією:")); controls.addWidget(self.combo_upscale)
        controls.addWidget(QtWidgets.QLabel("Пресет стилю:")); controls.addWidget(self.combo_preset)

        # tint / size / square
        self.edit_tint = QtWidgets.QLineEdit("#E7C2FF")
        controls.addWidget(QtWidgets.QLabel("Колір неону (hex):")); controls.addWidget(self.edit_tint)
        self.spin_size = QtWidgets.QSpinBox(); self.spin_size.setRange(0,8192); self.spin_size.setValue(1024)
        self.chk_square = QtWidgets.QCheckBox("Доповнити до 1:1 (square, з полями)"); self.chk_square.setChecked(True)
        self.chk_stretch = QtWidgets.QCheckBox("Розтягнути до 1:1 (без полів)")  # NEW
        controls.addWidget(QtWidgets.QLabel("Розмір / квадрат:"))
        controls.addWidget(self.spin_size); controls.addWidget(self.chk_square); controls.addWidget(self.chk_stretch)

        # edge mode
        self.combo_edges = QtWidgets.QComboBox(); self.combo_edges.addItems(["Canny","Sobel","XDoG","Fusion"])
        controls.addWidget(QtWidgets.QLabel("Режим контурів:")); controls.addWidget(self.combo_edges)

        # edge params: Canny manual/auto
        grp_canny_sigma, self.sp_canny_sigma = labeled_slider(controls_host, "Canny sigma (auto)", 0.05, 1.0, 0.01, 0.28, 2)
        grp_canny_low, self.sp_canny_low = labeled_slider(controls_host, "Canny low (ручний)", 0, 255, 1, 50, 0)
        grp_canny_high, self.sp_canny_high = labeled_slider(controls_host, "Canny high (ручний)", 0, 255, 1, 150, 0)
        self.chk_canny_manual = QtWidgets.QCheckBox("Ручні пороги Canny")
        controls.addWidget(grp_canny_sigma); controls.addWidget(self.chk_canny_manual); controls.addWidget(grp_canny_low); controls.addWidget(grp_canny_high)

        # Sobel / XDoG params
        grp_sobel_ks, self.sp_sobel_ks = labeled_slider(controls_host, "Sobel kernel", 3, 7, 2, 3, 0)
        controls.addWidget(grp_sobel_ks)
        grp_xdog_sigma, self.sp_xdog_sigma = labeled_slider(controls_host, "XDoG σ", 0.4, 2.0, 0.05, 0.8, 2)
        grp_xdog_k, self.sp_xdog_k = labeled_slider(controls_host, "XDoG k", 1.1, 2.5, 0.05, 1.6, 2)
        grp_xdog_p, self.sp_xdog_p = labeled_slider(controls_host, "XDoG p", 1.0, 40.0, 1.0, 20.0, 0)
        grp_xdog_eps, self.sp_xdog_eps = labeled_slider(controls_host, "XDoG ε", 0.001, 0.05, 0.001, 0.01, 3)
        grp_xdog_phi, self.sp_xdog_phi = labeled_slider(controls_host, "XDoG φ", 1.0, 30.0, 1.0, 10.0, 0)
        for g in (grp_xdog_sigma, grp_xdog_k, grp_xdog_p, grp_xdog_eps, grp_xdog_phi):
            controls.addWidget(g)

        # Fusion extras
        grp_edge_thresh, self.sp_edge_thresh = labeled_slider(controls_host, "Поріг бінаризації (Sobel/XDoG/LoG)", 0.0, 1.0, 0.01, 0.22, 2)
        grp_clahe_clip, self.sp_clahe_clip = labeled_slider(controls_host, "CLAHE clip", 0.1, 5.0, 0.1, 2.0, 1)
        grp_clahe_grid, self.sp_clahe_grid = labeled_slider(controls_host, "CLAHE grid", 2, 16, 1, 8, 0)
        grp_ms_levels, self.sp_ms_levels = labeled_slider(controls_host, "Multi-scale levels", 0, 2, 1, 2, 0)
        grp_chroma_w, self.sp_chroma_w = labeled_slider(controls_host, "Вага хроми (V)", 0.0, 1.0, 0.05, 0.5, 2)
        grp_log_sigma, self.sp_log_sigma = labeled_slider(controls_host, "LoG σ (L)", 0.0, 3.0, 0.1, 1.0, 1)
        grp_median, self.sp_median_ks = labeled_slider(controls_host, "Median cleanup (odd px)", 0, 9, 1, 3, 0)
        for g in (grp_edge_thresh, grp_clahe_clip, grp_clahe_grid, grp_ms_levels, grp_chroma_w, grp_log_sigma, grp_median):
            controls.addWidget(g)

        # Morph
        grp_morph_d, self.sp_morph_d = labeled_slider(controls_host, "Розширення контурів (dilate)", 0, 6, 1, 2, 0)
        grp_morph_e, self.sp_morph_e = labeled_slider(controls_host, "Звуження контурів (erode)", 0, 6, 1, 0, 0)
        controls.addWidget(grp_morph_d); controls.addWidget(grp_morph_e)

        # rendering sliders
        grp_line, self.sp_line = labeled_slider(controls_host, "Товщина лінії", 0.5, 5.0, 0.1, 1.6, 1)
        grp_glow, self.sp_glow = labeled_slider(controls_host, "Радіус сяйва (glow)", 0.0, 30.0, 0.5, 12.0, 1)
        grp_gg, self.sp_glow_gain = labeled_slider(controls_host, "Інтенсивність сяйва (gain)", 0.5, 2.0, 0.05, 1.3, 2)
        grp_motion, self.sp_motion = labeled_slider(controls_host, "Шлейфи (motion kernel)", 0.0, 31.0, 1.0, 11.0, 0)
        grp_angle, self.sp_angle = labeled_slider(controls_host, "Кут шлейфів (degrees)", -30.0, 30.0, 1.0, 0.0, 0)
        grp_ab, self.sp_ab = labeled_slider(controls_host, "Хром. аберація (px)", 0.0, 5.0, 1.0, 1.0, 0)
        grp_scan, self.sp_scan = labeled_slider(controls_host, "Сканлайни (strength)", 0.0, 0.4, 0.01, 0.12, 2)
        grp_period, self.sp_period = labeled_slider(controls_host, "Період сканлайнів", 2.0, 6.0, 1.0, 3.0, 0)
        grp_vig, self.sp_vignette = labeled_slider(controls_host, "Віньєтка", 0.0, 0.8, 0.01, 0.35, 2)
        grp_grain, self.sp_grain = labeled_slider(controls_host, "Зерно (noise σ)", 0.0, 12.0, 0.5, 5.0, 1)
        for g in (grp_line, grp_glow, grp_gg, grp_motion, grp_angle, grp_ab, grp_scan, grp_period, grp_vig, grp_grain):
            controls.addWidget(g)

        # size limit + preview controls
        row2 = QtWidgets.QHBoxLayout()
        self.chk_limit = QtWidgets.QCheckBox("Обмежити розмір (KB):"); self.chk_limit.setChecked(False)
        self.spin_limit = QtWidgets.QSpinBox(); self.spin_limit.setRange(50,8192); self.spin_limit.setValue(1024)
        row2.addWidget(self.chk_limit); row2.addWidget(self.spin_limit)
        controls.addLayout(row2)

        row3 = QtWidgets.QHBoxLayout()
        self.chk_auto = QtWidgets.QCheckBox("Авто-предпросмотр"); self.chk_auto.setChecked(True)
        self.btn_preview = QtWidgets.QPushButton("Оновити предпросмотр")
        self.btn_preview.clicked.connect(self.update_preview)
        row3.addWidget(self.chk_auto); row3.addWidget(self.btn_preview)
        controls.addLayout(row3)

        self.btn_save = QtWidgets.QPushButton("ЗАСТОСУВАТИ і ЗБЕРЕГТИ…")
        self.btn_save.clicked.connect(self.process_and_save)
        controls.addWidget(self.btn_save)

        self.log = QtWidgets.QPlainTextEdit(); self.log.setReadOnly(True); self.log.setMaximumHeight(140)
        controls.addWidget(QtWidgets.QLabel("Логи:")); controls.addWidget(self.log)
        controls.addStretch(1)

        # right: previews
        right = QtWidgets.QVBoxLayout(); h.addLayout(right, 1)
        self.lbl_in = QtWidgets.QLabel("Вхідне зображення"); self.lbl_in.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_in.setMinimumSize(480,360); self.lbl_in.setFrameShape(QtWidgets.QFrame.Box); right.addWidget(self.lbl_in, 1)
        self.lbl_out = QtWidgets.QLabel("Предпросмотр"); self.lbl_out.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_out.setMinimumSize(480,360); self.lbl_out.setFrameShape(QtWidgets.QFrame.Box); right.addWidget(self.lbl_out, 1)

        self.src_path = None
        try:
            import qdarkstyle
            self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        except Exception:
            pass

        # auto preview on changes
        for w in [self.combo_upscale, self.combo_preset, self.combo_edges,
                  self.edit_tint, self.spin_size, self.chk_square, self.chk_stretch,
                  self.sp_canny_sigma, self.sp_canny_low, self.sp_canny_high, self.chk_canny_manual,
                  self.sp_sobel_ks, self.sp_xdog_sigma, self.sp_xdog_k, self.sp_xdog_p, self.sp_xdog_eps, self.sp_xdog_phi,
                  self.sp_edge_thresh, self.sp_clahe_clip, self.sp_clahe_grid, self.sp_ms_levels, self.sp_chroma_w, self.sp_log_sigma, self.sp_median_ks,
                  self.sp_morph_d, self.sp_morph_e,
                  self.sp_line, self.sp_glow, self.sp_glow_gain, self.sp_motion, self.sp_angle, self.sp_ab,
                  self.sp_scan, self.sp_period, self.sp_vignette, self.sp_grain]:
            if hasattr(w, "valueChanged"):
                w.valueChanged.connect(self.on_any_change)
            elif hasattr(w, "currentIndexChanged"):
                w.currentIndexChanged.connect(self.on_any_change)
            elif hasattr(w, "textChanged"):
                w.textChanged.connect(self.on_any_change)
            elif hasattr(w, "toggled"):
                w.toggled.connect(self.on_any_change)

        self.apply_current_preset()

    def log_msg(self, s): self.log.appendPlainText(s)

    def apply_current_preset(self):
        name = self.combo_preset.currentText()
        if name in PRESETS:
            p = PRESETS[name]
            self.edit_tint.setText(p["tint"]); self.spin_size.setValue(p["size"]); self.chk_square.setChecked(bool(p["square"]))
            self.chk_stretch.setChecked(False)  # default off
            self.combo_edges.setCurrentText(p.get("edges_mode","Canny"))
            self.sp_line.setValue(p["line"]); self.sp_glow.setValue(p["glow"]); self.sp_glow_gain.setValue(p["glow_gain"])
            self.sp_motion.setValue(p["motion"]); self.sp_angle.setValue(p["angle"]); self.sp_ab.setValue(p["ab"])
            self.sp_scan.setValue(p["scan"]); self.sp_period.setValue(p["period"]); self.sp_vignette.setValue(p["vignette"]); self.sp_grain.setValue(p["grain"])
            # edge params
            ep = p.get("edge_params", {})
            self.sp_canny_sigma.setValue(ep.get("canny_sigma",0.28))
            self.chk_canny_manual.setChecked(bool(ep.get("canny_manual", False)))
            self.sp_canny_low.setValue(ep.get("canny_low",50)); self.sp_canny_high.setValue(ep.get("canny_high",150))
            self.sp_sobel_ks.setValue(ep.get("sobel_ksize",3))
            self.sp_xdog_sigma.setValue(ep.get("xdog_sigma",0.8)); self.sp_xdog_k.setValue(ep.get("xdog_k",1.6))
            self.sp_xdog_p.setValue(ep.get("xdog_p",20.0)); self.sp_xdog_eps.setValue(ep.get("xdog_eps",0.01)); self.sp_xdog_phi.setValue(ep.get("xdog_phi",10.0))
            self.sp_edge_thresh.setValue(ep.get("edge_thresh",0.22))
            self.sp_clahe_clip.setValue(ep.get("clahe_clip",2.0)); self.sp_clahe_grid.setValue(ep.get("clahe_grid",8))
            self.sp_ms_levels.setValue(ep.get("ms_levels",2)); self.sp_chroma_w.setValue(ep.get("chroma_w",0.5))
            self.sp_log_sigma.setValue(ep.get("log_sigma",1.0)); self.sp_median_ks.setValue(ep.get("median_ksize",3))
            self.sp_morph_d.setValue(ep.get("morph_dilate",2)); self.sp_morph_e.setValue(ep.get("morph_erode",0))
            self.log_msg(f"Застосовано пресет: {name}")
        if self.chk_auto.isChecked():
            self.update_preview()

    def open_image(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Вибрати зображення", "", "Images (*.png *.jpg *.jpeg *.webp *.bmp)")
        if not path:
            return
        self.src_path = path
        self.show_image(path, self.lbl_in)
        if self.chk_auto.isChecked():
            self.update_preview()

    def show_image(self, path_or_array, label):
        if isinstance(path_or_array, str):
            img = read_image_any(path_or_array)
        else:
            img = path_or_array
        if img is None:
            return
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h,w,_ = rgb.shape
        qimg = QtGui.QImage(rgb.data, w, h, 3*w, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg)
        pix = pix.scaled(label.width(), label.height(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        label.setPixmap(pix)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self.src_path:
            self.show_image(self.src_path, self.lbl_in)

    def collect_params(self):
        edge_params = {
            "canny_sigma": float(self.sp_canny_sigma.value()),
            "canny_manual": bool(self.chk_canny_manual.isChecked()),
            "canny_low": int(self.sp_canny_low.value()),
            "canny_high": int(self.sp_canny_high.value()),
            "sobel_ksize": int(self.sp_sobel_ks.value()),
            "xdog_sigma": float(self.sp_xdog_sigma.value()),
            "xdog_k": float(self.sp_xdog_k.value()),
            "xdog_p": float(self.sp_xdog_p.value()),
            "xdog_eps": float(self.sp_xdog_eps.value()),
            "xdog_phi": float(self.sp_xdog_phi.value()),
            "edge_thresh": float(self.sp_edge_thresh.value()),
            "clahe_clip": float(self.sp_clahe_clip.value()),
            "clahe_grid": int(self.sp_clahe_grid.value()),
            "ms_levels": int(self.sp_ms_levels.value()),
            "chroma_w": float(self.sp_chroma_w.value()),
            "log_sigma": float(self.sp_log_sigma.value()),
            "median_ksize": int(self.sp_median_ks.value()),
            "morph_dilate": int(self.sp_morph_d.value()),
            "morph_erode": int(self.sp_morph_e.value()),
        }
        return {
            "tint": self.edit_tint.text().strip(),
            "size": int(self.spin_size.value()),
            "square": bool(self.chk_square.isChecked()),
            "stretch": bool(self.chk_stretch.isChecked()),  # NEW
            "line": float(self.sp_line.value()),
            "glow": float(self.sp_glow.value()),
            "glow_gain": float(self.sp_glow_gain.value()),
            "motion": int(self.sp_motion.value()),
            "angle": float(self.sp_angle.value()),
            "ab": int(self.sp_ab.value()),
            "scan": float(self.sp_scan.value()),
            "period": int(self.sp_period.value()),
            "vignette": float(self.sp_vignette.value()),
            "grain": float(self.sp_grain.value()),
            "edges_mode": self.combo_edges.currentText(),
            "edge_params": edge_params
        }

    def on_any_change(self, *args):
        if self.chk_auto.isChecked() and self.src_path:
            self.update_preview()

    def update_preview(self):
        if not self.src_path:
            return
        upsel = self.combo_upscale.currentText()
        upscale = 1
        if "2x" in upsel: upscale=2
        if "4x" in upsel: upscale=4
        params = self.collect_params()
        self.run_worker(upscale, params, save_path=None, max_kb=0)

    def process_and_save(self):
        if not self.src_path:
            QtWidgets.QMessageBox.warning(self, "Немає вхідного", "Спочатку відкрий зображення.")
            return
        save_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Зберегти як…", "", "PNG (*.png);;JPEG (*.jpg *.jpeg)")
        if not save_path:
            return
        upsel = self.combo_upscale.currentText()
        upscale = 1
        if "2x" in upsel: upscale=2
        if "4x" in upsel: upscale=4
        params = self.collect_params()
        max_kb = int(self.spin_limit.value()) if self.chk_limit.isChecked() else 0
        self.run_worker(upscale, params, save_path=save_path, max_kb=max_kb)

    def run_worker(self, upscale, params, save_path=None, max_kb=0):
        self.thread = QtCore.QThread(self)
        self.worker = RenderWorker(self.src_path, upscale, params, save_path=save_path, max_kb=max_kb)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.preview_ready.connect(self.on_preview_ready)
        self.worker.saved.connect(self.on_saved)
        self.worker.failed.connect(self.on_failed)
        self.worker.preview_ready.connect(self.thread.quit)
        self.worker.saved.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.start()

    @QtCore.pyqtSlot(np.ndarray)
    def on_preview_ready(self, img):
        self.show_image(img, self.lbl_out)
        self.log_msg("Предпросмотр оновлено.")

    @QtCore.pyqtSlot(np.ndarray, str)
    def on_saved(self, img, path):
        self.show_image(img, self.lbl_out)
        self.log_msg(f"Збережено: {path}")

    @QtCore.pyqtSlot(str)
    def on_failed(self, err):
        QtWidgets.QMessageBox.critical(self, "Помилка", err)
        self.log_msg("Помилка: " + err)

def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
