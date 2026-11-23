"""
Extracción de características: convierte una imagen de fruta en un vector de números
que los modelos clásicos (SVM, Random Forest, ...) pueden usar.

Se calculan SOLO sobre los píxeles de la fruta (máscara de segmentation.py):
    1. Color   -> histogramas HSV           (madurez, manchas, podredumbre cambian el color)
    2. Textura -> histograma LBP            (golpes, moho y arrugas cambian la textura)
    3. Forma   -> área, circularidad, ...   (deformaciones; base para el tamaño)

Uso:
    from src.features.extract import extract_features
    vector, nombres = extract_features(img_bgr)        # segmenta internamente
"""
import cv2
import numpy as np
from skimage.feature import local_binary_pattern

from src.features.segmentation import segment_fruit

# ----------------------------------------------------------------------------- color
HSV_BINS = 16            # cajones por canal: pocos = robusto, muchos = más detalle
# Rangos de OpenCV: H en [0, 180), S y V en [0, 256)
HSV_RANGES = {"h": 180, "s": 256, "v": 256}


def color_features(img_bgr, mask, bins=HSV_BINS):
    """
    Histograma de H, S y V (por separado) de los píxeles de la fruta.
    Cada histograma se normaliza para que sume 1: así no depende del tamaño de la
    fruta en la foto, solo de la PROPORCIÓN de cada color.
    Además se agregan media y desviación de cada canal.
    Devuelve 3*bins + 6 números.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    feats, names = [], []
    for i, ch in enumerate("hsv"):
        hist = cv2.calcHist([hsv], [i], mask, [bins], [0, HSV_RANGES[ch]]).ravel()
        hist = hist / (hist.sum() + 1e-8)
        feats.extend(hist)
        names.extend(f"{ch}_hist_{b:02d}" for b in range(bins))
    pix = hsv[mask > 0].astype(np.float32)
    if len(pix) == 0:
        pix = np.zeros((1, 3), np.float32)
    for i, ch in enumerate("hsv"):
        feats.extend([pix[:, i].mean(), pix[:, i].std()])
        names.extend([f"{ch}_mean", f"{ch}_std"])
    return np.array(feats, np.float32), names


# --------------------------------------------------------------------------- textura
LBP_POINTS = 8           # vecinos que se comparan con cada píxel
LBP_RADIUS = 1           # distancia de esos vecinos


def texture_features(img_bgr, mask, points=LBP_POINTS, radius=LBP_RADIUS):
    """
    LBP (Local Binary Patterns, Ojala et al., 2002): cada píxel se compara con sus
    vecinos y se codifica si son más claros o más oscuros. El histograma de esos
    códigos describe la textura: piel lisa vs. rugosa, con moho, arrugada.
    Con method='uniform' hay points + 2 códigos posibles -> 10 números.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, points, radius, method="uniform")
    n_codes = points + 2
    vals = lbp[mask > 0]
    hist, _ = np.histogram(vals, bins=n_codes, range=(0, n_codes))
    hist = hist.astype(np.float32) / (hist.sum() + 1e-8)
    return hist, [f"lbp_{k:02d}" for k in range(n_codes)]


# ----------------------------------------------------------------------------- forma
SHAPE_NAMES = ["area_ratio", "circularity", "solidity", "aspect_ratio", "diameter_norm"]


def shape_features(mask):
    """
    Describe la FORMA de la fruta a partir de la máscara. Devuelve 5 números:
        area_ratio   : fracción de la imagen que ocupa la fruta = píxeles de fruta / píxeles totales
        circularity  : 4*pi*area / perimetro**2      (1 = círculo perfecto, menor = más irregular)
        solidity     : area / area_del_casco_convexo  (1 = sin hendiduras ni mordidas)
        aspect_ratio : lado_largo / lado_corto del rectángulo mínimo que la encierra
                       (~1 naranja, alto para banano)
        diameter_norm: diámetro equivalente / max(alto, ancho) de la imagen
                       diámetro equivalente = 2*sqrt(area/pi)  (diámetro de un círculo de igual área)

    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.zeros(len(SHAPE_NAMES), np.float32), list(SHAPE_NAMES)

    c = max(contours, key=cv2.contourArea)          # contorno de la fruta (el más grande)
    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)
    hull_area = cv2.contourArea(cv2.convexHull(c))
    (_, _), (w, h), _ = cv2.minAreaRect(c)

    H, W = mask.shape[:2]
    area_ratio = area / (H * W)
    circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0.0
    solidity = area / hull_area if hull_area > 0 else 0.0
    aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0.0
    diameter_norm = 2 * np.sqrt(area / np.pi) / max(H, W)

    values = [area_ratio, circularity, solidity, aspect_ratio, diameter_norm]
    return np.array(values, np.float32), list(SHAPE_NAMES)


# ------------------------------------------------------------------------ todo junto
IMG_SIZE = (224, 224)    # mismo tamaño que usa preprocess.ipynb para todo el dataset


def extract_features(img_bgr, mask=None, include_shape=True, size=IMG_SIZE):
    """
    Redimensiona a 224x224 (igual que el dataset de Samuel, así una foto propia y una de
    Kaggle se miden en la misma escala), segmenta y concatena color + textura (+ forma).
    """
    if size is not None and img_bgr.shape[:2] != (size[1], size[0]):
        img_bgr = cv2.resize(img_bgr, size, interpolation=cv2.INTER_AREA)
        if mask is not None:
            mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    if mask is None:
        mask = segment_fruit(img_bgr)
    parts = [color_features(img_bgr, mask), texture_features(img_bgr, mask)]
    if include_shape:
        parts.append(shape_features(mask))
    vector = np.concatenate([p[0] for p in parts])
    names = [n for p in parts for n in p[1]]
    return vector, names
