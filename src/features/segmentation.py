"""
Segmentación de la fruta: separa la fruta del fondo y devuelve una máscara binaria.

Idea (GrabCut, Rother et al., 2004):
    1. Se asume que la fruta está cerca del centro y que los bordes de la imagen son fondo.
    2. GrabCut modela los colores de la fruta y del fondo con mezclas de gaussianas (GMM)
       y separa los píxeles con un corte de grafo que además favorece que vecinos
       parecidos queden en el mismo grupo.
    3. Pista extra: los píxeles grises / poco saturados (piso, estantes) se marcan como
       "probablemente fondo".
    4. Limpieza: apertura/cierre morfológico, se conserva el componente más grande y se
       rellenan los huecos.

Uso:
    from src.features.segmentation import segment_fruit
    mask = segment_fruit(img_bgr)          # uint8, 255 = fruta, 0 = fondo
"""
import cv2
import numpy as np

MAX_SIDE = 512          # se trabaja a esta resolución por velocidad
LOW_SATURATION = 45     # S (0-255) por debajo de esto se considera gris lo que significa que es probablemente un fondo


def _resize(img, max_side=MAX_SIDE):
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return img, 1.0
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA), scale


def _largest_component_filled(mask):
    """Conserva solo la región blanca más grande y rellena sus huecos."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return mask
    biggest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    out = np.where(labels == biggest, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(out)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled


def segment_fruit(img_bgr, iterations=5, margin=0.08):
    """
    Devuelve la máscara de la fruta (uint8, 0/255) del mismo tamaño que la imagen original.

    img_bgr   : imagen leída con cv2.imread (formato BGR).
    iterations: iteraciones de GrabCut (más = más preciso y más lento).
    margin    : fracción del borde que se marca como fondo seguro.
    """
    small, _ = _resize(img_bgr)
    h, w = small.shape[:2]

    # Máscara inicial: todo "probablemente fondo"
    gc = np.full((h, w), cv2.GC_PR_BGD, np.uint8)

    # Elipse central: "probablemente fruta"
    center = (w // 2, h // 2)
    axes = (int(w * 0.30), int(h * 0.30))
    cv2.ellipse(gc, center, axes, 0, 0, 360, cv2.GC_PR_FGD, -1)

    # Píxeles grises (poco saturados): fondo probable, aunque estén en el centro
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    gc[hsv[:, :, 1] < LOW_SATURATION] = cv2.GC_PR_BGD

    # Borde de la imagen: fondo seguro
    mh, mw = max(1, int(h * margin)), max(1, int(w * margin))
    gc[:mh, :] = cv2.GC_BGD
    gc[-mh:, :] = cv2.GC_BGD
    gc[:, :mw] = cv2.GC_BGD
    gc[:, -mw:] = cv2.GC_BGD

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(small, gc, None, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_MASK)
        mask = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    except cv2.error:
        # Si GrabCut falla (p. ej. imagen sin variación), se usa la elipse central
        mask = np.zeros((h, w), np.uint8)
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

    # Limpieza de la máscara para eliminar huecos y regiones pequeñas
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = _largest_component_filled(mask)

    # Volver al tamaño original
    H, W = img_bgr.shape[:2]
    return cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)


def mask_area_ratio(mask):
    """Fracción de la imagen ocupada por la fruta (sirve para detectar segmentaciones raras)."""
    return float((mask > 0).mean())
