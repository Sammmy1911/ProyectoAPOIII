"""
Módulo de estimación de tamaño: small / medium / large.

Problema: Kaggle y OneDrive NO traen etiquetas de tamaño, así que no se puede entrenar
un modelo supervisado. En su lugar se MIDE el tamaño y se compara con frutas del mismo tipo.

1. Medición (en extract.shape_features):
       diameter_norm = 2 * sqrt(area / pi) / max(alto, ancho)
   es el diámetro de un círculo con la misma área que la fruta, como fracción del lado mayor
   de la imagen. No depende de la resolución de la foto.

2. Umbrales por fruta (calibrate_thresholds):
       t1 = percentil 33 y t2 = percentil 66 de diameter_norm para esa fruta en el dataset.
   size = small si d < t1;  medium si t1 <= d < t2;  large si d >= t2.
   Es un tamaño RELATIVO a las demás frutas del mismo tipo (terciles), y por eso los
   umbrales son distintos para una manzana que para una granada.

Supuesto clave: todas las fotos se toman a una distancia parecida de la cámara. Si la
cámara se acerca, la fruta "crece" en píxeles. Por eso el protocolo de captura fija la
distancia (por ejemplo, celular apoyado a ~30 cm), y las fotos propias con peso en gramos
sirven para validar que el orden small < medium < large tenga sentido físico.

Uso:
    th = calibrate_thresholds(df_features)                 # df con columnas fruit, diameter_norm
    save_thresholds(th);  th = load_thresholds()
    estimate_size(0.45, "apple", th)  -> {"size": "medium", "diameter_norm": 0.45, ...}
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
THRESHOLDS_PATH = ROOT / "src" / "features" / "size_thresholds.json"
SIZES = ["small", "medium", "large"]
PERCENTILES = (33.3, 66.7)


def calibrate_thresholds(df, fruit_col="fruit", diam_col="diameter_norm", percentiles=PERCENTILES):
    """Calcula (t1, t2) por fruta con percentiles de diameter_norm. Agrega 'all' como respaldo."""
    th = {}
    for fruit, d in df.groupby(fruit_col)[diam_col]:
        d = d[d > 0]
        if len(d) >= 10:
            th[fruit] = [float(np.percentile(d, p)) for p in percentiles]
    d_all = df[diam_col][df[diam_col] > 0]
    th["all"] = [float(np.percentile(d_all, p)) for p in percentiles]
    return th


def estimate_size(diameter_norm, fruit=None, thresholds=None):
    """Clasifica un diámetro normalizado en small / medium / large usando los umbrales de su fruta."""
    thresholds = thresholds or load_thresholds()
    key = fruit if fruit in thresholds else "all"
    t1, t2 = thresholds[key]
    size = SIZES[0] if diameter_norm < t1 else SIZES[1] if diameter_norm < t2 else SIZES[2]
    return {"size": size, "diameter_norm": float(diameter_norm), "thresholds": [t1, t2],
            "reference": key}


def save_thresholds(th, path=THRESHOLDS_PATH):
    Path(path).write_text(json.dumps(th, indent=2), encoding="utf-8")


def load_thresholds(path=THRESHOLDS_PATH):
    return json.loads(Path(path).read_text(encoding="utf-8"))
