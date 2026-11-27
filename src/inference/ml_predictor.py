"""Predictores de ML clasico (archivos .joblib) con la misma salida que KerasPredictor.

Cada .joblib en models_saved/ debe ser un Pipeline de sklearn (o un dict con la
clave 'pipeline' o 'model') que recibe el vector de 73 caracteristicas de
src.features.extract.extract_features y devuelve la clase.
"""
from pathlib import Path

import numpy as np

from src.utils.config import CLASSES, MODELS_DIR

# nombre del archivo (sin .joblib) -> nombre visible en la interfaz
NOMBRES = {
    "random_forest": "Random Forest",
    "svm_linear": "SVM lineal",
    "baseline": "Línea base",
}


def _indice_clase(valor) -> int:
    """Convierte una etiqueta de sklearn (0/1/2 o texto) en indice de CLASSES."""
    if isinstance(valor, (int, np.integer)):
        return int(valor)
    texto = str(valor).strip()
    if texto.isdigit():
        return int(texto)
    primera = texto.split()[0].split("_")[0].lower()
    for i, c in enumerate(CLASSES):
        if c.split()[0].split("_")[0].lower() == primera:
            return i
    raise ValueError(f"No reconozco la clase '{valor}' del modelo.")


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


class MLPredictor:
    es_mock = False

    def __init__(self, ruta: Path):
        import joblib

        ruta = Path(ruta)
        obj = joblib.load(ruta)
        if isinstance(obj, dict):
            obj = obj.get("pipeline") or obj.get("model") or obj.get("clf")
            if obj is None:
                raise ValueError("El joblib es un dict sin 'pipeline', 'model' ni 'clf'.")
        self.modelo_sk = obj
        self.clave = ruta.stem
        self.nombre = NOMBRES.get(self.clave, self.clave)
        self.motivo = ""
        self.prob_aproximada = False

        clases = getattr(obj, "classes_", None)
        if clases is None and hasattr(obj, "steps"):
            clases = getattr(obj.steps[-1][1], "classes_", None)
        if clases is None:
            raise ValueError("El modelo no expone classes_; no se puede alinear con CLASSES.")
        self._idx = [_indice_clase(c) for c in clases]

    def _vector(self, image):
        import pandas as pd
        from src.features.extract import extract_features

        arr = np.asarray(image)
        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[..., :3]
        bgr = np.ascontiguousarray(arr[..., ::-1]).astype(np.uint8)  # RGB -> BGR
        vector, names = extract_features(bgr)

        # El modelo espera un DataFrame con las columnas en el orden de feature_names_in_
        nombres = getattr(self.modelo_sk, "feature_names_in_", None)
        if nombres is not None:
            # Construir dict en el orden que espera el modelo
            feat_dict = {n: float(vector[names.index(n)]) for n in nombres}
            return pd.DataFrame([feat_dict])
        # Fallback: array numpy simple (orden de extract_features)
        return np.asarray(vector, dtype=float).reshape(1, -1)

    def _probabilidades(self, x: np.ndarray) -> np.ndarray:
        m = self.modelo_sk
        try:
            return np.asarray(m.predict_proba(x)[0], dtype=float)
        except (AttributeError, NotImplementedError):
            pass
        if hasattr(m, "decision_function"):
            self.prob_aproximada = True
            d = np.asarray(m.decision_function(x))[0]
            return _softmax(d)
        pred = m.predict(x)[0]
        p = np.zeros(len(self._idx))
        p[list(m.classes_).index(pred)] = 1.0
        self.prob_aproximada = True
        return p

    def predict(self, image) -> dict:
        x = self._vector(image)
        p = self._probabilidades(x)
        probs = {c: 0.0 for c in CLASSES}
        for pos, idx in enumerate(self._idx):
            probs[CLASSES[idx]] = float(p[pos])
        mejor = max(probs, key=probs.get)
        return {
            "calidad": mejor,
            "probabilidad": probs[mejor],
            "probabilidades": probs,
            "tamaño": None,
            "modelo": self.nombre,
            "es_mock": False,
            "prob_aproximada": self.prob_aproximada,
        }


def cargar_modelos_ml(carpeta: Path = MODELS_DIR):
    """Devuelve (modelos, errores): dicts {clave: MLPredictor} y {clave: mensaje}."""
    modelos, errores = {}, {}
    carpeta = Path(carpeta)
    orden = list(NOMBRES)
    archivos = sorted(carpeta.glob("*.joblib"),
                      key=lambda f: orden.index(f.stem) if f.stem in orden else 99)
    for f in archivos:
        try:
            modelos[f.stem] = MLPredictor(f)
        except Exception as exc:  # noqa: BLE001 - se muestra al usuario en la interfaz
            errores[f.stem] = f"{type(exc).__name__}: {exc}"
    return modelos, errores