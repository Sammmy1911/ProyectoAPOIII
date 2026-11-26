"""Contrato de inferencia para la interfaz.

La interfaz solo conoce esta funcion:

    predictor = get_predictor()
    resultado = predictor.predict(imagen_pil)

y recibe siempre un diccionario con esta forma:

    {
        "calidad": "Good",                  # una de CLASSES
        "probabilidad": 0.87,               # probabilidad de la clase ganadora
        "probabilidades": {"Bad": .., "Regular": .., "Good": ..},
        "tamaño": "mediano" | None,         # None = aun no hay modelo de tamano
        "modelo": "CNN",
        "es_mock": False,
        "latencia_ms": 12.3,                # opcional, tiempo de inferencia
    }

Si existe models_saved/cnn_best.keras se usa la CNN real;
si no, se intenta modelo clásico (model.joblib + scaler.joblib + encoder.joblib);
si no, un mock determinista para poder desarrollar y demostrar la interfaz sin modelo.
Las imagenes se procesan en memoria: este modulo nunca escribe a disco.
"""
from __future__ import annotations

import hashlib
import io
import time
from pathlib import Path
from typing import TypedDict, NotRequired

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from src.utils.config import CLASSES, CNN_IMG_SIZE, MODELS_DIR

CNN_MODEL_PATH = MODELS_DIR / "cnn_best.keras"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_PIXELS = 40_000_000  # ~40 MP; por encima se rechaza para no agotar memoria
SIZES = ["pequeño", "mediano", "grande"]


class ImageError(ValueError):
    """Error de entrada con un mensaje listo para mostrar al usuario."""


# --------------------------------------------------------------------------
# Contrato tipado
# --------------------------------------------------------------------------
class PredictResult(TypedDict):
    calidad: str
    probabilidad: float
    probabilidades: dict[str, float]
    tamaño: str | None
    modelo: str
    es_mock: bool
    latencia_ms: NotRequired[float]


# --------------------------------------------------------------------------
# Entrada
# --------------------------------------------------------------------------
def load_image(source, name: str | None = None) -> Image.Image:
    """Convierte bytes o un archivo subido en una imagen PIL RGB, en memoria."""
    if hasattr(source, "getvalue"):
        data = source.getvalue()
    elif hasattr(source, "read"):
        data = source.read()
    else:
        data = bytes(source)

    name = name or getattr(source, "name", "") or ""
    ext = Path(name).suffix.lower()
    if ext and ext not in ALLOWED_EXTENSIONS:
        raise ImageError(
            f"Formato {ext} no soportado. Usa JPG, PNG o WEBP."
        )

    try:
        img = Image.open(io.BytesIO(data))
        if img.width * img.height > MAX_PIXELS:
            raise ImageError(
                "La imagen es demasiado grande. Reduce su resolución e inténtalo de nuevo."
            )
        img.load()
    except ImageError:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise ImageError(
            "No se pudo leer el archivo. Comprueba que sea una imagen válida."
        ) from None

    img = ImageOps.exif_transpose(img)  # fotos de celular rotadas
    if img.mode in ("RGBA", "LA") or "transparency" in img.info:
        rgba = img.convert("RGBA")
        fondo = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        fondo.alpha_composite(rgba)
        return fondo.convert("RGB")
    return img.convert("RGB")


def preprocess(image: Image.Image, img_size=CNN_IMG_SIZE) -> np.ndarray:
    """PIL -> array (1, alto, ancho, 3) float32 en rango 0-255, orden RGB.

    El modelo ya incluye Rescaling(1/255), asi que aqui NO se normaliza.
    El redimensionado usa cv2.resize (sin conservar proporcion), igual que el
    preprocesamiento del dataset, para que entrenamiento e inferencia coincidan.
    """
    import cv2  # import perezoso

    arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    alto, ancho = img_size
    arr = cv2.resize(arr, (ancho, alto))  # cv2 recibe (ancho, alto)
    return np.expand_dims(arr.astype("float32"), axis=0)


def _resultado(
    proba: np.ndarray,
    tamaño: str | None,
    modelo: str,
    es_mock: bool,
    latencia_ms: float | None = None,
) -> PredictResult:
    proba = np.asarray(proba, dtype=float)
    i = int(proba.argmax())
    result: PredictResult = {
        "calidad": CLASSES[i],
        "probabilidad": float(proba[i]),
        "probabilidades": {c: float(p) for c, p in zip(CLASSES, proba)},
        "tamaño": tamaño,
        "modelo": modelo,
        "es_mock": es_mock,
    }
    if latencia_ms is not None:
        result["latencia_ms"] = latencia_ms
    return result


# --------------------------------------------------------------------------
# Predictores
# --------------------------------------------------------------------------
class MockPredictor:
    """Prediccion falsa pero determinista: la misma imagen da la misma respuesta."""

    nombre = "Demostración"
    es_mock = True
    motivo = "Aún no hay un modelo entrenado en models_saved/."

    def predict(self, image: Image.Image) -> PredictResult:
        t0 = time.perf_counter()
        semilla = int(hashlib.md5(image.tobytes()).hexdigest(), 16) % (2**32)
        rng = np.random.default_rng(semilla)
        proba = rng.dirichlet(np.ones(len(CLASSES)))
        latencia_ms = (time.perf_counter() - t0) * 1000
        return _resultado(proba, str(rng.choice(SIZES)), "Demostración", True, latencia_ms)


class KerasPredictor:
    nombre = "CNN"
    es_mock = False
    motivo = ""

    def __init__(self, model_path: Path = CNN_MODEL_PATH):
        import tensorflow as tf  # import perezoso: la app corre sin TF si usa el mock

        self.model = tf.keras.models.load_model(model_path, compile=False)

    def predict(self, image: Image.Image) -> PredictResult:
        t0 = time.perf_counter()
        proba = self.model.predict(preprocess(image), verbose=0)[0]
        latencia_ms = (time.perf_counter() - t0) * 1000
        # tamaño: lo aportará el modulo de caracteristicas; None = pendiente
        return _resultado(proba, None, "CNN", False, latencia_ms)


class SklearnPredictor:
    """Predictor para modelo clásico (joblib + scaler + encoder)."""

    nombre = "Clásico"
    es_mock = False
    motivo = ""

    def __init__(self, model_dir: Path = MODELS_DIR):
        self.model_dir = model_dir
        self._model = None
        self._scaler = None
        self._encoder = None

    def _load_artifacts(self):
        """Cargar model.joblib, scaler.joblib, encoder.joblib desde model_dir."""
        import joblib  # import perezoso

        model_path = self.model_dir / "model.joblib"
        scaler_path = self.model_dir / "scaler.joblib"
        encoder_path = self.model_dir / "encoder.joblib"

        if not (model_path.exists() and scaler_path.exists() and encoder_path.exists()):
            raise FileNotFoundError(
                f"Artefactos del modelo clásico no encontrados en {self.model_dir}"
            )

        self._model = joblib.load(model_path)
        self._scaler = joblib.load(scaler_path)
        self._encoder = joblib.load(encoder_path)

    def predict(self, image: Image.Image) -> PredictResult:
        import time

        t0 = time.perf_counter()
        # 1. PIL RGB -> cv2 BGR
        import cv2  # import perezoso

        img_bgr = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)

        # 2. extract_features(img_bgr) -> vector
        # 3. scaler.transform([vector])
        # 4. model.predict_proba -> proba
        # 5. encoder.inverse_transform para mapear a CLASSES
        # 6. _resultado(proba, None, "Clásico", False, latencia_ms)
        raise NotImplementedError("Artefactos modelo clásico no disponibles aún")


class SizePredictor:
    """Predictor de tamaño basado en umbrales calibrados."""

    def __init__(
        self, thresholds_path: Path = Path(__file__).parent.parent / "features" / "size_thresholds.json"
    ):
        self.thresholds_path = thresholds_path

    def predict_size(self, image: Image.Image, fruit_hint: str | None = None) -> dict:
        # 1. PIL -> cv2 BGR
        # 2. segment_fruit(img_bgr) -> mask
        # 3. shape_features(mask) -> diameter_norm
        # 4. estimate_size(diameter_norm, fruit_hint or "all", thresholds)
        # 5. devuelve {"size": "small|medium|large", "diameter_norm": float, ...}
        raise NotImplementedError("Umbrales de tamaño no calibrados aún")


def get_predictor():
    """Cadena de prioridad:
    1. CNN real (cnn_best.keras existe + TF carga)
    2. Clásico real (model.joblib + scaler.joblib + encoder.joblib existen)
    3. Mock (fallback)
    Cada predictor tiene .motivo explicativo si falla.
    """
    # 1. CNN
    if CNN_MODEL_PATH.exists():
        try:
            return KerasPredictor(CNN_MODEL_PATH)
        except Exception as exc:  # TF ausente, version incompatible, archivo danado
            predictor = MockPredictor()
            predictor.motivo = f"No se pudo cargar la CNN ({type(exc).__name__})."
            return predictor

    # 2. Modelo clásico (sklearn)
    classic_model = MODELS_DIR / "model.joblib"
    classic_scaler = MODELS_DIR / "scaler.joblib"
    classic_encoder = MODELS_DIR / "encoder.joblib"
    if classic_model.exists() and classic_scaler.exists() and classic_encoder.exists():
        try:
            predictor = SklearnPredictor(MODELS_DIR)
            predictor._load_artifacts()
            return predictor
        except Exception as exc:
            predictor = MockPredictor()
            predictor.motivo = f"No se pudo cargar el modelo clásico ({type(exc).__name__})."
            return predictor

    # 3. Mock
    return MockPredictor()