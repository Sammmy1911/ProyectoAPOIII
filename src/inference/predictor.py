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
    }

Si existe models_saved/cnn_best.keras se usa la CNN real; si no, un mock
determinista para poder desarrollar y demostrar la interfaz sin modelo.
Las imagenes se procesan en memoria: este modulo nunca escribe a disco.
"""
from __future__ import annotations

import hashlib
import io
from pathlib import Path

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
    import cv2

    arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    alto, ancho = img_size
    arr = cv2.resize(arr, (ancho, alto))  # cv2 recibe (ancho, alto)
    return np.expand_dims(arr.astype("float32"), axis=0)


def _resultado(proba, tamaño, modelo: str, es_mock: bool) -> dict:
    proba = np.asarray(proba, dtype=float)
    i = int(proba.argmax())
    return {
        "calidad": CLASSES[i],
        "probabilidad": float(proba[i]),
        "probabilidades": {c: float(p) for c, p in zip(CLASSES, proba)},
        "tamaño": tamaño,
        "modelo": modelo,
        "es_mock": es_mock,
    }


# --------------------------------------------------------------------------
# Predictores
# --------------------------------------------------------------------------
class MockPredictor:
    """Prediccion falsa pero determinista: la misma imagen da la misma respuesta."""

    nombre = "Demostración"
    es_mock = True
    motivo = "Aún no hay un modelo entrenado en models_saved/."

    def predict(self, image: Image.Image) -> dict:
        semilla = int(hashlib.md5(image.tobytes()).hexdigest(), 16) % (2**32)
        rng = np.random.default_rng(semilla)
        proba = rng.dirichlet(np.ones(len(CLASSES)))
        return _resultado(proba, str(rng.choice(SIZES)), "Demostración", True)


class KerasPredictor:
    nombre = "CNN"
    es_mock = False
    motivo = ""

    def __init__(self, model_path: Path = CNN_MODEL_PATH):
        import tensorflow as tf  # import perezoso: la app corre sin TF si usa el mock

        self.model = tf.keras.models.load_model(model_path, compile=False)

    def predict(self, image: Image.Image) -> dict:
        proba = self.model.predict(preprocess(image), verbose=0)[0]
        # tamano: lo aportara el modulo de caracteristicas; None = pendiente
        return _resultado(proba, None, "CNN", False)


def get_predictor():
    """CNN real si existe el archivo del modelo; si no (o si falla), mock."""
    if CNN_MODEL_PATH.exists():
        try:
            return KerasPredictor(CNN_MODEL_PATH)
        except Exception as exc:  # TF ausente, version incompatible, archivo danado
            mock = MockPredictor()
            mock.motivo = f"No se pudo cargar la CNN ({type(exc).__name__})."
            return mock
    return MockPredictor()