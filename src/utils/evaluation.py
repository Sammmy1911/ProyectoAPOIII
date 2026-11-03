import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)

from src.utils.config import CLASSES, PREDICTIONS_DIR


def compute_metrics(y_true, y_pred) -> dict:
    """Métricas comunes para comparar todos los modelos."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", labels=CLASSES),
        "report": classification_report(
            y_true, y_pred, labels=CLASSES, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=CLASSES),
    }


def save_predictions(paths, y_true, y_pred, y_proba, model_name: str, split: str):
    """
    Guarda un CSV con el formato común que usará Samuel para la evaluación:
    path, y_true, y_pred, proba_<clase>... 
    y_proba debe tener columnas en el orden de CLASSES.
    """
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"path": paths, "y_true": y_true, "y_pred": y_pred})
    for i, cls in enumerate(CLASSES):
        df[f"proba_{cls}"] = np.asarray(y_proba)[:, i]
    out = PREDICTIONS_DIR / f"{model_name}_{split}.csv"
    df.to_csv(out, index=False)
    return out