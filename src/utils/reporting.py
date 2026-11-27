"""Tablas y gráficas para los resultados de la CNN (vectoriales: PDF y SVG)."""
import csv
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

CLASES_ES = {"Bad": "Mala", "Regular": "Regular", "Good": "Buena"}


def _guardar(fig, nombre, figures_dir):
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg"):                       # vectorial, como pide el enunciado
        fig.savefig(figures_dir / f"{nombre}.{ext}", bbox_inches="tight")
    return figures_dir / f"{nombre}.pdf"


def curvas_entrenamiento(history, figures_dir, nombre="cnn_curvas"):
    """Pérdida y exactitud de train vs validación. La brecha entre ambas indica sobreajuste."""
    h = history.history if hasattr(history, "history") else history
    ep = np.arange(1, len(h["loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for a, clave, titulo in ((ax[0], "loss", "Pérdida"), (ax[1], "accuracy", "Exactitud")):
        a.plot(ep, h[clave], label="Entrenamiento")
        a.plot(ep, h["val_" + clave], label="Validación")
        a.set_xlabel("Época"); a.set_title(titulo); a.grid(alpha=.3); a.legend()
    fig.tight_layout()
    _guardar(fig, nombre, figures_dir)
    return fig


def matriz_confusion(y_true, y_pred, clases, figures_dir, nombre="cnn_confusion", normalizar=True):
    """Matriz de confusión; normalizada por fila = recall de cada clase."""
    cm = confusion_matrix(y_true, y_pred, labels=clases)
    mostrar = cm / cm.sum(axis=1, keepdims=True) if normalizar else cm
    etiquetas = [CLASES_ES.get(c, c) for c in clases]
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    im = ax.imshow(mostrar, cmap="Greens", vmin=0, vmax=1 if normalizar else None)
    ax.set_xticks(range(len(clases)), etiquetas); ax.set_yticks(range(len(clases)), etiquetas)
    ax.set_xlabel("Predicha"); ax.set_ylabel("Real")
    for i in range(len(clases)):
        for j in range(len(clases)):
            txt = f"{mostrar[i, j]:.2f}" if normalizar else f"{cm[i, j]}"
            ax.text(j, i, txt, ha="center", va="center",
                    color="white" if mostrar[i, j] > .6 else "black")
    fig.colorbar(im, ax=ax, fraction=.046)
    fig.tight_layout()
    _guardar(fig, nombre, figures_dir)
    return fig


def tabla_metricas(y_true, y_pred, clases):
    """Precisión, recall y F1 por clase, más exactitud y F1 macro (para pegar en el informe)."""
    rep = classification_report(y_true, y_pred, labels=clases, output_dict=True, zero_division=0)
    filas = [{"Clase": CLASES_ES.get(c, c), "Precisión": rep[c]["precision"],
              "Recall": rep[c]["recall"], "F1": rep[c]["f1-score"], "Soporte": int(rep[c]["support"])}
             for c in clases]
    filas.append({"Clase": "Macro", "Precisión": rep["macro avg"]["precision"],
                  "Recall": rep["macro avg"]["recall"], "F1": rep["macro avg"]["f1-score"],
                  "Soporte": int(rep["macro avg"]["support"])})
    df = pd.DataFrame(filas).round(3)
    df.attrs["accuracy"] = round(accuracy_score(y_true, y_pred), 3)
    return df


def errores_frecuentes(paths, y_true, y_pred, proba_max, n=12):
    """Las n imágenes mal clasificadas con mayor confianza (suelen revelar etiquetas dudosas)."""
    df = pd.DataFrame({"path": paths, "real": y_true, "predicha": y_pred, "confianza": proba_max})
    return df[df.real != df.predicha].sort_values("confianza", ascending=False).head(n)


def registrar_experimento(archivo, **campos):
    """Añade una fila a results/experiments.csv (parámetros, métricas, tiempo, semilla)."""
    archivo = Path(archivo); archivo.parent.mkdir(parents=True, exist_ok=True)
    campos = {"fecha": time.strftime("%Y-%m-%d %H:%M"), **campos}
    nuevo = not archivo.exists()
    with open(archivo, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(campos)); 
        if nuevo: w.writeheader()
        w.writerow(campos)