"""Carga del split oficial de Samuel (dataset_splits.csv + dataset_metadata.json).

Las rutas del CSV son de otra máquina (C:\\Users\\sgall\\...). Aquí se reconvierten a rutas
válidas bajo la raíz del repo, tanto en Windows como en Linux/Colab.
Se usa la columna `split_path` (la copia de la imagen dentro de src/data/splits/<split>/<clase>/).
"""
import json
from pathlib import Path

import pandas as pd

SPLIT_ALIASES = {"validation": "val", "valid": "val", "validacion": "val"}


def resolve_path(p, root):
    """'C:\\...\\src\\data\\splits\\train\\X\\a.jpg' -> '<root>/src/data/splits/train/X/a.jpg'."""
    root = Path(root)
    p = str(p)
    if Path(p).exists():
        return str(Path(p))
    partes = p.replace("\\", "/").split("/")      # entiende rutas de Windows también en Linux
    if "data" in partes:
        rel = partes[partes.index("data"):]
        for base in (root, root / "src"):
            cand = base.joinpath(*rel)
            if cand.exists():
                return str(cand)
    return p                                       # no encontrada: se deja igual para contarla como rota


def load_splits(root, classes=None, check_files=True):
    """Devuelve df_train, df_val, df_test, meta.

    Cada DataFrame tiene las columnas: path, label, class (y las originales del CSV).
    meta["class_weight_int"] es el diccionario de pesos listo para Keras (claves enteras).
    """
    root = Path(root)
    carpeta = root / "src" / "data" / "splits"
    csv_path, json_path = carpeta / "dataset_splits.csv", carpeta / "dataset_metadata.json"
    for f in (csv_path, json_path):
        if not f.exists():
            raise FileNotFoundError(
                f"Falta {f}. Si estás en Colab: git checkout origin/Preprocess -- {f.relative_to(root).as_posix()}"
            )

    df = pd.read_csv(csv_path)
    meta = json.loads(json_path.read_text(encoding="utf-8"))
    df["split"] = df["split"].astype(str).str.lower().replace(SPLIT_ALIASES)
    df["path"] = df["split_path"].map(lambda p: resolve_path(p, root))

    if classes is not None and list(meta["classes"]) != list(classes):
        raise ValueError(f"El orden de clases no coincide.\n  metadata: {meta['classes']}\n  config:   {list(classes)}")

    if check_files:
        faltan = ~df["path"].map(lambda p: Path(p).exists())
        if faltan.any():
            raise FileNotFoundError(
                f"{int(faltan.sum())} de {len(df)} imágenes no existen en disco. Ejemplo: {df.loc[faltan, 'path'].iloc[0]}\n"
                "¿Descomprimiste la carpeta 'splits' dentro de src/data/?"
            )

    meta["class_weight_int"] = {int(k): float(v) for k, v in meta["class_weight_by_idx"].items()}
    partes = {s: df[df["split"] == s].reset_index(drop=True) for s in ("train", "val", "test")}
    for s, d in partes.items():
        if d.empty:
            raise ValueError(f"No hay filas para el split '{s}'. Valores en el CSV: {sorted(df['split'].unique())}")
    return partes["train"], partes["val"], partes["test"], meta