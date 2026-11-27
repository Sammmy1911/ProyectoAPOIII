"""
Extracción masiva de características para todo el dataset.

Recorre las imágenes ORIGINALES (sin aumentos) de:
    - Kaggle:   src/data/raw/kaggle/{Bad,Good} Quality_Fruits/<Fruta>_<Calidad>/*.jpg   (Mixed se ignora)
    - OneDrive: src/data/raw/OneDrive_2026-11-23/.../{Bad,Regular,Good} Quality_Fruits/<Fruta>_<Calidad>/*.jpg
    - Propias:  src/data/raw/own/*.{jpg,jpeg,png,webp} (fruta_calidad_tamaño_###)

y guarda UNA fila por imagen en src/data/processed/features_all.csv con:
    rel_path, source, class, label, fruit, <69 características>

Las características se calculan una sola vez y se cruzan después con cualquier split
(por ejemplo dataset_splits.csv de Samuel) usando la ruta relativa de la imagen.

Uso (desde la raíz del repo):
    .venv/Scripts/python.exe -m src.features.build_features            # todo
    .venv/Scripts/python.exe -m src.features.build_features --limit 50 # prueba rápida
"""
import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.features.extract import extract_features  # noqa: E402

RAW = ROOT / "src" / "data" / "raw"
OUT = ROOT / "src" / "data" / "processed" / "features_all.csv"

# Orden oficial de clases (igual que dataset_metadata.json de Samuel: 0, 1, 2)
CLASSES = ["Bad Quality_Fruits", "Regular Quality_Fruits", "Good Quality_Fruits"]
LABEL = {c: i for i, c in enumerate(CLASSES)}
OWN_QUALITY = {"bad": "Bad Quality_Fruits", "regular": "Regular Quality_Fruits",
               "good": "Good Quality_Fruits"}
EXT = {".jpg", ".jpeg", ".png", ".webp"}


def _find_onedrive_root():
    """Encuentra la carpeta que contiene las clases dentro de OneDrive_*/ (con o sin 'APO3 PROJECT')."""
    for base in sorted(RAW.glob("OneDrive_*")):
        for cand in [base / "APO3 PROJECT", base]:
            if (cand / CLASSES[0]).is_dir():
                return cand
    return None


def list_images():
    rows = []
    sources = [("kaggle", RAW / "kaggle", ["Bad Quality_Fruits", "Good Quality_Fruits"])]
    od = _find_onedrive_root()
    if od is not None:
        sources.append(("onedrive", od, CLASSES))
    for source, base, classes in sources:
        for cls in classes:
            for p in sorted((base / cls).rglob("*")):
                if p.suffix.lower() in EXT and not p.name.startswith("aug_"):
                    fruit = p.parent.name.split("_")[0].lower() if p.parent != base / cls else "unknown"
                    rows.append(dict(rel_path=p.relative_to(RAW).as_posix(), source=source,
                                     **{"class": cls}, label=LABEL[cls], fruit=fruit))
    for p in sorted((RAW / "own").iterdir()):
        if not p.is_file() or p.suffix.lower() not in EXT:
            continue
        fruit, quality = p.stem.split("_")[:2]
        cls = OWN_QUALITY[quality]
        rows.append(dict(rel_path=p.relative_to(RAW).as_posix(), source="own",
                         **{"class": cls}, label=LABEL[cls], fruit=fruit))
    return pd.DataFrame(rows)


def _read(path):
    """Las fotos de celular (varios MB) se leen ya reducidas a 1/4: ahorra memoria y tiempo,
    y no se pierde nada porque después todo se lleva a 224x224."""
    flag = cv2.IMREAD_REDUCED_COLOR_4 if os.path.getsize(path) > 1_000_000 else cv2.IMREAD_COLOR
    return cv2.imread(str(path), flag)


def _one(rel_path):
    try:
        img = _read(RAW / rel_path)
        if img is None:
            return None
        vec, _ = extract_features(img)
        return vec
    except Exception:
        return None


def main(limit=None, n_jobs=4, chunk=1000):
    df = list_images()
    if limit:
        df = df.groupby(["source", "class"], group_keys=False).head(limit).reset_index(drop=True)
    print(f"Imágenes en total: {len(df)}", flush=True)
    print(df.groupby(["source", "class"]).size().to_string(), flush=True)

    _, names = extract_features(np.zeros((224, 224, 3), np.uint8) + 128)
    target = OUT if not limit else OUT.with_name("features_sample.csv")
    target.parent.mkdir(parents=True, exist_ok=True)

    # Reanudable: si el archivo ya existe, se saltan las imágenes ya procesadas
    done = set()
    if target.exists():
        done = set(pd.read_csv(target, usecols=["rel_path"])["rel_path"])
        print(f"Reanudando: {len(done)} imágenes ya estaban procesadas", flush=True)
    todo = df[~df["rel_path"].isin(done)].reset_index(drop=True)

    t0, failed = time.time(), 0
    for start in range(0, len(todo), chunk):
        part = todo.iloc[start:start + chunk].reset_index(drop=True)
        vecs = Parallel(n_jobs=n_jobs, batch_size=16)(delayed(_one)(p) for p in part["rel_path"])
        ok = [v is not None for v in vecs]
        failed += len(ok) - sum(ok)
        feats = pd.DataFrame([v for v in vecs if v is not None], columns=names)
        out = pd.concat([part[ok].reset_index(drop=True), feats], axis=1)
        out.to_csv(target, mode="a", header=not target.exists(), index=False)
        n = min(start + chunk, len(todo))
        print(f"  {n}/{len(todo)} guardadas | {time.time() - t0:.0f} s | fallidas: {failed}", flush=True)

    print(f"Listo: {target}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="imágenes por (fuente, clase) para probar")
    ap.add_argument("--jobs", type=int, default=4, help="procesos en paralelo (más = más rápido y más memoria)")
    a = ap.parse_args()
    main(a.limit, a.jobs)
