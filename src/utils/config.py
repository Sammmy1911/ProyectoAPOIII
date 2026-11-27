from pathlib import Path

# Raíz del repo, calculada desde este archivo (no depende de dónde se lance Jupyter)
ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "src" / "data"
RAW_KAGGLE_DIR = DATA_DIR / "raw" / "kaggle"
RAW_OWN_DIR = DATA_DIR / "raw" / "own"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = ROOT / "results"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
MODELS_DIR = ROOT / "models_saved"

SEED = 42
IMG_SIZE = (224, 224)
CNN_IMG_SIZE = (224,224)

# Orden ordinal: de peor a mejor calidad
CLASSES = ["Bad Quality_Fruits", "Regular Quality_Fruits", "Good Quality_Fruits"]
CLASS_FOLDERS = {c: c for c in CLASSES}
# Resultado: {"Bad Quality_Fruits": "Bad Quality_Fruits", "Regular Quality_Fruits": "Regular Quality_Fruits", "Good Quality_Fruits": "Good Quality_Fruits"}