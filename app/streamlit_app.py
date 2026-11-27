"""Interfaz de clasificacion de calidad de frutas.

Ejecutar desde la raiz del repo:   python -m streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from src.inference.predictor import ImageError, get_predictor, load_image
from src.utils.config import CLASSES

LOW_CONFIDENCE = 0.60

# Presentacion de cada clase: (nombre visible, color, significado)
CLASS_UI = {
    "Bad": ("Mala", "#B8402B", "Deterioro o defectos graves. No apta para la venta."),
    "Regular": ("Regular", "#A56A00", "Madurez avanzada o defectos leves. Conviene vender pronto."),
    "Good": ("Buena", "#2F7D4F", "Sin defectos visibles. Apta para la venta."),
}

st.set_page_config(page_title="Calidad de frutas", page_icon="🍎", layout="wide")

# Toda la apariencia propia vive en este bloque y en .streamlit/config.toml
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=Public+Sans:wght@400;500;600&display=swap');

:root {
  --ink: #14261D; --muted: #52645A; --line: #D3DFD7;
  --paper: #F2F6F3; --card: #FFFFFF; --leaf: #2F7D4F;
}
html, body, .stApp, [class*="css"] { font-family: 'Public Sans', system-ui, sans-serif; color: var(--ink); }
.stApp { background: var(--paper); }
.block-container { max-width: 1120px; padding-top: 2.4rem; padding-bottom: 3rem; }
header[data-testid="stHeader"] { background: transparent; }

/* cabecera */
.app-title { font-family: 'Fraunces', Georgia, serif; font-size: 2.6rem; font-weight: 700;
             letter-spacing: -.01em; margin: 0; line-height: 1.1; }
.app-sub { color: var(--muted); margin: .5rem 0 1rem 0; max-width: 58ch; font-size: 1.05rem; line-height: 1.5; }
.pill { display: inline-block; padding: .25rem .8rem; border-radius: 999px; font-size: .85rem; font-weight: 500; }
.pill-demo { background: #FFF1D6; color: #7A5200; border: 1px solid #E8C679; }
.pill-real { background: #DDEFE3; color: #1F5E3B; border: 1px solid #A9D2B6; }

/* pestañas */
.stTabs [data-baseweb="tab-list"] { gap: 1.6rem; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] { height: 3rem; padding: 0; font-weight: 500; color: var(--muted); }
.stTabs [aria-selected="true"] { color: var(--ink); }
.stTabs [data-baseweb="tab-highlight"] { background: var(--leaf); height: 3px; }

/* selector de origen y cargador */
[data-testid="stFileUploaderDropzone"] { background: var(--card); border: 1.5px dashed #9DB8A7; border-radius: 12px; }
[data-testid="stCameraInput"] > div { border-radius: 12px; overflow: hidden; border: 1px solid var(--line); }
[data-testid="stImage"] img { border-radius: 12px; border: 1px solid var(--line); }

/* veredicto con sello */
.verdict { display: flex; align-items: center; gap: 1.5rem; background: var(--card);
           border: 1px solid var(--line); border-radius: 14px; padding: 1.3rem 1.5rem; }
.sello { flex: none; width: 7.4rem; height: 7.4rem; border-radius: 50%; display: flex;
         flex-direction: column; align-items: center; justify-content: center;
         border: 4px solid var(--grade); outline: 2px solid var(--grade); outline-offset: 4px;
         background: color-mix(in srgb, var(--grade) 9%, white); transform: rotate(-6deg); }
.sello b { font-family: 'Fraunces', Georgia, serif; font-size: 1.75rem; line-height: 1; color: var(--grade); }
.sello span { font-size: .8rem; color: var(--grade); margin-top: .25rem; font-weight: 600; }
.verdict-label { color: var(--muted); font-size: .9rem; }
.verdict-conf { font-family: 'Fraunces', Georgia, serif; font-size: 1.55rem; font-weight: 600; margin: .1rem 0 .3rem 0; }
.verdict-hint { color: var(--muted); line-height: 1.45; }

/* barras de probabilidad */
.prow { display: flex; align-items: center; gap: .7rem; margin: .5rem 0; }
.plabel { width: 4.6rem; font-size: .95rem; }
.ptrack { flex: 1; height: .65rem; background: #E3EBE5; border-radius: 999px; overflow: hidden; }
.pfill { height: 100%; border-radius: 999px; }
.pval { width: 3rem; text-align: right; font-variant-numeric: tabular-nums; font-size: .95rem; }

.meta { margin-top: 1rem; }
.meta small { color: var(--muted); }
.acuerdo { margin: .9rem 0 0 0; padding: .7rem 1rem; border-radius: 10px; font-size: .95rem; }
.acuerdo-ok { background: #E4F2E8; color: #1F5E3B; }
.acuerdo-no { background: #FBEBD3; color: #7A5200; }

/* tarjetas de comparación */
.ccard { background: var(--card); border: 1px solid var(--line); border-top: 5px solid var(--grade, #B7C5BC);
         border-radius: 12px; padding: 1rem 1.15rem 1.1rem 1.15rem; height: 100%; }
.ccard h4 { margin: 0; font-size: 1rem; font-weight: 600; }
.ccard .tag { color: var(--muted); font-size: .8rem; margin-bottom: .6rem; }
.ccard .grade { font-family: 'Fraunces', Georgia, serif; font-size: 1.9rem; font-weight: 700;
                color: var(--grade); line-height: 1.1; }
.ccard .conf { color: var(--muted); font-size: .9rem; margin-bottom: .4rem; }
.ccard .muted { color: var(--muted); font-size: .92rem; }

@media (max-width: 640px) {
  .verdict { flex-direction: column; align-items: flex-start; }
  .app-title { font-size: 2rem; }
}
</style>
""",
    unsafe_allow_html=True,
)


def ui(cls: str):
    """Presentación de una clase; acepta 'Bad' o 'Bad Quality_Fruits'."""
    return CLASS_UI[cls.split()[0].split("_")[0]]


@st.cache_resource(show_spinner="Cargando modelo…")
def load_predictor():
    return get_predictor()


@st.cache_resource(show_spinner="Cargando modelos clásicos…")
def load_ml():
    from src.inference.ml_predictor import cargar_modelos_ml
    return cargar_modelos_ml()


def prob_rows(probs: dict) -> str:
    filas = []
    for cls in CLASSES:
        nombre, color, _ = ui(cls)
        pct = probs.get(cls, 0.0) * 100
        filas.append(
            f'<div class="prow"><span class="plabel">{nombre}</span>'
            f'<div class="ptrack"><div class="pfill" style="width:{pct:.1f}%;background:{color}"></div></div>'
            f'<span class="pval">{pct:.0f}%</span></div>'
        )
    return "".join(filas)


def verdict_html(res: dict) -> str:
    nombre, color, pista = ui(res["calidad"])
    return (
        f'<div class="verdict" style="--grade:{color}">'
        f'<div class="sello"><b>{nombre}</b><span>calidad</span></div>'
        f'<div><div class="verdict-label">Calidad estimada</div>'
        f'<div class="verdict-conf">Confianza {res["probabilidad"] * 100:.0f}%</div>'
        f'<div class="verdict-hint">{pista}</div></div></div>'
    )


def size_html(res: dict) -> str:
    if res["tamaño"]:
        return (f'<div class="meta">Tamaño relativo: <b>{res["tamaño"]}</b><br>'
                '<small>Es relativo a la foto: depende de la distancia de la cámara.</small></div>')
    return ('<div class="meta">Tamaño: <b>aún no disponible</b><br>'
            '<small>Se activará cuando se integre el módulo de tamaño.</small></div>')


def model_card(titulo: str, etiqueta: str, res: dict | None, error: str | None = None,
               vacio: str = "Carga una imagen en la pestaña Predicción.") -> str:
    if error:
        cuerpo = f'<p class="muted">No se pudo usar este modelo: {error}</p>'
        estilo = ""
    elif res is None:
        cuerpo = f'<p class="muted">{vacio}</p>'
        estilo = ""
    else:
        nombre, color, _ = ui(res["calidad"])
        aprox = " (aproximada)" if res.get("prob_aproximada") else ""
        cuerpo = (f'<div class="grade">{nombre}</div>'
                  f'<div class="conf">Confianza {res["probabilidad"] * 100:.0f}%{aprox}</div>'
                  + prob_rows(res["probabilidades"]))
        estilo = f' style="--grade:{color}"'
    return f'<div class="ccard"{estilo}><h4>{titulo}</h4><div class="tag">{etiqueta}</div>{cuerpo}</div>'


# ---------------------------------------------------------------- cabecera
predictor = load_predictor()
ml_models, ml_errors = load_ml()

st.markdown('<p class="app-title">Calidad de frutas</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="app-sub">Sube o captura la foto de una sola fruta sobre un fondo liso '
    "y obtén su clase de calidad y su tamaño.</p>",
    unsafe_allow_html=True,
)
if predictor.es_mock:
    st.markdown(
        '<span class="pill pill-demo">Modo demostración: los resultados son simulados</span>',
        unsafe_allow_html=True,
    )
    if predictor.motivo:
        st.caption(predictor.motivo)
else:
    st.markdown(f'<span class="pill pill-real">Modelo activo: {predictor.nombre}</span>',
                unsafe_allow_html=True)
st.write("")

tab_pred, tab_comp, tab_priv = st.tabs(["Predicción", "Comparación de modelos", "Privacidad y límites"])

image, result = None, None
ml_results, ml_fallos = {}, {}

# ------------------------------------------------------------- predicción
with tab_pred:
    modo = st.radio("Origen de la foto", ["Subir imagen", "Usar la cámara"], horizontal=True,
                    label_visibility="collapsed")
    if modo == "Subir imagen":
        origen = st.file_uploader("Foto de la fruta", type=["jpg", "jpeg", "png", "webp"],
                                  label_visibility="collapsed")
    else:
        origen = st.camera_input("Foto de la fruta", label_visibility="collapsed")

    if origen is not None:
        try:
            image = load_image(origen)
        except ImageError as exc:
            st.error(str(exc))

    if image is not None:
        with st.spinner("Analizando…"):
            result = predictor.predict(image)
            for clave, modelo in ml_models.items():
                try:
                    ml_results[clave] = modelo.predict(image)
                except Exception as exc:  # noqa: BLE001 - se muestra en la tarjeta
                    ml_fallos[clave] = f"{type(exc).__name__}: {exc}"

        izq, der = st.columns([1, 1], gap="large")
        with izq:
            st.image(image)
        with der:
            st.markdown(verdict_html(result), unsafe_allow_html=True)
            st.markdown(prob_rows(result["probabilidades"]), unsafe_allow_html=True)
            st.markdown(size_html(result), unsafe_allow_html=True)
            if result["probabilidad"] < LOW_CONFIDENCE:
                st.warning("Confianza baja. Prueba con más luz, un fondo liso y una sola fruta centrada.")
            if result["es_mock"]:
                st.caption("Resultado simulado: aún no hay un modelo entrenado.")

            # acuerdo entre la CNN y los modelos clásicos (sin contar la línea base)
            votos = [result["calidad"]] + [r["calidad"] for k, r in ml_results.items() if k != "baseline"]
            if len(votos) > 1:
                if len(set(votos)) == 1:
                    st.markdown(f'<div class="acuerdo acuerdo-ok">Los {len(votos)} modelos coinciden: '
                                f'{ui(votos[0])[0].lower()}.</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="acuerdo acuerdo-no">Los modelos no coinciden. '
                                'Mira la pestaña Comparación y revisa la fruta a mano.</div>',
                                unsafe_allow_html=True)
    elif origen is None:
        st.info("Sube una imagen o toma una foto para ver el resultado.")

# ------------------------------------------------------------ comparación
with tab_comp:
    st.write("Cada modelo se evalúa sobre la misma foto.")
    tarjetas = [("Red convolucional", "Aprendizaje profundo", result, None)]
    for clave, modelo in ml_models.items():
        etiqueta = "Referencia: predice sin mirar la foto" if clave == "baseline" else "Aprendizaje clásico"
        tarjetas.append((modelo.nombre, etiqueta, ml_results.get(clave), ml_fallos.get(clave)))
    for clave, err in ml_errors.items():
        tarjetas.append((clave, "Aprendizaje clásico", None, err))
    if not ml_models and not ml_errors:
        tarjetas.append(("Modelos clásicos", "Aprendizaje clásico", None,
                         "no hay archivos .joblib en models_saved/."))

    for inicio in range(0, len(tarjetas), 4):
        grupo = tarjetas[inicio:inicio + 4]
        cols = st.columns(len(grupo), gap="medium")
        for col, (titulo, etiqueta, res, err) in zip(cols, grupo):
            with col:
                st.markdown(model_card(titulo, etiqueta, res, err), unsafe_allow_html=True)

# ---------------------------------------------------------------- privacidad
with tab_priv:
    st.markdown(
        """
**Privacidad.** La imagen se procesa en memoria y esta aplicación no la guarda en disco.
Si la aplicación se ejecuta en un servidor, la imagen llega a ese servidor para poder analizarla.

**Datos de entrenamiento.** El modelo se entrena con un conjunto público de Kaggle y con fotos propias
del equipo. Puede rendir peor con frutas, variedades, cámaras o iluminación poco representadas.

**Condiciones de uso.** Funciona mejor con una sola fruta, centrada y sobre un fondo liso.
Con varias frutas, fondos recargados o poca luz, el resultado pierde fiabilidad.

**Uso responsable.** La clasificación apoya una revisión visual; no sustituye el control de
inocuidad ni decisiones sanitarias. Ante una confianza baja, revisa la fruta a mano.
"""
    )