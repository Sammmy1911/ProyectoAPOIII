"""Interfaz de clasificacion de calidad de frutas.

Ejecutar desde la raiz del repo:   streamlit run app/streamlit_app.py
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
    "Regular": ("Regular", "#9A6700", "Madurez avanzada o defectos leves. Conviene vender pronto."),
    "Good": ("Buena", "#2F7D4F", "Sin defectos visibles. Apta para la venta."),
}

st.set_page_config(page_title="Calidad de frutas", page_icon="🍎", layout="wide")

# Estilos: toda la apariencia propia vive en este bloque y en .streamlit/config.toml
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600&display=swap');

.block-container { max-width: 1080px; padding-top: 2.2rem; }
.app-title { font-family: 'Fraunces', Georgia, serif; font-size: 2.1rem; font-weight: 600;
             color: #16231C; margin: 0; line-height: 1.15; }
.app-sub { color: #4A5B51; margin: .35rem 0 1.2rem 0; max-width: 60ch; }
.pill { display: inline-block; padding: .2rem .7rem; border-radius: 999px; font-size: .82rem; }
.pill-demo { background: #FFF1D6; color: #7A5200; border: 1px solid #E8C679; }
.pill-real { background: #DDEFE3; color: #1F5E3B; border: 1px solid #A9D2B6; }

.verdict { border-left: 8px solid var(--grade); background: #FFFFFF; border-radius: 6px;
           padding: 1.1rem 1.4rem; box-shadow: 0 1px 2px rgba(22,35,28,.08); }
.verdict-label { color: #4A5B51; font-size: .9rem; }
.verdict-grade { font-family: 'Fraunces', Georgia, serif; font-size: 3.4rem; font-weight: 600;
                 color: var(--grade); line-height: 1.05; margin: .1rem 0 .3rem 0; }
.verdict-conf { font-weight: 600; color: #16231C; }
.verdict-hint { color: #4A5B51; margin-top: .3rem; }

.prow { display: flex; align-items: center; gap: .7rem; margin: .45rem 0; }
.plabel { width: 4.6rem; color: #16231C; }
.ptrack { flex: 1; height: .7rem; background: #E1E9E3; border-radius: 999px; overflow: hidden; }
.pfill { height: 100%; border-radius: 999px; }
.pval { width: 3rem; text-align: right; font-variant-numeric: tabular-nums; color: #16231C; }

.meta { margin-top: 1rem; color: #16231C; }
.meta small { color: #4A5B51; }
.ccard { background: #FFFFFF; border-radius: 6px; padding: 1rem 1.2rem; height: 100%;
         box-shadow: 0 1px 2px rgba(22,35,28,.08); }
.ccard h4 { margin: 0 0 .5rem 0; font-size: 1.05rem; }
.ccard .muted { color: #4A5B51; }
</style>
""",
    unsafe_allow_html=True,
)


def ui(cls: str):
    """Presentación de una clase; acepta 'Bad' o 'Bad Quality_Fruits'."""
    return CLASS_UI[cls.split()[0].split("_")[0]]

def ui(cls: str):
    """Presentación de una clase; acepta 'Bad' o 'Bad Quality_Fruits'."""
    return CLASS_UI[cls.split()[0].split("_")[0]]

@st.cache_resource(show_spinner="Cargando modelo…")
def load_predictor():
    return get_predictor()


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
        f'<div class="verdict-label">Calidad estimada</div>'
        f'<div class="verdict-grade">{nombre}</div>'
        f'<div class="verdict-conf">Confianza {res["probabilidad"] * 100:.0f}%</div>'
        f'<div class="verdict-hint">{pista}</div></div>'
    )


def size_html(res: dict) -> str:
    if res["tamaño"]:
        return (f'<div class="meta">Tamaño relativo: <b>{res["tamaño"]}</b><br>'
                '<small>Es relativo a la foto: depende de la distancia de la cámara.</small></div>')
    return ('<div class="meta">Tamaño: <b>aún no disponible</b><br>'
            '<small>Se activará cuando se integre el módulo de tamaño.</small></div>')


# ---------------------------------------------------------------- cabecera
predictor = load_predictor()

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
    elif origen is None:
        st.info("Sube una imagen o toma una foto para ver el resultado.")

# ------------------------------------------------------------ comparación
with tab_comp:
    st.write("Cada modelo se evalúa sobre la misma foto.")
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        if result is None:
            cuerpo = '<p class="muted">Carga una imagen en la pestaña Predicción.</p>'
        else:
            cuerpo = verdict_html(result) + prob_rows(result["probabilidades"])
        st.markdown(f'<div class="ccard"><h4>Red convolucional</h4>{cuerpo}</div>',
                    unsafe_allow_html=True)
    for col, titulo in ((c2, "Modelo clásico 1"), (c3, "Modelo clásico 2")):
        with col:
            st.markdown(
                f'<div class="ccard"><h4>{titulo}</h4>'
                '<p class="muted">No disponible: pendiente de entrenamiento e integración.</p></div>',
                unsafe_allow_html=True,
            )

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