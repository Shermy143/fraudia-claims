"""
Dashboard principal de Fraudia — Detector de Posibles Fraudes en Siniestros.

Ejecutar con:
    streamlit run src/app/main.py

Prerequisitos:
    1. data/processed/siniestros_merged.csv  (generado por load_data.py)
    2. models/fraud_model.pkl                (generado por 02_modelo_fraude.ipynb)
    3. GROQ_API_KEY en .env
"""

import os
import sys

# Permite importar desde src/ sin instalar el paquete
RUTA_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUTA_BASE = os.path.dirname(RUTA_SRC)
sys.path.insert(0, RUTA_SRC)

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# pyrefly: ignore [missing-import]
from src.explainability.explain_score import explicar_dataframe
from src.ai_agent.claims_agent import ClaimsAgent, consulta_rapida

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------

RUTA_DATASET = os.path.join(RUTA_BASE, "data", "processed", "siniestros_merged.csv")
RUTA_MODELO  = os.path.join(RUTA_BASE, "models", "fraud_model.pkl")

# ---------------------------------------------------------------------------
# Configuración de la página
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Fraudia — Detector de Posibles Fraudes",
    page_icon="🔍",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Estilos del semáforo
# ---------------------------------------------------------------------------

COLORES = {"rojo": "#e74c3c", "amarillo": "#f39c12", "verde": "#27ae60"}
EMOJIS  = {"rojo": "🔴", "amarillo": "🟡", "verde": "🟢"}


def badge_semaforo(nivel: str) -> str:
    color = COLORES.get(nivel, "#888")
    emoji = EMOJIS.get(nivel, "⚪")
    return f'<span style="background:{color};color:white;padding:2px 10px;border-radius:12px;font-weight:bold">{emoji} {nivel.upper()}</span>'


# ---------------------------------------------------------------------------
# Carga y análisis — cacheados para no recalcular en cada interacción
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Cargando dataset...")
def cargar_dataset() -> pd.DataFrame:
    return pd.read_csv(RUTA_DATASET)


@st.cache_data(show_spinner="Analizando siniestros...")
def analizar(_df: pd.DataFrame) -> pd.DataFrame:
    """
    Underscore en _df para que Streamlit no lo incluya en el hash del cache
    (los DataFrames grandes son lentos de hashear).
    """
    return explicar_dataframe(_df)


@st.cache_resource
def obtener_agente(_df_analizado: pd.DataFrame) -> ClaimsAgent:
    return ClaimsAgent(_df_analizado)


# ---------------------------------------------------------------------------
# Verificaciones previas
# ---------------------------------------------------------------------------

def verificar_prerequisitos() -> bool:
    errores = []
    if not os.path.exists(RUTA_DATASET):
        errores.append(f"❌ Dataset no encontrado: `{RUTA_DATASET}`  \nEjecuta: `python src/ingestion/load_data.py`")
    if not os.path.exists(RUTA_MODELO):
        errores.append(f"❌ Modelo no encontrado: `{RUTA_MODELO}`  \nEjecuta: `02_modelo_fraude.ipynb` en Google Colab")
    if not os.getenv("GROQ_API_KEY"):
        errores.append("❌ `GROQ_API_KEY` no configurada en `.env`")

    if errores:
        st.error("### Prerequisitos faltantes")
        for e in errores:
            st.markdown(e)
        return False
    return True


# ---------------------------------------------------------------------------
# Sección: KPIs
# ---------------------------------------------------------------------------

def render_kpis(df: pd.DataFrame):
    total     = len(df)
    rojos     = (df["semaforo"] == "rojo").sum()
    amarillos = (df["semaforo"] == "amarillo").sum()
    verdes    = (df["semaforo"] == "verde").sum()
    score_avg = df["score_final"].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total siniestros", total)
    c2.metric("🔴 Alto riesgo",    rojos,     delta=f"{rojos/total:.1%}", delta_color="inverse")
    c3.metric("🟡 Riesgo medio",   amarillos, delta=f"{amarillos/total:.1%}", delta_color="off")
    c4.metric("🟢 Bajo riesgo",    verdes,    delta=f"{verdes/total:.1%}", delta_color="normal")
    c5.metric("Score promedio",   f"{score_avg:.1f}")


# ---------------------------------------------------------------------------
# Sección: Tabla de casos
# ---------------------------------------------------------------------------

def render_tabla(df: pd.DataFrame):
    st.subheader("Bandeja de casos — ordenados por riesgo")

    # Filtros en columnas
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        filtro_semaforo = st.multiselect(
            "Filtrar por semáforo",
            options=["rojo", "amarillo", "verde"],
            default=["rojo", "amarillo"],
        )
    with col_f2:
        ramos = ["Todos"] + sorted(df["ramo"].dropna().unique().tolist()) if "ramo" in df.columns else ["Todos"]
        filtro_ramo = st.selectbox("Filtrar por ramo", ramos)
    with col_f3:
        score_min = st.slider("Score mínimo", 0, 100, 0)

    # Aplicar filtros
    mask = df["semaforo"].isin(filtro_semaforo) & (df["score_final"] >= score_min)
    if filtro_ramo != "Todos" and "ramo" in df.columns:
        mask &= df["ramo"] == filtro_ramo
    df_filtrado = df[mask]

    st.caption(f"{len(df_filtrado)} casos encontrados")

    # Columnas a mostrar
    cols_tabla = [c for c in [
        "id_siniestro", "semaforo", "score_final", "ramo",
        "monto_reclamado", "sucursal", "reglas_activadas",
    ] if c in df_filtrado.columns]

    st.dataframe(
        df_filtrado[cols_tabla].rename(columns={
            "id_siniestro":     "ID Siniestro",
            "semaforo":         "Semáforo",
            "score_final":      "Score",
            "ramo":             "Ramo",
            "monto_reclamado":  "Monto ($)",
            "sucursal":         "Sucursal",
            "reglas_activadas": "Reglas activadas",
        }),
        use_container_width=True,
        hide_index=True,
    )

    return df_filtrado


# ---------------------------------------------------------------------------
# Sección: Detalle de un caso
# ---------------------------------------------------------------------------

def render_detalle(df: pd.DataFrame):
    st.subheader("Detalle del caso")

    ids = df.sort_values("score_final", ascending=False)["id_siniestro"].tolist()
    id_seleccionado = st.selectbox("Selecciona un siniestro", ids)

    fila = df[df["id_siniestro"] == id_seleccionado].iloc[0]

    col_izq, col_der = st.columns([1, 2])

    with col_izq:
        nivel = fila["semaforo"]
        st.markdown(f"**Nivel de riesgo:** {badge_semaforo(nivel)}", unsafe_allow_html=True)
        st.metric("Score final", f"{fila['score_final']:.1f} / 100")

        st.markdown("**Datos del siniestro:**")
        campos = {
            "Ramo":             "ramo",
            "Cobertura":        "cobertura",
            "Monto reclamado":  "monto_reclamado",
            "Monto estimado":   "monto_estimado",
            "Sucursal":         "sucursal",
            "Días desde inicio póliza": "dias_desde_inicio_poliza",
            "Días hasta reporte":       "dias_entre_ocurrencia_reporte",
        }
        for label, col in campos.items():
            if col in fila.index and pd.notna(fila[col]):
                st.markdown(f"- **{label}:** {fila[col]}")

    with col_der:
        # Alertas del motor de reglas
        alertas = fila.get("alertas", [])
        if alertas:
            st.markdown("**Señales del motor de reglas:**")
            for alerta in alertas:
                st.warning(f"⚠️ {alerta}")

        # Texto de alerta completo
        texto = fila.get("texto_alerta", "")
        if texto:
            st.markdown("**Texto de alerta para el analista:**")
            st.code(texto, language=None)

        # Factores SHAP
        factores = fila.get("factores_shap", [])
        if factores:
            st.markdown("**Factores principales del modelo IA (SHAP):**")
            for f in factores:
                st.markdown(
                    f"- `{f['descripcion']}` → impacto **+{f['shap']:.3f}**"
                )

    # Explicación del agente
    st.markdown("---")
    if st.button("🤖 Generar explicación con IA", key=f"btn_{id_seleccionado}"):
        with st.spinner("Consultando al agente..."):
            try:
                pregunta = (
                    f"Explica de forma clara por qué el siniestro {id_seleccionado} "
                    f"tiene score {fila['score_final']:.1f} y semáforo {fila['semaforo']}. "
                    f"Alertas activas: {alertas}. "
                    f"¿Qué debería revisar el analista primero?"
                )
                respuesta = consulta_rapida(pregunta, df)
                st.info(respuesta)
            except Exception as e:
                st.error(f"Error al consultar el agente: {e}")


# ---------------------------------------------------------------------------
# Sección: Agente conversacional
# ---------------------------------------------------------------------------

def render_agente(df: pd.DataFrame):
    st.subheader("🤖 Agente conversacional")
    st.caption("Consulta cualquier cosa sobre los siniestros analizados en lenguaje natural.")

    # Inicializar historial en session_state
    if "mensajes_chat" not in st.session_state:
        st.session_state.mensajes_chat = []
    if "agente" not in st.session_state:
        try:
            st.session_state.agente = ClaimsAgent(df)
        except EnvironmentError as e:
            st.error(str(e))
            return

    # Botones de preguntas rápidas
    st.markdown("**Preguntas sugeridas:**")
    preguntas_rapidas = [
        "¿Cuáles son los 10 siniestros con mayor riesgo?",
        "¿Qué proveedores concentran más alertas?",
        "Genera un resumen ejecutivo de los casos críticos.",
        "¿Qué ramos tienen mayor porcentaje de casos sospechosos?",
        "Recomienda qué casos revisar primero.",
    ]

    cols = st.columns(len(preguntas_rapidas))
    for i, (col, pregunta) in enumerate(zip(cols, preguntas_rapidas)):
        if col.button(f"💬 {pregunta[:30]}...", key=f"quick_{i}"):
            st.session_state.mensajes_chat.append({"role": "user", "content": pregunta})
            with st.spinner("Consultando..."):
                respuesta = st.session_state.agente.consultar(pregunta)
            st.session_state.mensajes_chat.append({"role": "assistant", "content": respuesta})

    st.markdown("---")

    # Historial de conversación
    for mensaje in st.session_state.mensajes_chat:
        with st.chat_message(mensaje["role"]):
            st.markdown(mensaje["content"])

    # Input del usuario
    if prompt := st.chat_input("Escribe tu pregunta aquí..."):
        st.session_state.mensajes_chat.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Consultando al agente..."):
                try:
                    respuesta = st.session_state.agente.consultar(prompt)
                    st.markdown(respuesta)
                    st.session_state.mensajes_chat.append({"role": "assistant", "content": respuesta})
                except Exception as e:
                    st.error(f"Error: {e}")

    # Botón para limpiar historial
    if st.session_state.mensajes_chat:
        if st.button("🗑️ Limpiar conversación"):
            st.session_state.mensajes_chat = []
            st.session_state.agente.reiniciar_historial()
            st.rerun()


# ---------------------------------------------------------------------------
# App principal
# ---------------------------------------------------------------------------

def main():
    st.title("🔍 Fraudia — Detector de Posibles Fraudes en Siniestros")
    st.caption("Sistema de alertas para revisión humana. No reemplaza el análisis del analista.")

    if not verificar_prerequisitos():
        st.stop()

    # Cargar y analizar
    df_raw      = cargar_dataset()
    df_analizado = analizar(df_raw)

    # KPIs siempre visibles
    render_kpis(df_analizado)
    st.markdown("---")

    # Tabs principales
    tab1, tab2, tab3 = st.tabs(["📋 Bandeja de casos", "🔎 Detalle del caso", "🤖 Agente IA"])

    with tab1:
        render_tabla(df_analizado)

    with tab2:
        render_detalle(df_analizado)

    with tab3:
        render_agente(df_analizado)


if __name__ == "__main__":
    main()
