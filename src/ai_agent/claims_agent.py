"""
Agente conversacional para consultas sobre detección de posibles fraudes.

Usa la API de Groq para responder preguntas en lenguaje natural sobre
los siniestros analizados. Responde las 12 preguntas de la sección 12
del documento del reto.

Requiere: GROQ_API_KEY en el archivo .env
"""

import os
import json
import textwrap
from groq import Groq
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

MODELO_GROQ        = "llama-3.3-70b-versatile"
MAX_TOKENS         = 1024
MAX_CASOS_CONTEXTO = 20   # Top casos enviados al agente como contexto
MAX_PROVEEDORES    = 10   # Top proveedores enviados como contexto


# ---------------------------------------------------------------------------
# Construcción del contexto
# ---------------------------------------------------------------------------

def _resumir_dataset(df: pd.DataFrame) -> str:
    """
    Genera un resumen compacto del dataset analizado para el system prompt.
    Incluye estadísticas generales, top casos, proveedores y ramos.
    """
    total     = len(df)
    rojos     = (df["semaforo"] == "rojo").sum()
    amarillos = (df["semaforo"] == "amarillo").sum()
    verdes    = (df["semaforo"] == "verde").sum()

    # Top casos por score
    cols_caso = ["id_siniestro", "score_final", "semaforo", "ramo",
                 "monto_reclamado", "sucursal", "alertas", "reglas_activadas"]
    cols_disponibles = [c for c in cols_caso if c in df.columns]
    top_casos = (
        df.sort_values("score_final", ascending=False)
        .head(MAX_CASOS_CONTEXTO)[cols_disponibles]
        .to_dict(orient="records")
    )

    # Ranking de proveedores con alertas
    ranking_proveedores = []
    if "beneficiario" in df.columns:
        ranking_proveedores = (
            df[df["semaforo"].isin(["rojo", "amarillo"])]
            .groupby("beneficiario")
            .agg(
                total_alertas  = ("id_siniestro", "count"),
                score_promedio = ("score_final", "mean"),
                casos_rojos    = ("semaforo", lambda x: (x == "rojo").sum()),
            )
            .sort_values("total_alertas", ascending=False)
            .head(MAX_PROVEEDORES)
            .reset_index()
            .to_dict(orient="records")
        )

    # Distribución por ramo
    dist_ramos = {}
    if "ramo" in df.columns:
        dist_ramos = (
            df.groupby("ramo")
            .agg(
                total          = ("id_siniestro", "count"),
                rojos          = ("semaforo", lambda x: (x == "rojo").sum()),
                score_promedio = ("score_final", "mean"),
            )
            .sort_values("rojos", ascending=False)
            .to_dict(orient="index")
        )

    # Distribución geográfica
    dist_geo = {}
    col_geo = "sucursal" if "sucursal" in df.columns else "ciudad" if "ciudad" in df.columns else None
    if col_geo:
        dist_geo = (
            df.groupby(col_geo)
            .agg(
                total = ("id_siniestro", "count"),
                rojos = ("semaforo", lambda x: (x == "rojo").sum()),
            )
            .sort_values("rojos", ascending=False)
            .head(10)
            .to_dict(orient="index")
        )

    resumen = {
        "estadisticas_generales": {
            "total_siniestros":  total,
            "rojos":             int(rojos),
            "amarillos":         int(amarillos),
            "verdes":            int(verdes),
            "pct_alto_riesgo":   round(rojos / total * 100, 1),
            "score_promedio":    round(df["score_final"].mean(), 2),
        },
        "top_casos_sospechosos":   top_casos,
        "ranking_proveedores":     ranking_proveedores,
        "distribucion_ramos":      dist_ramos,
        "distribucion_geografica": dist_geo,
    }

    return json.dumps(resumen, ensure_ascii=False, indent=2, default=str)


def _construir_system_prompt(df: pd.DataFrame) -> str:
    """Construye el system prompt con el contexto del dataset analizado."""
    resumen = _resumir_dataset(df)

    return textwrap.dedent(f"""
        Eres un agente especializado en análisis de posibles fraudes en siniestros de seguros.
        Tienes acceso al resultado del análisis de {len(df)} siniestros procesados por el
        sistema Fraudia, que combina un motor de reglas de negocio y un modelo XGBoost
        con explicabilidad SHAP.

        PRINCIPIO CLAVE: el sistema genera alertas de REVISIÓN, no acusaciones de fraude.
        Siempre comunica los resultados con lenguaje de "posible fraude" o "requiere revisión".
        Nunca afirmes que un siniestro ES fraude.

        DATOS DEL ANÁLISIS:
        {resumen}

        SEMÁFORO DE RIESGO:
        - Rojo     (score 76-100): Escalar a Unidad Antifraude para revisión especializada de campo.
        - Amarillo (score 41-75):  Escalar a Unidad Antifraude para revisión documental.
        - Verde    (score 0-40):   Continuar flujo normal.

        SCORE HÍBRIDO: Motor de reglas (60%) + Modelo XGBoost (40%).

        PREGUNTAS QUE PUEDES RESPONDER:
        1. ¿Cuáles son los 10 siniestros con mayor riesgo de posible fraude?
        2. ¿Por qué este siniestro fue marcado como alto riesgo?
        3. ¿Qué proveedores concentran más alertas?
        4. ¿Qué ramos tienen mayor porcentaje de casos sospechosos?
        5. ¿Qué ciudades presentan mayor concentración de alertas?
        6. ¿Qué asegurados tienen mayor frecuencia de reclamos?
        7. ¿Qué documentos faltan en los casos críticos?
        8. ¿Qué casos tienen montos atípicos?
        9. ¿Qué siniestros ocurrieron cerca del inicio de la póliza?
        10. ¿Qué patrones se repiten en los reclamos sospechosos?
        11. Genera un resumen ejecutivo de los casos críticos.
        12. Recomienda qué casos debería revisar primero el analista.

        Responde siempre en español, de forma clara y estructurada.
        Cuando cites siniestros específicos, incluye su ID y score.
        Si no tienes suficiente información para responder, dilo claramente.
    """).strip()


# ---------------------------------------------------------------------------
# Agente conversacional
# ---------------------------------------------------------------------------

class ClaimsAgent:
    """
    Agente conversacional multi-turno para consultas sobre siniestros.

    Uso básico:
        agente = ClaimsAgent(df_analizado)
        respuesta = agente.consultar("¿Cuáles son los 10 casos más sospechosos?")
        respuesta2 = agente.consultar("¿Por qué el SIN-000005 es rojo?")
    """

    def __init__(self, df_analizado: pd.DataFrame):
        """
        df_analizado: DataFrame con columnas de score, semáforo y alertas,
                      resultado de explicar_dataframe() o analizar_dataframe().
        """
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY no encontrada. "
                "Agrega la variable al archivo .env"
            )

        self.cliente       = Groq(api_key=api_key)
        self.df            = df_analizado
        self.system_prompt = _construir_system_prompt(df_analizado)
        self.historial     = []   # {role, content} para multi-turno

    def consultar(self, pregunta: str) -> str:
        """
        Envía una pregunta al agente y retorna la respuesta.
        Mantiene el historial para contexto multi-turno.
        """
        self.historial.append({"role": "user", "content": pregunta})

        # Groq recibe system como primer mensaje del array
        mensajes = [{"role": "system", "content": self.system_prompt}] + self.historial

        respuesta = self.cliente.chat.completions.create(
            model      = MODELO_GROQ,
            max_tokens = MAX_TOKENS,
            messages   = mensajes,
        )

        texto = respuesta.choices[0].message.content
        self.historial.append({"role": "assistant", "content": texto})
        return texto

    def reiniciar_historial(self):
        """Limpia el historial manteniendo el contexto del dataset."""
        self.historial = []

    def explicar_siniestro(self, id_siniestro: str) -> str:
        """
        Explicación detallada de un siniestro específico.
        Busca el ID en el dataset y construye la pregunta automáticamente.
        """
        fila = self.df[self.df["id_siniestro"] == id_siniestro]

        if fila.empty:
            return f"No se encontró el siniestro {id_siniestro} en el dataset analizado."

        datos = fila.iloc[0]
        contexto = (
            f"Analiza el siniestro {id_siniestro} con los siguientes datos:\n"
            f"- Score final: {datos.get('score_final', 'N/A')}\n"
            f"- Semáforo: {datos.get('semaforo', 'N/A')}\n"
            f"- Ramo: {datos.get('ramo', 'N/A')}\n"
            f"- Monto reclamado: ${datos.get('monto_reclamado', 'N/A')}\n"
            f"- Alertas: {datos.get('alertas', [])}\n"
            f"- Reglas activadas: {datos.get('reglas_activadas', [])}\n"
            f"- Texto de alerta: {datos.get('texto_alerta', 'N/A')}\n\n"
            f"Explica por qué fue marcado como {datos.get('semaforo', '')} "
            f"y qué debería revisar el analista."
        )

        return self.consultar(contexto)

    def resumen_ejecutivo(self) -> str:
        """Genera el resumen ejecutivo de casos críticos (pregunta 11 sección 12)."""
        return self.consultar(
            "Genera un resumen ejecutivo de los casos críticos (semáforo rojo). "
            "Incluye: total de casos, patrones comunes, proveedores recurrentes, "
            "ramos más afectados y recomendación de prioridad de revisión."
        )

    def prioridad_revision(self) -> str:
        """Recomienda el orden de revisión para el analista (pregunta 12 sección 12)."""
        return self.consultar(
            "Recomienda al analista qué casos debería revisar primero y en qué orden. "
            "Considera el score, las reglas críticas activadas y los montos involucrados."
        )


# ---------------------------------------------------------------------------
# Consulta rápida sin historial (para el dashboard)
# ---------------------------------------------------------------------------

def consulta_rapida(pregunta: str, df_analizado: pd.DataFrame) -> str:
    """
    Consulta sin estado para usar desde Streamlit o la API.
    Cada llamada es independiente — no mantiene historial.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY no encontrada en .env")

    cliente = Groq(api_key=api_key)
    system  = _construir_system_prompt(df_analizado)

    respuesta = cliente.chat.completions.create(
        model      = MODELO_GROQ,
        max_tokens = MAX_TOKENS,
        messages   = [
            {"role": "system", "content": system},
            {"role": "user",   "content": pregunta},
        ],
    )

    return respuesta.choices[0].message.content
