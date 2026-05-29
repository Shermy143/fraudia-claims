"""
Explicabilidad del score de posible fraude.

Convierte los valores SHAP del modelo XGBoost en texto legible
para el analista y el agente conversacional.

Flujo:
  fraud_model.py  →  explain_score.py  →  claims_agent.py
  (score numérico)   (texto de alerta)    (respuesta en lenguaje natural)
"""

import joblib
import shap
import numpy as np
import pandas as pd

from models.fraud_model import cargar_modelo, RUTA_MODELO
from features.build_features import construir_features, obtener_matriz_modelo, FEATURES_MODELO


# ---------------------------------------------------------------------------
# Descripción legible de cada feature para el analista
# ---------------------------------------------------------------------------

DESCRIPCION_FEATURES = {
    "ratio_monto_estimado":         "monto reclamado vs estimado",
    "ratio_monto_suma_asegurada":   "monto reclamado vs suma asegurada",
    "monto_reclamado_normalizado":  "monto reclamado elevado",
    "dias_desde_inicio_poliza":     "días desde inicio de póliza",
    "dias_entre_ocurrencia_reporte":"demora entre ocurrencia y reporte",
    "es_borde_vigencia":            "siniestro en borde de vigencia (≤30 días)",
    "es_borde_vigencia_extremo":    "siniestro en primeras 48 horas de vigencia",
    "reporte_tardio":               "reporte tardío (>7 días)",
    "sin_documentos":               "documentación incompleta",
    "es_robo":                      "cobertura de robo",
    "historial_siniestros_asegurado":"historial de siniestros del asegurado",
    "frecuencia_asegurado_alta":    "alta frecuencia de reclamos del asegurado",
    "proveedor_lista_restrictiva":  "proveedor en lista restrictiva",
    "casos_observados_proveedor":   "proveedor con casos observados previos",
    "similitud_narrativa_max":      "similitud textual con otros reclamos",
    "score_reglas":                 "score acumulado del motor de reglas",
}


# ---------------------------------------------------------------------------
# Caché del explainer SHAP (se crea una sola vez)
# ---------------------------------------------------------------------------

_explainer_cache = None


def _obtener_explainer() -> shap.TreeExplainer:
    global _explainer_cache
    if _explainer_cache is None:
        modelo = cargar_modelo()
        _explainer_cache = shap.TreeExplainer(modelo)
    return _explainer_cache


# ---------------------------------------------------------------------------
# Cálculo de SHAP para una fila o DataFrame
# ---------------------------------------------------------------------------

def calcular_shap(df_features: pd.DataFrame) -> np.ndarray:
    """
    Calcula los valores SHAP para un DataFrame ya procesado
    por construir_features() y obtener_matriz_modelo().

    Retorna array de shape (n_filas, n_features).
    """
    explainer = _obtener_explainer()
    X = obtener_matriz_modelo(df_features)
    return explainer.shap_values(X)


# ---------------------------------------------------------------------------
# Texto de explicación
# ---------------------------------------------------------------------------

def _factores_principales(shap_row: np.ndarray, X_row: pd.Series, top_n: int = 5) -> list[dict]:
    """
    Identifica los top_n features con mayor impacto positivo en el score
    (los que más empujaron hacia fraude).

    Retorna lista de dicts con: feature, descripcion, valor, shap.
    """
    factores = []
    for i, feature in enumerate(FEATURES_MODELO):
        valor_shap = shap_row[i]
        if valor_shap > 0:  # Solo los que empujan hacia fraude
            factores.append({
                "feature":     feature,
                "descripcion": DESCRIPCION_FEATURES.get(feature, feature),
                "valor":       float(X_row.iloc[i]),
                "shap":        float(valor_shap),
            })

    return sorted(factores, key=lambda x: x["shap"], reverse=True)[:top_n]


def generar_texto_alerta(
    id_siniestro: str,
    score_final: float,
    semaforo: str,
    alertas_reglas: list[str],
    factores_shap: list[dict],
) -> str:
    """
    Genera el texto de alerta completo para un siniestro.
    Este texto lo usa el dashboard y el agente conversacional.

    Formato:
      - Encabezado con nivel de riesgo
      - Señales del motor de reglas
      - Factores del modelo ML (SHAP)
      - Nota ética obligatoria
    """
    emojis = {"verde": "🟢", "amarillo": "🟡", "rojo": "🔴"}
    niveles = {"verde": "BAJO", "amarillo": "MEDIO", "rojo": "ALTO"}

    lineas = [
        f"ALERTA DE REVISIÓN — {emojis.get(semaforo, '')} Riesgo {niveles.get(semaforo, semaforo).upper()}",
        f"Siniestro: {id_siniestro} | Score: {score_final:.1f}/100",
        "",
    ]

    # Señales del motor de reglas
    if alertas_reglas:
        lineas.append("Señales detectadas por motor de reglas:")
        for alerta in alertas_reglas:
            lineas.append(f"  • {alerta}")
        lineas.append("")

    # Factores del modelo ML
    if factores_shap:
        lineas.append("Factores principales del modelo IA:")
        for f in factores_shap:
            lineas.append(f"  • {f['descripcion'].capitalize()} (impacto: +{f['shap']:.3f})")
        lineas.append("")

    # Nota ética — obligatoria según sección 17 del documento
    lineas.append(
        "⚠️  Esta alerta es una señal de revisión, no una acusación de fraude. "
        "Se requiere análisis humano antes de cualquier decisión."
    )

    return "\n".join(lineas)


# ---------------------------------------------------------------------------
# Pipeline completo para un siniestro individual
# ---------------------------------------------------------------------------

def explicar_siniestro(row: pd.Series) -> dict:
    """
    Pipeline completo de explicabilidad para un siniestro.

    Entrada: una fila del dataset (sin procesar).
    Salida: dict con score, semáforo, texto de alerta y factores SHAP.

    Usa fraud_model.py para el score y SHAP para la explicación.
    """
    from models.fraud_model import analizar_siniestro

    # Score y alertas del motor de reglas + ML
    resultado = analizar_siniestro(row)

    # SHAP sobre la misma fila
    df_row     = pd.DataFrame([row])
    df_features = construir_features(df_row)
    shap_vals  = calcular_shap(df_features)
    X_row      = obtener_matriz_modelo(df_features).iloc[0]

    factores = _factores_principales(shap_vals[0], X_row)

    texto = generar_texto_alerta(
        id_siniestro  = str(row.get("id_siniestro", "N/A")),
        score_final   = resultado["score_final"],
        semaforo      = resultado["semaforo"],
        alertas_reglas= resultado["alertas"],
        factores_shap = factores,
    )

    return {
        **resultado,
        "factores_shap": factores,
        "texto_alerta":  texto,
    }


# ---------------------------------------------------------------------------
# Pipeline completo para un DataFrame entero
# ---------------------------------------------------------------------------

def explicar_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega columnas de explicabilidad a todo el DataFrame.

    Columnas nuevas:
      - texto_alerta   : texto completo de la alerta para el analista
      - factores_shap  : lista de dicts con los factores principales

    Más eficiente que llamar explicar_siniestro() fila a fila
    porque calcula SHAP una sola vez sobre todo el DataFrame.
    """
    from models.fraud_model import analizar_dataframe

    # Score y semáforo para todo el DF
    df_scored = analizar_dataframe(df)

    # SHAP para todo el DF de una sola vez
    df_features = construir_features(df.copy())
    shap_vals   = calcular_shap(df_features)
    X_matriz    = obtener_matriz_modelo(df_features)

    textos   = []
    factores = []

    for i in range(len(df_scored)):
        fila_scored = df_scored.iloc[i]
        fila_shap   = shap_vals[i]
        fila_X      = X_matriz.iloc[i]

        top_factores = _factores_principales(fila_shap, fila_X)
        factores.append(top_factores)

        texto = generar_texto_alerta(
            id_siniestro   = str(fila_scored.get("id_siniestro", "N/A")),
            score_final    = fila_scored["score_final"],
            semaforo       = fila_scored["semaforo"],
            alertas_reglas = fila_scored.get("alertas", []),
            factores_shap  = top_factores,
        )
        textos.append(texto)

    df_scored["texto_alerta"]  = textos
    df_scored["factores_shap"] = factores

    return df_scored
