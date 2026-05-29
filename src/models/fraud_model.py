"""
Módulo de producción para detección de posibles fraudes.

Responsabilidades:
  - Cargar el modelo XGBoost entrenado (fraud_model.pkl)
  - Calcular la probabilidad de fraude por siniestro
  - Combinar con el score de reglas en un score híbrido (reglas 60% + ML 40%)
  - Clasificar cada siniestro en el semáforo: verde / amarillo / rojo

Principio clave: este módulo genera alertas de revisión,
NO acusaciones automáticas de fraude.

Nunca reentrena el modelo. El entrenamiento ocurre en:
  notebooks/02_modelo_fraude.ipynb
"""

import os
import joblib
import pandas as pd

from src.features.build_features import construir_features, obtener_matriz_modelo
from src.rules.fraud_rules import calcular_score_reglas


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

RUTA_BASE   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUTA_MODELO = os.path.join(RUTA_BASE, "models", "fraud_model.pkl")

# Pesos del score híbrido (deben sumar 1.0)
PESO_REGLAS = 0.6
PESO_ML     = 0.4

# Umbral de clasificación binaria del modelo ML
UMBRAL_ML = 0.3

# Rangos del semáforo (sección 13 del documento)
UMBRAL_VERDE    = 40
UMBRAL_AMARILLO = 75


# ---------------------------------------------------------------------------
# Carga del modelo
# ---------------------------------------------------------------------------

_modelo_cache = None


def cargar_modelo():
    """
    Carga el modelo desde disco con caché en memoria.
    Falla con mensaje claro si el .pkl no existe todavía.
    """
    global _modelo_cache
    if _modelo_cache is not None:
        return _modelo_cache

    if not os.path.exists(RUTA_MODELO):
        raise FileNotFoundError(
            f"Modelo no encontrado en '{RUTA_MODELO}'.\n"
            "Ejecuta primero: notebooks/02_modelo_fraude.ipynb"
        )

    _modelo_cache = joblib.load(RUTA_MODELO)
    return _modelo_cache


# ---------------------------------------------------------------------------
# Predicción
# ---------------------------------------------------------------------------

def predict_proba(df: pd.DataFrame) -> pd.Series:
    """
    Devuelve la probabilidad de fraude (0-1) del modelo XGBoost
    para cada fila del DataFrame.

    El DataFrame debe haber pasado por construir_features() primero,
    o tener las columnas crudas del dataset para que esta función
    lo haga internamente.
    """
    modelo = cargar_modelo()
    X = obtener_matriz_modelo(df)
    probabilidades = modelo.predict_proba(X)[:, 1]
    return pd.Series(probabilidades, index=df.index, name="proba_fraude")


# ---------------------------------------------------------------------------
# Score híbrido y semáforo
# ---------------------------------------------------------------------------

def _semaforo(score: float) -> str:
    """Clasifica un score numérico en el nivel de riesgo del semáforo."""
    if score <= UMBRAL_VERDE:
        return "verde"
    elif score <= UMBRAL_AMARILLO:
        return "amarillo"
    return "rojo"


def calcular_score_hibrido(score_reglas: float, proba_ml: float) -> float:
    """
    Combina el score del motor de reglas y la probabilidad del modelo ML
    en un score único de 0 a 100.

    score_reglas: 0-100  (sección 7 del documento)
    proba_ml:     0-1    (XGBoost predict_proba)
    """
    score = (score_reglas * PESO_REGLAS) + (proba_ml * 100 * PESO_ML)
    return round(min(score, 100), 2)


# ---------------------------------------------------------------------------
# Pipeline completo por fila
# ---------------------------------------------------------------------------

def analizar_siniestro(row: pd.Series) -> dict:
    """
    Ejecuta el pipeline completo para un único siniestro.

    Retorna un dict con:
      - score_reglas        int       puntaje del motor de reglas
      - proba_fraude        float     probabilidad del modelo ML (0-1)
      - score_final         float     score híbrido (0-100)
      - semaforo            str       'verde' | 'amarillo' | 'rojo'
      - nivel_riesgo        str       texto legible del nivel
      - alertas             list[str] señales activadas
      - reglas_activadas    list[str] códigos RF activados
      - clasificacion_regla str|None  'rojo'/'amarillo' por regla crítica
    """
    # 1. Motor de reglas
    resultado_reglas = calcular_score_reglas(row)

    # 2. Feature engineering + predicción ML
    df_row = pd.DataFrame([row])
    df_features = construir_features(df_row)
    proba = predict_proba(df_features).iloc[0]

    # 3. Score híbrido
    score_final = calcular_score_hibrido(resultado_reglas["score_reglas"], proba)

    # 4. Semáforo — las reglas críticas pueden forzar un nivel mínimo
    nivel = _semaforo(score_final)
    clasificacion_regla = resultado_reglas.get("clasificacion_reglas")
    if clasificacion_regla == "rojo":
        nivel = "rojo"
        score_final = max(score_final, 76.0)
    elif clasificacion_regla == "amarillo" and nivel == "verde":
        nivel = "amarillo"
        score_final = max(score_final, 41.0)

    niveles_texto = {
        "verde":    "🟢 Bajo — continuar flujo normal",
        "amarillo": "🟡 Medio — escalar a Unidad Antifraude para revisión documental",
        "rojo":     "🔴 Alto — escalar a Unidad Antifraude para revisión especializada de campo",
    }

    return {
        "score_reglas":         resultado_reglas["score_reglas"],
        "proba_fraude":         round(float(proba), 4),
        "score_final":          score_final,
        "semaforo":             nivel,
        "nivel_riesgo":         niveles_texto[nivel],
        "alertas":              resultado_reglas["alertas"],
        "reglas_activadas":     resultado_reglas["reglas_activadas"],
        "clasificacion_regla":  clasificacion_regla,
    }


# ---------------------------------------------------------------------------
# Pipeline completo para un DataFrame entero
# ---------------------------------------------------------------------------

def analizar_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ejecuta el pipeline completo para todos los siniestros del DataFrame.

    Agrega columnas al DataFrame original:
      score_reglas, proba_fraude, score_final, semaforo, nivel_riesgo,
      alertas, reglas_activadas

    Retorna el DataFrame ordenado por score_final descendente.
    """
    # Feature engineering una sola vez sobre todo el DF (más eficiente)
    df_features = construir_features(df.copy())
    probas = predict_proba(df_features)

    # Score de reglas fila a fila
    resultados_reglas = df.apply(calcular_score_reglas, axis=1, result_type="expand")

    df_out = df.copy()
    df_out["score_reglas"]      = resultados_reglas["score_reglas"]
    df_out["proba_fraude"]      = probas.values
    df_out["score_final"]       = df_out.apply(
        lambda r: calcular_score_hibrido(r["score_reglas"], r["proba_fraude"]), axis=1
    )
    df_out["clasificacion_regla"] = resultados_reglas["clasificacion_reglas"]
    df_out["semaforo"]          = df_out.apply(
        lambda r: _semaforo_con_override(r["score_final"], r["clasificacion_regla"]), axis=1
    )
    df_out["alertas"]           = resultados_reglas["alertas"]
    df_out["reglas_activadas"]  = resultados_reglas["reglas_activadas"]

    return df_out.sort_values("score_final", ascending=False).reset_index(drop=True)


def _semaforo_con_override(score: float, clasificacion_regla) -> str:
    """
    Aplica el semáforo con override por reglas críticas.
    Versión vectorizable para usar en apply().
    """
    nivel = _semaforo(score)
    if clasificacion_regla == "rojo":
        return "rojo"
    if clasificacion_regla == "amarillo" and nivel == "verde":
        return "amarillo"
    return nivel
