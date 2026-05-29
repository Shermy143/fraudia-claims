"""
Feature engineering para el modelo XGBoost.
Toma el DataFrame de siniestros (ya cruzado con pólizas, asegurados,
proveedores y documentos) y construye las variables que usará el modelo.

Principio: XGBoost necesita columnas numéricas. Aquí transformamos
las señales del documento en features que el modelo pueda aprender.
"""

import pandas as pd
import numpy as np

from rules.fraud_rules import aplicar_a_dataframe


# ---------------------------------------------------------------------------
# Lista de features que consume el modelo
# Importante: este orden debe ser el mismo en entrenamiento y predicción
# ---------------------------------------------------------------------------
FEATURES_MODELO = [
    # Ratios y montos
    "ratio_monto_estimado",
    "ratio_monto_suma_asegurada",
    "monto_reclamado_normalizado",

    # Tiempos de vigencia y reporte
    "dias_desde_inicio_poliza",
    "dias_entre_ocurrencia_reporte",

    # Indicadores binarios de señales
    "es_borde_vigencia",
    "es_borde_vigencia_extremo",
    "reporte_tardio",
    "sin_documentos",
    "es_robo",

    # Historial y comportamiento
    "historial_siniestros_asegurado",
    "frecuencia_asegurado_alta",

    # Proveedor
    "proveedor_lista_restrictiva",
    "casos_observados_proveedor",

    # Score del motor de reglas (enfoque híbrido)
    "score_reglas",
]


# ---------------------------------------------------------------------------
# Funciones de construcción de features
# ---------------------------------------------------------------------------

def _ratios_monto(df: pd.DataFrame) -> pd.DataFrame:
    """Construye los ratios derivados de los montos."""
    df = df.copy()

    # Evitar división por cero con un valor mínimo
    df["ratio_monto_estimado"] = df["monto_reclamado"] / df["monto_estimado"].replace(0, 1)
    df["ratio_monto_suma_asegurada"] = df["monto_reclamado"] / df["suma_asegurada"].replace(0, 1)

    # Normalización por log para que XGBoost maneje mejor montos muy grandes
    df["monto_reclamado_normalizado"] = np.log1p(df["monto_reclamado"])

    return df


def _indicadores_binarios(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte señales del documento en columnas 0/1."""
    df = df.copy()

    df["es_borde_vigencia"] = (df["dias_desde_inicio_poliza"] <= 30).astype(int)
    df["es_borde_vigencia_extremo"] = (df["dias_desde_inicio_poliza"] < 2).astype(int)
    df["reporte_tardio"] = (df["dias_entre_ocurrencia_reporte"] > 7).astype(int)
    df["sin_documentos"] = (
        df["documentos_completos"].astype(str).str.strip().str.lower() == "no"
    ).astype(int)
    df["es_robo"] = (df["cobertura"].astype(str).str.lower() == "robo").astype(int)

    return df


def _features_comportamiento(df: pd.DataFrame) -> pd.DataFrame:
    """Features derivadas del historial del asegurado y proveedor."""
    df = df.copy()

    df["frecuencia_asegurado_alta"] = (df["historial_siniestros_asegurado"] >= 2).astype(int)

    # El proveedor en lista restrictiva ya viene marcado por las reglas,
    # pero lo replicamos como feature numérica para el modelo
    from rules.fraud_rules import LISTA_RESTRICTIVA
    df["proveedor_lista_restrictiva"] = (
        df["beneficiario"].astype(str).str.upper().isin(LISTA_RESTRICTIVA)
    ).astype(int)

    return df


def _validar_columnas_requeridas(df: pd.DataFrame) -> None:
    """Verifica que el DataFrame tenga las columnas mínimas."""
    requeridas = {
        "monto_reclamado", "monto_estimado", "suma_asegurada",
        "dias_desde_inicio_poliza", "dias_entre_ocurrencia_reporte",
        "documentos_completos", "cobertura", "beneficiario",
        "historial_siniestros_asegurado", "casos_observados_proveedor",
        "inconsistencia_detectada",
    }
    faltantes = requeridas - set(df.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas en el DataFrame: {faltantes}")


# ---------------------------------------------------------------------------
# Función principal
# ---------------------------------------------------------------------------

def construir_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye todas las features para el modelo XGBoost.

    El DataFrame de entrada debe tener las columnas del dataset sintético
    ya cruzadas (siniestros + pólizas + proveedores + documentos).

    Retorna el DataFrame original con las features agregadas y la columna
    score_reglas del motor de reglas.
    """
    _validar_columnas_requeridas(df)

    df = _ratios_monto(df)
    df = _indicadores_binarios(df)
    df = _features_comportamiento(df)

    # El motor de reglas agrega: score_reglas, alertas, clasificacion_reglas, reglas_activadas
    df = aplicar_a_dataframe(df)

    return df


def obtener_matriz_modelo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Devuelve solo las columnas que entran al modelo XGBoost,
    en el orden definido por FEATURES_MODELO.

    Usar después de construir_features().
    """
    faltantes = set(FEATURES_MODELO) - set(df.columns)
    if faltantes:
        raise ValueError(
            f"Features faltantes: {faltantes}. "
            f"¿Ejecutaste construir_features() primero?"
        )
    return df[FEATURES_MODELO].copy()
