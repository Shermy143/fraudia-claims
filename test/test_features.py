"""
Tests para build_features.py
Ejecutar con: python -m pytest tests/test_features.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import pytest   
from features.build_features import (
    construir_features,
    obtener_matriz_modelo,
    FEATURES_MODELO,
)


def make_df(n=None, **overrides) -> pd.DataFrame:
    """Crea un DataFrame mínimo con todas las columnas requeridas.
    Si se pasa una lista en overrides, n se infiere de su longitud."""
    if n is None:
        # Inferir n de la primera lista en overrides, o usar 3 por defecto
        n = 3
        for v in overrides.values():
            if isinstance(v, list):
                n = len(v)
                break

    base = {
        "id_siniestro": [f"SIN-{i:03d}" for i in range(n)],
        "cobertura": ["choque"] * n,
        "dias_desde_inicio_poliza": [180] * n,
        "dias_entre_ocurrencia_reporte": [1] * n,
        "historial_siniestros_asegurado": [0] * n,
        "documentos_completos": ["Si"] * n,
        "monto_reclamado": [1000] * n,
        "monto_estimado": [1000] * n,
        "suma_asegurada": [10000] * n,
        "beneficiario": ["PROV-001"] * n,
        "casos_observados_proveedor": [0] * n,
        "inconsistencia_detectada": ["No"] * n,
    }
    for k, v in overrides.items():
        base[k] = v if isinstance(v, list) else [v] * n
    return pd.DataFrame(base)


class TestConstruirFeatures:
    def test_columnas_creadas(self):
        df = make_df()
        resultado = construir_features(df)
        for feature in FEATURES_MODELO:
            assert feature in resultado.columns, f"Falta feature: {feature}"

    def test_ratio_monto_estimado(self):
        df = make_df(monto_reclamado=[1500], monto_estimado=[1000])
        resultado = construir_features(df)
        assert resultado["ratio_monto_estimado"].iloc[0] == 1.5

    def test_division_por_cero_no_rompe(self):
        df = make_df(monto_estimado=[0], suma_asegurada=[0])
        resultado = construir_features(df)
        # No debe lanzar excepción, solo devolver el monto reclamado
        assert resultado["ratio_monto_estimado"].iloc[0] == 1000

    def test_es_borde_vigencia(self):
        df = make_df(dias_desde_inicio_poliza=[5, 25, 60])
        resultado = construir_features(df)
        assert list(resultado["es_borde_vigencia"]) == [1, 1, 0]

    def test_es_borde_vigencia_extremo(self):
        df = make_df(dias_desde_inicio_poliza=[1, 5])
        resultado = construir_features(df)
        assert list(resultado["es_borde_vigencia_extremo"]) == [1, 0]

    def test_sin_documentos(self):
        df = make_df(documentos_completos=["Si", "No", "no"])
        resultado = construir_features(df)
        assert list(resultado["sin_documentos"]) == [0, 1, 1]

    def test_es_robo(self):
        df = make_df(cobertura=["choque", "robo", "incendio"])
        resultado = construir_features(df)
        assert list(resultado["es_robo"]) == [0, 1, 0]

    def test_proveedor_lista_restrictiva(self):
        df = make_df(beneficiario=["PROV-001", "PROV-999", "PROV-888"])
        resultado = construir_features(df)
        assert list(resultado["proveedor_lista_restrictiva"]) == [0, 1, 1]

    def test_score_reglas_se_agrega(self):
        df = make_df()
        resultado = construir_features(df)
        assert "score_reglas" in resultado.columns
        assert "alertas" in resultado.columns

    def test_falta_columna_lanza_error(self):
        df = make_df().drop(columns=["suma_asegurada"])
        with pytest.raises(ValueError, match="Faltan columnas"):
            construir_features(df)


class TestObtenerMatrizModelo:
    def test_solo_features_definidas(self):
        df = make_df()
        df_con_features = construir_features(df)
        matriz = obtener_matriz_modelo(df_con_features)
        assert list(matriz.columns) == FEATURES_MODELO

    def test_orden_consistente(self):
        df = make_df(n=5)
        df_con_features = construir_features(df)
        matriz = obtener_matriz_modelo(df_con_features)
        # El orden debe ser el de FEATURES_MODELO, no el de df
        for i, col in enumerate(FEATURES_MODELO):
            assert matriz.columns[i] == col

    def test_sin_construir_features_lanza_error(self):
        df = make_df()
        with pytest.raises(ValueError, match="Features faltantes"):
            obtener_matriz_modelo(df)
