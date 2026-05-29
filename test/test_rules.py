"""
Tests unitarios para fraud_rules.py
Ejecutar con: python -m pytest tests/test_rules.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from rules.fraud_rules import (
    calcular_score_reglas,
    señal_borde_vigencia,
    señal_reporte_tardio,
    señal_documentos_incompletos,
    señal_monto_atipico,
    señal_proveedor_recurrente,
    regla_RF03_lista_restrictiva,
    regla_RF05_borde_vigencia_extremo,
)


def make_row(**kwargs) -> pd.Series:
    """Crea una fila base con valores seguros (sin señales)."""
    base = {
        "id_siniestro": "SIN-001",
        "cobertura": "choque",
        "dias_desde_inicio_poliza": 180,
        "dias_entre_ocurrencia_reporte": 1,
        "historial_siniestros_asegurado": 0,
        "documentos_completos": "Si",
        "monto_reclamado": 1000,
        "monto_estimado": 1000,
        "suma_asegurada": 10000,
        "beneficiario": "PROV-001",
        "casos_observados_proveedor": 0,
        "inconsistencia_detectada": "No",
    }
    base.update(kwargs)
    return pd.Series(base)


# ---------------------------------------------------------------------------
# Tests señales individuales
# ---------------------------------------------------------------------------

class TestSenalBordeVigencia:
    def test_menos_de_10_dias(self):
        row = make_row(dias_desde_inicio_poliza=5)
        puntos, desc = señal_borde_vigencia(row)
        assert puntos == 8
        assert "≤10 días" in desc

    def test_entre_11_y_30_dias(self):
        row = make_row(dias_desde_inicio_poliza=20)
        puntos, desc = señal_borde_vigencia(row)
        assert puntos == 4

    def test_mas_de_30_dias(self):
        row = make_row(dias_desde_inicio_poliza=60)
        puntos, desc = señal_borde_vigencia(row)
        assert puntos == 0
        assert desc == ""


class TestSenalReporteTardio:
    def test_mas_de_7_dias(self):
        row = make_row(dias_entre_ocurrencia_reporte=10)
        puntos, desc = señal_reporte_tardio(row)
        assert puntos == 5

    def test_entre_4_y_7_dias(self):
        row = make_row(dias_entre_ocurrencia_reporte=5)
        puntos, desc = señal_reporte_tardio(row)
        assert puntos == 3

    def test_3_dias_o_menos(self):
        row = make_row(dias_entre_ocurrencia_reporte=2)
        puntos, desc = señal_reporte_tardio(row)
        assert puntos == 0


class TestSenalDocumentosIncompletos:
    def test_sin_documentos(self):
        row = make_row(documentos_completos="No")
        puntos, desc = señal_documentos_incompletos(row)
        assert puntos == 4

    def test_con_documentos(self):
        row = make_row(documentos_completos="Si")
        puntos, desc = señal_documentos_incompletos(row)
        assert puntos == 0


class TestSenalMontoAtipico:
    def test_supera_95_suma_asegurada(self):
        row = make_row(monto_reclamado=9600, suma_asegurada=10000, monto_estimado=5000)
        puntos, desc = señal_monto_atipico(row)
        assert puntos == 4
        assert "95%" in desc

    def test_supera_50_estimado(self):
        row = make_row(monto_reclamado=1600, monto_estimado=1000, suma_asegurada=50000)
        puntos, desc = señal_monto_atipico(row)
        assert puntos == 4
        assert "+50%" in desc

    def test_monto_normal(self):
        row = make_row(monto_reclamado=800, monto_estimado=1000, suma_asegurada=10000)
        puntos, desc = señal_monto_atipico(row)
        assert puntos == 0


class TestSenalProveedorRecurrente:
    def test_en_lista_restrictiva(self):
        row = make_row(beneficiario="PROV-999")
        puntos, desc = señal_proveedor_recurrente(row)
        assert puntos == 10
        assert "lista restrictiva" in desc

    def test_mas_de_2_casos_observados(self):
        row = make_row(beneficiario="PROV-100", casos_observados_proveedor=3)
        puntos, desc = señal_proveedor_recurrente(row)
        assert puntos == 5

    def test_proveedor_normal(self):
        row = make_row(beneficiario="PROV-001", casos_observados_proveedor=0)
        puntos, desc = señal_proveedor_recurrente(row)
        assert puntos == 0


# ---------------------------------------------------------------------------
# Tests reglas críticas
# ---------------------------------------------------------------------------

class TestReglaRF03:
    def test_proveedor_en_lista(self):
        row = make_row(beneficiario="PROV-888")
        clasificacion, desc = regla_RF03_lista_restrictiva(row)
        assert clasificacion == "rojo"

    def test_proveedor_normal(self):
        row = make_row(beneficiario="PROV-001")
        clasificacion, desc = regla_RF03_lista_restrictiva(row)
        assert clasificacion is None


class TestReglaRF05:
    def test_primeras_48_horas(self):
        row = make_row(dias_desde_inicio_poliza=1)
        clasificacion, desc = regla_RF05_borde_vigencia_extremo(row)
        assert clasificacion == "amarillo"

    def test_fuera_de_48_horas(self):
        row = make_row(dias_desde_inicio_poliza=5)
        clasificacion, desc = regla_RF05_borde_vigencia_extremo(row)
        assert clasificacion is None


# ---------------------------------------------------------------------------
# Tests del score completo
# ---------------------------------------------------------------------------

class TestCalcularScoreReglas:
    def test_siniestro_limpio(self):
        row = make_row()
        resultado = calcular_score_reglas(row)
        assert resultado["score_reglas"] == 0
        assert resultado["alertas"] == []
        assert resultado["clasificacion_reglas"] is None

    def test_siniestro_con_multiples_señales(self):
        row = make_row(
            dias_desde_inicio_poliza=5,        # 8 pts
            documentos_completos="No",          # 4 pts
            dias_entre_ocurrencia_reporte=10,   # 5 pts
        )
        resultado = calcular_score_reglas(row)
        assert resultado["score_reglas"] == 17
        assert len(resultado["alertas"]) == 3

    def test_score_maximo_100(self):
        row = make_row(
            dias_desde_inicio_poliza=5,
            documentos_completos="No",
            dias_entre_ocurrencia_reporte=10,
            monto_reclamado=9600,
            suma_asegurada=10000,
            monto_estimado=1000,
            beneficiario="PROV-999",
            historial_siniestros_asegurado=3,
        )
        resultado = calcular_score_reglas(row)
        assert resultado["score_reglas"] <= 100

    def test_rojo_tiene_prioridad(self):
        row = make_row(
            beneficiario="PROV-999",            # RF-03 → rojo
            dias_desde_inicio_poliza=1,         # RF-05 → amarillo
        )
        resultado = calcular_score_reglas(row)
        assert resultado["clasificacion_reglas"] == "rojo"
