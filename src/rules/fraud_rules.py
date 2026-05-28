"""
Motor de reglas de negocio para detección de posibles fraudes.
Basado en las señales de la sección 7 y reglas críticas de la sección 8
del documento hackIAthon - Reto Aseguradora del Sur.

Principio clave: este módulo genera alertas de revisión,
NO acusaciones automáticas de fraude.
"""

import pandas as pd


# ---------------------------------------------------------------------------
# Lista restrictiva de proveedores/beneficiarios (simulada)
# En producción vendría de una tabla externa
# ---------------------------------------------------------------------------
LISTA_RESTRICTIVA = {"PROV-999", "PROV-888", "PROV-777"}


# ---------------------------------------------------------------------------
# Señales individuales — Sección 7
# Cada función recibe una fila del DataFrame y retorna (puntos, descripción)
# ---------------------------------------------------------------------------

def señal_borde_vigencia(row) -> tuple[int, str]:
    """
    Reclamo cercano al borde de vigencia.
    ≤ 10 días desde inicio: 8 pts
    11 a 30 días:           4 pts
    > 30 días:              0 pts
    """
    dias = row.get("dias_desde_inicio_poliza", 9999)
    if dias <= 10:
        return 8, f"Siniestro ocurrido {dias} días después del inicio de la póliza (≤10 días)"
    elif dias <= 30:
        return 4, f"Siniestro ocurrido {dias} días después del inicio de la póliza (11-30 días)"
    return 0, ""


def señal_demora_denuncia_robo(row) -> tuple[int, str]:
    """
    Demora en denuncia para reclamos de tipo Robo.
    > 48 horas:   8 pts
    24 a 48 hrs:  4 pts
    < 24 horas:   0 pts
    Solo aplica si cobertura es 'robo'.
    """
    if str(row.get("cobertura", "")).lower() != "robo":
        return 0, ""
    horas = row.get("dias_entre_ocurrencia_reporte", 0) * 24
    if horas > 48:
        return 8, f"Denuncia de robo con {horas:.0f} horas de demora (>48 horas)"
    elif horas >= 24:
        return 4, f"Denuncia de robo con {horas:.0f} horas de demora (24-48 horas)"
    return 0, ""


def señal_frecuencia_asegurado(row) -> tuple[int, str]:
    """
    Alta frecuencia de reclamos por asegurado en los últimos 18 meses.
    ≥ 3 siniestros: 8 pts
    2 siniestros:   4 pts
    0-1:            0 pts
    """
    historial = row.get("historial_siniestros_asegurado", 0)
    if historial >= 3:
        return 8, f"Asegurado con {historial} siniestros previos en 18 meses (≥3)"
    elif historial == 2:
        return 4, f"Asegurado con {historial} siniestros previos en 18 meses"
    return 0, ""


def señal_documentos_incompletos(row) -> tuple[int, str]:
    """
    Falta de documentación obligatoria.
    documentos_completos = 'No': 4 pts
    """
    if str(row.get("documentos_completos", "Si")).strip().lower() == "no":
        return 4, "Documentación incompleta o faltante"
    return 0, ""


def señal_reporte_tardio(row) -> tuple[int, str]:
    """
    Reporte tardío del siniestro.
    > 7 días:   5 pts
    4 a 7 días: 3 pts
    ≤ 3 días:   0 pts
    """
    dias = row.get("dias_entre_ocurrencia_reporte", 0)
    if dias > 7:
        return 5, f"Siniestro reportado {dias} días después del evento (>7 días)"
    elif dias >= 4:
        return 3, f"Siniestro reportado {dias} días después del evento (4-7 días)"
    return 0, ""


def señal_monto_atipico(row) -> tuple[int, str]:
    """
    Monto reclamado cercano o superior a la suma asegurada.
    > 95% de la suma asegurada: 4 pts
    +50% del monto estimado:    4 pts
    """
    reclamado = row.get("monto_reclamado", 0)
    estimado = row.get("monto_estimado", 1)
    suma_asegurada = row.get("suma_asegurada", 1)

    if suma_asegurada > 0 and (reclamado / suma_asegurada) > 0.95:
        return 4, f"Monto reclamado ({reclamado}) supera el 95% de la suma asegurada ({suma_asegurada})"
    if estimado > 0 and reclamado > estimado * 1.5:
        return 4, f"Monto reclamado ({reclamado}) supera en +50% el estimado ({estimado})"
    return 0, ""


def señal_proveedor_recurrente(row) -> tuple[int, str]:
    """
    Proveedor o beneficiario en lista restrictiva o con alta concentración.
    En lista restrictiva:       10 pts
    En >2 casos observados:      5 pts
    """
    proveedor = str(row.get("beneficiario", "")).strip().upper()
    if proveedor in LISTA_RESTRICTIVA:
        return 10, f"Proveedor/beneficiario '{proveedor}' en lista restrictiva"
    casos_observados = row.get("casos_observados_proveedor", 0)
    if casos_observados > 2:
        return 5, f"Proveedor con {casos_observados} casos observados este año (>2)"
    return 0, ""


# ---------------------------------------------------------------------------
# Reglas críticas de negocio — Sección 8 (RF-01 a RF-07)
# Retornan (clasificacion, descripcion) si aplican, o (None, "") si no
# ---------------------------------------------------------------------------

def regla_RF01_perdida_total_robo(row) -> tuple[str | None, str]:
    """RF-01: Cobertura Pérdida Total por Robo → Rojo"""
    cobertura = str(row.get("cobertura", "")).lower()
    if "pérdida total" in cobertura or "ptxrb" in cobertura or "perdida total" in cobertura:
        return "rojo", "RF-01: Cobertura de Pérdida Total por Robo detectada"
    return None, ""


def regla_RF02_falsificacion_documental(row) -> tuple[str | None, str]:
    """RF-02: Evidencia de Falsificación o Adulteración Documental → Rojo"""
    inconsistencia = str(row.get("inconsistencia_detectada", "")).strip().lower()
    if inconsistencia in {"si", "sí", "true", "1"}:
        return "rojo", "RF-02: Inconsistencia documental detectada (posible falsificación)"
    return None, ""


def regla_RF03_lista_restrictiva(row) -> tuple[str | None, str]:
    """RF-03: Coincidencia con Lista Restrictiva → Rojo"""
    proveedor = str(row.get("beneficiario", "")).strip().upper()
    if proveedor in LISTA_RESTRICTIVA:
        return "rojo", f"RF-03: Proveedor '{proveedor}' en lista restrictiva"
    return None, ""


def regla_RF05_borde_vigencia_extremo(row) -> tuple[str | None, str]:
    """RF-05: Siniestro en las primeras 48 horas de la póliza → Amarillo"""
    dias = row.get("dias_desde_inicio_poliza", 9999)
    if 0 <= dias < 2:
        return "amarillo", f"RF-05: Siniestro ocurrido en las primeras 48 horas de vigencia"
    return None, ""


def regla_RF06_demora_atipica_robo(row) -> tuple[str | None, str]:
    """RF-06: Demora atípica en denuncia de robo (>4 días) → Amarillo"""
    if str(row.get("cobertura", "")).lower() != "robo":
        return None, ""
    dias = row.get("dias_entre_ocurrencia_reporte", 0)
    if dias > 4:
        return "amarillo", f"RF-06: Denuncia de robo con {dias} días de demora (>4 días)"
    return None, ""


# ---------------------------------------------------------------------------
# Función principal — aplica todas las señales y reglas a una fila
# ---------------------------------------------------------------------------

def calcular_score_reglas(row: pd.Series) -> dict:
    """
    Aplica todas las señales (sección 7) y reglas críticas (sección 8)
    a un siniestro individual.

    Retorna un dict con:
    - score_reglas (int 0-100): puntaje acumulado por señales
    - alertas (list[str]): descripciones de señales activadas
    - clasificacion_reglas (str): 'rojo' | 'amarillo' | None (por reglas críticas)
    - reglas_activadas (list[str]): códigos RF activados
    """
    score = 0
    alertas = []
    clasificacion_critica = None
    reglas_activadas = []

    # --- Señales de puntuación (sección 7) ---
    señales = [
        señal_borde_vigencia,
        señal_demora_denuncia_robo,
        señal_frecuencia_asegurado,
        señal_documentos_incompletos,
        señal_reporte_tardio,
        señal_monto_atipico,
        señal_proveedor_recurrente,
    ]

    for señal in señales:
        puntos, descripcion = señal(row)
        score += puntos
        if descripcion:
            alertas.append(descripcion)

    # Limitar el score de reglas a 100
    score = min(score, 100)

    # --- Reglas críticas (sección 8) ---
    reglas = [
        ("RF-01", regla_RF01_perdida_total_robo),
        ("RF-02", regla_RF02_falsificacion_documental),
        ("RF-03", regla_RF03_lista_restrictiva),
        ("RF-05", regla_RF05_borde_vigencia_extremo),
        ("RF-06", regla_RF06_demora_atipica_robo),
    ]

    for codigo, regla in reglas:
        clasificacion, descripcion = regla(row)
        if clasificacion:
            reglas_activadas.append(codigo)
            alertas.append(descripcion)
            # Rojo tiene prioridad sobre amarillo
            if clasificacion_critica != "rojo":
                clasificacion_critica = clasificacion

    return {
        "score_reglas": score,
        "alertas": alertas,
        "clasificacion_reglas": clasificacion_critica,
        "reglas_activadas": reglas_activadas,
    }


def aplicar_a_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica el motor de reglas a todo el DataFrame de siniestros.
    Agrega columnas: score_reglas, alertas, clasificacion_reglas, reglas_activadas.
    """
    resultados = df.apply(calcular_score_reglas, axis=1, result_type="expand")
    return pd.concat([df, resultados], axis=1)
