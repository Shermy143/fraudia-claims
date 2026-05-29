# Reglas de Negocio

Todas las reglas están implementadas en `src/rules/fraud_rules.py` y cubiertas por 22 tests en `tests/test_rules.py`.

---

## Reglas Críticas (Override de Semáforo)

Estas reglas ignoran el score numérico y fuerzan el nivel directamente. Se aplican antes de calcular el score híbrido.

### RF-01 — Proveedor + Borde de Vigencia → Rojo

Se activa cuando un siniestro ocurre en los primeros 10 días de vigencia Y el proveedor tiene más de 2 casos observados previos. La combinación de ambos factores es una señal de coordinación entre asegurado y proveedor.

### RF-02 — Inconsistencia Documental → Rojo

Se activa si la columna `inconsistencia_detectada` tiene valor "Sí". Cubre casos donde el perito o el sistema detectó documentos alterados, fechas incoherentes, o datos contradictorios entre fuentes.

### RF-03 — Proveedor en Lista Restrictiva → Rojo

Lista fija de proveedores con historial de fraudes confirmados. En el dataset sintético: `PROV-999`, `PROV-888`, `PROV-777`. En producción, esta lista la mantiene la Unidad Antifraude.

### RF-05 — Siniestro en Primeras 48 Horas → Amarillo

Un siniestro que ocurre en las primeras 48 horas desde el inicio de la póliza es estadísticamente anómalo. Por sí solo no alcanza para rojo, pero requiere revisión documental.

### RF-06 — Robo Sin Denuncia Policial → Amarillo

Aplica solo a coberturas de robo. Si no hay número de parte policial registrado y la demora entre ocurrencia y reporte supera los 5 días, se activa la regla.

---

## Señales Acumulativas (Suman Puntos)

Estas señales se suman al score. Ninguna por sí sola lleva a rojo, pero su combinación sí.

| Señal | Condición | Puntos |
|-------|-----------|--------|
| Borde de vigencia | Siniestro en ≤30 días desde inicio | 15 |
| Borde extremo | Siniestro en ≤2 días desde inicio | 25 adicionales |
| Demora denuncia robo | Robo sin parte policial + >5 días | 20 |
| Alta frecuencia | Asegurado con >3 siniestros previos | 15 |
| Documentos incompletos | Sin documentación registrada | 20 |
| Reporte tardío | Más de 7 días entre ocurrencia y reporte | 10 |
| Monto atípico | Monto reclamado > 90% de la suma asegurada | 15 |
| Proveedor recurrente | Proveedor con >5 casos en alertas amarillo/rojo | 10 |

---

## Score del Motor de Reglas

El score de reglas va de 0 a 100. Se calcula sumando los puntos de todas las señales activas, con techo en 100.

Ejemplos:
- Borde de vigencia (15) + documentos incompletos (20) + reporte tardío (10) = **45 → amarillo**
- Borde extremo (25+15) + RF-03 (override) → **rojo directo**
- Sin señales = **0 → verde**

---

## Override de Semáforo

Cuando una regla crítica activa rojo pero el score híbrido calculado es menor a 76, el sistema fuerza el semáforo a rojo y ajusta el score a mínimo 76. Lo mismo para el override a amarillo (mínimo 41).

Esto garantiza que casos con evidencia documental clara (RF-02: documentos alterados, RF-03: lista restrictiva) nunca terminen en semáforo verde, independientemente de lo que diga el modelo ML.

---

## Lista Restrictiva de Proveedores

En el dataset sintético: `{"PROV-999", "PROV-888", "PROV-777"}`.

Para actualizar la lista en producción, editar la constante `LISTA_RESTRICTIVA` en `fraud_rules.py`. No requiere reentrenar el modelo — las reglas son independientes del ML.

---

## Extensión de Reglas

Para agregar una nueva regla:

1. Crear la función `regla_RF0X_nombre(row) -> tuple[str | None, str]` en `fraud_rules.py`
2. Registrarla en `calcular_score_reglas()`, en la sección de reglas críticas o señales según corresponda
3. Agregar tests en `tests/test_rules.py`

El motor de reglas no requiere cambios en ningún otro módulo.
