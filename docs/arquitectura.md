# Arquitectura del Sistema

## Visión General

Fraudia es un sistema de alertas para detectar posibles fraudes en siniestros de seguros. No toma decisiones automáticas: genera alertas que un analista humano revisa antes de actuar.

El sistema combina dos enfoques complementarios: un motor de reglas de negocio y un modelo de Machine Learning. Ambos producen un score que determina la prioridad de revisión.

---

## Flujo de Datos

```
Datos de entrada
      │
      ▼
load_data.py          Merge de tres fuentes → siniestros_merged.csv
      │
      ▼
build_features.py     16 variables para el modelo
      │
      ├──────────────────────────┐
      ▼                          ▼
fraud_rules.py           fraud_model.py
Motor de reglas          XGBoost (pkl cargado)
Score 0-100              Probabilidad 0-1
      │                          │
      └──────────┬───────────────┘
                 ▼
         Score Híbrido
     reglas × 0.6 + ML × 0.4
                 │
                 ▼
        explain_score.py     SHAP + texto de alerta
                 │
                 ▼
        claims_agent.py      Agente Groq LLaMA-3.3-70b
                 │
                 ▼
            main.py          Dashboard Streamlit
```

---

## Componentes

### Motor de Reglas (`fraud_rules.py`)

Aplica reglas de negocio basadas en el documento del reto (sección 7 y 8). Cada regla suma puntos al score o lo fuerza directamente a un nivel de riesgo.

Reglas críticas (fuerzan el nivel sin importar el score):
- **RF-01**: proveedor + borde de vigencia → rojo
- **RF-02**: inconsistencia documental → rojo
- **RF-03**: proveedor en lista restrictiva → rojo
- **RF-05**: siniestro en primeras 48 horas → amarillo
- **RF-06**: robo sin denuncia policial → amarillo

Señales acumulativas (suman puntos):
- Borde de vigencia (≤30 días desde inicio de póliza)
- Demora en denuncia de robo
- Alta frecuencia de reclamos del asegurado
- Documentación incompleta
- Reporte tardío (>7 días)
- Monto atípico
- Proveedor recurrente en alertas

### Modelo ML (`fraud_model.py`)

Carga `fraud_model.pkl` generado por el notebook 02. Nunca reentrena en producción.

El modelo usa `scale_pos_weight` para compensar el desbalance de clases (5.67x más casos normales que fraudes). El umbral de decisión es 0.3 en lugar del 0.5 por defecto, porque en seguros es peor no detectar un fraude que investigar un caso de más.

### Score Híbrido

```
score_final = score_reglas × 0.6 + proba_xgboost × 100 × 0.4
```

Los pesos reflejan que el motor de reglas es más confiable con los datos disponibles. El modelo ML complementa con patrones no explícitos.

Las reglas críticas tienen **override**: si RF-03 activa rojo pero el score híbrido da 30, el semáforo se fuerza a rojo de todas formas.

### Semáforo

| Nivel | Score | Acción |
|-------|-------|--------|
| 🟢 Verde | 0–40 | Continuar flujo normal |
| 🟡 Amarillo | 41–75 | Revisión documental |
| 🔴 Rojo | 76–100 | Revisión especializada de campo |

### Explicabilidad (`explain_score.py`)

Usa SHAP (SHapley Additive exPlanations) para explicar por qué el modelo asignó ese score a cada siniestro. Traduce los valores SHAP a texto legible por el analista, indicando qué variables empujaron el score hacia arriba y cuánto.

### Agente Conversacional (`claims_agent.py`)

Usa la API de Groq con el modelo LLaMA-3.3-70b-versatile. Recibe como contexto un resumen del dataset analizado (estadísticas, top casos, ranking de proveedores) y responde preguntas en lenguaje natural. Mantiene historial multi-turno.

---

## Decisiones de Diseño

**¿Por qué XGBoost y no una red neuronal?**
XGBoost funciona bien con tablas pequeñas (~1000 registros), es más rápido de entrenar, y SHAP lo soporta nativamente. Una red neuronal necesitaría más datos para ser útil.

**¿Por qué reglas al 60% y ML al 40%?**
El AUC-ROC del modelo es 0.56 (datos sintéticos), lo que lo hace poco confiable solo. Las reglas son deterministas y validables. El 40% ML aporta patrones no capturados por las reglas.

**¿Por qué Groq en lugar de otra API?**
Groq ofrece free tier con velocidad de inferencia muy alta (~400 tokens/seg). Para la demo en vivo, la latencia baja es importante.

**¿Por qué Streamlit y no React + FastAPI?**
Con Streamlit se reduce el tiempo de desarrollo a un solo archivo Python. Para un hackathon de 48 horas, la velocidad importa más que la flexibilidad del stack.

---

## Limitaciones de la Arquitectura

- El modelo reentrena desde cero si se actualiza el dataset (no hay fine-tuning incremental).
- El agente no tiene acceso a los PDFs de documentación — solo al dataset analizado.
- El sistema no persiste estado entre reinicios de Streamlit.
