# Uso de Inteligencia Artificial

El sistema usa IA en dos capas distintas: un modelo de clasificación para detectar patrones, y un agente conversacional para responder preguntas en lenguaje natural.

---

## Capa 1 — Modelo de Clasificación (XGBoost)

### Algoritmo

XGBoost (eXtreme Gradient Boosting) con clasificación binaria supervisada. El modelo entrena con 16 features y la etiqueta `etiqueta_fraude_simulada` (0 = normal, 1 = posible fraude).

Gradient boosting construye árboles de decisión en secuencia, donde cada árbol corrige los errores del anterior. El resultado final es la suma ponderada de 100 árboles con profundidad máxima 4.

### Parámetros Clave

| Parámetro | Valor | Justificación |
|-----------|-------|---------------|
| `n_estimators` | 100 | Balance entre capacidad y riesgo de sobreajuste |
| `max_depth` | 4 | Árboles poco profundos generalizan mejor |
| `learning_rate` | 0.1 | Estándar para XGBoost |
| `scale_pos_weight` | 5.67 | Compensa el desbalance 85.6% / 14.4% |
| `eval_metric` | auc | AUC-ROC es la métrica más informativa para datos desbalanceados |

### Métricas del Modelo Entrenado

```
Dataset:   1000 registros | 144 fraudes (14.4%)
Split:     80% train / 20% test (estratificado)
Features:  16

AUC-ROC:   0.5566

Umbral 0.5 (por defecto):
  Precision fraude:  18.75%
  Recall fraude:     20.69%
  F1-Score:          0.1967

Umbral 0.3 (ajustado al negocio):
  Precision fraude:  18.18%
  Recall fraude:     55.17%
  F1-Score:          0.2735
```

El AUC-ROC de 0.56 refleja que el modelo fue entrenado con datos 100% sintéticos. Con datos reales de siniestros confirmados, el rendimiento esperado es AUC-ROC > 0.75.

### Umbral Ajustado al Negocio

El umbral de 0.3 (en lugar del 0.5 por defecto) aumenta el recall de fraude de 21% a 55%. El trade-off consciente: más falsos positivos (casos normales marcados) a cambio de menos fraudes no detectados. En seguros, un fraude que pasa desapercibido tiene un costo mayor que revisar un caso de más.

### Explicabilidad con SHAP

Cada predicción se acompaña de valores SHAP (SHapley Additive exPlanations), que miden cuánto contribuyó cada feature al score final. Features con SHAP positivo empujaron el score hacia fraude; con SHAP negativo, hacia normal.

Los cinco factores con mayor impacto identificados en el dataset:
1. `ratio_monto_suma_asegurada` — monto reclamado cercano al límite de cobertura
2. `dias_entre_ocurrencia_reporte` — demora en reportar el siniestro
3. `casos_observados_proveedor` — proveedor con historial de alertas
4. `historial_siniestros_asegurado` — asegurado con múltiples reclamos previos
5. `ratio_monto_estimado` — monto reclamado mayor al estimado por el perito

---

## Capa 2 — Agente Conversacional (LLaMA vía Groq)

### Modelo

LLaMA 3.3 70B Versatile, accedido vía API de Groq. Groq usa hardware especializado (LPU) que permite velocidades de inferencia de ~400 tokens/segundo, reduciendo la latencia en la demo en vivo.

### Funcionamiento

El agente recibe en cada consulta un resumen del dataset analizado (estadísticas, top 20 casos por score, ranking de proveedores, distribución por ramo y ciudad). Con ese contexto responde preguntas en lenguaje natural sobre los siniestros.

Preguntas que puede responder:
1. ¿Cuáles son los 10 siniestros con mayor riesgo?
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
12. Recomienda qué casos revisar primero.

### Principio Ético en el Agente

El system prompt instruye al agente a usar siempre lenguaje de "posible fraude" o "requiere revisión". Nunca puede afirmar que un siniestro ES fraude. Esta restricción no puede ser removida por el usuario durante la conversación.

---

## Score Híbrido — Integración de las Dos Capas

```
score_final = score_reglas × 0.6 + proba_xgboost × 100 × 0.4
```

El motor de reglas tiene mayor peso porque es determinista y validable. El modelo ML aporta patrones que las reglas no capturan. Si una regla crítica detecta algo grave (RF-02 falsificación, RF-03 lista restrictiva), fuerza el semáforo a rojo sin importar qué diga el modelo.

---

## Lo que la IA No Hace

- No rechaza siniestros automáticamente.
- No accede a datos externos en tiempo real.
- No modifica los registros del sistema de la aseguradora.
- No almacena información de conversaciones entre sesiones.
- No reemplaza el análisis del ajustador o el investigador.
