# Limitaciones del Sistema

Documentar las limitaciones del sistema es parte del diseño responsable. Esta sección cubre lo que el sistema no puede hacer, por qué, y qué riesgo implica cada limitación.

---

## Limitaciones del Modelo ML

### Datos Sintéticos

El modelo fue entrenado con 1000 registros generados artificialmente. Los patrones aprendidos pueden no coincidir con los de fraudes reales. El AUC-ROC resultante (0.5566) lo confirma: el modelo es apenas mejor que una predicción aleatoria.

**Impacto:** el modelo ML aporta poco poder predictivo real. El sistema lo compensa con el motor de reglas (60% del score híbrido).

**Solución en producción:** reentrenar con datos reales de siniestros con fraudes confirmados. Con 5000+ casos etiquetados se espera AUC-ROC > 0.75.

### Desbalance de Clases

Solo el 14.4% del dataset son casos de posible fraude. Aunque XGBoost compensa con `scale_pos_weight`, el número absoluto de ejemplos de fraude (144) es pequeño para que el modelo aprenda patrones complejos.

**Impacto:** tasa de falsos positivos alta (42% con umbral 0.3) y falsos negativos también alta (44.8%).

### Features Limitadas

Las 16 variables del modelo se derivan de las columnas disponibles en los CSVs. No incluyen señales que los expertos conocen y que podrían mejorar el modelo, como ubicación GPS del evento, imágenes del daño, o historial de pagos del asegurado.

### Similitud Narrativa Parcial

La feature `similitud_narrativa_max` solo está disponible para 500 de los 1000 registros (los que coinciden con el dataset del organizador). Los otros 500 tienen NaN. XGBoost lo maneja, pero el modelo tiene información incompleta para la mitad del dataset.

---

## Limitaciones del Motor de Reglas

### Lista Restrictiva Estática

La lista de proveedores en `LISTA_RESTRICTIVA` es fija en el código. En producción necesita sincronizarse con una base de datos actualizable por la Unidad Antifraude, sin necesidad de redeploy.

### Umbrales Sin Validación Histórica

Los pesos de cada señal (15 puntos por borde de vigencia, 20 por documentos incompletos, etc.) se definieron basándose en el documento del reto, no en análisis estadístico de casos reales. Pueden no reflejar el peso real de cada señal en la aseguradora.

### Reglas sin Contexto Temporal

Las reglas no consideran tendencias en el tiempo. Por ejemplo, un proveedor puede aparecer en muchos siniestros legítimos en temporadas de lluvia, y el motor lo marcaría igual que en períodos normales.

---

## Limitaciones del Sistema Completo

### No Persiste Estado

El sistema no guarda el historial de revisiones ni las decisiones del analista. Si el analista descarta una alerta como falso positivo, el sistema la volverá a marcar en la próxima ejecución.

### Agente sin Memoria entre Sesiones

El agente conversacional pierde el historial al reiniciar el dashboard. No recuerda conversaciones anteriores ni el contexto de decisiones pasadas.

### No Integrado con Sistemas de la Aseguradora

El sistema lee CSVs y escribe alertas en pantalla. No se conecta a los sistemas de gestión de siniestros, ni actualiza registros, ni envía notificaciones automáticas.

### Contexto del Agente Limitado

El agente recibe un resumen compacto del dataset (top 20 casos, estadísticas generales), no los 1000 registros completos. Para preguntas muy específicas sobre casos individuales fuera del top 20, puede no tener la información necesaria.

---

## Lo que el Sistema NO Es

- No es un sistema de decisión automática. Toda alerta requiere revisión humana.
- No es una acusación de fraude. El lenguaje del sistema es siempre de "posible fraude" o "requiere revisión".
- No es un sistema en tiempo real. Procesa el dataset completo en cada ejecución.
- No reemplaza la investigación de campo. El sistema prioriza qué revisar, no cierra casos.

---

## Riesgos y Mitigaciones

| Riesgo | Probabilidad | Mitigación |
|--------|-------------|------------|
| Falsos positivos afectan a asegurados honestos | Alta | Revisar manualmente antes de actuar |
| Modelo obsoleto con el tiempo | Media | Reentrenar periódicamente con casos nuevos |
| Lista restrictiva desactualizada | Media | Proceso de actualización con la Unidad Antifraude |
| Sesgo en datos de entrenamiento | Alta (datos sintéticos) | Auditar con expertos antes de producción |
| Asegurado o proveedor identifican el patrón | Baja | Combinar con investigación de campo |
