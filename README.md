# Fraudia — Detector de Posibles Fraudes en Siniestros

Sistema de alertas de revisión para siniestros de seguros basado en Inteligencia Artificial. Desarrollado para el hackIAthon 2026 — Reto Aseguradora del Sur.

> **Principio clave:** el sistema genera alertas de revisión, no acusaciones automáticas de fraude. Toda alerta requiere análisis humano antes de cualquier decisión.

---

## Arquitectura

```
Dataset sintético (CSV + Excel + PDFs)
        ↓
load_data.py          → data/processed/siniestros_merged.csv
        ↓
build_features.py     → features para XGBoost (15 variables)
        ↓
fraud_rules.py        → motor de reglas RF-01 a RF-07 (score 0-100)
        ↓
fraud_model.py        → XGBoost + score híbrido (reglas 60% + ML 40%)
        ↓
explain_score.py      → SHAP → texto de alerta legible
        ↓
claims_agent.py       → agente Groq LLaMA-3.3-70b (lenguaje natural)
        ↓
main.py               → Dashboard Streamlit (semáforo + chat)
```

**Score híbrido:** `score_reglas × 0.6 + proba_xgboost × 100 × 0.4`

**Semáforo:**
- 🔴 Rojo (76-100): revisión especializada de campo
- 🟡 Amarillo (41-75): revisión documental
- 🟢 Verde (0-40): flujo normal

---

## Estructura del repositorio

```
fraudia-claims/
├── README.md
├── requirements.txt
├── .env.example
├── data/
│   ├── synthetic/          ← datasets originales (CSV, Excel, PDFs)
│   └── processed/
│       └── siniestros_merged.csv   ← generado por load_data.py
├── notebooks/
│   ├── 01_exploracion_datos.ipynb
│   ├── 02_modelo_fraude.ipynb      ← entrena y guarda fraud_model.pkl
│   └── 03_evaluacion_modelo.ipynb  ← SHAP, ROC, métricas
├── src/
│   ├── ingestion/
│   │   └── load_data.py            ← merge de las 3 fuentes
│   ├── features/
│   │   └── build_features.py       ← feature engineering
│   ├── rules/
│   │   └── fraud_rules.py          ← motor de reglas RF-01 a RF-07
│   ├── models/
│   │   └── fraud_model.py          ← carga pkl, score híbrido, semáforo
│   ├── explainability/
│   │   └── explain_score.py        ← SHAP → texto de alerta
│   ├── ai_agent/
│   │   └── claims_agent.py         ← agente conversacional Groq
│   └── app/
│       └── main.py                 ← dashboard Streamlit
├── models/
│   └── fraud_model.pkl             ← generado por notebook 02
├── docs/
│   ├── arquitectura.md
│   ├── modelo_datos.md
│   ├── reglas_negocio.md
│   ├── uso_ia.md
│   └── limitaciones.md
├── tests/
│   ├── test_rules.py               ← 22 tests del motor de reglas
│   └── test_features.py            ← 13 tests del feature engineering
└── presentation/
    └── pitch.pdf
```

---

## Instalación

```bash
git clone https://github.com/Shermy143/fraudia-claims.git
cd fraudia-claims
pip install -r requirements.txt
```

Copia el archivo de variables de entorno y agrega tu API key de Groq:

```bash
cp .env.example .env
# Edita .env y agrega tu GROQ_API_KEY
# Obtén una clave gratis en https://console.groq.com
```

---

## Ejecución paso a paso

### 1. Merge del dataset

Asegúrate de tener en `data/synthetic/`:
- `siniestros_sintetico.csv`
- `Evento_Datasets_Sinteticos_Fraude_500_v2.xlsx`
- PDFs en subcarpetas (PARTE POLICIAL, FACTURAS, DECLARACIÓN DE ACCIDENTE)

```bash
python src/ingestion/load_data.py
```

Genera `data/processed/siniestros_merged.csv` con 1000 registros y 58 columnas.

### 2. Entrenamiento del modelo

Abre `notebooks/02_modelo_fraude.ipynb` en Google Colab:

```
https://colab.research.google.com/github/Shermy143/fraudia-claims/blob/main/notebooks/02_modelo_fraude.ipynb
```

Ejecuta todas las celdas. Descarga `fraud_model.pkl` y colócalo en `models/`.

### 3. Evaluación (opcional antes de la demo)

Abre `notebooks/03_evaluacion_modelo.ipynb` para ver SHAP, curva ROC y métricas detalladas.

### 4. Dashboard

```bash
streamlit run src/app/main.py
```

---

## Tests

```bash
python -m pytest tests/ -v
```

35 tests en total: 22 para el motor de reglas, 13 para feature engineering.

---

## Fuentes de datos

| Fuente | Descripción |
|--------|-------------|
| `siniestros_sintetico.csv` | 1000 siniestros sintéticos generados por el equipo. Contiene `etiqueta_fraude_simulada` (0/1) requerida para el entrenamiento supervisado. |
| `Evento_Datasets_Sinteticos_Fraude_500_v2.xlsx` | Dataset oficial del organizador. 5 hojas: Siniestros, Pólizas, Asegurados, Proveedores, Documentos. Incluye `similitud_narrativa_max` pre-calculada. |
| PDFs | 11 documentos sintéticos (partes policiales, facturas, declaraciones) vinculados al dataset por `SIN-XXXX`. |

Todos los datos son **100% sintéticos**. Ningún dato corresponde a personas o siniestros reales.

---

## Modelo de IA

**Algoritmo:** XGBoost (clasificación binaria supervisada, gradient boosting)

**Features:** 16 variables derivadas de los datos crudos, incluyendo ratios de montos, indicadores binarios de señales del documento y el score del motor de reglas.

**Umbral de decisión:** 0.3 (ajustado al negocio — prioriza recall sobre precision para minimizar fraudes no detectados)

**Agente conversacional:** LLaMA 3.3 70B vía Groq API. Responde las 12 preguntas de la sección 12 del documento del reto en lenguaje natural, con contexto del dataset analizado.

**Explicabilidad:** SHAP (TreeExplainer). Cada alerta incluye los factores que más contribuyeron al score, en texto legible por el analista.

---

## Limitaciones

- El modelo fue entrenado con datos sintéticos. El rendimiento en datos reales puede variar.
- La similitud narrativa (`similitud_narrativa_max`) solo está disponible para 500 de los 1000 registros.
- Las reglas de negocio son referenciales. Deben ser validadas y ajustadas con expertos de la aseguradora antes de cualquier uso en producción.
- El sistema no toma decisiones automáticas de pago o rechazo. Es una herramienta de priorización para el analista.
- Los falsos positivos son esperables. Todo caso marcado debe pasar por revisión humana.

---

## Seguridad y ética

- No se usan datos personales reales ni información confidencial.
- Las API keys no están en el repositorio (usar `.env`).
- El lenguaje de las alertas es siempre de "posible fraude" o "requiere revisión", nunca de acusación.
- El sistema está diseñado para apoyar al analista, no para reemplazarlo.

---

## Equipo

hackIAthon 2026 — Reto Aseguradora del Sur
