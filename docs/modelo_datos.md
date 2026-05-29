# Modelo de Datos

## Origen del Dataset

El proyecto usa exclusivamente datos sintéticos. Ningún registro corresponde a personas, pólizas o siniestros reales.

### Dataset base — `siniestros_sintetico.csv`

Generado mediante una solicitud directa a Claude (Anthropic) usando como contexto el documento oficial del reto HackIAthon 2026, secciones 6 (estructura de tablas) y 7 (señales de fraude). Se indicó explícitamente generar 1000 registros respetando:

- Las columnas y tipos de datos definidos en la sección 6.1 del documento
- Una prevalencia de fraude de aproximadamente 14% (`etiqueta_fraude_simulada`)
- Distribuciones realistas de montos, fechas y coberturas para seguros en Ecuador
- Señales de fraude de la sección 7 presentes en los casos etiquetados (borde de vigencia, documentos incompletos, reporte tardío, etc.)

El resultado fue un CSV de 1000 filas y 51 columnas, validado contra la estructura del documento antes de usarse para entrenamiento.

### Dataset del organizador — `Evento_Datasets_Sinteticos_Fraude_500_v2.xlsx`

Provisto directamente por los organizadores del hackathon. Contiene 500 registros en 5 hojas normalizadas (Siniestros, Pólizas, Asegurados, Proveedores, Documentos) y PDFs sintéticos de soporte (partes policiales, facturas, declaraciones de accidente).

### Consideraciones sobre los datos sintéticos

Al ser generados con IA, los patrones de fraude simulados pueden no reflejar exactamente la realidad operativa de la aseguradora. El sistema debe reentrenarse con datos reales confirmados antes de cualquier uso en producción.

---

## Fuentes

El dataset final (`siniestros_merged.csv`) combina tres fuentes:

| Fuente | Registros | Descripción |
|--------|-----------|-------------|
| `siniestros_sintetico.csv` | 1000 | Base de entrenamiento con etiqueta de fraude |
| `Evento_Datasets_Sinteticos_Fraude_500_v2.xlsx` | 500 | Dataset del organizador, 5 hojas normalizadas |
| PDFs en subcarpetas | 11 archivos | Documentos vinculados por ID de siniestro |

El merge es un left join sobre el ID de siniestro normalizado (formato SIN-XXXX). Los 500 registros sin match en el organizador conservan sus datos originales con NaN en las columnas nuevas.

---

## Tablas del Excel del Organizador

### 1_Siniestros
Columnas usadas: `ID Siniestro`, `Similitud Narrativa Máx.`, `Número Parte Policial`

### 2_Pólizas
No se incorpora directamente — los datos relevantes ya están en el CSV de la compañera.

### 3_Asegurados
No se incorpora — idem.

### 4_Proveedores
No se puede hacer join por ID: el CSV usa `PRV-XXXX` y el Excel usa `TALLER-XXX`. Son sistemas distintos.

### 5_Documentos
Columnas usadas: `ID Siniestro`, `Tipo Documento`, `Inconsistencia Detectada`. Se agrega el conteo de documentos por siniestro y los tipos presentes.

---

## Dataset Merged — Columnas Principales

### Identificación
| Columna | Tipo | Descripción |
|---------|------|-------------|
| `id_siniestro` | str | ID original del CSV (SIN-000001) |
| `sin_norm` | str | ID normalizado para joins (SIN-0001) |

### Etiqueta
| Columna | Tipo | Descripción |
|---------|------|-------------|
| `etiqueta_fraude_simulada` | int (0/1) | 1 = posible fraude (simulado). 14.4% del dataset |

### Datos del Siniestro
| Columna | Tipo | Descripción |
|---------|------|-------------|
| `fecha_ocurrencia` | date | Fecha del evento |
| `fecha_reporte` | date | Fecha en que se reportó |
| `fecha_inicio_poliza` | date | Inicio de vigencia de la póliza |
| `fecha_fin_poliza` | date | Fin de vigencia de la póliza |
| `monto_reclamado` | float | Monto pedido por el asegurado |
| `monto_estimado` | float | Estimado por el perito |
| `suma_asegurada` | float | Límite de cobertura de la póliza |
| `ramo` | str | Tipo de seguro (Vehículos, Salud, Hogar…) |
| `cobertura` | str | Cobertura específica activada |
| `sucursal` | str | Oficina que gestiona el siniestro |

### Datos del Asegurado y Proveedor
| Columna | Tipo | Descripción |
|---------|------|-------------|
| `id_asegurado` | str | ID del asegurado |
| `historial_siniestros` | int | Cantidad de siniestros previos |
| `beneficiario` | str | Proveedor o taller asignado |
| `inconsistencia_detectada` | str | Si/No — hay inconsistencia documental |

### Columnas Añadidas por el Merge
| Columna | Tipo | Fuente | Cobertura |
|---------|------|--------|-----------|
| `similitud_narrativa_max` | float | Excel organizador | 500/1000 |
| `numero_parte_policial` | str | Excel organizador | 6/1000 |
| `total_docs_tabla` | int | Excel organizador (hoja 5) | 1000/1000 |
| `tipos_documentos` | str | Excel organizador | 500/1000 |
| `tiene_pdf_fisico` | bool | PDFs físicos | 1000/1000 |
| `count_pdfs_fisicos` | int | PDFs físicos | 1000/1000 |

---

## Features del Modelo (16 variables)

Las features se calculan en `build_features.py` a partir de las columnas crudas.

| Feature | Cálculo | Señal esperada |
|---------|---------|----------------|
| `ratio_monto_estimado` | monto_reclamado / monto_estimado | >1.2 es sospechoso |
| `ratio_monto_suma_asegurada` | monto_reclamado / suma_asegurada | Cercano a 1.0 es sospechoso |
| `monto_reclamado_normalizado` | log1p(monto_reclamado) | Valores extremos |
| `dias_desde_inicio_poliza` | fecha_ocurrencia - fecha_inicio | <30 días es sospechoso |
| `dias_entre_ocurrencia_reporte` | fecha_reporte - fecha_ocurrencia | >7 días es sospechoso |
| `es_borde_vigencia` | bool: ≤30 días desde inicio | Señal directa |
| `es_borde_vigencia_extremo` | bool: ≤2 días desde inicio | Señal fuerte |
| `reporte_tardio` | bool: >7 días hasta reporte | Señal directa |
| `sin_documentos` | bool: total_docs_tabla == 0 | Señal directa |
| `es_robo` | bool: cobertura contiene "robo" | Modifica otras señales |
| `historial_siniestros_asegurado` | historial_siniestros (raw) | >3 es sospechoso |
| `frecuencia_asegurado_alta` | bool: historial > umbral | Señal directa |
| `proveedor_lista_restrictiva` | bool: ID en lista negra | Señal crítica (RF-03) |
| `casos_observados_proveedor` | conteo por beneficiario | Concentración de alertas |
| `similitud_narrativa_max` | float 0-1 del organizador | >0.8 es sospechoso |
| `score_reglas` | calculado por fraud_rules.py | 0-100 |

---

## Notas de Calidad

- **NaN en similitud_narrativa_max**: 500 registros sin match en el organizador. XGBoost los maneja nativamente sin imputación.
- **Columna renombrada**: en el CSV de la compañera la columna es `documento_inconsistente`. En el merge se renombra a `inconsistencia_detectada` para consistencia con el documento del reto.
- **IDs de proveedor incompatibles**: no se puede hacer join entre `PRV-XXXX` y `TALLER-XXX`. Los datos de proveedor del organizador no se incorporan.
- **PDFs físicos**: solo 2 siniestros tienen PDF vinculado por nombre de archivo. La información de documentos viene principalmente de la hoja 5 del Excel.
