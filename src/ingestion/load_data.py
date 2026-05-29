"""
Ingesta y merge de las tres fuentes de datos del proyecto fraudia-claims.

Fuentes:
  1. siniestros_sintetico.csv       → base de entrenamiento (1000 registros, tiene etiqueta)
  2. Evento_Datasets_Sinteticos_Fraude_500_v2.xlsx → tablas normalizadas del organizador
  3. PDFs en subcarpetas            → referenciados en la columna nombre_archivo_pdf

Salida:
  data/processed/siniestros_merged.csv → dataset listo para build_features.py y XGBoost

Notas sobre el merge:
  - IDs de siniestro: distintos formatos (SIN-000001 vs SIN-0001), mapeables por número.
    500 de 1000 registros de la compañera coinciden con el organizador.
  - IDs de proveedor: sistemas completamente distintos (PRV-XXXX vs TALLER-XXX),
    no se puede hacer join. Se conservan los datos de proveedor de la compañera.
  - PDFs: se registra cuáles siniestros tienen documentos PDF vinculados.
"""

import os
import re
import pandas as pd


# ---------------------------------------------------------------------------
# Rutas — relativas a la raíz del repositorio
# ---------------------------------------------------------------------------
RUTA_BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RUTA_CSV_COMPANERA   = os.path.join(RUTA_BASE, "data", "synthetic", "siniestros_sintetico.csv")
RUTA_EXCEL_ORG       = os.path.join(RUTA_BASE, "data", "synthetic", "Evento_Datasets_Sinteticos_Fraude_500_v2.xlsx")
RUTA_CARPETA_PDFS    = os.path.join(RUTA_BASE, "data", "synthetic")
RUTA_SALIDA          = os.path.join(RUTA_BASE, "data", "processed", "siniestros_merged.csv")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalizar_id_siniestro(serie: pd.Series) -> pd.Series:
    """Convierte SIN-000001 o SIN-0001 al formato canónico SIN-XXXX (4 dígitos)."""
    numeros = serie.str.extract(r"(\d+)", expand=False).astype(int)
    return numeros.apply(lambda n: f"SIN-{n:04d}")


def _escanear_pdfs(carpeta: str) -> dict[str, list[str]]:
    """
    Recorre las subcarpetas buscando archivos PDF.
    Extrae el SIN-XXXX del nombre del archivo y devuelve un dict
    { sin_norm: [nombre_pdf1, nombre_pdf2, ...] }
    """
    patron_sin = re.compile(r"SIN-(\d+)", re.IGNORECASE)
    pdfs_por_siniestro: dict[str, list[str]] = {}

    for raiz, _, archivos in os.walk(carpeta):
        for archivo in archivos:
            if not archivo.lower().endswith(".pdf"):
                continue
            match = patron_sin.search(archivo)
            if match:
                sin_norm = f"SIN-{int(match.group(1)):04d}"
                pdfs_por_siniestro.setdefault(sin_norm, []).append(archivo)

    return pdfs_por_siniestro


# ---------------------------------------------------------------------------
# Carga de fuentes
# ---------------------------------------------------------------------------

def cargar_companera() -> pd.DataFrame:
    """
    Carga el CSV base de la compañera y normaliza el ID de siniestro.
    También estandariza nombres de columna para que coincidan con
    los esperados por fraud_rules.py y build_features.py.
    """
    df = pd.read_csv(RUTA_CSV_COMPANERA)
    df["sin_norm"] = _normalizar_id_siniestro(df["id_siniestro"])

    # Mapeo de nombres del CSV de la compañera al estándar del proyecto
    mapeo_columnas = {
        "documento_inconsistente": "inconsistencia_detectada",
    }
    df.rename(columns={k: v for k, v in mapeo_columnas.items() if k in df.columns}, inplace=True)

    print(f"[1] Compañera cargada:     {len(df):>5} registros | {len(df.columns)} columnas")
    return df


def cargar_organizador() -> dict[str, pd.DataFrame]:
    """
    Carga las hojas relevantes del Excel del organizador.
    Retorna un dict con claves: siniestros, proveedores, documentos.
    """
    xl = pd.ExcelFile(RUTA_EXCEL_ORG)

    # Hoja Siniestros — solo columnas que aportan algo nuevo
    df_sin = xl.parse("1_Siniestros", usecols=[
        "ID Siniestro",
        "Similitud Narrativa Máx.",
        "Número Parte Policial",
    ])
    df_sin.rename(columns={
        "ID Siniestro":            "sin_norm_org",
        "Similitud Narrativa Máx.": "similitud_narrativa_max",
        "Número Parte Policial":    "numero_parte_policial",
    }, inplace=True)
    df_sin["sin_norm"] = _normalizar_id_siniestro(df_sin["sin_norm_org"])
    df_sin.drop(columns=["sin_norm_org"], inplace=True)

    # Hoja Documentos — agregada por siniestro
    df_docs = xl.parse("5_Documentos", usecols=[
        "ID Siniestro",
        "Tipo Documento",
        "Nombre Archivo PDF",
    ])
    df_docs.rename(columns={"ID Siniestro": "sin_norm_org"}, inplace=True)
    df_docs["sin_norm"] = _normalizar_id_siniestro(df_docs["sin_norm_org"])
    df_docs.drop(columns=["sin_norm_org"], inplace=True)

    print(f"[2] Organizador cargado:   {len(df_sin):>5} siniestros | {len(df_docs):>5} documentos en tabla")
    return {"siniestros": df_sin, "documentos": df_docs}


# ---------------------------------------------------------------------------
# Transformaciones intermedias
# ---------------------------------------------------------------------------

def _agregar_documentos(df_docs: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega la tabla de documentos por siniestro.
    Genera columnas:
      - total_docs_tabla: total de registros de documentos en la tabla
      - tiene_pdf_fisico: si existe al menos un PDF en la columna Nombre Archivo PDF
      - tipos_documentos: lista de tipos como texto (para el agente)
    """
    agg = df_docs.groupby("sin_norm").agg(
        total_docs_tabla=("Tipo Documento", "count"),
        tiene_pdf_fisico=("Nombre Archivo PDF", lambda x: int(x.notna().any())),
        tipos_documentos=("Tipo Documento", lambda x: " | ".join(sorted(set(x.dropna())))),
    ).reset_index()
    return agg


def _registrar_pdfs_fisicos(carpeta: str) -> pd.DataFrame:
    """
    Escanea las subcarpetas del dataset buscando PDFs y genera un DataFrame
    con sin_norm y la lista de archivos encontrados.
    """
    pdfs = _escanear_pdfs(carpeta)
    if not pdfs:
        return pd.DataFrame(columns=["sin_norm", "archivos_pdf", "count_pdfs_fisicos"])

    rows = [
        {
            "sin_norm":           sin_norm,
            "archivos_pdf":       " | ".join(archivos),
            "count_pdfs_fisicos": len(archivos),
        }
        for sin_norm, archivos in pdfs.items()
    ]
    df = pd.DataFrame(rows)
    print(f"[3] PDFs físicos:          {len(pdfs):>5} siniestros con PDF | {df['count_pdfs_fisicos'].sum()} archivos")
    return df


# ---------------------------------------------------------------------------
# Merge principal
# ---------------------------------------------------------------------------

def construir_dataset_merged() -> pd.DataFrame:
    """
    Une las tres fuentes y guarda el resultado en data/processed/.

    Columnas nuevas que se agregan a la base de la compañera:
      Del organizador (siniestros):
        similitud_narrativa_max   → score 0-1 de similitud con otros reclamos
        numero_parte_policial     → número de parte policial si existe

      Del organizador (documentos agregados):
        total_docs_tabla          → cuántos registros de docs tiene en la tabla
        tiene_pdf_fisico          → 1 si hay PDF referenciado en la tabla
        tipos_documentos          → tipos de documentos como texto

      De PDFs físicos escaneados:
        archivos_pdf              → nombres de los PDFs encontrados
        count_pdfs_fisicos        → cuántos PDFs físicos existen

    Nota: los 500 registros de la compañera que NO tienen match con el
    organizador quedarán con NaN en similitud_narrativa_max y numero_parte_policial.
    El modelo XGBoost maneja NaN nativamente.
    """
    print("\n=== INICIANDO MERGE ===\n")
    os.makedirs(os.path.dirname(RUTA_SALIDA), exist_ok=True)

    # 1. Cargar fuentes
    df_base        = cargar_companera()
    fuentes_org    = cargar_organizador()
    df_pdfs_fisicos = _registrar_pdfs_fisicos(RUTA_CARPETA_PDFS)

    # 2. Agregar documentos del organizador
    df_docs_agg = _agregar_documentos(fuentes_org["documentos"])

    # 3. Merge: base + siniestros organizador (left join — conserva los 1000)
    df = df_base.merge(
        fuentes_org["siniestros"],
        on="sin_norm",
        how="left",
    )
    matches_sin = df["similitud_narrativa_max"].notna().sum()
    print(f"\n[merge 1] Siniestros org → compañera: {matches_sin}/1000 con similitud_narrativa_max")

    # 4. Merge: + documentos agregados
    df = df.merge(df_docs_agg, on="sin_norm", how="left")
    matches_docs = df["total_docs_tabla"].notna().sum()
    print(f"[merge 2] Documentos org → compañera: {matches_docs}/1000 con info de documentos")

    # 5. Merge: + PDFs físicos (si existen)
    if not df_pdfs_fisicos.empty:
        df = df.merge(df_pdfs_fisicos, on="sin_norm", how="left")
        matches_pdfs = df["count_pdfs_fisicos"].notna().sum()
        print(f"[merge 3] PDFs físicos  → compañera: {matches_pdfs}/1000 con PDFs vinculados")
    else:
        df["archivos_pdf"]        = None
        df["count_pdfs_fisicos"]  = 0
        print("[merge 3] PDFs físicos: no se encontraron archivos en la carpeta")

    # 6. Limpiar columna auxiliar
    df.drop(columns=["sin_norm"], inplace=True)

    # 7. Rellenar NaN en columnas numéricas nuevas con 0
    df["total_docs_tabla"]    = df["total_docs_tabla"].fillna(0).astype(int)
    df["tiene_pdf_fisico"]    = df["tiene_pdf_fisico"].fillna(0).astype(int)
    df["count_pdfs_fisicos"]  = df["count_pdfs_fisicos"].fillna(0).astype(int)

    # 8. Guardar
    df.to_csv(RUTA_SALIDA, index=False)

    print(f"\n=== MERGE COMPLETO ===")
    print(f"Registros finales:  {len(df)}")
    print(f"Columnas finales:   {len(df.columns)}")
    print(f"Archivo guardado:   {RUTA_SALIDA}")
    print(f"\nColumnas nuevas añadidas:")
    nuevas = [
        "similitud_narrativa_max", "numero_parte_policial",
        "total_docs_tabla", "tiene_pdf_fisico", "tipos_documentos",
        "archivos_pdf", "count_pdfs_fisicos",
    ]
    for col in nuevas:
        no_nulos = df[col].notna().sum()
        print(f"  {col:<30} {no_nulos:>5} registros con valor")

    print(f"\nNota: similitud_narrativa_max tiene {df['similitud_narrativa_max'].isna().sum()} NaN")
    print("      (los 500 registros de la compañera sin match en el organizador)")
    print("      XGBoost maneja NaN nativamente — no requiere imputación.")

    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df_merged = construir_dataset_merged()
    print(f"\nPrimeras filas del dataset merged:")
    print(df_merged[["id_siniestro", "etiqueta_fraude_simulada",
                      "similitud_narrativa_max", "numero_parte_policial",
                      "total_docs_tabla", "count_pdfs_fisicos"]].head(10).to_string())
