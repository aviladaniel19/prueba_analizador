"""
Módulo de servicios.
Actúa como puente entre los endpoints de la API y las clases de limpieza.
"""

import uuid
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from fastapi import HTTPException
from scipy import stats as scipy_stats

from .store import datasets_db, metadata_db
from .schemas import (
    StatsResult,
    ColumnaInfo,
    ImputacionResult,
    DetalleColumnaImputada,
    RecomendacionResult,
    ExploracionResult,
    EstadisticaColumna,
)
from limpieza_datos import (
    AnalizadorDatos,
    ImputadorAvanzado,
    LimpiadorCompleto,
    EvaluadorMetodos,
)


def create_dataset(df: pd.DataFrame, name: str, description: Optional[str] = None) -> str:
    """Crea un nuevo dataset y devuelve su ID."""
    dataset_id = str(uuid.uuid4())
    datasets_db[dataset_id] = df
    metadata_db[dataset_id] = {"name": name, "description": description}
    return dataset_id


def delete_dataset(dataset_id: str) -> bool:
    """Elimina un dataset de la memoria."""
    if dataset_id in datasets_db:
        del datasets_db[dataset_id]
        if dataset_id in metadata_db:
            del metadata_db[dataset_id]
        return True
    return False


def get_stats(dataset_id: str) -> StatsResult:
    """Obtiene estadísticas del dataset usando AnalizadorDatos."""
    df = datasets_db[dataset_id]
    analizador = AnalizadorDatos(df)
    reporte = analizador.generar_reporte_completo()

    columnas_api = []
    for col in df.columns:
        serie = df[col]
        columnas_api.append(ColumnaInfo(
            nombre=col,
            tipo_detectado=analizador._detectar_tipo_variable(col),
            n_nulos=int(serie.isna().sum()),
            pct_nulos=float((serie.isna().sum() / len(df)) * 100),
            unicos=int(serie.nunique())
        ))

    return StatsResult(
        dataset_id=dataset_id,
        total_filas=len(df),
        total_columnas=len(df.columns),
        columnas=columnas_api,
        reporte_nulos=reporte['datos_faltantes'].to_dict(orient='records')
    )


def get_exploracion(dataset_id: str) -> ExploracionResult:
    """Genera análisis exploratorio descriptivo completo por columna."""
    df = datasets_db[dataset_id]
    analizador = AnalizadorDatos(df)
    estadisticas = []

    for col in df.columns:
        serie = df[col]
        tipo = analizador._detectar_tipo_variable(col)
        n_nulos = int(serie.isna().sum())
        pct_nulos = float((n_nulos / len(df)) * 100)
        serie_clean = serie.dropna()

        if pd.api.types.is_numeric_dtype(serie) and tipo not in ('booleanas',):
            # Estadísticas numéricas
            try:
                skew = float(scipy_stats.skew(serie_clean))
                kurt = float(scipy_stats.kurtosis(serie_clean))
            except Exception:
                skew, kurt = None, None

            # Histograma: 15 bins
            try:
                counts, edges = np.histogram(serie_clean, bins=15)
                distribucion = [
                    {"bin": f"{edges[i]:.2f}–{edges[i+1]:.2f}", "count": int(counts[i])}
                    for i in range(len(counts))
                ]
            except Exception:
                distribucion = []

            estadisticas.append(EstadisticaColumna(
                columna=col,
                tipo=tipo,
                count=int(serie_clean.count()),
                media=float(serie_clean.mean()),
                mediana=float(serie_clean.median()),
                desv_std=float(serie_clean.std()),
                minimo=float(serie_clean.min()),
                maximo=float(serie_clean.max()),
                q25=float(serie_clean.quantile(0.25)),
                q75=float(serie_clean.quantile(0.75)),
                skewness=skew,
                kurtosis=kurt,
                n_nulos=n_nulos,
                pct_nulos=pct_nulos,
                top_valores=None,
                distribucion=distribucion,
            ))

        elif tipo in ('categoricas_nominales', 'categoricas_ordinales', 'booleanas', 'texto'):
            # Frecuencias top 10
            top = serie_clean.value_counts().head(10)
            top_valores = [
                {"valor": str(k), "frecuencia": int(v), "pct": float(v / len(serie_clean) * 100)}
                for k, v in top.items()
            ]
            distribucion = top_valores  # reutilizamos para gráfico de barras

            estadisticas.append(EstadisticaColumna(
                columna=col,
                tipo=tipo,
                count=int(serie_clean.count()),
                media=None, mediana=None, desv_std=None,
                minimo=None, maximo=None, q25=None, q75=None,
                skewness=None, kurtosis=None,
                n_nulos=n_nulos,
                pct_nulos=pct_nulos,
                top_valores=top_valores,
                distribucion=distribucion,
            ))

        elif tipo == 'fechas':
            serie_dt = pd.to_datetime(serie_clean, errors='coerce').dropna()
            estadisticas.append(EstadisticaColumna(
                columna=col,
                tipo=tipo,
                count=int(serie_dt.count()),
                media=None, mediana=None, desv_std=None,
                minimo=str(serie_dt.min()) if len(serie_dt) else None,
                maximo=str(serie_dt.max()) if len(serie_dt) else None,
                q25=None, q75=None, skewness=None, kurtosis=None,
                n_nulos=n_nulos,
                pct_nulos=pct_nulos,
                top_valores=None,
                distribucion=None,
            ))

    return ExploracionResult(
        dataset_id=dataset_id,
        total_filas=len(df),
        total_columnas=len(df.columns),
        estadisticas=estadisticas,
    )


def apply_imputation(dataset_id: str, metodo: str, columnas: List[str], params: Dict) -> ImputacionResult:
    """Aplica imputación y devuelve detalle por columna."""
    df = datasets_db[dataset_id]

    # Capturar nulos y estadísticas ANTES
    nulos_antes = {col: int(df[col].isna().sum()) for col in columnas if col in df.columns}
    media_antes = {}
    mediana_antes = {}
    for col in columnas:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            media_antes[col] = float(df[col].mean()) if df[col].notna().any() else None
            mediana_antes[col] = float(df[col].median()) if df[col].notna().any() else None

    imputador = ImputadorAvanzado(df)

    if metodo == 'knn':
        df_new = imputador.imputar_knn(columnas, **params)
    elif metodo == 'mice':
        df_new = imputador.imputar_mice(columnas, **params)
    elif metodo == 'interpolacion':
        df_new = imputador.imputar_interpolacion(columnas, **params)
    elif metodo == 'regresion':
        df_new = imputador.imputar_regresion(columnas)
    else:
        raise ValueError(f"Método {metodo} no soportado")

    datasets_db[dataset_id] = df_new

    # Detalle por columna
    detalle = []
    total_imputados = 0
    for col in columnas:
        if col not in df.columns:
            continue
        n_imp = nulos_antes.get(col, 0)
        total_imputados += n_imp
        media_d = float(df_new[col].mean()) if pd.api.types.is_numeric_dtype(df_new[col]) and df_new[col].notna().any() else None
        mediana_d = float(df_new[col].median()) if pd.api.types.is_numeric_dtype(df_new[col]) and df_new[col].notna().any() else None
        detalle.append(DetalleColumnaImputada(
            columna=col,
            n_imputados=n_imp,
            media_antes=media_antes.get(col),
            media_despues=media_d,
            mediana_antes=mediana_antes.get(col),
            mediana_despues=mediana_d,
        ))

    return ImputacionResult(
        dataset_id=dataset_id,
        metodo_aplicado=metodo,
        columnas_afectadas=columnas,
        total_imputados=total_imputados,
        nulos_restantes=int(df_new.isna().sum().sum()),
        detalle_por_columna=detalle,
    )


def compare_methods(dataset_id: str, columnas: List[str], porcentaje_nulos: float) -> RecomendacionResult:
    """Compara métodos y devuelve recomendaciones."""
    df = datasets_db[dataset_id]

    # Filtrar solo columnas numéricas
    columnas_numericas = [c for c in columnas if pd.api.types.is_numeric_dtype(df[c])]

    if not columnas_numericas:
        raise HTTPException(
            status_code=400,
            detail=f"Ninguna de las columnas seleccionadas es numérica. Columnas recibidas: {columnas}"
        )

    # El evaluador necesita un df SIN nulos en las columnas a evaluar
    df_clean = df[columnas_numericas].dropna()

    if len(df_clean) < 20:
        raise HTTPException(
            status_code=400,
            detail=f"No hay suficientes filas completas ({len(df_clean)}). Se necesitan al menos 20."
        )

    df_eval = df.loc[df_clean.index].copy()

    evaluador = EvaluadorMetodos(df_eval)
    comparacion = evaluador.comparar_metodos(columnas_numericas, porcentaje_nulos=porcentaje_nulos)

    if comparacion.empty or 'Columna' not in comparacion.columns:
        raise HTTPException(
            status_code=400,
            detail="No se pudieron evaluar los métodos. Intenta con otras columnas o reduce el % de nulos simulados."
        )

    recomendaciones = evaluador.recomendar_mejor_metodo(comparacion)

    return RecomendacionResult(
        dataset_id=dataset_id,
        recomendaciones=recomendaciones
    )
