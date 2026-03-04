"""
Módulo de esquemas de datos usando Pydantic
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class DatasetInput(BaseModel):
    """Esquema para la creación/subida de un dataset."""
    name: str = Field(..., example="Dataset Vendedores")
    description: Optional[str] = Field(None, example="Datos de ventas del Q1")


class DatasetInfo(BaseModel):
    """Información resumida de un dataset."""
    id: str
    name: str
    rows: int


class ColumnaInfo(BaseModel):
    """Información básica de una columna."""
    nombre: str
    tipo_detectado: str
    n_nulos: int
    pct_nulos: float
    unicos: int


class StatsResult(BaseModel):
    """Modelo para resultados estadísticos del dataset."""
    dataset_id: str
    total_filas: int
    total_columnas: int
    columnas: List[ColumnaInfo]
    reporte_nulos: List[Dict]


class ImputacionRequest(BaseModel):
    """Petición de imputación para un dataset."""
    metodo: str = Field(..., example="knn")
    columnas: List[str] = Field(..., example=["edad", "ingreso"])
    params: Optional[Dict] = Field(default={}, example={"n_neighbors": 5})


class DetalleColumnaImputada(BaseModel):
    """Detalle de imputación por columna."""
    columna: str
    n_imputados: int
    media_antes: Optional[float]
    media_despues: Optional[float]
    mediana_antes: Optional[float]
    mediana_despues: Optional[float]


class ImputacionResult(BaseModel):
    """Resultado de la operación de imputación con detalle por columna."""
    dataset_id: str
    metodo_aplicado: str
    columnas_afectadas: List[str]
    total_imputados: int
    nulos_restantes: int
    detalle_por_columna: List[DetalleColumnaImputada]


class PipelineRequest(BaseModel):
    """Petición para ejecutar el pipeline completo."""
    metodo_imputacion: str = "knn"
    eliminar_nulos_categoricos: bool = True
    corregir_fechas: bool = True


class ComparacionRequest(BaseModel):
    """Petición para comparar métodos."""
    columnas: List[str]
    porcentaje_nulos: float = 20.0


class RecomendacionResult(BaseModel):
    """Resultado de la recomendación del mejor método."""
    dataset_id: str
    recomendaciones: Dict[str, Dict]


class EstadisticaColumna(BaseModel):
    """Estadísticas descriptivas de una columna."""
    columna: str
    tipo: str
    count: int
    media: Optional[float] = None
    mediana: Optional[float] = None
    desv_std: Optional[float] = None
    minimo: Optional[Any] = None
    maximo: Optional[Any] = None
    q25: Optional[float] = None
    q75: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    n_nulos: int
    pct_nulos: float
    top_valores: Optional[List[Dict]] = None
    distribucion: Optional[List[Dict]] = None


class ExploracionResult(BaseModel):
    """Resultado del análisis exploratorio descriptivo."""
    dataset_id: str
    total_filas: int
    total_columnas: int
    estadisticas: List[EstadisticaColumna]
