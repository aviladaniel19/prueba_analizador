"""
Módulo Principal de la API.
"""

import io
import pandas as pd
from typing import List, Dict, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.responses import StreamingResponse

# FASE 4: Importación de Pydantic y Schemas
from .schemas import (
    DatasetInput,
    DatasetInfo,
    StatsResult,
    ImputacionRequest,
    ImputacionResult,
    ComparacionRequest,
    RecomendacionResult,
)
from . import services
from .store import datasets_db, metadata_db

# FASE 5: Síncrono vs Asíncrono (FastAPI gestiona el loop de eventos asíncrono)
app = FastAPI(
    title="API de Limpieza Avanzada de Datos",
    description="API para análisis, imputación y limpieza de datasets.",
    version="1.0.0"
)

# FASE 4: Decoradores @app.get, @app.post, @app.delete

@app.post("/datasets/", status_code=status.HTTP_201_CREATED, response_model=Dict)
async def upload_dataset(name: str, description: Optional[str] = None, file: UploadFile = File(...)):
    """
    FASE 2 & 4: Subida de archivos y creación de recursos (CRUD).
    Recibe un archivo CSV y lo almacena en memoria.
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Solo se admiten archivos .csv")

    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))
    
    dataset_id = services.create_dataset(df, name, description)
    return {"id": dataset_id, "message": "Dataset creado correctamente"}


@app.get("/datasets/", response_model=List[Dict])
async def list_datasets():
    """FASE 4: Verbo GET para listar recursos."""
    return [
        {"id": k, "name": metadata_db[k]["name"], "rows": len(datasets_db[k])}
        for k in datasets_db.keys()
    ]


@app.get("/datasets/{dataset_id}", response_model=Dict, status_code=status.HTTP_200_OK)
async def get_dataset_info(dataset_id: str):
    """FASE 4: GET con parámetros de ruta."""
    if dataset_id not in datasets_db:
        raise HTTPException(status_code=404, detail="Dataset no encontrado")
    
    df = datasets_db[dataset_id]
    meta = metadata_db[dataset_id]
    
    return {
        "id": dataset_id,
        "name": meta["name"],
        "description": meta["description"],
        "shape": df.shape,
        "nulls": int(df.isna().sum().sum())
    }


@app.delete("/datasets/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_dataset(dataset_id: str):
    """FASE 4: Verbo DELETE para eliminar recursos."""
    if not services.delete_dataset(dataset_id):
        raise HTTPException(status_code=404, detail="Dataset no encontrado")
    return None


@app.post("/datasets/{dataset_id}/analizar", response_model=StatsResult)
async def analyze_dataset(dataset_id: str):
    """FASE 4: POST para ejecutar procesos."""
    if dataset_id not in datasets_db:
        raise HTTPException(status_code=404, detail="Dataset no encontrado")
    
    return services.get_stats(dataset_id)


@app.post("/datasets/{dataset_id}/imputar", response_model=ImputacionResult)
async def impute_data(dataset_id: str, request: ImputacionRequest):
    """FASE 2 & 4: Uso de modelos Pydantic en el body del POST."""
    if dataset_id not in datasets_db:
        raise HTTPException(status_code=404, detail="Dataset no encontrado")
    
    try:
        return services.apply_imputation(
            dataset_id, 
            request.metodo, 
            request.columnas, 
            request.params
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/datasets/{dataset_id}/comparar", response_model=RecomendacionResult)
async def compare_imputation_methods(dataset_id: str, request: ComparacionRequest):
    """FASE 4: Endpoint para comparación estadística."""
    if dataset_id not in datasets_db:
        raise HTTPException(status_code=404, detail="Dataset no encontrado")
    
    return services.compare_methods(dataset_id, request.columnas, request.porcentaje_nulos)


@app.get("/datasets/{dataset_id}/descargar")
async def download_dataset(dataset_id: str):
    """FASE 4: GET para descargar archivos procesados."""
    if dataset_id not in datasets_db:
        raise HTTPException(status_code=404, detail="Dataset no encontrado")
    
    df = datasets_db[dataset_id]
    stream = io.StringIO()
    df.to_csv(stream, index=False)
    
    response = StreamingResponse(
        iter([stream.getvalue()]),
        media_type="text/csv"
    )
    response.headers["Content-Disposition"] = f"attachment; filename=dataset_{dataset_id}_limpio.csv"
    return response


# FASE 5: Ejecución con Uvicorn
if __name__ == "__main__":
    import uvicorn
    # FASE 5: Uvicorn server
    uvicorn.run(app, host="0.0.0.0", port=8000)
