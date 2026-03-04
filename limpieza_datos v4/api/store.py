"""
Módulo de almacenamiento en memoria para los datasets.
Simula una base de datos simple (CRUD).
"""

from typing import Dict
import pandas as pd

# Diccionario global para almacenar DataFrames por ID
# Key: dataset_id (str), Value: pd.DataFrame
datasets_db: Dict[str, pd.DataFrame] = {}

# Almacén de metadatos
metadata_db: Dict[str, dict] = {}
