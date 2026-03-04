# 🧹 Módulo de Limpieza Avanzada de Datos (con FastAPI)

Paquete Python modular y reutilizable para limpieza de datos, ahora con una **API REST profesional** integrada.

## 🚀 Fases del Proyecto (Parámetros del Curso)

El código incluye anotaciones para identificar las fases implementadas:
1. **Fase 1**: Setup (Entornos, `requirements.txt`)
2. **Fase 2**: Modelos Pydantic (Validación)
3. **Fase 4**: Routing, CRUD y Decoradores (`GET`, `POST`, `DELETE`)
4. **Fase 5**: Síncrono/Asíncrono y Servidor Uvicorn

## 📦 Instalación y Ejecución de la API

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Ejecutar la API
python -m api.main
```

Una vez ejecutado, accede a la documentación interactiva en:
👉 **Swagger UI:** [http://localhost:8000/docs](http://localhost:8000/docs)

## 📡 Endpoints Disponibles

| Verbo | Ruta | Acción |
|-------|------|--------|
| `POST` | `/datasets/` | Subir CSV y crear recurso |
| `GET` | `/datasets/` | Listar todos los datasets |
| `GET` | `/datasets/{id}` | Obtener info de un dataset |
| `DELETE` | `/datasets/{id}` | Eliminar dataset |
| `POST` | `/datasets/{id}/analizar` | Análisis estadístico detallado |
| `POST` | `/datasets/{id}/imputar` | Ejecutar imputación avanzada |
| `POST` | `/datasets/{id}/comparar` | Comparar métodos (tests estadísticos) |
| `GET` | `/datasets/{id}/descargar` | Descargar archivo procesado |

## 🔧 Uso como Librería (Pipeline Local)

```python
import pandas as pd
from limpieza_datos import LimpiadorCompleto

# Cargar tus datos (CSV, Excel, SQL, etc.)
df = pd.read_csv('mi_base_de_datos.csv')
# df = pd.read_excel('mi_base.xlsx')
# df = pd.read_sql('SELECT * FROM tabla', conexion)

# Ejecutar pipeline completo
limpiador = LimpiadorCompleto(df)
df_limpio = limpiador.pipeline_completo(metodo_imputacion='knn')
```

## 🔧 Uso por Módulos Individuales

### 1. Analizar datos

```python
from limpieza_datos import AnalizadorDatos

analizador = AnalizadorDatos(df)

# Clasificar tipos de variables
clasificacion = analizador.identificar_tipos_variables()

# Analizar datos faltantes
faltantes = analizador.analizar_datos_faltantes()

# Reporte completo
reporte = analizador.generar_reporte_completo()
```

### 2. Imputar valores faltantes

```python
from limpieza_datos import ImputadorAvanzado

imputador = ImputadorAvanzado(df)

# Opción A: KNN
df_limpio = imputador.imputar_knn(['col1', 'col2'], n_neighbors=5)

# Opción B: MICE
df_limpio = imputador.imputar_mice(['col1', 'col2'], max_iter=10)

# Opción C: Interpolación
df_limpio = imputador.imputar_interpolacion(['col1'], metodo='linear')

# Opción D: Regresión lineal
df_limpio = imputador.imputar_regresion(['col1', 'col2'])

# Eliminar categóricos nulos
df_limpio = imputador.eliminar_categoricos_nulos(umbral_porcentaje=50)
```

### 3. Corregir formatos de fecha

```python
from limpieza_datos import CorreccionFormatos

corrector = CorreccionFormatos(df)

# Detección automática
df_corregido = corrector.corregir_fechas()

# Con formato específico
df_corregido = corrector.corregir_fechas(
    columnas=['fecha_registro'],
    formato='%d/%m/%Y'
)
```

## ⚙️ Opciones del Pipeline Completo

```python
df_limpio = limpiador.pipeline_completo(
    metodo_imputacion='knn',          # 'knn', 'mice', 'interpolacion', 'regresion'
    eliminar_nulos_categoricos=True,
    corregir_fechas=True,
    n_neighbors=5,                     # Para KNN
    max_iter=10,                       # Para MICE
    metodo_interpolacion='linear',     # Para interpolación
    umbral_categoricos=50.0,           # Umbral % para eliminar columnas
)
```

## 🔬 Evaluación y Selección de Método de Imputación

### 4. Comparar métodos estadísticamente

```python
from limpieza_datos import EvaluadorMetodos

evaluador = EvaluadorMetodos(df_completo)  # DataFrame sin nulos (referencia)

# Comparar KNN, MICE, regresión e interpolación
comparacion = evaluador.comparar_metodos(
    columnas=['col1', 'col2'],
    porcentaje_nulos=20.0,
    metodos=['knn', 'mice', 'regresion', 'interpolacion'],
)

# Obtener recomendación automática
recomendaciones = evaluador.recomendar_mejor_metodo(comparacion)
```

### 5. Visualizar resultados

```python
from limpieza_datos import (
    visualizar_comparacion_metodos,
    visualizar_distribucion_errores,
    crear_heatmap_metricas,
)

# Barras comparativas (MAE, RMSE, R², MAPE)
visualizar_comparacion_metodos(comparacion)

# Heatmap de métricas
crear_heatmap_metricas(comparacion)

# Distribución de errores (scatter, histograma, residuales)
visualizar_distribucion_errores(
    df_original, df_imputado, 'col1', 'knn', mascara_nulos
)

# Guardar figuras a disco
visualizar_comparacion_metodos(comparacion, guardar_como='comparacion.png')
```

## 📁 Estructura del Paquete

```
limpieza_datos/
├── __init__.py            # API pública
├── analizador.py          # AnalizadorDatos
├── imputador.py           # ImputadorAvanzado
├── formatos.py            # CorreccionFormatos
├── limpiador.py           # LimpiadorCompleto (orquestador)
├── evaluador.py           # EvaluadorMetodos (tests estadísticos)
└── visualizaciones.py     # Funciones de visualización
```

## 📊 Bases de Datos Compatibles

| Fuente         | Cómo cargar                                  |
| -------------- | -------------------------------------------- |
| CSV            | `pd.read_csv('archivo.csv')`                 |
| Excel          | `pd.read_excel('archivo.xlsx')`              |
| SQL            | `pd.read_sql('SELECT * FROM t', conexion)`   |
| JSON           | `pd.read_json('archivo.json')`               |
| Parquet        | `pd.read_parquet('archivo.parquet')`          |
| Google Sheets  | `pd.read_csv(url_publica)`                   |

