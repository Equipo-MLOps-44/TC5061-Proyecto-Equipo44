# TC5061 - Proyecto MiBici | Equipo 44

Proyecto integrador de la materia de MLOps (TC5061), desarrollado a partir de los datos abiertos del sistema de bicicletas públicas MiBici, en Guadalajara, México.

El objetivo es construir un flujo reproducible de ingesta de datos, Feature Engineering, entrenamiento de modelos de Machine Learning y seguimiento de experimentos con MLflow.

**Repositorio:** https://github.com/Equipo-MLOps-44/TC5061-Proyecto-Equipo44

## 1. Preguntas del proyecto

El proyecto busca responder tres preguntas mediante diferentes técnicas de Machine Learning.

| Pregunta | Objetivo | Tipo de problema |
|---|---|---|
| P1 | Predecir cuántos viajes se iniciarán en una estación durante una hora futura. | Regresión |
| P2 | Estimar si una estación estará en riesgo de quedarse sin bicicletas o sin espacios disponibles. | Clasificación |
| P3 | Identificar grupos de estaciones que comparten patrones de utilización similares. | Clustering |

P1 busca apoyar la planeación de la demanda, P2 el rebalanceo de bicicletas y P3 el análisis de estaciones para la planeación del sistema.

## 2. Datos utilizados

**Fuente:** [Portal de datos abiertos de MiBici](https://mibici.net/es/datos-abiertos/).

El proyecto utiliza registros mensuales de viajes y el archivo de nomenclatura de estaciones, que contiene identificadores, ubicaciones y capacidades.

En la ejecución realizada se procesaron:

| Elemento | Cantidad |
|---|---:|
| Periodo | Octubre 2024 - septiembre 2026 |
| Archivos mensuales | 24 |
| Viajes procesados | 9,043,925 |
| Registros por estación y hora | 6,662,294 |
| Perfiles de estaciones | 472 |

La ingesta normaliza los nombres de columnas, las codificaciones y los campos de fecha. Los datos se procesan por lotes y se almacenan en formato Parquet.

El programa de descarga selecciona los 24 meses más recientes disponibles, por lo que el periodo puede cambiar en ejecuciones futuras.

No se incluyen datos individuales de MiBici en GitHub. Se utilizan las variables necesarias para el análisis y se evita publicar información personal en el repositorio.

### Versionamiento de datos

Se implementó un mecanismo de identificación de versiones mediante hashes SHA-256.

El módulo `data_version.py` calcula los hashes de los archivos CSV y del Parquet procesado, junto con sus cantidades de registros. Esta información se guarda en `references/data_manifest.json`.

La versión generada para la ejecución documentada es:

`e01da28b62056c69`

El manifiesto permite identificar y comprobar la integridad de los archivos utilizados sin incluir los datasets en Git. Para reconstruir exactamente esta versión es necesario conservar los archivos originales correspondientes al periodo indicado.

## 3. Estructura del repositorio

El repositorio utiliza la plantilla Cookiecutter Data Science.

```text
TC5061-Proyecto-Equipo44/
|
|-- data/
|   |-- external/
|   |-- interim/
|   |-- processed/
|   |-- raw/
|
|-- notebooks/
|-- references/
|   |-- data_manifest.json
|-- reports/
|   |-- figures/
|-- models/
|-- docs/
|
|-- tc5061_proyecto_equipo_44/
|   |-- config.py
|   |-- dataset.py
|   |-- data_version.py
|   |-- features.py
|   |-- modeling/
|       |-- train.py
|       |-- predict.py
|
|-- requirements.txt
|-- pyproject.toml
|-- .gitignore
|-- README.md
```

Las carpetas `data/`, `models/` y `reports/` separan los datos, los modelos y los resultados. La lógica principal de procesamiento y entrenamiento se encuentra en el módulo `tc5061_proyecto_equipo_44/`.

La carpeta `notebooks/` se utiliza para el análisis exploratorio, mientras que `references/` contiene materiales de apoyo y metadatos del proyecto.

## 4. Instalación

Se requiere Python, Git y `uv`.

**Instalar `uv` en Windows** (si el comando `uv` no está disponible):

```powershell
winget install --id astral-sh.uv -e
```

Cierra y vuelve a abrir PowerShell y comprueba la instalación:

```powershell
uv --version
```

**Clonar el repositorio:**

```powershell
git clone https://github.com/Equipo-MLOps-44/TC5061-Proyecto-Equipo44.git
cd TC5061-Proyecto-Equipo44
```

**Crear y activar el entorno virtual (PowerShell):**

```powershell
uv venv --python 3.10
.\.venv\Scripts\Activate.ps1
```

**Instalar las dependencias**, incluido MLflow:

```powershell
uv pip install -r requirements.txt
```

Confirma que MLflow quedó disponible en el entorno activo:

```powershell
python -m mlflow --version
```

`requirements.txt` instala el proyecto en modo editable y contiene MLflow; no hace falta instalarlo por separado.

## 5. Ejecución del pipeline

Los comandos se ejecutan desde la carpeta principal del repositorio.

### Ejecutar el MLproject con MLflow

El archivo `MLproject` define los entry points `ingest`, `features` y `train`.
`python_env.yaml` especifica Python 3.10 y las dependencias del proyecto para MLflow;
los comandos de abajo usan el entorno virtual ya instalado y activado.

```powershell
python -m mlflow run . -e ingest --env-manager=local --experiment-name mibici-equipo44
python -m mlflow run . -e features --env-manager=local --experiment-name mibici-equipo44
python -m mlflow run . -e train --env-manager=local --experiment-name mibici-equipo44 -P question=p1 -P model=baseline
```

Para ejecutar otra pregunta, sustituye `p1` por `p2` o `p3`. P3 requiere `model=kmeans`:

```powershell
python -m mlflow run . -e train --env-manager=local --experiment-name mibici-equipo44 -P question=p3 -P model=kmeans
```

`--env-manager=local` ejecuta los entry points en el entorno virtual activo. Mantén ese entorno
activado en la terminal donde corres estos comandos. `--experiment-name mibici-equipo44` asigna
tanto la ejecución de MLflow Projects como sus ejecuciones anidadas al experimento del proyecto;
sin esta opción, MLflow Projects puede usar `Default`.

### 5.1. Ingesta

```powershell
uv run --no-sync python -m tc5061_proyecto_equipo_44.dataset
```

Descarga y procesa los registros de MiBici y genera:

`data/processed/dataset.parquet`

El código también incorpora la generación del manifiesto de versionamiento al finalizar la ingesta.

Para generar el manifiesto por separado, utilizando datos ya descargados:

```powershell
uv run --no-sync python -m tc5061_proyecto_equipo_44.data_version
```

### 5.2. Feature Engineering

```powershell
uv run --no-sync python -m tc5061_proyecto_equipo_44.features
```

Este proceso genera:

- `data/processed/hourly_features.parquet`: información por estación y hora para P1 y P2.
- `data/processed/station_profiles.parquet`: características agregadas de las estaciones para P3.

Las transformaciones se encuentran implementadas en `features.py` como funciones reutilizables.

Para P2 se aproxima el inventario a partir de la capacidad de cada estación y del flujo de salidas y llegadas. Se supone inicialmente un inventario equivalente al 75 % de la capacidad y se utilizan umbrales del 20 % para generar las etiquetas de riesgo.

Estas etiquetas son estimadas, ya que los datos disponibles no contienen mediciones reales de bicicletas o espacios libres.

### 5.3. Entrenamiento de modelos

**P1 - Regresión:**

```powershell
uv run --no-sync python -c "from tc5061_proyecto_equipo_44.modeling.train import train_p1; train_p1()"
```

**P2 - Clasificación:**

```powershell
uv run --no-sync python -c "from tc5061_proyecto_equipo_44.modeling.train import train_p2; train_p2()"
```

**P3 - Clustering:**

```powershell
uv run --no-sync python -c "from tc5061_proyecto_equipo_44.modeling.train import train_p3; train_p3()"
```

Los modelos baseline utilizados son:

| Pregunta | Modelo | Métricas de evaluación |
|---|---|---|
| P1 | DummyRegressor | MAE y RMSE |
| P2 | DummyClassifier | Accuracy, F1 y Recall |
| P3 | KMeans | Silhouette Score |

Para P1 y P2 se utiliza una división cronológica entre entrenamiento y prueba, con el objetivo de evitar incorporar información futura en el entrenamiento.

En P3 se utilizan perfiles históricos agregados para identificar grupos de estaciones.

## 6. Experimentos con MLflow

El seguimiento de experimentos se realiza mediante MLflow Tracking.

**Nombre del experimento:** `mibici-equipo44`

Se registraron cuatro ejecuciones baseline: una para P1, dos para P2 (una por cada variable de riesgo) y una para P3.

Los runs incluyen parámetros, métricas, etiquetas y artefactos, así como referencias al código utilizado. El registro explícito de la versión del manifiesto de datos en MLflow queda pendiente.

Para consultar la interfaz:

```powershell
uv run --no-sync mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Abrir en el navegador:

http://127.0.0.1:5000

Los archivos locales de experimentos y modelos no se incluyen en GitHub.

## 7. Buenas prácticas y reproducibilidad

El proyecto aplica las siguientes prácticas:

- **Separación del código:** ingesta, características y entrenamiento se implementan en módulos independientes.
- **Configuración:** las rutas y variables compartidas se administran mediante `config.py`.
- **Datos:** los archivos originales y procesados se mantienen separados.
- **Versionamiento:** se utiliza un manifiesto con hashes SHA-256 para identificar los datos procesados.
- **Evaluación temporal:** P1 y P2 utilizan una partición cronológica para reducir el riesgo de fuga de información futura.
- **Experimentos:** MLflow permite consultar parámetros, métricas, etiquetas y artefactos.
- **Git:** `.gitignore` excluye datos, modelos serializados, archivos locales de MLflow y credenciales.

Los archivos generados localmente no deben incorporarse al historial de Git.

## 8. Flujo de trabajo con Git y GitHub

El equipo desarrolla sus cambios mediante ramas independientes y Pull Requests.

El flujo de trabajo consiste en:

1. Actualizar la rama compartida `develop`.
2. Crear una rama para la funcionalidad o corrección.
3. Implementar y probar los cambios.
4. Crear commits descriptivos.
5. Subir la rama a GitHub.
6. Abrir un Pull Request hacia `develop`.
7. Revisar e integrar los cambios.

Se utilizan nombres descriptivos para las ramas, como `fix/modeling-dependencies` y `docs/project-documentation`, así como convenciones de commits como `feat:`, `fix:` y `docs:`.

La actividad establece que la rama principal debe estar protegida y que cada integrante debe abrir al menos un Pull Request y realizar al menos una revisión. Estas condiciones deben comprobarse en GitHub antes de la entrega.

## 9. Integrantes del equipo

| Integrante | Matrícula | Responsabilidad principal |
|---|---|---|
| Axel Dair Torrecilla Hernández | A01797810 | Desarrollo y entrenamiento de los modelos baseline |
| Guillermo Navarro García | A01282490 | Feature Engineering para P1, P2 y P3 |
| José Andrés Orantes Guillén | A01174130 | Análisis exploratorio de datos (EDA) |
| José Luis Navarro Hernandez | A01840092 | Ingesta, normalización y preparación de datos |
