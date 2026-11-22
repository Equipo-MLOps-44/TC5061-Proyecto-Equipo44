
# TC5061 - Proyecto MiBici | Equipo 44

Proyecto integrador de la materia de MLOps (TC5061), desarrollado a partir de los datos abiertos del sistema de bicicletas públicas MiBici, en Guadalajara, México.

El objetivo es construir un flujo reproducible de ingesta de datos, *feature engineering*, entrenamiento de modelos de *machine learning*, validación de calidad de datos y seguimiento de experimentos con MLflow.

**Repositorio:** https://github.com/Equipo-MLOps-44/TC5061-Proyecto-Equipo44

## Guía rápida: levantar el sistema y ejecutar el pipeline

Resumen de punta a punta en PowerShell, desde la raíz del repositorio. Cada paso remite a la sección con el detalle.

**1. Preparar el entorno** (sección 4). Requiere Python 3.11, Git, `uv` y Podman.

```powershell
git clone https://github.com/Equipo-MLOps-44/TC5061-Proyecto-Equipo44.git
cd TC5061-Proyecto-Equipo44
uv venv --python 3.11
.\.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
```

**2. Ejecutar el pipeline de ML** (sección 5): ingesta → features → entrenamiento, registrado en MLflow.

```powershell
python -m mlflow run . -e ingest   --env-manager=local --experiment-name mibici-equipo44
python -m mlflow run . -e features --env-manager=local --experiment-name mibici-equipo44
python -m mlflow run . -e train    --env-manager=local --experiment-name mibici-equipo44 -P question=p1 -P model=rf
python -m mlflow run . -e train    --env-manager=local --experiment-name mibici-equipo44 -P question=p2 -P model=baseline
python -m mlflow run . -e train    --env-manager=local --experiment-name mibici-equipo44 -P question=p3 -P model=k_means
python -m pytest tests -v
```

La ingesta descarga ~9 millones de viajes, por lo que tarda varios minutos. Para revisar los experimentos: `python -m mlflow ui` y abrir http://127.0.0.1:5000.

**3. Levantar el sistema** (secciones 10 y 13): API de inferencia y servidor MLflow.

```powershell
podman machine start          # solo en Windows/macOS, si la máquina no está corriendo
podman compose up -d --build
Invoke-RestMethod http://127.0.0.1:8000/health
```

- API (Swagger): http://127.0.0.1:8000/docs
- MLflow: http://127.0.0.1:5000

La imagen de la API incluye los modelos de `models/serving/`, así que el sistema se puede levantar sin ejecutar antes el pipeline. Si `podman compose up` falla con un error de `netavark`/`nftables`, consulta la solución en la sección 13.

**4. Pruebas de carga y análisis FinOps** (secciones 13 y 14).

```powershell
uv venv .venv-locust --python 3.11
uv pip install --python .\.venv-locust\Scripts\python.exe -r .\requirements-locust.txt
.\.venv-locust\Scripts\python.exe -m locust -f locustfile.py --host http://127.0.0.1:8000 --headless -u 25 -r 5 -t 3m --csv reports/locust/mi_corrida/locust_25u --html reports/locust/mi_corrida/locust_25u.html
```

**5. Detener el sistema:** `podman compose down`.

## 1. Preguntas del proyecto

El proyecto busca responder tres preguntas mediante diferentes técnicas de *machine learning*.

| Pregunta | Objetivo | Tipo de problema |
|---|---|---|
| P1 | Predecir cuántos viajes se iniciarán en una estación durante una hora futura. | Regresión |
| P2 | Estimar si una estación estará en riesgo de quedarse sin bicicletas o sin espacios disponibles. | Clasificación |
| P3 | Identificar grupos de estaciones que comparten patrones de utilización similares. | Clustering |

P1 busca apoyar la planeación de la demanda, P2 el rebalanceo de bicicletas y P3 el análisis de estaciones para la planeación del sistema.

## 2. Datos utilizados

**Fuente:** [Portal de datos abiertos de MiBici](https://mibici.net/es/datos-abiertos/).

El proyecto utiliza registros mensuales de viajes y el archivo de nomenclatura de estaciones, que contiene identificadores, ubicaciones y capacidades.

En la ejecución documentada se procesaron:

| Elemento | Cantidad |
|---|---:|
| Periodo | Octubre de 2024 – septiembre de 2026 |
| Archivos mensuales | 24 |
| Viajes procesados | 9,043,925 |
| Registros por estación y hora | 6,662,294 |
| Perfiles de estaciones | 472 |

La ingesta normaliza los nombres de columnas, las codificaciones y los campos de fecha. Los datos se procesan por lotes y se almacenan en formato Parquet.

**Periodo fijo de ingesta.** El módulo `dataset.py` selecciona exactamente los 24 archivos mensuales de **octubre de 2024 a septiembre de 2026**, además de la nomenclatura de **septiembre de 2026**. Ya no utiliza los últimos 24 meses disponibles. Si falta alguno de los archivos esperados en la página de MiBici, el proceso informa un error antes de comenzar las descargas. Esta selección se comprueba mediante una prueba automatizada en `tests/test_data.py`, que también simula la publicación de un mes posterior.

No se incluyen datos individuales de MiBici en GitHub. Se utilizan las variables necesarias para el análisis y se evita publicar información personal en el repositorio.

### Versionamiento de datos

Se implementó un mecanismo de identificación de versiones mediante hashes SHA-256.

El módulo `data_version.py` calcula los hashes de los archivos CSV y del Parquet procesado, junto con sus cantidades de registros. Esta información se guarda en `references/data_manifest.json`.

La versión generada para la ejecución documentada es:

`e01da28b62056c69`

El manifiesto permite identificar y comprobar la integridad de los archivos utilizados sin incluir los datasets en Git. **Fijar las fechas no garantiza que el proveedor no modifique el contenido de los CSV**: los hashes del manifiesto permiten detectar esos cambios. Para reconstruir exactamente una versión es necesario conservar los archivos de origen correspondientes.

## 3. Estructura del repositorio

El repositorio utiliza la plantilla Cookiecutter Data Science.

```text
TC5061-Proyecto-Equipo44/
├── data/
│   ├── external/
│   ├── interim/
│   ├── processed/
│   └── raw/
├── notebooks/
├── references/
│   └── data_manifest.json
├── reports/
│   ├── figures/
│   └── deepchecks/                  # Reportes locales, ignorados por Git
├── models/
├── docs/
├── scripts/
│   ├── validate_mibici.py
│   ├── validate_mibici_drift.py
│   └── ab_test_p1.py                # A/B testing de versiones del modelo P1
├── tests/
│   └── test_data.py
├── tc5061_proyecto_equipo_44/
│   ├── config.py
│   ├── dataset.py
│   ├── data_version.py
│   ├── features.py
│   └── modeling/
│       ├── train.py
│       ├── p1_trainer.py            # Entrenamiento P1 (incluye ventanas y registro)
│       ├── mlflow_utils.py
│       └── predict.py
├── MLproject
├── python_env.yaml
├── requirements.txt
├── requirements-deepchecks.txt
├── pyproject.toml
├── .gitignore
└── README.md
```

Las carpetas `data/`, `models/` y `reports/` separan los datos, los modelos y los resultados. La lógica principal de procesamiento y entrenamiento se encuentra en `tc5061_proyecto_equipo_44/`. Los scripts de DeepChecks están en `scripts/` y las pruebas automatizadas en `tests/`.

## 4. Instalación

Se requiere Python, Git y `uv`.

**Instalar `uv` en Windows** (si el comando no está disponible):

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

**Crear y activar el entorno virtual principal (PowerShell):**

```powershell
uv venv --python 3.11
.\.venv\Scripts\Activate.ps1
```

**Instalar las dependencias**, incluido MLflow:

```powershell
uv pip install -r requirements.txt
```

Comprueba la instalación de MLflow:

```powershell
python -m mlflow --version
```

`requirements.txt` instala el proyecto en modo editable y contiene MLflow; no hace falta instalarlo por separado. **DeepChecks se instala en otro entorno**, siguiendo la sección 9, para evitar modificar las versiones del entorno principal.

## 5. Ejecución del pipeline

Los comandos se ejecutan desde la carpeta principal del repositorio.

### Ejecutar el MLproject con MLflow

El archivo `MLproject` define los entry points `ingest`, `features` y `train`. `python_env.yaml` especifica Python 3.11 y las dependencias para MLflow. Los siguientes comandos usan el entorno virtual principal ya instalado y activado.

```powershell
python -m mlflow run . -e ingest --env-manager=local --experiment-name mibici-equipo44
python -m mlflow run . -e features --env-manager=local --experiment-name mibici-equipo44
python -m mlflow run . -e train --env-manager=local --experiment-name mibici-equipo44 -P question=p1 -P model=baseline
```

Para ejecutar P3 se requiere `model=k_means`:

```powershell
python -m mlflow run . -e train --env-manager=local --experiment-name mibici-equipo44 -P question=p3 -P model=k_means
```

`--env-manager=local` ejecuta los entry points con el entorno virtual activo. `--experiment-name mibici-equipo44` asigna la ejecución de MLflow Projects al experimento del equipo.

### 5.1. Ingesta

```powershell
uv run --no-sync python -m tc5061_proyecto_equipo_44.dataset
```

Descarga los **24 CSV fijos de octubre de 2024 a septiembre de 2026**, junto con `nomenclatura_2026_09.csv`; procesa los viajes y genera `data/processed/dataset.parquet` y el manifiesto de versionamiento.

**Atención:** ejecutar este comando descarga y vuelve a procesar los archivos. Para comprobar la selección de fechas sin conectarse a internet, ejecuta la prueba indicada en la sección 9.

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

Para P2 se aproxima el inventario a partir de la capacidad de cada estación y del flujo de salidas y llegadas. Se supone inicialmente un inventario equivalente al 75 % de la capacidad y se utilizan umbrales del 20 % para generar las etiquetas de riesgo. Estas etiquetas son estimadas, ya que los datos disponibles no contienen mediciones reales de bicicletas o espacios libres.

### 5.3. Entrenamiento de modelos

Las ejecuciones se realizan mediante el entry point `train` del archivo `MLproject`.

**P1 — Regresión, baseline:**

```powershell
uv run --no-sync mlflow run . -e train -P question=p1 -P model=baseline --env-manager local --experiment-name mibici-equipo44
```

**P1 — Tres configuraciones adicionales de Random Forest:**

```powershell
uv run --no-sync mlflow run . -e train -P question=p1 -P model=rf -P n_estimators=10 -P max_depth=6 --env-manager local --experiment-name mibici-equipo44 --run-name P1_RF_10_depth6
uv run --no-sync mlflow run . -e train -P question=p1 -P model=rf -P n_estimators=20 -P max_depth=8 --env-manager local --experiment-name mibici-equipo44 --run-name P1_RF_20_depth8
uv run --no-sync mlflow run . -e train -P question=p1 -P model=rf -P n_estimators=40 -P max_depth=10 --env-manager local --experiment-name mibici-equipo44 --run-name P1_RF_40_depth10
```

**P1 — Reentrenamiento con ventanas de datos y registro en el Model Registry (Continuous Training):**

Por defecto, P1 usa una división cronológica 70/30 (comportamiento original). Para reentrenar con otro periodo, por ejemplo al incorporar datos recientes, se definen ventanas explícitas de entrenamiento y evaluación y se registra el modelo como una nueva versión de `P1_Departures_Model`:

```powershell
uv run --no-sync mlflow run . -e train -P question=p1 -P model=rf -P n_estimators=300 -P max_depth=15 -P train_start=2024-10-01 -P train_end=2026-10-16 -P eval_start=2026-10-16 -P register=true --env-manager local --experiment-name mibici-equipo44 --run-name P1_RF_v2
```

Equivalente sin MLflow Projects:

```powershell
uv run --no-sync python -m tc5061_proyecto_equipo_44.modeling.train --question p1 --model rf --n-estimators 300 --max-depth 15 --train-start 2024-10-01 --train-end 2026-10-16 --eval-start 2026-10-16 --register
```

| Parámetro (`-P` / flag) | Descripción | Valor por defecto |
|---|---|---|
| `train_start` / `--train-start` | Inicio de la ventana de entrenamiento (inclusive). | `none` (desde el primer dato) |
| `train_end` / `--train-end` | Fin del entrenamiento (exclusivo). Si se indica, se activa la división por ventanas en lugar de la 70/30. | `none` |
| `eval_start` / `--eval-start` | Inicio del conjunto de evaluación (hold-out). | `none` (igual a `train_end`) |
| `eval_end` / `--eval-end` | Fin del conjunto de evaluación (exclusivo). | `none` (hasta el último dato) |
| `register` / `--register` | Registra el modelo como nueva versión de `P1_Departures_Model`. Solo aplica a `model=rf`. | `false` |

Con `register=true`, la versión queda etiquetada con `data_train_start`, `data_train_end`, `train_rows`, `n_estimators` y `max_depth`. La primera versión registrada recibe el alias `champion` y las siguientes el alias `challenger`. Si el periodo de evaluación no existe en `hourly_features.parquet`, el entrenamiento se detiene con un mensaje indicando que hay que volver a ejecutar la ingesta y el Feature Engineering.

**P2 — Clasificación, baseline:**

```powershell
uv run --no-sync mlflow run . -e train -P question=p2 -P model=baseline --env-manager local --experiment-name mibici-equipo44
```

**P3 — Clustering:**

```powershell
uv run --no-sync mlflow run . -e train -P question=p3 -P model=k_means --env-manager local --experiment-name mibici-equipo44
```

| Pregunta | Modelo baseline | Métricas de evaluación |
|---|---|---|
| P1 | DummyRegressor | MAE y RMSE |
| P2 | DummyClassifier | Accuracy, F1 y Recall |
| P3 | KMeans | Silhouette Score |

Para P1 y P2 se utiliza una división cronológica entre entrenamiento y prueba, con el objetivo de evitar incorporar información futura en el entrenamiento. En P3 se utilizan perfiles históricos agregados para identificar grupos de estaciones.

## 6. Experimentos con MLflow

El seguimiento de experimentos se realiza mediante MLflow Tracking.

**Nombre del experimento:** `mibici-equipo44`.

Se registraron cuatro ejecuciones baseline: una para P1, dos para P2 (una por cada variable de riesgo) y una para P3. También se ejecutaron tres configuraciones de Random Forest para P1.

| Modelo P1 | Árboles | Profundidad máxima | Test MAE | Test RMSE |
|---|---:|---:|---:|---:|
| DummyRegressor | — | — | 1.5655 | 2.4704 |
| Random Forest 1 | 10 | 6 | 0.8759 | 1.5687 |
| Random Forest 2 | 20 | 8 | 0.8449 | 1.5079 |
| Random Forest 3 | 40 | 10 | **0.8248** | **1.4711** |

La configuración de 40 árboles y profundidad 10 obtuvo el menor error en estas pruebas. Es un candidato provisional: para una selección definitiva conviene utilizar una validación separada del conjunto de prueba.

Los runs incluyen parámetros, métricas, etiquetas y artefactos, así como referencias al código utilizado. El módulo de P1 incorpora el manifiesto de datos a los experimentos nuevos; la disponibilidad de ese artefacto en corridas anteriores debe comprobarse en el servidor MLflow correspondiente.

Para consultar la interfaz local:

```powershell
uv run --no-sync mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Abrir en el navegador: http://127.0.0.1:5000

Los archivos locales de experimentos y modelos no se incluyen en GitHub.

## 7. Buenas prácticas y reproducibilidad

El proyecto aplica las siguientes prácticas:

- **Separación del código:** ingesta, características y entrenamiento se implementan en módulos independientes.
- **Configuración:** las rutas y variables compartidas se administran mediante `config.py`.
- **Datos:** los archivos originales y procesados se mantienen separados.
- **Versionamiento:** se utiliza un manifiesto con hashes SHA-256 para identificar los datos procesados.
- **Ingesta fija:** se selecciona siempre el periodo octubre de 2024 a septiembre de 2026.
- **Evaluación temporal:** P1 y P2 utilizan una partición cronológica para reducir el riesgo de fuga de información futura.
- **Validación de calidad:** DeepChecks verifica características de integridad y cambios de distribución sobre muestras reproducibles.
- **Experimentos:** MLflow permite consultar parámetros, métricas, etiquetas y artefactos.
- **Git:** `.gitignore` excluye datos, modelos serializados, archivos locales de MLflow, credenciales, entornos virtuales y reportes de DeepChecks.

Los archivos generados localmente no deben incorporarse al historial de Git.

## 8. Flujo de trabajo con Git y GitHub

El equipo desarrolla sus cambios mediante ramas independientes y Pull Requests.

1. Actualizar la rama compartida `develop`.
2. Crear una rama para la funcionalidad o corrección.
3. Implementar y probar los cambios.
4. Crear commits descriptivos.
5. Subir la rama a GitHub.
6. Abrir un Pull Request hacia `develop`.
7. Revisar e integrar los cambios.

Se utilizan nombres descriptivos para las ramas, como `fix/modeling-dependencies` y `docs/project-documentation`, así como convenciones de commits como `feat:`, `fix:` y `docs:`.

La actividad establece que la rama principal debe estar protegida y que cada integrante debe abrir al menos un Pull Request y realizar al menos una revisión. Estas condiciones deben comprobarse en GitHub antes de la entrega.

## 9. Validación de datos con DeepChecks

Se integró **DeepChecks** para revisar la integridad de los registros de viajes y detectar posibles cambios de distribución (*data drift*) entre las muestras de entrenamiento y prueba de P1. Esta validación no modifica los archivos originales ni reentrena modelos.

### 9.1. Entorno independiente

DeepChecks se ejecuta en un entorno separado para evitar conflictos con NumPy, Pandas y Scikit-learn del proyecto principal. **No es necesario cambiar las dependencias del entorno de MLflow.**

Desde la raíz del repositorio, en PowerShell:

```powershell
uv venv .venv-deepchecks --python 3.10
uv pip install --python .\.venv-deepchecks\Scripts\python.exe -r requirements-deepchecks.txt
```

El archivo `requirements-deepchecks.txt` fija las versiones utilizadas en las validaciones locales: DeepChecks 0.19.1, NumPy 1.26.4, Pandas 2.2.3, Scikit-learn 1.5.2, Setuptools 80.9.0 y PyArrow 25.0.1. El entorno `.venv-deepchecks/` está excluido de Git.

### 9.2. Validación de integridad de viajes

```powershell
.\.venv-deepchecks\Scripts\python.exe scripts/validate_mibici.py
```

El script lee `data/processed/dataset.parquet` por lotes y selecciona una **muestra aleatoria reproducible de 10,000 registros (semilla 42)**. Para evitar exponer identificadores individuales en el reporte, se analizan únicamente las columnas `genero`, `ano_de_nacimiento`, `inicio_del_viaje`, `fin_del_viaje`, `origen_id` y `destino_id`.

Los chequeos ejecutados son `ColumnsInfo`, `PercentOfNulls`, `MixedNulls`, `MixedDataTypes` y `DataDuplicates`.

**Resultados observados en la muestra:**

| Chequeo | Hallazgo |
|---|---|
| Valores nulos formales | 0 % en las columnas evaluadas por `PercentOfNulls` |
| Valores faltantes codificados como texto | `genero`: 79 valores `"NULL"` (0.79 %); `ano_de_nacimiento`: 7 valores `"NA"` (0.07 %) |
| Tipos mixtos | `ano_de_nacimiento` contiene principalmente años expresados como texto numérico, además de `"NA"` |
| Duplicados | 0 % de filas duplicadas **en las columnas muestreadas y analizadas** |

**Reportes locales:**

- `reports/deepchecks/integridad_viajes.html`: exportación HTML original de DeepChecks (puede visualizarse en blanco en determinados navegadores o entornos).
- `reports/deepchecks/integridad_viajes_simple.html`: versión estática alternativa con los resultados textuales, sin JavaScript.

Para abrir la versión estática en Windows:

```powershell
Invoke-Item .\reports\deepchecks\integridad_viajes_simple.html
```

Los porcentajes corresponden a la muestra analizada, **no al conjunto completo de 9,043,925 viajes**. Los textos `"NULL"` y `"NA"` requieren atención porque no se consideran automáticamente valores nulos formales.

### 9.3. Análisis de data drift

```powershell
.\.venv-deepchecks\Scripts\python.exe scripts/validate_mibici_drift.py
```

Se utiliza `data/processed/hourly_features.parquet` y un corte temporal de referencia en **2026-02-24**: los registros anteriores corresponden a entrenamiento y los posteriores o iguales a prueba. Se toman muestras reproducibles con semilla 42, de hasta 10,000 filas de cada periodo.

La validación compara las variables de P1: `departures_lag_1h`, `departures_lag_24h`, `departures_lag_168h`, `departures_rolling_mean_24h`, `hour_of_day`, `day_of_week`, `month` e `is_weekend`.

**Resultados observados:**

| Variable | Drift score | Método |
|---|---:|---|
| `month` | 0.6138 | Cramér's V |
| `departures_rolling_mean_24h` | 0.0614 | Kolmogorov-Smirnov |
| `departures_lag_168h` | 0.0366 | Kolmogorov-Smirnov |
| `departures_lag_24h` | 0.0313 | Kolmogorov-Smirnov |
| `departures_lag_1h` | 0.0262 | Kolmogorov-Smirnov |
| `hour_of_day` | 0.0123 | Cramér's V |
| `day_of_week` | 0.0000 | Cramér's V |
| `is_weekend` | 0.0000 | Cramér's V |

La diferencia más marcada dentro de las variables categóricas se observó en `month`, lo cual es compatible con una partición cronológica que abarca meses diferentes. **Los puntajes de Cramér's V y Kolmogorov-Smirnov utilizan medidas distintas y no deben compararse directamente como si tuvieran la misma escala.** La presencia de drift no demuestra por sí sola una caída en el rendimiento predictivo.

El reporte estático se guarda en:

`reports/deepchecks/data_drift.html`

Para abrirlo en Windows:

```powershell
Invoke-Item .\reports\deepchecks\data_drift.html
```

### 9.4. Pruebas automatizadas

Para ejecutar todas las pruebas existentes del repositorio:

```powershell
uv run --no-sync python -m pytest -q
```

Para ejecutar las pruebas de entrenamiento P1, P2 y P3:

```powershell
uv run --no-sync python -m pytest tests/test_training.py -v
```

Para ejecutar únicamente las pruebas de una pregunta:

```powershell
uv run --no-sync python -m pytest tests/test_training.py -k train_p1 -v
uv run --no-sync python -m pytest tests/test_training.py -k train_p2 -v
uv run --no-sync python -m pytest tests/test_training.py -k train_p3 -v
```

Las pruebas de entrenamiento usan datos sintéticos y aíslan los artefactos en directorios temporales; no requieren descargar datos ni iniciar una ejecución real de MLflow.

Para comprobar específicamente que la ingesta usa los 24 meses fijos y descarta publicaciones posteriores, **sin descargar archivos reales**:

```powershell
uv run --no-sync python -m pytest tests/test_data.py -v
```

En la verificación local con Python 3.11.17 se aprobaron **29 pruebas** (con dos advertencias de MLflow). Las capturas de PyTest, DeepChecks y la auditoría de seguridad pueden adjuntarse como evidencia al Pull Request o a la entrega.

### 9.5. Auditoría básica de seguridad

Para revisar vulnerabilidades conocidas en las dependencias **instaladas** en el entorno principal de Python 3.11:

```powershell
python -m pip_audit --local
```

La revisión local con `pip-audit` no encontró vulnerabilidades conocidas en los paquetes que pudo auditar. El paquete local `tc5061-proyecto-equipo-44` fue omitido porque no está publicado en PyPI y no puede evaluarse de ese modo.

**Nota:** auditar directamente `requirements.txt` con `python -m pip_audit -r requirements.txt` puede fallar porque la exportación contiene hashes y la instalación editable `-e .`. Por ello, la auditoría documentada se ejecutó sobre el entorno instalado. Conviene guardar una captura del resultado.

También se revisaron los archivos versionados en busca de nombres habituales de archivos con credenciales y referencias a contraseñas o tokens:

```powershell
git ls-files | Select-String -Pattern '(^|/)\.env($|\.)|\.pem$|\.key$|\.p12$|\.pfx$|id_rsa$|id_ed25519$'
git grep -il -E "api[_-]?key|secret|password|access[_-]?token|private[_-]?key|BEGIN .*PRIVATE KEY"
```

Estas búsquedas no arrojaron coincidencias en la revisión realizada. Son comprobaciones básicas y **no garantizan** la ausencia total de secretos o vulnerabilidades.

### 9.6. Limitaciones y reproducibilidad

- Los chequeos de DeepChecks usan **muestras**, por lo que no descartan problemas en filas no seleccionadas.
- La comprobación de duplicados utiliza únicamente las seis columnas disponibles en la muestra y no equivale a comprobar duplicados por `viaje_id` sobre todo el dataset.
- La fecha de corte debe verificarse contra el código de entrenamiento vigente si el equipo modifica la partición temporal.
- Los reportes se generan localmente y la carpeta `reports/deepchecks/` está excluida de Git; los scripts y las dependencias sí deben versionarse.

## 10. Despliegue de la API con Podman

Se desarrolló una API REST con FastAPI para consultar los modelos de MiBici mediante peticiones HTTP.

### Modelo para pregunta 1 utilizado

- Nombre en MLflow Model Registry: `mibici-p1-rf`
- Versión: `1`
- Algoritmo: RandomForestRegressor
- Objetivo: predecir el número de viajes que iniciarán en una estación durante una hora.

El modelo se recuperó desde el Model Registry de MLflow y se guardó en `models/serving/`.

Se eligió empaquetar el modelo dentro de la imagen para realizar inferencia en tiempo real sin depender de un servidor MLflow durante las predicciones.

### Modelo para pregunta 3 utilizado

- Ubicación: `models/serving/p3_kmeans_5_clusters.joblib`
- Algoritmo: `KMeans` (dentro de un `Pipeline` de scikit-learn)
- Configuración: 5 clústeres, inicialización `random`.
- Objetivo: agrupar las estaciones del sistema MiBici basándose en patrones de uso, capacidad y geografía.

El modelo se cargó directamente mediante la librería `joblib` desde el sistema de archivos. Se eligió empaquetar el modelo dentro de la imagen para realizar inferencia en tiempo real sin depender de servicios externos durante las predicciones.

### Archivos principales

- `api/main.py`: código de FastAPI y validación de entradas con Pydantic.
- `api/requirements-api.txt`: dependencias del servicio.
- `Containerfile`: instrucciones de construcción de la imagen.
- `.containerignore`: exclusión de archivos innecesarios.
- `models/serving/`: modelo registrado y sus metadatos.

### Construir la imagen
Desde la raíz del repositorio:

```powershell
podman build -t mibici-api -f Containerfile .
```

### Ejecutar el contenedor

```powershell
podman run --rm --name mibici-api-test -p 8001:8000 mibici-api
```

La API estará disponible en:

http://127.0.0.1:8001/docs

### Endpoints

- `GET /health`: verifica el funcionamiento del servicio.
- `POST /predict`: recibe las ocho variables del modelo y devuelve una predicción.
- `POST /predict/p3`: recibe las características de una estación y devuelve el clúster al que pertenece

### Ejemplo de petición a POST /predict

```json
{
  "departures_lag_1h": 2.0,
  "departures_lag_24h": 3.0,
  "departures_lag_168h": 4.0,
  "departures_rolling_mean_24h": 3.0,
  "hour_of_day": 14,
  "day_of_week": 0,
  "month": 8,
  "is_weekend": 0
}
```

### Respuesta obtenida

```json
{
  "predicted_departures": 4.268983060280685
}
```

### Ejemplo de petición a POST /predict/p3
```json
{
  "total_departures": 23,
  "total_arrivals": 10,
  "morning_departures": 20,
  "weekend_departures": 3,
  "dpcapacity": 0,
  "avg_daily_departures": 0,
  "morning_share": 0.5,
  "weekend_share": 0.2,
  "latitude": 0,
  "longitude": 0
}
```

### Respuesta obtenida

```json
{
  "predicted_cluster": 2
}
```

La prueba realizada desde Podman respondió con código HTTP 200.

Los datos de entrada se validan con Pydantic. Las peticiones que no cumplen las restricciones establecidas reciben una respuesta de error HTTP 422.

### Evidencia

La API se ejecutó dentro de un contenedor Podman y respondió
correctamente a una petición POST /predict con código HTTP 200.

![Predicción de MiBici ejecutada en Podman](docs/evidence/podman_predict.png)

### Estrategia de despliegue

Se eligió inferencia en tiempo real mediante una API REST. Este enfoque permite que otras aplicaciones envíen las características de una estación y obtengan una predicción cuando la necesiten.

La imagen contiene el modelo previamente registrado en MLflow, sus dependencias y el código de FastAPI, para facilitar su ejecución en otras computadoras compatibles con Podman o Docker.
## 11. Monitoreo de data drift con Evidently

Se implemento el monitoreo de datos para P1 (regresion),
P2 (clasificacion) y P3 (clustering).

### Periodos analizados

- Referencia: 24 meses históricos (ej. 2024-10-01 a 2026-10-01). Representa el conjunto completo de datos con el que se entrenó el modelo (Champion), abarcando todas las estacionalidades.
- Actual: Último mes en producción (ej. 2026-10-01 a 2026-11-01).
- P1 y P2: 10,000 registros por periodo para optimización de memoria.
- P3: 468 estaciones compartidas.

Justificación técnica de la ventana de tiempo:

En MLOps, contrastar la memoria histórica completa del modelo (24 meses) contra exclusivamente el mes actual aislado (1 mes) evita la dilución de la señal y los falsos positivos por estacionalidad. Esto garantiza que cualquier drift detectado refleje un cambio genuino en las distribuciones (ej. alteración en los rezagos o valores nulos nuevos) frente a los datos que la API recibe hoy.

### Instalacion

```powershell
uv venv .venv-evidently --python 3.11
uv pip install --python .\.venv-evidently\Scripts\python.exe -r .\requirements-evidently.txt
```

### Ejecucion

Desde la carpeta principal del proyecto:

```powershell
.\.venv-evidently\Scripts\python.exe .\scripts\monitor_p1_evidently.py
.\.venv-evidently\Scripts\python.exe .\scripts\monitor_p2_evidently.py
.\.venv-evidently\Scripts\python.exe -m scripts.monitor_p3_evidently
```

### Resultados

| Modelo | Variables con drift | Drift global |
|---|---|---|
| P1 | 5 de 8 (month, departures_rolling_mean_24h, departures_lag_24h, departures_lag_1h, departures_lag_168h) | Detectado (62.5%) |
| P2 | 6 de 11 (month, departures_rolling_mean_24h, departures_lag_24h, departures_lag_1h, departures_lag_168h, arrivals_lag_1h) | Detectado (54.5%) |
| P3 | 1 de 10 (avg_daily_departures, total_arrivals, weekend_departures, morning_departures, total_departures) | Detectado (50%) |

### Evidencia

- reports/evidently/p1_drift_v2.html
- reports/evidently/p2_drift_v2.html
- reports/evidently/p3_drift_v2.html

<!-- La interpretacion y las limitaciones se encuentran
en docs/evidently_drift_results.md. -->

## 12. A/B testing de versiones del modelo P1

Para decidir si conviene reentrenar el modelo con datos recientes (Continuous Training), se comparan dos versiones de `P1_Departures_Model` sobre el mismo conjunto de evaluación, que ninguna de las dos utilizó para entrenar.

### Versiones y periodos

| Versión | Datos de entrenamiento | Alias inicial |
|---|---|---|
| v1 | Octubre 2024 – septiembre 2026 (24 meses) | `champion` |
| v2 | Los mismos 24 meses más el 1–15 de octubre de 2026 | `challenger` |

- **Conjunto de evaluación (hold-out):** del 16 de octubre de 2026 en adelante, común para ambas versiones.
- Los hiperparámetros de v2 deben ser los mismos que los de v1, para que la diferencia se deba a los datos y no al modelo.

### Preparación

1. Tener ingestados y procesados los datos de octubre de 2026 (secciones 5.1 y 5.2). Se comprueba con:

```powershell
uv run --no-sync python -c "import pandas as pd; d=pd.read_parquet('data/processed/hourly_features.parquet',columns=['hour']); print(d.hour.max())"
```

2. Levantar el servidor de MLflow (sección 6) y registrar la versión v1 una sola vez, indicando el Run ID del run de Random Forest de 24 meses:

```powershell
uv run --no-sync python -c "import mlflow; from mlflow.tracking import MlflowClient as C; mlflow.set_tracking_uri('http://127.0.0.1:5000'); mv=mlflow.register_model('runs:/<RUN_ID>/model','P1_Departures_Model'); c=C(); c.set_model_version_tag('P1_Departures_Model',mv.version,'data_train_start','2024-10-01'); c.set_model_version_tag('P1_Departures_Model',mv.version,'data_train_end','2026-10-01'); c.set_registered_model_alias('P1_Departures_Model','champion',mv.version)"
```

3. Entrenar y registrar v2 con los parámetros de ventana descritos en la sección 5.3:

```powershell
uv run --no-sync python -m tc5061_proyecto_equipo_44.modeling.train --question p1 --model rf --n-estimators 300 --max-depth 15 --train-start 2024-10-01 --train-end 2026-10-16 --eval-start 2026-10-16 --register
```

### Ejecución de la prueba A/B

```powershell
uv run --no-sync python scripts/ab_test_p1.py --eval-start 2026-10-16
```

`--eval-start` debe coincidir con el `--train-end` / `--eval-start` usado para entrenar v2. Por defecto, v1 es la versión con alias `champion` y v2 la que tiene alias `challenger`. Para comparar otras versiones:

```powershell
uv run --no-sync python scripts/ab_test_p1.py --eval-start 2026-10-16 --v1 1 --v2 2
```

Para mover el alias `champion` a la versión ganadora, agregar `--promote`:

```powershell
uv run --no-sync python scripts/ab_test_p1.py --eval-start 2026-10-16 --promote
```

| Opción | Descripción | Valor por defecto |
|---|---|---|
| `--eval-start` | Inicio del hold-out (obligatorio). | — |
| `--eval-end` | Fin del hold-out (exclusivo). | Último dato disponible |
| `--v1`, `--v2` | Versiones a comparar. | Alias `champion` y `challenger` |
| `--min-improvement` | Mejora relativa mínima del MAE para que v2 reemplace a v1. | 0.02 (2 %) |
| `--peak-tolerance` | Empeoramiento máximo tolerado del MAE en horas pico. | 0.02 (2 %) |
| `--n-boot` | Remuestreos del bootstrap. | 2000 |
| `--promote` | Mueve el alias `champion` a la versión ganadora. | Desactivado |
| `--allow-overlap` | Permite continuar si una versión entrenó con datos del hold-out. | Desactivado |

### Métricas y criterio de decisión

El script calcula, para cada versión, MAE, RMSE, R², sesgo y MAE en horas de alta demanda (percentil 90 de las salidas reales). La significancia de la diferencia de MAE se estima con un bootstrap pareado que remuestrea horas completas.

La versión v2 reemplaza a v1 solo si cumple las tres condiciones:

1. Mejora el MAE al menos `--min-improvement`.
2. El intervalo de confianza del 95 % de la diferencia de MAE (v2 − v1) queda completamente por debajo de 0.
3. El MAE en horas pico no empeora más de `--peak-tolerance`.

Antes de evaluar, el script verifica mediante las etiquetas `data_train_end` que ninguna versión haya entrenado con datos del hold-out, y se detiene si detecta una fuga.

### Resultados

Archivos generados:

- Un run en MLflow con la etiqueta `stage_of_work=ab_test`, con parámetros y métricas de ambas versiones.
- `reports/p1_ab_test.json`: métricas, intervalo de confianza, criterios y versión ganadora.
- `reports/figures/p1_ab_mae_by_hour.png`: MAE por hora del día para cada versión.

**Resultado obtenido:** la versión v1 obtuvo el mejor desempeño y se mantiene como versión vigente (alias `champion`).

### Limitaciones

- El hold-out cubre un periodo corto (la segunda quincena de octubre de 2026), por lo que la variable `month` es constante en la evaluación y no se puede valorar el comportamiento estacional.
- El bootstrap remuestrea horas, que dentro de un mismo día están correlacionadas; por ello el intervalo de confianza puede resultar más estrecho de lo real.
- Para una conclusión más robusta sobre Continuous Training conviene repetir la comparación en varios periodos consecutivos (backtest).

## 13. Orquestación con Docker Compose y pruebas de carga con Locust

### Docker Compose

Se implementó la orquestación de los servicios del proyecto MiBici mediante Docker Compose, utilizando el Containerfile desarrollado anteriormente.

El sistema incluye dos servicios:

- **FastAPI:** API REST para realizar predicciones con P1 y P3.
- **MLflow:** servidor para el seguimiento de experimentos y modelos.

Para construir e iniciar los servicios desde la carpeta principal del proyecto:

```powershell
podman compose up -d --build
```

Para verificar que los contenedores estén activos:

```powershell
podman compose ps
```

Servicios disponibles:

- FastAPI: http://localhost:8000/docs
- MLflow: http://localhost:5000

Para detener los servicios:

```powershell
podman compose down
```

MLflow utiliza un volumen persistente independiente. La API carga los modelos incluidos en su imagen, por lo que iniciar MLflow no modifica automáticamente los modelos utilizados para las predicciones.

Se comprobó que ambos servicios funcionan y que los endpoints de P1 y P3 responden correctamente.

#### Solución de problemas: error `netavark` / `nftables` en Windows (WSL)

En algunos equipos con Windows, `podman compose up` falla al iniciar los contenedores con:

```text
Error response from daemon: netavark (exit code 1): nftables error: "nft" did not return successfully while applying ruleset
```

El fallo ocurre al crear la red propia del proyecto y los alias de red que Compose agrega a cada servicio. La red por defecto de Podman sí funciona. Primero reinicia la máquina de Podman (`podman machine stop` y luego `podman machine start`). Si el error persiste, levanta los mismos dos servicios con `podman run`. La API carga los modelos desde su imagen y no se comunica con MLflow para predecir, así que el comportamiento es el mismo:

```powershell
podman compose build
podman volume create tc5061-proyecto-equipo44_mlflow_data
podman run -d --name mibici-mlflow -p 5000:5000 -v tc5061-proyecto-equipo44_mlflow_data:/mlflow mibici-api:compose mlflow server --backend-store-uri sqlite:////mlflow/mlflow.db --artifacts-destination /mlflow/artifacts --host 0.0.0.0 --port 5000
podman run -d --name mibici-api -p 8000:8000 mibici-api:compose
```

Para detenerlos: `podman rm -f mibici-api mibici-mlflow`.

### Pruebas de carga con Locust

Se implementó un escenario de pruebas de carga utilizando Locust para analizar el rendimiento de la API bajo diferentes cantidades de usuarios simultáneos.

El archivo `locustfile.py` simula peticiones a los endpoints:

- `POST /predict`: modelo P1 (regresión).
- `POST /predict/p3`: modelo P3 (clustering).

El escenario utiliza una proporción aproximada de tres peticiones a P1 por cada petición a P3.

### Instalación de Locust

Crear un entorno independiente e instalar las dependencias:

```powershell
uv venv .venv-locust --python 3.11
uv pip install --python .\.venv-locust\Scripts\python.exe -r .\requirements-locust.txt
```

### Ejecución de las pruebas

Con Docker Compose funcionando, ejecutar:

```powershell
.\.venv-locust\Scripts\python.exe -m locust -f locustfile.py --host http://127.0.0.1:8000
```

Abrir http://localhost:8089 para configurar los usuarios simulados e iniciar las pruebas.

Para ejecuciones reproducibles se recomienda el modo *headless*, que guarda CSV y un reporte HTML (`-u`: usuarios, `-r`: usuarios nuevos por segundo, `-t`: duración):

```powershell
.\.venv-locust\Scripts\python.exe -m locust -f locustfile.py --host http://127.0.0.1:8000 `
  --headless -u 25 -r 5 -t 3m `
  --csv reports/locust/mi_corrida/locust_25u --html reports/locust/mi_corrida/locust_25u.html
```

### Monitoreo de CPU y memoria con `podman stats`

En otra terminal, mientras corre Locust:

```powershell
podman stats --no-reset --interval 2 `
  --format "{{.Name}},{{.CPUPerc}},{{.MemUsage}},{{.MemPerc}},{{.NetIO}},{{.PIDs}}" `
  mibici-api mibici-mlflow > reports/locust/mi_corrida/podman_stats_25u.csv
```

Ajusta los nombres de los contenedores a los de `podman ps`. Con Compose suelen ser `tc5061-proyecto-equipo44-api-1` y `tc5061-proyecto-equipo44-mlflow-1`.

**Importante:** no uses `podman stats --no-stream` para medir la carga. Cada lectura en ese modo reporta el promedio de CPU desde que arrancó el contenedor, no la carga del momento, y el valor decae aunque la carga sea constante. En modo streaming (`--interval`) descarta la primera lectura, que también es acumulada. 100 % de CPU equivale a un núcleo.

### Escenarios evaluados

Cada usuario simulado espera entre 1 y 3 s entre peticiones (≈ 0.5 req/s) y cada escenario dura 3 minutos.

| Escenario | Usuarios | Ramp-up | Workers de uvicorn |
|---|---:|---:|---:|
| Carga baja | 10 | 2 usuarios/s | 1 |
| Carga media | 25 | 5 usuarios/s | 1 |
| Carga alta | 100 | 10 usuarios/s | 1 |
| Estrés | 250 | 10 usuarios/s | 1 |
| Estrés optimizado | 250 | 10 usuarios/s | 4 |

Durante las pruebas se analizaron el número de peticiones, las peticiones por segundo, los percentiles de latencia, el porcentaje de errores y el consumo de CPU y memoria de cada contenedor.

### Resultados y evidencia

Corrida del 21 de noviembre de 2026 (métricas en la ventana estable de cada prueba):

| Escenario | Peticiones | Errores | Req/s | Latencia p50 / p95 | CPU API | RAM API |
|---|---:|---:|---:|---|---:|---:|
| 10 usuarios | 870 | 0 | 4.9 | 62 / 68 ms | 7 % | 249 MB |
| 25 usuarios | 2,131 | 0 | 11.9 | 62 / 72 ms | 16 % | 233 MB |
| 100 usuarios | 8,506 | 0 | 48.5 | 65 / 110 ms | 63 % | 250 MB |
| 250 usuarios | 15,450 | 0 | 89.8 | 720 / 1,300 ms | 124 % | 257 MB |
| 250 usuarios, 4 workers | 20,338 | 0 | 121.3 | 64 / 86 ms | 162 % | 904 MB |

- No hubo errores en ningún escenario.
- Cada inferencia consume ≈ 13 ms de CPU, y el consumo crece linealmente con el tráfico.
- Con un solo worker la API se satura en ≈ 90 req/s: las peticiones se encolan y la mediana de latencia sube a 720 ms. Con 4 workers vuelve a 64 ms.
- El servidor MLflow ocupa ≈ 2.3 GB de RAM aunque no participa en la inferencia.

Evidencia:

- `reports/locust/run_2026-11-21_streaming/`: reportes HTML de Locust, CSV de estadísticas e historial, y muestreos de `podman stats` por escenario.
- `reports/figures/load_*.png`: gráficas de rendimiento, latencia, CPU y memoria.
- `reports/reporte_carga_finops.docx`: reporte completo de las pruebas de carga y del análisis FinOps.

## 14. Análisis FinOps

Con el consumo medido se estimó el costo por inferencia para cuatro opciones de despliegue. Se usaron precios de lista on-demand de referencia: AWS us-east-1 (t3.small $0.0208/h, t3.medium $0.0416/h, c6i.large $0.085/h) y Google Cloud Run ($0.000024 por vCPU-s, $0.0000025 por GiB-s y $0.40 por millón de peticiones). Las cifras cubren solo cómputo.

Para regenerar las métricas, las gráficas y `reports/finops_summary.json` a partir de una corrida:

```powershell
python scripts/finops_load_analysis.py --run-dir reports/locust/run_2026-11-21_streaming
```

Los precios y supuestos se editan al inicio de `scripts/finops_load_analysis.py`.

| Opción | Demanda natural (0.13 req/s) | 25 usuarios (11.9 req/s) | 80 req/s |
|---|---:|---:|---:|
| A. Actual: API + MLflow en t3.medium | $87.76 / millón · $30.37 / mes | $0.97 / millón | $0.34 / millón |
| B. Solo API en t3.small | $43.88 / millón · $15.18 / mes | **$0.48 / millón** | **$0.19 / millón** |
| C. Solo API (2 workers) en c6i.large | $179.32 / millón · $62.05 / mes | $1.98 / millón | $0.30 / millón |
| D. Serverless con escala a cero (Cloud Run) | **$2.93 / millón · $1.01 / mes** | $2.52 / millón | $1.03 / millón |

La demanda natural supone una predicción por estación por hora (474 × 24 ≈ 341 mil inferencias al mes).

![Costo por millón de inferencias](reports/figures/finops_cost_per_million.png)

Conclusiones y optimizaciones:

1. **Separar MLflow del serving.** Sin MLflow, la API cabe en una t3.small: −50 % del costo fijo ($30.37 → $15.18 al mes).
2. **Elegir el despliegue según la demanda.** Con la demanda natural, la infraestructura está ociosa más del 99 % del tiempo y serverless con escala a cero cuesta ≈ $1 al mes. Por encima de ≈ 2 req/s sostenidas (≈ 5.2 millones de inferencias al mes) conviene una VM bien dimensionada.
3. **Workers y autoescalado.** Conviene usar un worker por vCPU y escalar al superar el 70 % de CPU (≈ 63 req/s por worker).
4. **Predicción por lotes.** Las 474 predicciones de cada hora pueden atenderse con un endpoint por lotes o un job programado, en lugar de 474 peticiones HTTP individuales.
5. **Modelo e imagen más ligeros.** Un Random Forest más pequeño o exportado a ONNX abarataría la inferencia, pero debe evaluarse contra el MAE. La imagen de 954 MB puede reducirse cargando el modelo skops sin MLflow completo.

El detalle de supuestos, limitaciones y estrategias está en `reports/reporte_carga_finops.docx`.

## 15. Integrantes del equipo

| Integrante | Matrícula | Responsabilidad principal |
|---|---|---|
| Axel Dair Torrecilla Hernández | A01797810 | Desarrollo y entrenamiento de los modelos baseline |
| Guillermo Navarro García | A01282490 | Feature Engineering para P1, P2 y P3 |
| José Andrés Orantes Guillén | A01174130 | Análisis exploratorio de datos (EDA) |
| José Luis Navarro Hernandez | A01840092 | Ingesta, normalización y preparación de datos |
