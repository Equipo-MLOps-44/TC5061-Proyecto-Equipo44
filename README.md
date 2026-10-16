# TC5061 Proyecto Equipo 44

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Proyecto integrador de la materia de MLOps que toma los datos de mibici para realizar un flujo de MLOps.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         tc5061_proyecto_equipo_44 and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── tc5061_proyecto_equipo_44   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes tc5061_proyecto_equipo_44 a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

## Data ingestion

Run `python -m tc5061_proyecto_equipo_44.dataset` from the project directory.
The script reads the download links published on
[MiBici's open-data page](https://mibici.net/es/datos-abiertos/), downloads the
latest 24 months of trip CSVs and the newest nomenclature CSV into
`data/external`, and combines the trip data into
`data/processed/dataset.parquet`. The nomenclature file is kept separately and
is not mixed into the trip dataset. CSV rows are processed in bounded batches
to limit memory use. Column names are normalized to lowercase ASCII
`snake_case`, text is decoded from UTF-8 (or Windows-1252 for legacy files) and
stored as Unicode, and the `inicio_del_viaje` and `fin_del_viaje` columns are
written as Parquet timestamps. Other columns remain strings so identifiers
retain their original formatting.

--------
