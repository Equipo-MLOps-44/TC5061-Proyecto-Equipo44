# Monitoreo de Data Drift con Evidently - MiBici

## 1. Objetivo

Analizar los cambios en las distribuciones de las variables utilizadas por los modelos P1 (regresion), P2 (clasificacion) y P3 (clustering) mediante Evidently.

## 2. Metodologia

- Periodo de referencia: julio-septiembre de 2025.
- Periodo actual: julio-septiembre de 2026.
- P1 y P2: muestras reproducibles de 10,000 registros por periodo (random_state=42).
- P3: perfiles de 368 estaciones presentes en ambos periodos.
- Herramienta: Evidently, utilizando DataDriftPreset y DataSummaryPreset.

Los periodos fueron seleccionados para comparar la misma temporada de dos anos distintos. Esta seleccion debe validarse con el equipo frente a los datos utilizados en el entrenamiento de cada modelo.

## 3. Resultados

### P1 - Regresion

- Variables analizadas: 8.
- Variables con drift: 0.
- Drift global: no detectado.
- Variable con mayor diferencia: departures_rolling_mean_24h.
- Distancia Wasserstein normalizada: 0.09378.
- Valores faltantes: 0 en referencia y 94 en actual.
- Principal variable con faltantes: departures_lag_168h (84 registros).

Interpretacion:
No se detecto drift significativo bajo los criterios configurados en Evidently. Sin embargo, existen diferencias menores en las distribuciones y nuevos valores faltantes que conviene investigar.

### P2 - Clasificacion

- Variables analizadas: 11.
- Variables con drift: 1.
- Drift global: no detectado.
- Variable afectada: dpcapacity.
- Distancia Wasserstein normalizada: 0.104526.
- Capacidad promedio en referencia: 16.93.
- Capacidad promedio en actual: 17.03.
- Valores faltantes: 0 en referencia y 99 en actual.

Interpretacion:
Se detecto un cambio en la distribucion de la capacidad de las estaciones. Una posible causa es que haya cambiado la composicion de estaciones representadas en los datos. Esto debe verificarse antes de atribuirle una causa especifica.

### P3 - Clustering

- Variables analizadas: 10.
- Variables con drift: 1.
- Drift global: no detectado.
- Variable afectada: weekend_share.
- Prueba: Kolmogorov-Smirnov.
- Valor p: 0.000953.
- Promedio de weekend_share en referencia: 0.20.
- Promedio de weekend_share en actual: 0.22.
- Estaciones comparadas: 368.
- Valores faltantes: 0 en ambos conjuntos.

Interpretacion:
La distribucion de la proporcion de salidas durante fines de semana cambio entre los periodos. Esto podria modificar la forma en que KMeans agrupa las estaciones, pero no demuestra por si solo una perdida de desempeno.

## 4. Resumen comparativo

| Modelo | Variables | Con drift | Drift global |
|---|---:|---:|---|
| P1 - Regresion | 8 | 0 | No detectado |
| P2 - Clasificacion | 11 | 1 | No detectado |
| P3 - Clustering | 10 | 1 | No detectado |

## 5. Conclusiones

No se detecto drift global en ninguno de los tres modelos bajo los umbrales utilizados.

Sin embargo, se identificaron cambios individuales en dpcapacity (P2) y weekend_share (P3), asi como nuevos valores faltantes en P1 y P2.

Se recomienda continuar el monitoreo de estas variables, revisar la calidad de los datos y considerar los resultados del A/B testing antes de tomar una decision sobre el reentrenamiento.

## 6. Limitaciones

- Los periodos comparados todavia deben validarse frente a los datos de entrenamiento de las versiones registradas en MLflow.
- P1 y P2 utilizan muestras de 10,000 registros por periodo.
- P3 compara unicamente estaciones presentes en ambos periodos.
- El monitoreo mide cambios en los datos, no directamente el desempeno predictivo.
- El resultado de drift global depende de los umbrales configurados.

## 7. Evidencia

- reports/evidently/p1_drift.html
- reports/evidently/p2_drift.html
- reports/evidently/p3_drift.html

Los archivos HTML incluyen las graficas y estadisticas generadas por Evidently.
