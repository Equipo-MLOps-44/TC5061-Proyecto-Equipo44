"""Analisis de pruebas de carga (Locust + podman stats) y estimacion FinOps.

Lee los CSV generados por Locust (--csv) y los muestreos de `podman stats`
en modo streaming, calcula metricas de desempeno y consumo, estima el costo
por inferencia con precios de referencia y genera las graficas del reporte.

Uso:
    python scripts/finops_load_analysis.py \
        --run-dir reports/locust/run_2026-11-21_streaming \
        --fig-dir reports/figures \
        --out reports/finops_summary.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------------------------
# Escenarios ejecutados (etiqueta de archivo -> descripcion)
# ---------------------------------------------------------------------------
SCENARIOS = [
    # key, prefijo locust, prefijo stats, usuarios, workers
    ("10u", "locust_10u", "podman_stats_10u", 10, 1),
    ("25u", "locust_25u", "podman_stats_25u", 25, 1),
    ("100u", "locust_100u", "podman_stats_100u", 100, 1),
    ("250u", "locust_250u", "podman_stats_250u", 250, 1),
    ("w4_250u", "locust_w4_250u", "podman_stats_w4_250u", 250, 4),
]

# Ventana estable: se excluye el ramp-up y el apagado.
STEADY_START_S = 40
STEADY_END_S = 170

# ---------------------------------------------------------------------------
# Precios de referencia (on-demand, Linux, USD). Ajustar a precios vigentes.
# ---------------------------------------------------------------------------
HOURS_PER_MONTH = 730
PRICES = {
    "t3.small": {"usd_h": 0.0208, "vcpu": 2, "gib": 2, "baseline_vcpu": 0.4},
    "t3.medium": {"usd_h": 0.0416, "vcpu": 2, "gib": 4, "baseline_vcpu": 0.4},
    "c6i.large": {"usd_h": 0.085, "vcpu": 2, "gib": 4, "baseline_vcpu": None},
}
T3_UNLIMITED_USD_PER_VCPU_H = 0.05  # cargo por uso sostenido sobre la linea base
CLOUD_RUN = {
    "usd_vcpu_s": 0.000024,
    "usd_gib_s": 0.0000025,
    "usd_per_million_req": 0.40,
    "vcpu": 1,
    "gib": 0.5,
    "billing_granularity_s": 0.1,
}

# Demanda natural del caso de uso: una prediccion por estacion por hora.
STATIONS = 474

# Paleta (referencia dataviz, modo claro)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
VIOLET = "#6250d6"


def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)


def _new_fig(w=7.0, h=3.6):
    fig, ax = plt.subplots(figsize=(w, h), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    _style_axes(ax)
    return fig, ax


def _title(ax, text):
    ax.set_title(text, loc="left", fontsize=11, color=INK, fontweight="bold", pad=10)


def _cpu_float(series: pd.Series) -> pd.Series:
    return series.astype(str).str.rstrip("%").astype(float)


def _mem_mb(text: str) -> float:
    used = str(text).split("/")[0].strip()
    for unit, factor in (("GB", 1000.0), ("MB", 1.0), ("kB", 0.001)):
        if used.endswith(unit):
            return float(used[: -len(unit)]) * factor
    return float("nan")


def load_scenario(run_dir: Path, key, locust_prefix, stats_prefix, users, workers):
    stats = pd.read_csv(run_dir / f"{locust_prefix}_stats.csv")
    agg = stats[stats["Name"] == "Aggregated"].iloc[0]
    per_ep = stats[stats["Name"] != "Aggregated"].set_index("Name")

    hist = pd.read_csv(run_dir / f"{locust_prefix}_stats_history.csv")
    hist = hist[hist["Name"] == "Aggregated"].copy()
    hist["t"] = hist["Timestamp"] - hist["Timestamp"].iloc[0]
    steady_hist = hist[(hist["t"] >= STEADY_START_S) & (hist["t"] <= STEADY_END_S)]
    steady_rps = float(steady_hist["Requests/s"].mean())

    ps = pd.read_csv(run_dir / f"{stats_prefix}.csv")
    ps["cpu"] = _cpu_float(ps["cpu_perc"])
    ps["mem_mb"] = ps["mem_usage"].map(_mem_mb)
    api = ps[ps["name"].str.startswith("mibici-api")]
    api_steady = api[(api["elapsed_s"] >= STEADY_START_S) & (api["elapsed_s"] <= STEADY_END_S)]
    mlflow = ps[ps["name"] == "mibici-mlflow"]

    cpu_cores = float(api_steady["cpu"].mean()) / 100.0
    return {
        "key": key,
        "users": users,
        "workers": workers,
        "requests": int(agg["Request Count"]),
        "failures": int(agg["Failure Count"]),
        "rps_avg_run": float(agg["Requests/s"]),
        "rps_steady": steady_rps,
        "lat_avg_ms": float(agg["Average Response Time"]),
        "lat_p50_ms": float(agg["50%"]),
        "lat_p95_ms": float(agg["95%"]),
        "lat_p99_ms": float(agg["99%"]),
        "p1_p50_ms": float(per_ep.loc["P1 - Regression", "50%"]),
        "p3_p50_ms": float(per_ep.loc["P3 - Clustering", "50%"]),
        "api_cpu_cores_steady": cpu_cores,
        "api_cpu_cores_max": float(api["cpu"].max()) / 100.0,
        "api_mem_mb": float(api["mem_mb"].median()),
        "mlflow_mem_mb": float(mlflow["mem_mb"].median()) if len(mlflow) else None,
        "mlflow_cpu_cores": float(mlflow["cpu"].mean()) / 100.0 if len(mlflow) else None,
        "cpu_ms_per_request": 1000.0 * cpu_cores / steady_rps,
        "_hist": hist,
        "_api": api,
    }


# ---------------------------------------------------------------------------
# Modelo de costos
# ---------------------------------------------------------------------------
def vm_cost_per_hour(instance: str, used_vcpu: float) -> float:
    p = PRICES[instance]
    extra = 0.0
    if p["baseline_vcpu"] is not None and used_vcpu > p["baseline_vcpu"]:
        extra = (used_vcpu - p["baseline_vcpu"]) * T3_UNLIMITED_USD_PER_VCPU_H
    return p["usd_h"] + extra


def vm_cost_per_million(instance, rps, cpu_s_per_req, capacity_rps, fixed_vcpu=0.0):
    """Costo por millon de inferencias en VMs 24/7 escalando por instancias."""
    n_inst = max(1, math.ceil(rps / capacity_rps))
    used_vcpu_each = fixed_vcpu + (rps / n_inst) * cpu_s_per_req
    usd_h = n_inst * vm_cost_per_hour(instance, used_vcpu_each)
    return usd_h / (rps * 3600) * 1e6, usd_h


def cloud_run_cost_per_million(latency_s: float, rps: float | None = None,
                               capacity_rps: float | None = None) -> float:
    """Costo serverless por millon de inferencias.

    Cota superior (trafico bajo, sin peticiones simultaneas): cada peticion se
    factura por su duracion redondeada. Con trafico alto las peticiones
    comparten instancia (concurrencia) y se factura el tiempo de instancia
    activa; se toma el menor de ambos.
    """
    g = CLOUD_RUN["billing_granularity_s"]
    usd_s = CLOUD_RUN["vcpu"] * CLOUD_RUN["usd_vcpu_s"] + CLOUD_RUN["gib"] * CLOUD_RUN["usd_gib_s"]
    billed_s = math.ceil(latency_s / g) * g
    per_request = billed_s * usd_s * 1e6
    if rps and capacity_rps:
        n_inst = max(1, math.ceil(rps / capacity_rps))
        per_instance_time = n_inst * usd_s * 3600 / (rps * 3600) * 1e6
        per_request = min(per_request, per_instance_time)
    return per_request + CLOUD_RUN["usd_per_million_req"]


def build_cost_model(res: dict) -> dict:
    cpu_s = (res["25u"]["cpu_ms_per_request"] + res["100u"]["cpu_ms_per_request"]) / 2000.0
    cap_1w = res["250u"]["rps_steady"]  # saturacion medida con 1 worker
    cap_1w_safe = 0.7 * cap_1w
    cap_2vcpu_safe = 0.7 * 2 / cpu_s  # 2 workers en 2 vCPU al 70 %
    mlflow_vcpu = res["25u"]["mlflow_cpu_cores"] or 0.0

    natural_rps = STATIONS * 24 / 86400
    observed = res["25u"]["rps_steady"]

    options = {
        "A_compose_t3.medium": dict(
            label="A. Actual: API (1 worker) + MLflow en t3.medium",
            instance="t3.medium", capacity=cap_1w_safe, fixed_vcpu=mlflow_vcpu,
        ),
        "B_api_t3.small": dict(
            label="B. Solo API (1 worker) en t3.small",
            instance="t3.small", capacity=cap_1w_safe, fixed_vcpu=0.0,
        ),
        "C_api_c6i.large": dict(
            label="C. Solo API (2 workers) en c6i.large",
            instance="c6i.large", capacity=cap_2vcpu_safe, fixed_vcpu=0.0,
        ),
    }

    table = []
    for key, o in options.items():
        row = {"key": key, "label": o["label"], "instance": o["instance"],
               "usd_h_list": PRICES[o["instance"]]["usd_h"],
               "capacity_rps_70pct": o["capacity"]}
        for name, rps in (("natural", natural_rps), ("observed_25u", observed),
                          ("high_80rps", 80.0)):
            cpm, usd_h = vm_cost_per_million(o["instance"], rps, cpu_s, o["capacity"], o["fixed_vcpu"])
            row[f"usd_per_million_{name}"] = cpm
            row[f"usd_month_{name}"] = usd_h * HOURS_PER_MONTH
        table.append(row)

    lat_s = res["25u"]["lat_p50_ms"] / 1000.0
    cr_cap = 0.7 * CLOUD_RUN["vcpu"] / cpu_s
    cr_cpm = cloud_run_cost_per_million(lat_s)
    cr_row = {
        "key": "D_cloud_run",
        "label": "D. Solo API serverless (Cloud Run, escala a cero)",
        "instance": "Cloud Run 1 vCPU / 0.5 GiB",
        "capacity_rps_70pct": cr_cap,
    }
    for name, rps in (("natural", natural_rps), ("observed_25u", observed), ("high_80rps", 80.0)):
        cpm = cloud_run_cost_per_million(lat_s, rps, cr_cap)
        cr_row[f"usd_per_million_{name}"] = cpm
        cr_row[f"usd_month_{name}"] = cpm * rps * 3600 * HOURS_PER_MONTH / 1e6
    table.append(cr_row)

    b_month = PRICES["t3.small"]["usd_h"] * HOURS_PER_MONTH
    breakeven_req_month = b_month / (cr_cpm / 1e6)

    return {
        "cpu_s_per_request": cpu_s,
        "capacity_1worker_rps_measured": cap_1w,
        "capacity_1worker_rps_70pct": cap_1w_safe,
        "capacity_2vcpu_rps_70pct": cap_2vcpu_safe,
        "mlflow_vcpu_idle": mlflow_vcpu,
        "natural_rps": natural_rps,
        "natural_requests_month": STATIONS * 24 * 30,
        "observed_rps_25u": observed,
        "cloud_run_usd_per_million": cr_cpm,
        "cloud_run_capacity_rps_70pct": cr_cap,
        "_cr_lat_s": lat_s,
        "breakeven_B_vs_D_requests_month": breakeven_req_month,
        "breakeven_B_vs_D_rps": breakeven_req_month / (HOURS_PER_MONTH * 3600),
        "marginal_cpu_usd_per_million_c6i": cpu_s * PRICES["c6i.large"]["usd_h"] / 2 / 3600 * 1e6,
        "options": table,
        "prices": PRICES,
        "cloud_run_prices": CLOUD_RUN,
        "t3_unlimited_usd_per_vcpu_h": T3_UNLIMITED_USD_PER_VCPU_H,
        "_opts": options,
    }


# ---------------------------------------------------------------------------
# Graficas
# ---------------------------------------------------------------------------
def fig_throughput(res, path):
    one = [res[k] for k in ("10u", "25u", "100u", "250u")]
    w4 = res["w4_250u"]
    fig, ax = _new_fig()
    xs = [r["users"] for r in one]
    ys = [r["rps_steady"] for r in one]
    demand = [u / 2.0 for u in xs]  # wait_time between(1,3) -> ~0.5 req/s por usuario
    ax.plot(xs, demand, color=INK_2, linewidth=1.2, linestyle=(0, (4, 3)))
    ax.plot(xs, ys, color=BLUE, linewidth=2, marker="o", markersize=6,
            markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax.plot([w4["users"]], [w4["rps_steady"]], color=ORANGE, marker="D", markersize=7,
            linestyle="none", markeredgecolor=SURFACE, markeredgewidth=1.5)
    for r in one[:-1]:
        ax.annotate(f"{r['rps_steady']:.1f}", (r["users"], r["rps_steady"]),
                    textcoords="offset points", xytext=(-6, 8), ha="right", fontsize=8, color=INK)
    ax.annotate(f"1 worker, saturado: {ys[-1]:.1f}", (xs[-1], ys[-1]),
                textcoords="offset points", xytext=(-10, 8), ha="right", va="bottom",
                fontsize=8, color=INK)
    ax.annotate(f"4 workers: {w4['rps_steady']:.1f}", (w4["users"], w4["rps_steady"]),
                textcoords="offset points", xytext=(-10, 0), ha="right", va="center",
                fontsize=8, color=INK)
    ax.text(150, 82, "demanda generada\n(~0.5 req/s por usuario)", ha="right", va="bottom",
            fontsize=8, color=INK_2)
    ax.set_xlabel("Usuarios concurrentes simulados")
    ax.set_ylabel("Peticiones por segundo (estable)")
    ax.set_xlim(0, 265)
    ax.set_ylim(0, 140)
    _title(ax, "Rendimiento: la API con 1 worker se satura cerca de 90 req/s")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def fig_latency(res, path):
    keys = ["10u", "25u", "100u", "250u", "w4_250u"]
    labels = ["10 usuarios", "25 usuarios", "100 usuarios", "250 usuarios", "250 usuarios\n4 workers"]
    p50 = [res[k]["lat_p50_ms"] for k in keys]
    p95 = [res[k]["lat_p95_ms"] for k in keys]
    fig, ax = _new_fig()
    x = range(len(keys))
    w = 0.36
    b1 = ax.bar([i - w / 2 - 0.01 for i in x], p50, width=w, color=BLUE, label="Mediana (p50)")
    b2 = ax.bar([i + w / 2 + 0.01 for i in x], p95, width=w, color=ORANGE, label="Percentil 95")
    for bars in (b1, b2):
        for b in bars:
            ax.annotate(f"{b.get_height():.0f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                        textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8, color=INK)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Tiempo de respuesta (ms)")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    _title(ax, "Latencia por escenario (ms)")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def fig_cpu_timeseries(res, path):
    series = [("10u", "10 usuarios", AQUA), ("25u", "25 usuarios", BLUE),
              ("100u", "100 usuarios", VIOLET), ("250u", "250 usuarios", ORANGE)]
    fig, ax = _new_fig(h=3.8)
    for key, label, color in series:
        api = res[key]["_api"]
        ax.plot(api["elapsed_s"], api["cpu"], color=color, linewidth=1.6, label=label)
    ax.axhline(100, color=INK_2, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(178, 103, "100 % = 1 núcleo", fontsize=8, color=INK_2, ha="right")
    ax.set_xlabel("Tiempo desde el inicio de la prueba (s)")
    ax.set_ylabel("CPU del contenedor API (%)")
    ax.set_ylim(0, 140)
    ax.legend(frameon=False, fontsize=8.5, ncol=4, loc="upper left", bbox_to_anchor=(0, -0.2))
    _title(ax, "CPU de la API (podman stats, cada 2 s), 1 worker")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def fig_cpu_per_rps(res, path):
    one = [res[k] for k in ("10u", "25u", "100u", "250u")]
    fig, ax = _new_fig()
    xs = [r["rps_steady"] for r in one]
    ys = [r["api_cpu_cores_steady"] for r in one]
    cpu_s = res["_cost"]["cpu_s_per_request"]
    line_x = [0, 100]
    ax.plot(line_x, [v * cpu_s for v in line_x], color=INK_2, linewidth=1.2, linestyle=(0, (4, 3)))
    ax.plot(xs, ys, color=BLUE, marker="o", markersize=7, linestyle="none",
            markeredgecolor=SURFACE, markeredgewidth=1.5)
    w4 = res["w4_250u"]
    ax.plot([w4["rps_steady"]], [w4["api_cpu_cores_steady"]], color=ORANGE, marker="D",
            markersize=7, linestyle="none", markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax.text(w4["rps_steady"] - 3, w4["api_cpu_cores_steady"], "4 workers", ha="right",
            va="center", fontsize=8, color=ORANGE)
    ax.text(20, 1.0, f"Pendiente ≈ {cpu_s * 1000:.1f} ms de CPU por inferencia",
            fontsize=8.5, color=INK_2)
    ax.set_xlabel("Peticiones por segundo (estable)")
    ax.set_ylabel("Núcleos de CPU usados (API)")
    ax.set_xlim(0, 125)
    ax.set_ylim(0, 2.0)
    _title(ax, "El consumo de CPU crece linealmente con el tráfico")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def fig_memory(res, path):
    items = [
        ("MLflow server (inactivo)", res["25u"]["mlflow_mem_mb"], ORANGE),
        ("API, 4 workers", res["w4_250u"]["api_mem_mb"], BLUE),
        ("API, 1 worker", res["25u"]["api_mem_mb"], BLUE),
    ]
    fig, ax = _new_fig(h=2.6)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    names = [i[0] for i in items]
    vals = [i[1] for i in items]
    bars = ax.barh(names, vals, color=[i[2] for i in items], height=0.55)
    for b, v in zip(bars, vals):
        ax.text(v + 30, b.get_y() + b.get_height() / 2, f"{v:,.0f} MB", va="center",
                fontsize=8.5, color=INK)
    ax.set_xlabel("Memoria usada (MB)")
    ax.set_xlim(0, 2900)
    _title(ax, "Memoria por contenedor (MB)")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def fig_cost_curve(res, path):
    cost = res["_cost"]
    cpu_s = cost["cpu_s_per_request"]
    rps_grid = [10 ** (e / 20) for e in range(-40, 47)]  # 0.01 .. ~200 req/s
    fig, ax = _new_fig(h=4.0)
    colors = {"A_compose_t3.medium": ORANGE, "B_api_t3.small": BLUE, "C_api_c6i.large": VIOLET}
    for key, o in cost["_opts"].items():
        ys = [vm_cost_per_million(o["instance"], r, cpu_s, o["capacity"], o["fixed_vcpu"])[0]
              for r in rps_grid]
        ax.plot(rps_grid, ys, color=colors[key], linewidth=2, label=o["label"])
    cr = [cloud_run_cost_per_million(cost["_cr_lat_s"], r, cost["cloud_run_capacity_rps_70pct"])
          for r in rps_grid]
    ax.plot(rps_grid, cr, color=AQUA, linewidth=2,
            label="D. Serverless (Cloud Run, escala a cero)")
    ax.axvline(cost["natural_rps"], color=INK_2, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(cost["natural_rps"] * 1.12, 2.2e3, "demanda natural\n474 estaciones × 24 h",
            fontsize=8, color=INK_2)
    ax.axvline(cost["observed_rps_25u"], color=INK_2, linewidth=1, linestyle=(0, (1, 2)))
    ax.text(cost["observed_rps_25u"] * 1.12, 2.2e3, "prueba\n25 usuarios", fontsize=8, color=INK_2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_xlabel("Tráfico promedio sostenido (req/s, escala log)")
    ax.set_ylabel("USD por millón de inferencias (log)")
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.17), ncol=2)
    _title(ax, "Costo por inferencia según tráfico y opción de despliegue")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="reports/locust/run_2026-11-21_streaming")
    parser.add_argument("--fig-dir", default="reports/figures")
    parser.add_argument("--out", default="reports/finops_summary.json")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    res = {s[0]: load_scenario(run_dir, *s) for s in SCENARIOS}
    res["_cost"] = build_cost_model(res)

    figures = {
        "throughput": fig_dir / "load_throughput.png",
        "latency": fig_dir / "load_latency.png",
        "cpu_timeseries": fig_dir / "load_cpu_timeseries.png",
        "cpu_per_rps": fig_dir / "load_cpu_per_rps.png",
        "memory": fig_dir / "load_memory.png",
        "cost_curve": fig_dir / "finops_cost_per_million.png",
    }
    fig_throughput(res, figures["throughput"])
    fig_latency(res, figures["latency"])
    fig_cpu_timeseries(res, figures["cpu_timeseries"])
    fig_cpu_per_rps(res, figures["cpu_per_rps"])
    fig_memory(res, figures["memory"])
    fig_cost_curve(res, figures["cost_curve"])

    summary = {
        "scenarios": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                      for k, v in res.items() if not k.startswith("_")},
        "finops": {k: v for k, v in res["_cost"].items() if not k.startswith("_")},
        "figures": {k: str(v).replace("\\", "/") for k, v in figures.items()},
    }
    Path(args.out).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
