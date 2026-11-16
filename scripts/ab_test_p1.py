"""A/B test offline de dos versiones del modelo P1 sobre el mismo hold-out.

Compara MAE, RMSE, R2, sesgo y MAE en horas de alta demanda (>= percentil 90 real),
con un bootstrap pareado por hora (las filas de una misma hora estan correlacionadas)
para saber si la diferencia de MAE es significativa. Registra todo en MLflow y, con
--promote, mueve el alias `champion` a la version ganadora.

Regla de decision (parametrizable): v2 reemplaza a v1 solo si
  1) mejora relativa del MAE >= --min-improvement,
  2) el IC 95% de la diferencia de MAE (v2 - v1) es totalmente < 0, y
  3) el MAE en horas pico no empeora mas de --peak-tolerance.

Uso:
    python scripts/ab_test_p1.py --eval-start 2026-10-16 --v1 1 --v2 2
    python scripts/ab_test_p1.py --eval-start 2026-10-16 --promote
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from loguru import logger
from mlflow.tracking import MlflowClient

from tc5061_proyecto_equipo_44.config import REPORTS_DIR
from tc5061_proyecto_equipo_44.modeling.mlflow_utils import TEAM_TAG, configure_mlflow
from tc5061_proyecto_equipo_44.modeling.p1_trainer import (
    HOURLY_PATH, P1_MODEL_FEATURES as FEATURES, P1_REGISTERED_MODEL_NAME as MODEL_NAME,
)

TARGET = "departures"


def load_holdout(start: str, end: str | None) -> pd.DataFrame:
    """Complete rows of the hourly table with start <= hour < end."""
    filters = [("hour", ">=", pd.Timestamp(start))]
    if end:
        filters.append(("hour", "<", pd.Timestamp(end)))
    df = pd.read_parquet(HOURLY_PATH, columns=["hour", TARGET, *FEATURES], filters=filters)
    df["hour"] = pd.to_datetime(df["hour"])
    df = df.dropna(subset=[TARGET, *FEATURES]).sort_values("hour").reset_index(drop=True)
    if len(df) < 1000:
        last = pd.read_parquet(HOURLY_PATH, columns=["hour"])["hour"].max()
        raise SystemExit(
            f"Hold-out con solo {len(df)} filas; la tabla termina en {last}. "
            "Si el periodo actual no esta, amplia END_MONTH en dataset.py y vuelve a "
            "correr ingesta y features."
        )
    return df


def metrics(y: np.ndarray, pred: np.ndarray, peak: np.ndarray) -> dict:
    err = y - pred
    ss_res, ss_tot = float(np.sum(err**2)), float(np.sum((y - y.mean()) ** 2))
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "r2": 1 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "bias": float(np.mean(pred - y)),
        "mae_peak": float(np.mean(np.abs(err[peak]))),
    }


def paired_bootstrap(diff: np.ndarray, groups: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    """IC 95% de mean(diff) remuestreando grupos (horas) completos."""
    _, inv = np.unique(groups, return_inverse=True)
    sums = np.bincount(inv, weights=diff)
    counts = np.bincount(inv).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sums), size=(n_boot, len(sums)))
    means = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval-start", required=True, help="Inicio del hold-out (el mismo --eval-start/--train-end del entrenamiento de v2).")
    p.add_argument("--eval-end", default=None, help="Fin del hold-out (exclusivo).")
    p.add_argument("--v1", default=None, help="Version vigente (por defecto: alias champion).")
    p.add_argument("--v2", default=None, help="Version alternativa (por defecto: alias challenger).")
    p.add_argument("--min-improvement", type=float, default=0.02)
    p.add_argument("--peak-tolerance", type=float, default=0.02)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--promote", action="store_true", help="Mover alias champion si v2 gana.")
    p.add_argument("--allow-overlap", action="store_true",
                   help="No abortar si alguna version vio el hold-out durante el entrenamiento.")
    args = p.parse_args()

    configure_mlflow()
    client = MlflowClient()
    v1 = str(args.v1 or client.get_model_version_by_alias(MODEL_NAME, "champion").version)
    v2 = str(args.v2 or client.get_model_version_by_alias(MODEL_NAME, "challenger").version)

    # Anti-fuga: ninguna version debe haber entrenado con datos del hold-out.
    for v in (v1, v2):
        end = client.get_model_version(MODEL_NAME, v).tags.get("data_train_end")
        if end is None:
            logger.warning("v{} no tiene el tag data_train_end; no se puede verificar fuga.", v)
        elif pd.Timestamp(end) > pd.Timestamp(args.eval_start) and not args.allow_overlap:
            raise SystemExit(f"v{v} entreno hasta {end} > eval-start {args.eval_start}: hold-out contaminado.")

    df = load_holdout(args.eval_start, args.eval_end)
    info = {"rows": len(df), "start": str(df["hour"].min()), "end": str(df["hour"].max())}
    logger.info("Hold-out: {}", info)

    X, y = df[FEATURES].astype("float64"), df[TARGET].to_numpy()
    peak = y >= np.percentile(y, 90)
    if peak.sum() == 0:  # p90 == 0 (casi todo ceros): usar > 0
        peak = y > 0
    preds = {
        v: mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{v}").predict(X) for v in (v1, v2)
    }
    res = {v: metrics(y, preds[v], peak) for v in (v1, v2)}

    diff = np.abs(y - preds[v2]) - np.abs(y - preds[v1])  # <0 => v2 mejor
    ci_lo, ci_hi = paired_bootstrap(diff, df["hour"].to_numpy(), args.n_boot, args.seed)
    rel = (res[v2]["mae"] - res[v1]["mae"]) / res[v1]["mae"]
    peak_rel = (res[v2]["mae_peak"] - res[v1]["mae_peak"]) / res[v1]["mae_peak"]

    checks = {
        f"mejora MAE >= {args.min_improvement:.0%}": -rel >= args.min_improvement,
        "IC95% diferencia MAE < 0": ci_hi < 0,
        f"MAE pico no empeora > {args.peak_tolerance:.0%}": peak_rel <= args.peak_tolerance,
    }
    v2_wins = all(checks.values())
    winner = v2 if v2_wins else v1

    out = {
        "model": MODEL_NAME, "v1": v1, "v2": v2, "holdout": info,
        "metrics": {f"v{v}": m for v, m in res.items()},
        "mae_diff_v2_minus_v1": float(diff.mean()), "mae_diff_ci95": [ci_lo, ci_hi],
        "mae_rel_change": rel, "mae_peak_rel_change": peak_rel,
        "checks": {k: bool(c) for k, c in checks.items()}, "winner_version": winner,
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fig_dir = REPORTS_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    json_path, fig_path = REPORTS_DIR / "p1_ab_test.json", fig_dir / "p1_ab_mae_by_hour.png"
    json_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8, 5))
    hours = df["hour_of_day"].to_numpy()
    for v in (v1, v2):
        mae_h = [np.mean(np.abs(y - preds[v])[hours == h]) for h in range(24)]
        ax.plot(range(24), mae_h, marker="o", label=f"v{v} (MAE={res[v]['mae']:.3f})")
    ax.set_xlabel("Hora del dia")
    ax.set_ylabel("MAE (salidas/hora)")
    ax.set_title("P1 A/B: MAE por hora en el hold-out")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    with mlflow.start_run(run_name="P1_ab_test"):
        mlflow.set_tags({"problem": "P1", "stage_of_work": "ab_test", "team": TEAM_TAG,
                         "winner_version": winner})
        mlflow.log_params({"v1": v1, "v2": v2, "eval_start": args.eval_start,
                           "eval_rows": info["rows"], "min_improvement": args.min_improvement,
                           "peak_tolerance": args.peak_tolerance})
        for v, m in res.items():
            mlflow.log_metrics({f"v{v}_{k}": val for k, val in m.items()})
        mlflow.log_metrics({"mae_rel_change": rel, "mae_diff_ci95_low": ci_lo,
                            "mae_diff_ci95_high": ci_hi, "mae_peak_rel_change": peak_rel})
        mlflow.log_artifact(str(json_path), "ab_test")
        mlflow.log_artifact(str(fig_path), "ab_test")

    print(f"\n{'metrica':<10}{'v' + v1:>12}{'v' + v2:>12}")
    for k in ("mae", "rmse", "r2", "bias", "mae_peak"):
        print(f"{k:<10}{res[v1][k]:>12.4f}{res[v2][k]:>12.4f}")
    print(f"\nDif. MAE (v2-v1): {diff.mean():+.4f}  IC95% [{ci_lo:+.4f}, {ci_hi:+.4f}]  ({rel:+.1%})")
    for k, c in checks.items():
        print(f"  [{'OK' if c else 'NO'}] {k}")
    print(f"\n=> Version vigente recomendada: v{winner}")

    if args.promote:
        client.set_registered_model_alias(MODEL_NAME, "champion", winner)
        loser = v1 if winner == v2 else v2
        client.set_registered_model_alias(MODEL_NAME, "challenger", loser)
        print(f"Alias champion -> v{winner}")


if __name__ == "__main__":
    main()
