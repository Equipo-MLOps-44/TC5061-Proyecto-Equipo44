# tc5061_proyecto_equipo_44/modeling/train.py
import argparse
from loguru import logger

from tc5061_proyecto_equipo_44.config import PROCESSED_DATA_DIR
from .mlflow_utils import configure_mlflow
from .p1_trainer import train_p1
from .p2_trainer import train_p2, search_p2
from .p3_trainer import train_p3


HOURLY_PATH = PROCESSED_DATA_DIR / "hourly_features.parquet"
PROFILES_PATH = PROCESSED_DATA_DIR / "station_profiles.parquet"


def main() -> None:
    """Train only the selected question/model and track it in MLflow."""

    # Lectura del cli
    parser = argparse.ArgumentParser(description="Train a selected MiBici model.")
    parser.add_argument("--question", choices=["p1", "p2", "p3"], default="p1")
    parser.add_argument("--model",choices=["baseline", "rf", "rf_search", "k_means", "search"],default="baseline",)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=15)
    parser.add_argument("--class-weight", type=str, default="balanced")
    parser.add_argument("--n-clusters", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--n-init", type=int, default=10)
    parser.add_argument("--init", type=str, default="k-means++", choices=["k-means++", "random"])
    # Continuous training (P1): explicit data windows and Model Registry registration.
    # "none" (the MLproject default) is treated as "not provided".
    opt_date = lambda v: None if v is None or v.lower() == "none" else v  # noqa: E731
    parser.add_argument("--train-start", type=opt_date, default=None, help="P1: inicio de la ventana de entrenamiento (inclusive).")
    parser.add_argument("--train-end", type=opt_date, default=None, help="P1: fin de entrenamiento (exclusivo). Activa el split por ventanas.")
    parser.add_argument("--eval-start", type=opt_date, default=None, help="P1: inicio del hold-out (por defecto, --train-end).")
    parser.add_argument("--eval-end", type=opt_date, default=None, help="P1: fin del hold-out (exclusivo).")
    parser.add_argument("--register", nargs="?", const=True, default=False,
                        type=lambda v: str(v).lower() in ("1", "true", "yes"),
                        help="P1 rf: registrar como nueva version en el Model Registry.")
    args = parser.parse_args()

    allowed = {"p1": {"baseline", "rf", "search"}, "p2": {"baseline", "rf", "rf_search"}, "p3": {"k_means", "search"}}
    if args.model not in allowed[args.question]:
        parser.error(f"model={args.model} is not valid for question={args.question}; choose {sorted(allowed[args.question])}.")
    if args.n_estimators < 1 or args.max_depth < 1 or args.n_clusters < 2 or args.n_init < 1:
        parser.error("n_estimators, max_depth, n_init must be positive and n_clusters must be at least 2.")

    configure_mlflow()
    required_path = PROFILES_PATH if args.question == "p3" else HOURLY_PATH
    if not required_path.exists():
        raise FileNotFoundError(f"Required feature dataset not found: {required_path}")

    if args.question == "p1":
        result = train_p1(
            args.model, args.n_estimators, args.max_depth, args.random_state,
            train_start=args.train_start, train_end=args.train_end,
            eval_start=args.eval_start, eval_end=args.eval_end, register=args.register,
        )
    elif args.question == "p2":
        if args.model == "rf_search":
            result = search_p2(n_estimators=args.n_estimators,random_state=args.random_state,)
        else:
            result = train_p2(model_name=args.model,n_estimators=args.n_estimators,max_depth=args.max_depth,class_weight=args.class_weight,random_state=args.random_state,)
    else:
        result = train_p3(args.model, args.n_clusters, args.random_state, args.n_init, args.init)

    logger.info("Training completed: question={}, model={}", args.question, args.model)
    logger.info("Result summary: {}", result)


if __name__ == "__main__":
    main()