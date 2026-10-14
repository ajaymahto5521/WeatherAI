"""Copy a registered model's joblib file into models/.

Usage:  python scripts/pull_model.py [--name weatherai-forecaster] [--alias candidate] [--out models/forecaster.joblib]
Uses MLFLOW_TRACKING_URI, defaulting to the local mlflow.db.
"""

import argparse
import os
import shutil
import tempfile
from pathlib import Path

import mlflow

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="weatherai-forecaster")
    parser.add_argument("--alias", default="candidate")
    parser.add_argument("--out", default=str(ROOT / "models" / "forecaster.joblib"))
    args = parser.parse_args()

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"))
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(mlflow.artifacts.download_artifacts(f"models:/{args.name}@{args.alias}", dst_path=tmp))
        src = next(local.rglob("*.joblib"))
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, args.out)
    print(f"Pulled {args.name}@{args.alias} -> {args.out}")


if __name__ == "__main__":
    main()
