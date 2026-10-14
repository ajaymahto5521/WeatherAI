"""Minimal MLflow model wrapper so a joblib artifact can be versioned in the registry.

Serving doesn't load through MLflow; the registry is the versioned source that
scripts/pull_model.py copies models/<file>.joblib out of.
"""

from mlflow.models import set_model
from mlflow.pyfunc import PythonModel


class JoblibArtifact(PythonModel):
    def predict(self, context, model_input, params=None):
        raise NotImplementedError("Load the joblib artifact with app.models.forecasting instead.")


set_model(JoblibArtifact())
