# app.py

from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
from model import __version__ as model_version
from model import preprocess

# -------------------------------------------------------------------
# 1. Inicialización de la app
# -------------------------------------------------------------------
app = FastAPI(
    title="Cancellation Prediction API",
    description="API para predecir cancelaciones usando XGBoost",
    version="1.0.0",
)

# -------------------------------------------------------------------
# 2. Carga del modelo al arrancar
# -------------------------------------------------------------------
MODEL_PATH = "xgboost_model.pkl"
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    raise RuntimeError(f"No se pudo cargar el modelo en {MODEL_PATH}: {e}")

# -------------------------------------------------------------------
# 3. Definición del esquema de entrada con Pydantic
#    Aquí definimos un payload genérico como dict de key/valor,
#    pero podrías crear atributos concretos si conoces tus features.
# -------------------------------------------------------------------
class PredictionRequest(BaseModel):
    records: List[Dict[str, Any]]

# -------------------------------------------------------------------
# 4. Endpoints
# -------------------------------------------------------------------
@app.get("/health", tags=["salud"])
def health():
    """
    Verifica que la API está corriendo.
    """
    return {"status": "ok", "model_version": model_version}


@app.post("/predict", tags=["predicción"])
def predict(request: PredictionRequest):
    """
    Recibe una lista de registros (records) en JSON,
    los convierte en DataFrame, aplica preprocesamiento
    y devuelve las predicciones.
    """
    try:
        # 4.1 Construir DataFrame
        df = pd.DataFrame(request.records)
        # 4.2 Preprocesar (si aplica)
        df_processed = preprocess(df)
        # 4.3 Predecir
        preds = model.predict(df_processed)
        # 4.4 Formatear resultado
        return {
            "predictions": preds.tolist(),
            "n_records": len(preds),
            "model_version": model_version
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en predicción: {e}")

# -------------------------------------------------------------------
# 5. Para arrancar localmente:
#    uvicorn app:app --reload --host 0.0.0.0 --port 8001
# -------------------------------------------------------------------
