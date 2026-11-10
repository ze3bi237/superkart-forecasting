# Flask backend that serves the serialized SuperKart sales model
import io
import os

import joblib
import pandas as pd
from flask import Flask, jsonify, request

# Features the model was trained on (order matters only for readability - the pipeline selects columns by name)
FEATURES = [
    "Product_Weight",
    "Product_Sugar_Content",
    "Product_Allocated_Area",
    "Product_MRP",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
    "Product_Id_char",
    "Store_Age_Years",
    "Product_Type_Category",
]
NUMERIC = ["Product_Weight", "Product_Allocated_Area", "Product_MRP", "Store_Age_Years"]

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "superkart_model.joblib")
model = joblib.load(MODEL_PATH)

superkart_api = Flask("SuperKart Sales Predictor")


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Validate columns, coerce numeric types and return the frame in model order."""
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature(s): {missing}")
    df = df[FEATURES].copy()
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="raise")
    return df


@superkart_api.get("/")
def home():
    return "Welcome to the SuperKart Sales Prediction API!"


@superkart_api.get("/health")
def health():
    return jsonify({"status": "ok"})


@superkart_api.post("/v1/predict")
def predict():
    """Online inference: one JSON object with the ten features -> predicted sales for that product in that store."""
    try:
        payload = request.get_json(force=True)
        if not isinstance(payload, dict):
            raise ValueError("Request body must be a single JSON object.")
        frame = _clean(pd.DataFrame([payload]))
        prediction = float(model.predict(frame)[0])
        return jsonify({"Predicted_Sales": round(max(prediction, 0.0), 2)})
    except Exception as exc:  # noqa: BLE001 - report any bad input as a 400
        return jsonify({"error": str(exc)}), 400


@superkart_api.post("/v1/predictbatch")
def predict_batch():
    """Batch inference: a CSV upload (form field 'file') -> {row_index: predicted_sales}."""
    try:
        upload = request.files.get("file")
        if upload is None:
            raise ValueError("Upload a CSV under the form field 'file'.")
        frame = _clean(pd.read_csv(io.BytesIO(upload.read())))
        predictions = model.predict(frame).clip(min=0)
        return jsonify({str(i): round(float(p), 2) for i, p in zip(frame.index, predictions)})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    superkart_api.run(host="0.0.0.0", port=7860)
