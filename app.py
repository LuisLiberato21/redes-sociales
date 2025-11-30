from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import joblib
import pandas as pd
import numpy as np
import os

app = FastAPI()

# Habilitar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, reemplazar "*" por dominio real
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar modelo
model = joblib.load("modelo_emociones.pkl")

# Mapeo emociones
emotion_map = {
    0: "Ira",
    1: "Ansiedad",
    2: "Aburrimiento",
    3: "Felicidad",
    4: "Neutral",
    5: "Tristeza"
}

# Orden correcto de columnas del modelo
FEATURE_COLUMNS = [
    "Age",
    "Gender",
    "Platform",
    "Daily_Usage_Time (minutes)",
    "Posts_Per_Day",
    "Likes_Received_Per_Day",
    "Comments_Received_Per_Day",
    "Messages_Sent_Per_Day"
]

# Servir frontend
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")

@app.get("/")
def root():
    return FileResponse(os.path.join("frontend", "index.html"))

@app.get("/prediccion.html")
def prediccion():
    return FileResponse(os.path.join("frontend", "prediccion.html"))


@app.post("/predict")
def predict(data: dict):

    # Crear DataFrame con orden EXACTO de columnas
    df = pd.DataFrame([[data[col] for col in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)

    # Predicción
    pred = model.predict(df)[0]

    # Probabilidades (si el modelo las soporta)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(df)[0]
        probabilities = {
            emotion_map[i]: float(probs[i]) for i in range(len(probs))
        }
    else:
        probabilities = None

    return {
        "prediction": emotion_map[int(pred)],
        "probabilities": probabilities
    }
