from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import joblib
import numpy as np
import os

app = FastAPI()

# Habilitar CORS (para permitir fetch desde el frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Para producción, reemplaza "*" con tu dominio
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar modelo
model = joblib.load("modelo_emociones.pkl")

# Diccionario de emociones en español según tu encoding
emotion_map = {
    0: "Ira",
    1: "Ansiedad",
    2: "Aburrimiento",
    3: "Felicidad",
    4: "Neutral",
    5: "Tristeza"
}

# Servir frontend
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
def root():
    return FileResponse(os.path.join("frontend", "index.html"))

@app.post("/predict")
def predict(data: dict):
    # Convertir a array en el mismo orden que tu modelo usa
    valores = np.array([list(data.values())])
    pred = model.predict(valores)[0]

    # Obtener probabilidades si el modelo soporta predict_proba
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(valores)[0]
        probabilities = {emotion_map[i]: float(probs[i]) for i in range(len(probs))}
    else:
        probabilities = None

    # Devolver emoción en español y probabilidades
    return {"prediction": emotion_map[int(pred)], "probabilities": probabilities}

