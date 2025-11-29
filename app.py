from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import joblib
import numpy as np
import os

app = FastAPI()

# Cargar modelo
model = joblib.load("modelo_emociones.pkl")

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
    return {"prediction": int(pred)}
