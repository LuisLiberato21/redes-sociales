from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

# Cargar modelo
model = joblib.load("modelo_emociones.pkl")

@app.get("/")
def root():
    return {"status": "API working successfully 🎉"}

@app.post("/predict")
def predict(data: dict):

    # Convertir a array en el mismo orden que tu modelo usa
    valores = np.array([list(data.values())])

    pred = model.predict(valores)[0]

    return {
        "prediction": int(pred)
    }
