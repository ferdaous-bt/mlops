# ===== 1. Importer les bibliothèques =====
from typing import Literal

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# ===== 2. Créer l'application =====
app = FastAPI(title="API Churn")

# ===== 3. Charger le modèle une seule fois =====
mlflow.set_tracking_uri("sqlite:///mlflow.db")
model = mlflow.sklearn.load_model("models:/churn-model@champion")


# ===== 4. Décrire les données attendues =====
class Client(BaseModel):
    age: int
    salaire: float
    abonnement: Literal["Basique", "Standard", "Premium"]
    anciennete: float
    consommation: float


# ===== 5. Vérifier que l'API fonctionne =====
@app.get("/health")
def health():
    return {"status": "ok"}


# ===== 6. Prédire le départ d'un client =====
@app.post("/predict")
def predict(client: Client):
    # Transformer les données en tableau d'une ligne
    ligne = pd.DataFrame([client.model_dump()])

    # Encoder l'abonnement en colonnes contenant 0 ou 1
    ligne = pd.get_dummies(
        ligne,
        columns=["abonnement"],
        dtype=int,
    )

    # Retrouver les colonnes et l'ordre utilisés à l'entraînement
    ligne = ligne.reindex(
        columns=model.feature_names_in_,
        fill_value=0,
    )

    # Prédire : 1 = part, 0 = reste
    resultat = int(model.predict(ligne)[0])

    return {
        "churn": resultat,
        "message": (
            "Le client va partir"
            if resultat == 1
            else "Le client reste"
        ),
    }