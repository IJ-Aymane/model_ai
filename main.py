from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import uvicorn

# ============================================================
# INITIALISATION
# ============================================================
app = FastAPI(
    title="🏥 API Diagnostic Médical",
    description="API de prédiction de maladies basée sur les symptômes (BernoulliNB - 85% accuracy)",
    version="1.0.0",
)

# --- CORS CONFIGURATION ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chargement du modèle au démarrage
print("🔄 Chargement du modèle médical...")
try:
    model_data = joblib.load("bernoulli_nb_medical_model.pkl")
    model = model_data['model']
    le = model_data['label_encoder']
    symptoms_list = model_data['symptoms']
    diseases_list = model_data['diseases']
    metrics = model_data['metrics']
    
    print(f"✅ Modèle chargé: {metrics['accuracy']*100:.1f}% accuracy")
    print(f"📋 Nombre de symptômes disponibles: {len(symptoms_list)}")
    print(f"🏥 Nombre de maladies: {len(diseases_list)}")
except FileNotFoundError:
    print("❌ Erreur: Fichier 'bernoulli_nb_medical_model.pkl' non trouvé!")
    model = None
    le = None
    symptoms_list = []
    diseases_list = []
    metrics = {'accuracy': 0.0}

# ============================================================
# PYDANTIC MODELS
# ============================================================
class PredictionRequest(BaseModel):
    symptoms: List[str] = Field(..., min_items=1, description="Liste des symptômes du patient")
    top_n: Optional[int] = Field(default=3, ge=1, le=10, description="Nombre de diagnostics à retourner")

class DiagnosisResult(BaseModel):
    disease: str
    probability: float
    rank: int

class PredictionResponse(BaseModel):
    predicted_disease: str
    confidence: float
    top_diagnoses: List[DiagnosisResult]
    symptoms_checked: List[str]
    symptoms_unknown: List[str]
    timestamp: str

class SymptomResponse(BaseModel):
    symptoms: List[str]
    count: int

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    accuracy: float
    f1_score: Optional[float] = None
    total_symptoms: int
    total_diseases: int

# ============================================================
# FONCTIONS UTILITAIRES
# ============================================================
def predict_disease_api(symptom_list: List[str], top_n: int = 3) -> Dict:
    """Prédit les maladies probables basées sur les symptômes"""
    
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Le modèle n'est pas chargé. Vérifiez que le fichier .pkl existe."
        )
    
    # 1. Normalize list for comparison
    available_symptoms_lower = [s.lower() for s in symptoms_list]
    
    valid_symptoms = []
    invalid_symptoms = []
    
    for sym in symptom_list:
        if sym.lower() in available_symptoms_lower:
            # Find the exact original case name from the list
            idx = available_symptoms_lower.index(sym.lower())
            valid_symptoms.append(symptoms_list[idx])
        else:
            invalid_symptoms.append(sym)
    
    if not valid_symptoms:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Aucun symptôme valide trouvé parmi: {symptom_list}"
        )
    
    # 2. Prepare prediction data
    patient_df = pd.DataFrame(np.zeros((1, len(symptoms_list))), columns=symptoms_list)
    for sym in valid_symptoms:
        patient_df[sym] = 1
    
    # 3. Predict
    probabilities = model.predict_proba(patient_df)[0]
    top_indices = np.argsort(probabilities)[-top_n:][::-1]
    
    results = []
    for rank, idx in enumerate(top_indices, 1):
        disease = le.inverse_transform([idx])[0]
        prob = float(probabilities[idx])
        results.append({
            "disease": disease,
            "probability": round(prob, 4),
            "rank": rank
        })
    
    return {
        "predicted_disease": results[0]["disease"],
        "confidence": results[0]["probability"],
        "top_diagnoses": results,
        "symptoms_checked": valid_symptoms,
        "symptoms_unknown": invalid_symptoms,
        "timestamp": datetime.now().isoformat()
    }

# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/", tags=["Info"])
async def root():
    """Page d'accueil de l'API"""
    return {
        "message": "🏥 API Diagnostic Médical",
        "version": "1.0.0",
        "status": "online",
        "docs": "/docs",
        "endpoints": {
            "health": "/health",
            "symptoms": "/symptoms",
            "predict": "/predict",
            "diseases": "/diseases"
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["Info"])
async def health_check():
    """Vérifier l'état de l'API et du modèle"""
    return {
        "status": "healthy" if model is not None else "model_not_loaded",
        "model_loaded": model is not None,
        "accuracy": metrics.get('accuracy', 0.0),
        "f1_score": metrics.get('f1_score', None),
        "total_symptoms": len(symptoms_list),
        "total_diseases": len(diseases_list)
    }

@app.get("/symptoms", response_model=SymptomResponse, tags=["Data"])
async def get_symptoms():
    """Récupérer la liste complète des symptômes disponibles"""
    if not symptoms_list:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="La liste des symptômes n'est pas disponible"
        )
    
    return {
        "symptoms": sorted(symptoms_list),  # Sort alphabetically for easier use
        "count": len(symptoms_list)
    }

@app.get("/diseases", tags=["Data"])
async def get_diseases():
    """Récupérer la liste des maladies que le modèle peut diagnostiquer"""
    if not diseases_list:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="La liste des maladies n'est pas disponible"
        )
    
    return {
        "diseases": sorted(diseases_list),
        "count": len(diseases_list)
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_disease(request: PredictionRequest):
    """
    Prédire les maladies probables basées sur les symptômes fournis
    
    - **symptoms**: Liste des symptômes (ex: ["fever", "cough", "headache"])
    - **top_n**: Nombre de diagnostics à retourner (défaut: 3)
    """
    try:
        result = predict_disease_api(request.symptoms, request.top_n)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de la prédiction: {str(e)}"
        )

@app.get("/symptoms/search/{query}", tags=["Data"])
async def search_symptoms(query: str):
    """Rechercher des symptômes par mot-clé"""
    if not symptoms_list:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="La liste des symptômes n'est pas disponible"
        )
    
    query_lower = query.lower()
    matching = [s for s in symptoms_list if query_lower in s.lower()]
    
    return {
        "query": query,
        "matches": matching,
        "count": len(matching)
    }

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🏥 Démarrage du serveur API Diagnostic Médical")
    print("="*60)
    print(f"📍 URL: http://127.0.0.1:8000")
    print(f"📖 Documentation: http://127.0.0.1:8000/docs")
    print(f"🔬 Modèle: {'Chargé' if model else 'Non chargé'}")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
