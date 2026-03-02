import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score
import time
import os
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("🏥 ENTRAÎNEMENT BERNOULLINB - DIAGNOSTIC MÉDICAL")
print("=" * 60)

# 1. Chargement
print("\n Chargement...")
df = pd.read_csv("Diseases_and_Symptoms.csv")
print(f"   • {len(df):,} échantillons")
print(f"   • {df['diseases'].nunique()} maladies")
print(f"   • {len(df.columns)-1} symptômes")

# 2. Préparation
X = df.drop(columns=["diseases"])
y = df["diseases"]

# Filtrer classes rares
min_samples = 2
counts = y.value_counts()
valid_classes = counts[counts >= min_samples].index
mask = y.isin(valid_classes)
X, y = X[mask], y[mask]

le = LabelEncoder()
y_enc = le.fit_transform(y)
print(f"   • {len(le.classes_)} maladies après filtrage")

# 3. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)
print(f"\n  Split: {len(X_train):,} train / {len(X_test):,} test")

# 4. Entraînement
print("\n  Entraînement BernoulliNB...")
start = time.time()

model = BernoulliNB(alpha=1.0)
model.fit(X_train, y_train)

print(f"    Terminé en {time.time() - start:.1f}s")

# 5. Évaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"\n📊 Résultats:")
print(f"   Accuracy : {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"   F1-Score : {f1:.4f}")

# 6. Sauvegarde
print(f"\n Sauvegarde...")
model_data = {
    "model": model,
    "label_encoder": le,
    "symptoms": list(X.columns),
    "diseases": list(le.classes_),
    "metrics": {
        "accuracy": float(accuracy),
        "f1_score": float(f1),
        "train_samples": len(X_train),
        "test_samples": len(X_test)
    }
}

filename = "bernoulli_nb_medical_model.pkl"
joblib.dump(model_data, filename)


file_size = os.path.getsize(filename) / (1024 * 1024)
print(f"    Modèle sauvegardé: {filename}")
print(f"    Taille fichier: {file_size:.1f} MB")

print("\n" + "=" * 60)
print(" ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS")
print("=" * 60)
