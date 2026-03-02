# Backend IA – FastAPI

## Table des matières
1. [Présentation générale](#présentation-générale)
2. [Architecture du service IA](#architecture-du-service-ia)
3. [Communication avec le Frontend](#communication-avec-le-frontend)
4. [Modèle d’Intelligence Artificielle](#modèle-dintelligence-artificielle)
   - [Problématique](#problématique)
   - [Dataset](#dataset)
   - [Algorithme : Bernoulli Naïve Bayes](#algorithme--bernoulli-naïve-bayes)
   - [Avantages théoriques](#avantages-théoriques)
   - [Interprétation médicale](#interprétation-médicale)
   - [Limites actuelles](#limites-actuelles)
   - [Améliorations futures](#améliorations-futures)
5. [Installation et exécution](#installation-et-exécution)

---

## Présentation générale
Le backend dédié à l’IA a été développé avec **FastAPI**, un framework Python moderne permettant de créer des API rapides et performantes.

### Avantages
- Documentation automatique (Swagger)
- Validation des données en entrée
- Architecture légère et performante

Le service IA fonctionne indépendamment du backend principal afin de gérer les traitements lourds liés au modèle IA sans ralentir la logique métier.

---

## Architecture du service IA
Le service IA fonctionne comme un **microservice autonome** :
1. Reçoit des requêtes HTTP depuis le backend principal
2. Exécute les prédictions ou traitements via le modèle IA
3. Renvoie les résultats au backend pour intégration dans l’application

### Bénéfices
- Scalabilité indépendante
- Isolation des composants critiques
- Meilleure performance globale

---

## Communication avec le Frontend
Le frontend (React + TSX) interagit directement avec le service IA via des **API REST** :
- Les données nécessaires au modèle sont envoyées depuis le frontend
- Le backend IA renvoie les résultats sous forme JSON
- Le backend principal reste indépendant pour d’autres fonctionnalités métier

Cette approche assure une communication directe et rapide tout en maintenant la modularité et l’isolation des services.

---

## Modèle d’Intelligence Artificielle

### Problématique
Le diagnostic médical est une tâche de **classification multi-classes** :
- Identifier une pathologie parmi **K = 754 maladies**
- À partir d’un vecteur de symptômes \( x \in \{0,1\}^{D} \), \( D = 377 \) dimensions

**Complexité du problème :**
- Grande dimensionnalité (377 symptômes)
- Déséquilibre des classes
- Sparsité des données (98,5% de valeurs nulles)
- Inférence en temps réel pour usage clinique

**Objectifs :**
1. Sélectionner l’algorithme offrant le meilleur compromis précision/vitesse/interprétabilité
2. Déployer une solution logicielle robuste via une API REST professionnelle

---

### Dataset
- Fichier : `Diseases_and_Symptoms.csv`
- Structure : vecteurs binaires indiquant la présence/absence des symptômes

---

### Algorithme : Bernoulli Naïve Bayes (BernoulliNB)
- Modèle génératif basé sur le théorème de Bayes
- Hypothèse d’indépendance conditionnelle des features

**Protocole expérimental :**
- Métriques : Accuracy, F1-score pondéré
- Validation : Stratified Train/Test Split (80/20) + Validation croisée 5-fold
- Environnement : Python 3.12, scikit-learn 1.3.2

---

### Avantages théoriques
1. Adapté aux données binaires
2. Gestion de la sparsité (matrice de covariance diagonale dominante)
3. Résistant à la malédiction de la dimensionnalité (complexité linéaire O(N D))

---

### Interprétation médicale
- Probabilités calibrées interprétables cliniquement
- La confusion entre maladies aux symptômes similaires reflète la réalité clinique (ex : Cystite vs Prostatite 55% / 44%)

---

### Limites actuelles
1. Indépendance des symptômes (corrélations non prises en compte)
2. Classes rares (25 maladies exclues)
3. Contexte patient non intégré (âge, sexe, antécédents)

---

### Améliorations futures
- Intégration d’un modèle Bayésien naïf structuré (TAN)
- Utilisation de ComplementNB pour gérer le déséquilibre des classes
- Mécanisme de fallback vers un médecin en cas de confiance < 50%

---

## Installation et exécution

### Prérequis
- Python >= 3.12
- pip installé
- Fichier `Diseases_and_Symptoms.csv` disponible dans le projet

### Installation des dépendances
```bash
# Créer un environnement virtuel (optionnel mais recommandé)
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows

# Installer les dépendances exactes
pip install fastapi==0.104.1 uvicorn[standard]==0.24.0 pydantic==2.5.0 joblib==1.3.2 numpy>=1.26.4 pandas==2.1.3 scikit-learn==1.3.2 python-multipart==0.0.6
