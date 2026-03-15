# Profit Prediction App

Application web de prédiction du profit d'une entreprise basée sur la **Régression Linéaire Multiple**, développée avec **Streamlit**.

---

## Description

Cette application permet de :
- Charger un jeu de données CSV
- Inspecter et nettoyer les valeurs manquantes
- Encoder la variable catégorielle `State`
- Normaliser les variables numériques
- Visualiser la matrice de corrélation
- Appliquer une réduction dimensionnelle (ACP)
- Entraîner un modèle de Régression Linéaire Multiple
- Évaluer le modèle (R², MAE, MSE, RMSE)
- Prédire le profit d'une nouvelle entreprise

---

## Structure du projet

```
profit-prediction-app/
├── app/
│   └── app_streamlit.py      # Application Streamlit
├── src/
│   ├── preprocessing.py      # Chargement et nettoyage des données
│   ├── encoding.py           # Encodage de la variable State
│   ├── normalization.py      # Normalisation des variables
│   ├── model.py              # ACP, découpage, entraînement, prédiction
│   ├── evaluation.py         # Métriques d'évaluation
│   └── visualization.py      # Graphiques et visualisations
├── data/
│   └── profitentr.csv        # Jeu de données
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/mohamed-bouich/profit-prediction-app.git
cd profit-prediction-app
pip install -r requirements.txt
```

---

## Lancer l'application

```bash
cd app
streamlit run app_streamlit.py
```

---

## Bibliothèques utilisées

| Bibliothèque | Utilisation |
|---|---|
| pandas | Manipulation des données |
| numpy | Calcul numérique |
| scikit-learn | Modèles ML et prétraitement |
| matplotlib | Visualisation |
| seaborn | Heatmap de corrélation |
| streamlit | Interface web |
| category_encoders | Encodage One-Hot et Binary |


---

## Application en ligne

👉 https://mohamed-bouich-profit-prediction-app-appapp-streamlit-yuyuax.streamlit.app/

---

## Informations académiques

- **Université** : Abdelmalek Essaâdi — FST Al-Hoceima
- **Département** : Informatique
- **Filière** : LST-IDLL
- **Module** : Intelligence Artificielle
- **Professeur** : Pr. ZANNOU Abderrahim
- **Année universitaire** : 2025/2026