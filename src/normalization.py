import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler



# NORMALISATION DES VARIABLES

def normaliser_variables(df, methode='standard', colonnes_exclues=None):
    
    df_normalise = df.copy()
    colonnes_exclues = colonnes_exclues or []

    # Sélectionner uniquement les colonnes numériques non exclues
    colonnes_a_normaliser = [
        col for col in df_normalise.columns
        if col not in colonnes_exclues
        and df_normalise[col].dtype in ['float64', 'int64', 'int32', 'float32']
    ]

    # ── Choix du normaliseur ─────────────────
    if methode == 'standard':
        normaliseur = StandardScaler()
    elif methode == 'minmax':
        normaliseur = MinMaxScaler()
    elif methode == 'robust':
        normaliseur = RobustScaler()
    else:
        raise ValueError(f"Méthode de normalisation inconnue : {methode}")

    # ── Ajustement et transformation ─────────
    df_normalise[colonnes_a_normaliser] = normaliseur.fit_transform(
        df_normalise[colonnes_a_normaliser]
    )
    """

    Retourne :
        df_normalise    : DataFrame normalisé
        normaliseur     : objet scaler ajusté (pour réutilisation lors de la prédiction)
        colonnes_norm   : liste des colonnes qui ont été normalisées

        """
    return df_normalise, normaliseur, colonnes_a_normaliser