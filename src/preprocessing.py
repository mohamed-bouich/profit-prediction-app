import pandas as pd

VARIABLES_INDEPENDANTES = ['R&D Spend', 'Administration', 'Marketing Spend', 'State']
VARIABLE_CIBLE = 'Profit'
VALEURS_STATE           = ['New York', 'California'] 

# CHARGEMENT DU JEU DE DONNÉES

def charger_donnees(fichier_importe):
    """
    Charger un fichier CSV depuis un objet Streamlit uploadé.
    Retourne un DataFrame pandas.
    """
    df = pd.read_csv(fichier_importe)
    return df

# INFORMATIONS GÉNÉRALES

def obtenir_infos_generales(df):
  
    return {
        "nb_lignes":              df.shape[0],
        "nb_colonnes":            df.shape[1],
        "variables_independantes": VARIABLES_INDEPENDANTES,
        "variable_cible":         VARIABLE_CIBLE,
        "colonnes":               list(df.columns),
    }

# DÉTECTION DES VALEURS MANQUANTES

def obtenir_infos_manquantes(df):
    """
    Détecter les valeurs manquantes dans le jeu de données.
    Retourne :
    - la liste des colonnes contenant des valeurs manquantes
    - le nombre de lignes contenant au moins une valeur manquante
    - le nombre de valeurs manquantes par colonne
    """
    colonnes_manquantes = df.columns[df.isnull().any()].tolist()
    lignes_manquantes   = int(df.isnull().any(axis=1).sum())
    comptage_manquants  = (
                            df[colonnes_manquantes].isnull().sum().to_dict()
                            if colonnes_manquantes else {}
    )
    return {
        "colonnes_manquantes": colonnes_manquantes,
        "lignes_manquantes":   lignes_manquantes,
        "comptage_manquants":  comptage_manquants,
    }

def nettoyer_valeurs_manquantes(df, strategie='mean'):
    """
    Remplacer les valeurs manquantes :
        - Colonnes numériques    → selon la stratégie choisie (mean / median / mode)
        - Colonnes catégorielles → toujours par le mode
    """
    df_nettoye = df.copy()
 
    # Colonnes numériques
    cols_num = df_nettoye.select_dtypes(include='number').columns
    for col in cols_num:
        if strategie == 'mean':
            df_nettoye[col].fillna(df_nettoye[col].mean(), inplace=True)
        elif strategie == 'median':
            df_nettoye[col].fillna(df_nettoye[col].median(), inplace=True)
        elif strategie == 'mode':
            df_nettoye[col].fillna(df_nettoye[col].mode()[0], inplace=True)
 
    # Colonnes catégorielles → toujours mode
    cols_cat = df_nettoye.select_dtypes(include='object').columns
    for col in cols_cat:
        df_nettoye[col].fillna(df_nettoye[col].mode()[0], inplace=True)
 
    return df_nettoye