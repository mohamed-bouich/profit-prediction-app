import pandas as pd
import category_encoders as ce
from sklearn.preprocessing import LabelEncoder


def encoder_state(df, methode='onehot'):

    df_encode = df.copy()
    colonne_categorielle = 'State'
    nouvelles_colonnes = []

    #  One-Hot Encoding 
    if methode == 'onehot':
        encodeur = ce.OneHotEncoder(
            cols=[colonne_categorielle],
            use_cat_names=True,
            drop_invariant=False
        )
        df_encode = encodeur.fit_transform(df_encode)
        nouvelles_colonnes = [
            col for col in df_encode.columns
            if col.startswith(colonne_categorielle + '_')
        ]

    #  Binary Encoding 
    elif methode == 'binary':
        encodeur = ce.BinaryEncoder(cols=[colonne_categorielle])
        df_encode = encodeur.fit_transform(df_encode)
        nouvelles_colonnes = [
            col for col in df_encode.columns
            if col.startswith(colonne_categorielle + '_')
        ]

    #  Label Encoding (commence à 1) 
    elif methode == 'label':
        le = LabelEncoder()
        df_encode[colonne_categorielle] = le.fit_transform(
            df_encode[colonne_categorielle]
        ) + 1
        nouvelles_colonnes = []

    else:
        raise ValueError(f"Méthode d'encodage inconnue : {methode}")

    return df_encode, nouvelles_colonnes