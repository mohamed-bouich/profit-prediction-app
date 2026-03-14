
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


# RÉDUCTION DE DIMENSION — ACP (PCA)


def appliquer_acp(X, nb_composantes=0.95):
    """
    Paramètres :
        X               : tableau numpy (variables normalisées)
        nb_composantes  : float → variance à conserver (défaut 95%)
                          int   → nombre exact de composantes
    """
    acp = PCA(n_components=nb_composantes)
    X_reduit = acp.fit_transform(X)
    """
    Retourne :
        X_reduit : tableau transformé
        acp      : objet ACP ajusté (pour réutilisation lors de la prédiction)
    """
    return X_reduit, acp


# DÉCOUPAGE ENTRAÎNEMENT / TEST


def decouper_donnees(X, y, taille_entrainement=0.8):
    """
    Diviser les données en ensembles d'entraînement et de test.

    Paramètres :
        X                    : tableau ou DataFrame des variables
        y                    : série ou tableau de la variable cible
        taille_entrainement  : proportion pour l'entraînement (ex: 0.8 = 80%)

    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=round(1.0 - taille_entrainement, 2),
        random_state=42
    )
    return X_train, X_test, y_train, y_test


# ENTRAÎNEMENT DU MODÈLE


def entrainer_modele(X_train, y_train):
    
    modele = LinearRegression()
    modele.fit(X_train, y_train)
    """
    Retourne :
        modele : objet LinearRegression ajusté
    """
    return modele


# PRÉDICTION
def predire(modele, X_test):
    """
    Générer les prédictions à partir du modèle entraîné.

    Retourne :
        y_pred : tableau numpy des valeurs prédites
    """
    return modele.predict(X_test)

