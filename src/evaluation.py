import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ─────────────────────────────────────────────
# MÉTRIQUES D'ÉVALUATION DU MODÈLE
# ─────────────────────────────────────────────
def evaluer_modele(y_reel, y_predit):
    """
    Retourne :
        dictionnaire contenant :
            - R²   : coefficient de détermination (plus proche de 1 = meilleur)
            - MAE  : erreur absolue moyenne
            - MSE  : erreur quadratique moyenne
            - RMSE : racine de l'erreur quadratique moyenne
    """
    r2   = r2_score(y_reel, y_predit)
    mae  = mean_absolute_error(y_reel, y_predit)
    mse  = mean_squared_error(y_reel, y_predit)
    rmse = np.sqrt(mse)

    return {
        "R²"   : round(r2,   4),
        "MAE"  : round(mae,  2),
        "MSE"  : round(mse,  2),
        "RMSE" : round(rmse, 2),
    }