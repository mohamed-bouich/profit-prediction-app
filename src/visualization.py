import matplotlib.pyplot as plt
import seaborn as sns



# MATRICE DE CORRÉLATION

def tracer_matrice_correlation(df, colonnes_exclues=None):
   
    colonnes_exclues = colonnes_exclues or []

    # Garder uniquement les colonnes numériques non exclues
    df_numerique = df.select_dtypes(include='number').drop(
        columns=colonnes_exclues, errors='ignore'
    )

    matrice_corr = df_numerique.corr()

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(
        matrice_corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        square=True,
        ax=ax
    )
    ax.set_title(
        "Matrice de Corrélation — Variables Numériques",
        fontsize=14,
        fontweight='bold'
    )
    plt.tight_layout()
    """
    Retourne :
        fig : objet Figure matplotlib
    """   
    return fig


# VALEURS RÉELLES VS VALEURS PRÉDITES


def tracer_reel_vs_predit(y_reel, y_predit):

    valeurs_reelles = y_reel.values if hasattr(y_reel, 'values') else y_reel
    indices = range(len(valeurs_reelles))

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(indices, valeurs_reelles,
            label='Valeurs Réelles',
            marker='o', color='steelblue', linewidth=2)
    ax.plot(indices, y_predit,
            label='Valeurs Prédites',
            marker='x', color='tomato', linewidth=2, linestyle='--')

    # Zone entre les deux courbes
    ax.fill_between(indices, valeurs_reelles, y_predit,
                    alpha=0.1, color='gray')

    ax.set_title("Valeurs Réelles vs Valeurs Prédites — Profit",
                 fontsize=14, fontweight='bold')
    ax.set_xlabel("Index de l'échantillon", fontsize=11)
    ax.set_ylabel("Profit ($)", fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    """
    Retourne :
        fig : objet Figure matplotlib
    """
    return fig


# GRAPHE DES RÉSIDUS


def tracer_residus(y_reel, y_predit):
    """
    Tracer les résidus (erreurs) du modèle pour analyser
    la qualité des prédictions.
    """
    valeurs_reelles = y_reel.values if hasattr(y_reel, 'values') else y_reel
    residus = valeurs_reelles - y_predit

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.scatter(y_predit, residus,
               color='darkorange', alpha=0.7,
               edgecolors='black', linewidths=0.4)
    ax.axhline(y=0, color='navy', linewidth=1.5, linestyle='--')

    ax.set_title("Graphe des Résidus", fontsize=14, fontweight='bold')
    ax.set_xlabel("Valeurs Prédites", fontsize=11)
    ax.set_ylabel("Résidus", fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    return fig