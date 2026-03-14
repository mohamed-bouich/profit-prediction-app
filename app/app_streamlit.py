import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import numpy as np

from src.preprocessing import (
    charger_donnees,
    obtenir_infos_generales,
    obtenir_infos_manquantes,
    nettoyer_valeurs_manquantes,
    VALEURS_STATE,
)
from src.encoding      import encoder_state
from src.normalization import normaliser_variables
from src.model         import (
    appliquer_acp,
    decouper_donnees,
    entrainer_modele,
    predire,
)
from src.evaluation    import evaluer_modele
from src.visualization import (
    tracer_matrice_correlation,
    tracer_reel_vs_predit,
    tracer_residus,
)

# ══════════════════════════════════════════════════════
#  CONFIGURATION DE LA PAGE
# ══════════════════════════════════════════════════════
st.set_page_config(
    page_title="Prédiction du Profit",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Prédiction du Profit d'une Entreprise")
st.markdown("**Régression Linéaire Multiple** — Application Web Streamlit")
st.markdown("LST-IDLL — Intelligence Artificielle — A.U 2025/2026 ")
st.divider()

# ══════════════════════════════════════════════════════
#  INITIALISATION DE L'ÉTAT DE SESSION
# ══════════════════════════════════════════════════════
valeurs_par_defaut = {
    "df":                 None,
    "df_nettoye":         None,
    "df_encode":          None,
    "df_normalise":       None,
    "X_final":            None,
    "y":                  None,
    "X_train":            None,
    "X_test":             None,
    "y_train":            None,
    "y_test":             None,
    "modele":             None,
    "y_predit":           None,
    "normaliseur":        None,
    "acp_modele":         None,
    "methode_encodage":   None,
    "nouvelles_colonnes": [],
    "methode_norm":       None,
    "colonnes_norm":      [],
    "utiliser_acp":       None,
    "etape_chargement":   False,
    "etape_nettoyage":    False,
    "etape_encodage":     False,
    "etape_normalisation":False,
    "etape_correlation":  False,
    "etape_acp":          False,
    "etape_decoupage":    False,
    "etape_entrainement": False,
    "etape_evaluation":   False,
}
for cle, valeur in valeurs_par_defaut.items():
    if cle not in st.session_state:
        st.session_state[cle] = valeur

# ══════════════════════════════════════════════════════
#  BARRE LATÉRALE — SUIVI DE LA PROGRESSION
# ══════════════════════════════════════════════════════
st.sidebar.header(" Suivi de la Progression")
st.sidebar.divider()

etapes_progression = [
    ("etape_chargement",    "1. Chargement des données"),
    ("etape_nettoyage",     "2. Nettoyage des données"),
    ("etape_encodage",      "3. Encodage"),
    ("etape_normalisation", "4. Normalisation"),
    ("etape_correlation",   "5. Matrice de corrélation"),
    ("etape_acp",           "6. Réduction dimensionnelle (ACP)"),
    ("etape_decoupage",     "7. Découpage Train / Test"),
    ("etape_entrainement",  "8. Entraînement du modèle"),
    ("etape_evaluation",    "9. Évaluation du modèle"),
]
for cle_etape, libelle in etapes_progression:
    if st.session_state[cle_etape]:
        st.sidebar.success(f" {libelle}")
    else:
        st.sidebar.info(f" {libelle}")

st.sidebar.divider()
st.sidebar.markdown("FST Al-Hoceima — Département Informatique")

# ══════════════════════════════════════════════════════
#  ÉTAPE 1 — CHARGEMENT
# ══════════════════════════════════════════════════════
with st.expander("📂 Étape 1 — Chargement du jeu de données",
                 expanded=not st.session_state["etape_chargement"]):

    fichier_importe = st.file_uploader(
        "Importer votre fichier CSV (profitentr.csv)", type=["csv"]
    )
    if fichier_importe is not None:
        df = charger_donnees(fichier_importe)
        st.session_state["df"] = df
        st.session_state["etape_chargement"] = True
        infos = obtenir_infos_generales(df)

        st.success("✅ Fichier chargé avec succès !")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Nombre de lignes",        infos["nb_lignes"])
        col2.metric("Nombre de colonnes",      infos["nb_colonnes"])
        col3.metric("Variables indépendantes", len(infos["variables_independantes"]))
        col4.metric("Variable cible",          infos["variable_cible"])

        gauche, droite = st.columns(2)
        with gauche:
            st.write("**Variables indépendantes (X) :**")
            for var in infos["variables_independantes"]:
                st.write(f"  ▸ `{var}`")
        with droite:
            st.write(f"**Variable cible (y) :** `{infos['variable_cible']}`")

        st.subheader("Aperçu des premières lignes")
        st.dataframe(df.head(10), use_container_width=True)

# ══════════════════════════════════════════════════════
#  ÉTAPE 2 — NETTOYAGE
# ══════════════════════════════════════════════════════
if not st.session_state["etape_chargement"]:
    st.warning(" **Étape 2 verrouillée** — Veuillez d'abord charger le jeu de données.")
else:
    with st.expander("🧹 Étape 2 — Inspection et Nettoyage des données",
                     expanded=not st.session_state["etape_nettoyage"]):

        df = st.session_state["df"]
        infos_manquantes = obtenir_infos_manquantes(df)

        if infos_manquantes["colonnes_manquantes"]:
            st.warning(
                f"  **{infos_manquantes['lignes_manquantes']}** ligne(s) "
                f"contiennent des valeurs manquantes."
            )
            st.dataframe(pd.DataFrame({
                "Colonne":             list(infos_manquantes["comptage_manquants"].keys()),
                "Nombre de manquants": list(infos_manquantes["comptage_manquants"].values()),
            }), use_container_width=True)
        else:
            st.success("✅ Aucune valeur manquante détectée.")

        options_strategie = {
            "Moyenne (mean)":   "mean",
            "Médiane (median)": "median",
            "Mode (mode)":      "mode",
        }
        choix_strategie = st.selectbox(
            "Remplacer les valeurs manquantes par :",
            list(options_strategie.keys()),
            key="select_strategie_nettoyage"
        )

        if st.button("✅ Appliquer le nettoyage", key="btn_nettoyage"):
            df_nettoye = nettoyer_valeurs_manquantes(
                df, strategie=options_strategie[choix_strategie]
            )
            st.session_state["df_nettoye"]      = df_nettoye
            st.session_state["etape_nettoyage"] = True
            st.success(f" Valeurs manquantes remplacées par : **{choix_strategie}**")
            st.dataframe(df_nettoye.head(10), use_container_width=True)

        if st.session_state["df_nettoye"] is None:
            st.session_state["df_nettoye"] = df.copy()
# ══════════════════════════════════════════════════════
#  ÉTAPE 3 — ENCODAGE
# ══════════════════════════════════════════════════════
if not st.session_state["etape_nettoyage"]:
    st.warning(" **Étape 3 verrouillée** — Veuillez d'abord nettoyer les données.")
else:
    with st.expander("🔤 Étape 3 — Encodage de la variable 'State'",
                     expanded=not st.session_state["etape_encodage"]):

        options_encodage = {
            "One-Hot Encoding":              "onehot",
            "Binary Encoding":               "binary",
            "Label Encoding (commence à 1)": "label",
        }
        descriptions_encodage = {
            "One-Hot Encoding":              "Crée une colonne binaire par catégorie.",
            "Binary Encoding":               "Encode en colonnes binaires compactes (log2 colonnes).",
            "Label Encoding (commence à 1)": "Remplace chaque catégorie par un entier à partir de 1.",
        }
        choix_encodage = st.selectbox(
            "Méthode d'encodage :", list(options_encodage.keys()), key="select_encodage"
        )
        st.caption(descriptions_encodage[choix_encodage])

        if st.button(" Appliquer l'encodage", key="btn_encodage"):
            df_nettoye = st.session_state["df_nettoye"]
            X = df_nettoye.drop(columns=["Profit"])
            y = df_nettoye["Profit"]

            methode_enc = options_encodage[choix_encodage]
            X_encode, nouvelles_colonnes = encoder_state(X, methode=methode_enc)

            st.session_state["df_encode"]          = X_encode
            st.session_state["y"]                  = y
            st.session_state["methode_encodage"]   = methode_enc
            st.session_state["nouvelles_colonnes"] = nouvelles_colonnes
            st.session_state["etape_encodage"]     = True

            st.success(f"✅ Encodage appliqué : **{choix_encodage}**")
            st.dataframe(X_encode.head(10), use_container_width=True)


# ══════════════════════════════════════════════════════
#  ÉTAPE 4 — NORMALISATION
# ══════════════════════════════════════════════════════
if not st.session_state["etape_encodage"]:
    st.warning(" **Étape 4 verrouillée** — Veuillez d'abord encoder les données.")
else:
    with st.expander("📐 Étape 4 — Normalisation des variables",
                     expanded=not st.session_state["etape_normalisation"]):

        options_normalisation = {
            "StandardScaler  — Moyenne nulle, variance unitaire":    "standard",
            "MinMaxScaler    — Mise à l'échelle entre [0, 1]":        "minmax",
            "RobustScaler    — Robuste aux valeurs aberrantes (IQR)": "robust",
        }
        choix_normalisation = st.selectbox(
            "Méthode de normalisation :", list(options_normalisation.keys()), key="select_normalisation"
        )

        methode_enc        = st.session_state["methode_encodage"]
        nouvelles_colonnes = st.session_state["nouvelles_colonnes"]

        if methode_enc in ["onehot", "binary"]:
            st.info(
                f"ℹ️  Les colonnes encodées ({methode_enc}) **ne seront pas normalisées** "
                f"car elles sont déjà binaires (0 ou 1)."
            )

        if st.button(" Appliquer la normalisation", key="btn_normalisation"):
            X_encode         = st.session_state["df_encode"]
            colonnes_exclues = nouvelles_colonnes if methode_enc in ["onehot", "binary"] else []
            methode_norm     = options_normalisation[choix_normalisation]

            X_normalise, normaliseur, colonnes_norm = normaliser_variables(
                X_encode, methode=methode_norm, colonnes_exclues=colonnes_exclues
            )
            st.session_state["df_normalise"]        = X_normalise
            st.session_state["normaliseur"]         = normaliseur
            st.session_state["methode_norm"]        = methode_norm
            st.session_state["colonnes_norm"]       = colonnes_norm
            st.session_state["etape_normalisation"] = True

            st.success(f"✅ Normalisation appliquée : **{choix_normalisation}**")
            st.caption(f"Colonnes normalisées : {colonnes_norm}")
            st.dataframe(X_normalise.head(10), use_container_width=True)



# ══════════════════════════════════════════════════════
#  ÉTAPE 5 — MATRICE DE CORRÉLATION
# ══════════════════════════════════════════════════════
if not st.session_state["etape_normalisation"]:
    st.warning(" **Étape 5 verrouillée** — Veuillez d'abord normaliser les données.")
else:
    with st.expander("📊 Étape 5 — Matrice de Corrélation",
                     expanded=not st.session_state["etape_correlation"]):

        st.write("Corrélations entre les variables numériques de X (colonne 'State' exclue).")

        if st.button(" Afficher la matrice de corrélation", key="btn_correlation"):
            X_normalise        = st.session_state["df_normalise"]
            methode_enc        = st.session_state["methode_encodage"]
            nouvelles_colonnes = st.session_state["nouvelles_colonnes"]

            colonnes_a_exclure = (
                nouvelles_colonnes if methode_enc in ["onehot", "binary"]
                else ["State"]     if methode_enc == "label"
                else []
            )
            st.pyplot(tracer_matrice_correlation(X_normalise, colonnes_exclues=colonnes_a_exclure))
            st.session_state["etape_correlation"] = True



# ══════════════════════════════════════════════════════
#  ÉTAPE 6 — ACP
# ══════════════════════════════════════════════════════
if not st.session_state["etape_correlation"]:
    st.warning(" **Étape 6 verrouillée** — Veuillez d'abord afficher la matrice de corrélation.")
else:
    with st.expander("🔬 Étape 6 — Réduction Dimensionnelle (ACP)",
                     expanded=not st.session_state["etape_acp"]):

        choix_acp = st.radio(
            "Souhaitez-vous appliquer une réduction dimensionnelle ?",
            ["Sans réduction (utiliser toutes les variables)",
             "Avec ACP (réduction à 95% de variance)"],
            key="radio_acp"
        )

        if st.button("✅ Confirmer le choix ACP", key="btn_acp"):
            tableau_X = st.session_state["df_normalise"].values

            if "Avec ACP" in choix_acp:
                X_final, acp_modele = appliquer_acp(tableau_X)
                st.session_state["acp_modele"] = acp_modele
                variance = np.sum(acp_modele.explained_variance_ratio_) * 100
                st.success(
                    f"✅ ACP appliquée — **{X_final.shape[1]}** composante(s). "
                    f"Variance conservée : **{variance:.1f}%**"
                )
                st.dataframe(pd.DataFrame({
                    "Composante":             [f"CP{i+1}" for i in range(len(acp_modele.explained_variance_ratio_))],
                    "Variance expliquée (%)": np.round(acp_modele.explained_variance_ratio_ * 100, 2),
                }), use_container_width=True)
            else:
                X_final = tableau_X
                st.success(" Aucune réduction — toutes les variables conservées.")

            st.session_state["X_final"]      = X_final
            st.session_state["utiliser_acp"] = choix_acp
            st.session_state["etape_acp"]    = True

            st.subheader("Aperçu du jeu de données final (X)")
            st.dataframe(pd.DataFrame(X_final).head(10), use_container_width=True)



# ══════════════════════════════════════════════════════
#  ÉTAPE 7 — DÉCOUPAGE
# ══════════════════════════════════════════════════════
if not st.session_state["etape_acp"]:
    st.warning(" **Étape 7 verrouillée** — Veuillez d'abord confirmer le choix ACP.")
else:
    with st.expander("✂️ Étape 7 — Découpage Entraînement / Test",
                     expanded=not st.session_state["etape_decoupage"]):

        pourcentage_entrainement = st.slider(
            "Pourcentage de l'ensemble d'entraînement :",
            min_value=50, max_value=90, value=80, step=5,
            key="slider_decoupage"
        )
        col_gauche, col_droite = st.columns(2)
        col_gauche.metric("Ensemble d'entraînement", f"{pourcentage_entrainement}%")
        col_droite.metric("Ensemble de test",         f"{100 - pourcentage_entrainement}%")

        if st.button("✅ Appliquer le découpage", key="btn_decoupage"):
            X_train, X_test, y_train, y_test = decouper_donnees(
                st.session_state["X_final"],
                st.session_state["y"],
                taille_entrainement=pourcentage_entrainement / 100
            )
            st.session_state["X_train"]         = X_train
            st.session_state["X_test"]          = X_test
            st.session_state["y_train"]         = y_train
            st.session_state["y_test"]          = y_test
            st.session_state["etape_decoupage"] = True

            st.success(
                f"✅ Découpage effectué — "
                f"**Entraînement : {len(X_train)} échantillons** | "
                f"**Test : {len(X_test)} échantillons**"
            )



# ══════════════════════════════════════════════════════
#  ÉTAPE 8 — ENTRAÎNEMENT
# ══════════════════════════════════════════════════════
if not st.session_state["etape_decoupage"]:
    st.warning(" **Étape 8 verrouillée** — Veuillez d'abord effectuer le découpage.")
else:
    with st.expander("🏋️ Étape 8 — Entraînement du Modèle",
                     expanded=not st.session_state["etape_entrainement"]):

        st.write("Modèle utilisé : **Régression Linéaire Multiple** (`LinearRegression` de scikit-learn)")

        if st.button(" Lancer l'entraînement du modèle", key="btn_entrainement"):
            modele = entrainer_modele(
                st.session_state["X_train"],
                st.session_state["y_train"]
            )
            st.session_state["modele"]             = modele
            st.session_state["etape_entrainement"] = True

            st.success("✅ Modèle entraîné avec succès !")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Ordonnée à l'origine (intercept)", f"{modele.intercept_:,.4f}")
            with col2:
                st.write("**Coefficients du modèle :**")
                st.write(np.round(modele.coef_, 4))



# ══════════════════════════════════════════════════════
#  ÉTAPE 9 — ÉVALUATION
# ══════════════════════════════════════════════════════
if not st.session_state["etape_entrainement"]:
    st.warning(" **Étape 9 verrouillée** — Veuillez d'abord entraîner le modèle.")
else:
    with st.expander("🧪 Étape 9 — Évaluation du Modèle",
                     expanded=not st.session_state["etape_evaluation"]):

        if st.button("🧪 Tester le modèle", key="btn_evaluation"):
            y_predit = predire(st.session_state["modele"], st.session_state["X_test"])
            st.session_state["y_predit"]         = y_predit
            st.session_state["etape_evaluation"] = True

            metriques = evaluer_modele(st.session_state["y_test"], y_predit)
            st.success("✅ Évaluation terminée !")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("R²",   metriques["R²"])
            col2.metric("MAE",  f"${metriques['MAE']:,}")
            col3.metric("MSE",  f"${metriques['MSE']:,}")
            col4.metric("RMSE", f"${metriques['RMSE']:,}")

        if st.session_state["y_predit"] is not None:
            st.subheader("📉 Visualisations des résultats")
            onglet1, onglet2 = st.tabs(["Valeurs Réelles vs Prédites", "Graphe des Résidus"])
            with onglet1:
                st.pyplot(tracer_reel_vs_predit(
                    st.session_state["y_test"], st.session_state["y_predit"]
                ))
            with onglet2:
                st.pyplot(tracer_residus(
                    st.session_state["y_test"], st.session_state["y_predit"]
                ))


# ══════════════════════════════════════════════════════
#  ÉTAPE 10 — PRÉDICTION SUR DE NOUVELLES DONNÉES
# ══════════════════════════════════════════════════════
if not st.session_state["etape_evaluation"]:
    st.warning(" **Étape 10 verrouillée** — Veuillez d'abord évaluer le modèle.")
else:
    with st.expander("🔮 Étape 10 — Prédiction sur de nouvelles données", expanded=True):

        st.write("Saisir les données d'une nouvelle entreprise pour prédire son profit :")

        col_gauche, col_droite = st.columns(2)
        with col_gauche:
            depenses_rd        = st.number_input("Dépenses R&D ($)",       min_value=0.0, value=float(st.session_state["df"]["R&D Spend"].mean()), step=1000.0)
            administration     = st.number_input("Administration ($)",      min_value=0.0, value=float(st.session_state["df"]["Administration"].mean()), step=1000.0)
        with col_droite:
            depenses_marketing = st.number_input("Dépenses Marketing ($)",  min_value=0.0, value=float(st.session_state["df"]["Marketing Spend"].mean()), step=1000.0)
            etat               = st.selectbox("État (State)", VALEURS_STATE)

        if st.button("🔮 Prédire le profit", key="btn_prediction"):
            try:
                methode_enc        = st.session_state["methode_encodage"]
                nouvelles_colonnes = st.session_state["nouvelles_colonnes"]
                normaliseur        = st.session_state["normaliseur"]
                colonnes_norm      = st.session_state["colonnes_norm"]
                modele             = st.session_state["modele"]
                utiliser_acp       = st.session_state["utiliser_acp"]
                acp_modele         = st.session_state["acp_modele"]
                df_normalise_ref   = st.session_state["df_normalise"]

                nouvelles_donnees = pd.DataFrame({
                    "R&D Spend":       [depenses_rd],
                    "Administration":  [administration],
                    "Marketing Spend": [depenses_marketing],
                    "State":           [etat],
                })

                donnees_encodees, _ = encoder_state(nouvelles_donnees, methode=methode_enc)
                donnees_encodees    = donnees_encodees.reindex(
                    columns=df_normalise_ref.columns, fill_value=0
                )
                donnees_encodees[colonnes_norm] = normaliseur.transform(
                    donnees_encodees[colonnes_norm]
                )

                tableau_saisie = donnees_encodees.values
                if utiliser_acp == "Avec ACP (réduction à 95% de variance)" and acp_modele is not None:
                    tableau_saisie = acp_modele.transform(tableau_saisie)

                valeur_predite = modele.predict(tableau_saisie)[0]
                st.success(f" Profit prédit : **${valeur_predite:,.2f}**")

            except Exception as erreur:
                st.error(f"Erreur lors de la prédiction : {erreur}")