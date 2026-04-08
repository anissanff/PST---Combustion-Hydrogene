# PST : Modélisation des émissions de NOx (Approche Physique)

Ce dossier contient les ressources relatives à la prédiction des émissions de NOx pour les swirlers **S066** et **S082**. L'objectif de cette partie du projet est de proposer un modèle prédictif robuste, physiquement cohérent et capable de généraliser entre différentes géométries.

## 🔬 Démarche Scientifique et Méthodologie

Contrairement aux approches purement statistiques de type "boîte noire", nous avons privilégié une **approche informée par la physique** (*Physics-Informed Machine Learning*) :

1. **Loi d'Arrhenius & Mécanisme de Zeldovich** : La formation des NOx est modélisée en s'appuyant sur la cinétique chimique. Nous utilisons l'inverse de la température maximale ($1/T_{max}$) comme caractéristique principale, linéarisant ainsi la réponse du modèle par rapport au logarithme des concentrations.
2. **Transition d'Architecture** : Initialement testé avec XGBoost (arbres de décision), le modèle présentait des artefacts de prédiction discrets ("effet d'escalier"). Nous avons migré vers une **Régression Ridge (linéaire régularisée)** pour garantir une interpolation **continue et lisse**, indispensable pour représenter fidèlement les phénomènes de mécanique des fluides.
3. **Master Modèle** : Un modèle unifié intégrant le type de swirler (via One-Hot Encoding) permet d'atteindre une précision globale de **R² = 0.98**.

## 📂 Contenu du Dossier

- **`train_multi_models.py`** : Script principal d'entraînement. Il effectue le nettoyage des données, l'ingénierie des caractéristiques physiques, l'entraînement des modèles individuels et du Master Modèle, et sauvegarde les modèles au format `.pkl`.
- **`generate_presentation_visuals.py`** : Script dédié à la génération des graphiques de performance et d'analyse (Corrélation Réel/Prédit, Importance des variables, et comparaison XGBoost/Ridge).
- **`S066_Clean_all.xlsx` & `S082_Clean_all.xlsx`** : Jeux de données expérimentaux synchronisés et nettoyés (températures thermocouples, richesse, taux d'eau, mesures NOx).

## 🚀 Utilisation

Pour reproduire les résultats :
1. Installer les dépendances : `pip install pandas numpy scikit-learn matplotlib xgboost`
2. Exécuter l'entraînement : `python train_multi_models.py`
3. Générer les visuels : `python generate_presentation_visuals.py`

Les visuels seront générés dans un dossier `presentation_visuals/` et les modèles sauvegardés dans `outputs_multi_swirlers/`.