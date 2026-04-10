# Jumeau Numérique 'Zéro Flamme' - Prédiction des Émissions de NOx
*Projet Scientifique & Technique (PST) - Centrale*

Ce dépôt contient le code source d'un **Jumeau Numérique** développé pour modéliser une flamme hydrogène dans un brûleur de type Swirler (Géométries S066 et S082). Ce modèle est dit "Zéro Flamme" car il est intégralement fonctionnel de manière macroscopique : il prédit les rejets environnementaux (NOx) à partir des pures commandes opérateurs, sans aucun besoin d'allumer le gaz ni de recourir à des capteurs de température.

## Architecture & Intelligence Artificielle
- **Algorithme Central** : Régression Ridge (Pénalisation L2 pour stabilité mathématique).
- **Surface de Réponse Mathématique** : Polynomial Features (Degré 3) permettant au solver mathématique de cartographier l'espace des données de commande et d'émuler indirectement la courbe de cinétique chimique Arrhenius sans sonde de température réelle.
- **Précision (Testée avec GroupKFold strict pour éviter l'overfitting temporel)** : R² = **98.51%**

## Contenu du Répertoire
1. **`simulateur_interactif.py`** : (Recommandé) L'interface de commande en ligne (CLI). Lancez ce script pour entrer vos propres valeurs de richesse et d'eau en direct et obtenir les prédictions NOx instantanées.
2. **`predict_zero_flamme.py`** : L'entraînement rigoureux du Jumeau Numérique et l'évaluation GroupKFold avec tests scénarisés.
3. **`visu_3D_surface.py`** : Génération de l'empreinte mathématique de l'IA (Une Nappe 3D topographique remplaçant le poids des features classiques).
4. **`lance_le_simulateur.bat`** : Fichier batch de raccourci Windows (Double-clic) pour lancer l'interface sans requérir d'EDI.

## Comment Utiliser (Installation)
1. Installez les packages requis via votre terminal :
   ```bash
   pip install -r requirements.txt
   ```
2. Double-cliquez sur `lance_le_simulateur.bat` (Windows) ou exécutez `python simulateur_interactif.py` (Linux/Mac).

## Note Scientifique (Enveloppe Opératoire)
Ce Jumeau Numérique est qualifié pour opérer avec haute précision (Interpolation) dans l'enceinte de données [Richesse 0.65 - 0.90] et [Taux d'eau 0 - 22.5%]. Au-delà de ces plages expérimentalement validées, la fonction polynomiale divergera vers des prédictions mathématiques infaisables en laboratoire, ignorant par nature le soufflage asymétrique ou l'extinction réelle de la combustion (Risque de perte de représentativité physique en extrapolation libre).
