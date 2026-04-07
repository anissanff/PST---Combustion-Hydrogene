import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# --- VISUALISATION DE LA STRUCTURE ---

def visualize_forest_structure(model, feature_names=None, class_names=None):
    """
    Visualise le premier arbre du RandomForest.
    """
    # On récupère le premier arbre (index 0)
    estimator = model.estimators_[0]
    
    plt.figure(figsize=(20, 10))
    
    # On trace l'arbre
    # max_depth=3 pour que ce soit lisible, sinon l'arbre est trop immense
    plot_tree(estimator, 
              max_depth=3, 
              feature_names=feature_names,
              class_names=class_names,
              filled=True, 
              rounded=True, 
              fontsize=10)
    
    plt.title("Structure simplifiée du 1er arbre de la forêt (Profondeur max affichée : 3)")
    plt.show()

# --- UTILISATION DANS VOTRE MAIN ---
# Ajoutez ceci juste après : model.fit(X_train, y_train)

# On crée des noms génériques pour les pixels si vous n'en avez pas
# Pour un pooling 10x10 sur 1024x1024, vous avez ~10404 features
feature_cols = [f"pixel_{i}" for i in range(X_train.shape[1])]
target_classes = [str(c) for c in class_names] # vos labels 0.65, 0.75...

visualize_forest_structure(model, feature_names=feature_cols, class_names=target_classes)