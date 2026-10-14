# type: ignore

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import yaml
import json

## Loading and Importing Data ##
def read_models_from_yaml(file_path):
    '''
    Read model configurations from a YAML file
    '''
    with open(file_path, 'r') as f:
        models = yaml.safe_load(f)
        model_names = []
    try:
        if not isinstance(models, dict):
            raise ValueError("YAML file must contain a dictionary of models.")
        for model_name, config in models.items():
            if 'type' not in config or 'params' not in config:
                raise ValueError(f"Model '{model_name}' must have 'type' and 'params' fields.")
    except Exception as e:
        raise ValueError(f"Error in YAML format: {str(e)}")
    return models

def read_models_from_json(file_path):
    '''
    Read model configurations from a JSON file
    '''
    with open(file_path, 'r') as f:
        models = json.load(f)
        model_names = []
    try:
        if not isinstance(models, dict):
            raise ValueError("JSON file must contain a dictionary of models.")
        for model_name, config in models.items():
            if 'type' not in config or 'params' not in config:
                raise ValueError(f"Model '{model_name}' must have 'type' and 'params' fields.")
    except Exception as e:
        raise ValueError(f"Error in JSON format: {str(e)}")
    return models

## Statistical Tests ##
def generate_permutation_ids(n_samples, n_permutations):
    '''
    Generate permutation IDs for permutation tests
    '''
    rng = np.random.default_rng(seed=42)  # Set a random seed for reproducibility
    permutation_ids = []
    for _ in range(n_permutations):
        permuted_indices = rng.permutation(n_samples)
        permutation_ids.append(permuted_indices)
    return permutation_ids

## Visualizations ##
def plot_PCA_tSNE_feature_space(features, labels=None, predictions=None, title="Feature Space"):
    """Visualize feature space with PCA/t-SNE"""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Reduce dementionality
    if features.shape[1] > 2:
        # PCA
        pca = PCA(n_components=2)
        features_2d_pca = pca.fit_transform(features)
        
        # t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)-1))
        features_2d_tsne = tsne.fit_transform(features)
    else:
        features_2d_pca = features
        features_2d_tsne = features
        pca = None
    
    # Определяем цвета для точек
    if predictions is not None:
        colors = ['blue'] * (len(features) - len(predictions)) + ['red'] * len(predictions)
        label_points = 'Train' if len(predictions) < len(features) else None
        pred_points = 'Predictions' if len(predictions) > 0 else None
    elif labels is not None:    
        # Используем непрерывную цветовую схему для регрессии
        colors = labels
        label_points = 'Train'
        pred_points = None
    else:
        colors = 'blue'
        label_points = 'Data'
        pred_points = None
    
    # PCA plot
    scatter1 = axes[0].scatter(features_2d_pca[:, 0], features_2d_pca[:, 1], 
                              c=colors if predictions is None else colors,
                              alpha=0.6, edgecolors='w', linewidth=0.5)
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')
    axes[0].set_title(f'{title} - PCA')
    
    if pca and hasattr(pca, 'explained_variance_ratio_'):
        axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    
    # t-SNE plot
    scatter2 = axes[1].scatter(features_2d_tsne[:, 0], features_2d_tsne[:, 1], 
                              c=colors if predictions is None else colors,
                              alpha=0.6, edgecolors='w', linewidth=0.5)
    axes[1].set_xlabel('t-SNE1')
    axes[1].set_ylabel('t-SNE2')
    axes[1].set_title(f'{title} - t-SNE')
    
    # Legend
    if predictions is not None and len(predictions) > 0:
        import matplotlib.patches as mpatches
        train_patch = mpatches.Patch(color='blue', label='Training data')
        pred_patch = mpatches.Patch(color='red', label='Prediction data')
        axes[0].legend(handles=[train_patch, pred_patch])
        axes[1].legend(handles=[train_patch, pred_patch])
    
    plt.tight_layout()
    return fig, features_2d_pca, features_2d_tsne

## Quality Control ##
def detect_outliers_and_ad(features, contamination=0.1):
    """Define outliers and applicability domain"""
    
    # Local Outlier Factor for outliers detection
    lof = LocalOutlierFactor(n_neighbors=min(20, len(features)-1), 
                            contamination=contamination)
    outlier_labels = lof.fit_predict(features)
    
    # Nearest Neighbors for applicability domain detection
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=min(5, len(features)-1))
    nn.fit(features)
    distances, _ = nn.kneighbors(features)
    mean_distances = distances.mean(axis=1)
    
    return outlier_labels, mean_distances
