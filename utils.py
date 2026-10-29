# type: ignore

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import LocalOutlierFactor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.base import clone
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
import json

## Loading and Importing Data ##
def read_models_from_yaml(file_path):
    '''
    Read model configurations from a YAML file
    **Depreciated**: Use `read_models_from_json` instead.
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

## Train Model ##
def train_model(
    model_class,
    model_params,
    X,
    y,
    cat_features = [],
    gridsearch=False,
    gridsearch_params=None,
    cv=5,
    n_jobs=1
):
    '''Train a machine learning model using the provided data and parameters.

    Parameters:
    - model_class: The class of the model to be trained.
    - model_params: Dictionary of parameters for the model.
    - X: Feature data (DataFrame).
    - y: Target variable (Series or array).
    - gridsearch: Boolean indicating whether to perform grid search for hyperparameter tuning.
    - gridsearch_params: Parameter candidates selected for grid search.
    - cv: Number of cross-validation folds.
    - n_jobs: Number of parallel jobs for cross-validation.

    Returns:
    - model: The trained model (Pipeline).
    - cv_scores: The cross-validation scores of the trained model.
    '''
    scale_features = [
        column for column in X.columns
        if column not in cat_features
    ]
    scaler = ColumnTransformer(
        [("standard_scaler", StandardScaler(), scale_features)],
        remainder="passthrough"
    )
    model = Pipeline([
        ("scaler", scaler),
        ("model", model_class(random_state=42))
    ])
    model.set_params(**model_params)

    if gridsearch:
        gs_model = GridSearchCV(
            model,
            gridsearch_params,
            cv=cv,
            n_jobs=n_jobs,
            refit=True
        )
        gs_model.fit(X, y)
        model = gs_model.best_estimator_

    cv_scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        return_train_score=True,
        n_jobs=n_jobs
    )
    model.fit(X, y)
    return model, cv_scores # Return the trained model, the cross-validation scores, and the model parameters

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

def permutation_test(
    model,
    X,
    y,
    covariate_exclusion = False,
    covariate_variables=None,
    cv=5,
    n_permutations=1000,
    n_jobs=1,
    progress_callback=None
):
    '''
    Perform a permutation test to assess the significance of the model's performance.

    Parameters:
    - model: The trained model (Pipeline).
    - X: Feature data (DataFrame).
    - y: Target variable (Series or array).
    - covariate_exclusion: Boolean indicating whether to exclude covariate variables.
    - covariate_variables: List of covariate variables to exclude if covariate_exclusion is True.
    - cv: Number of cross-validation folds.
    - n_permutations: Number of permutations to perform.
    - n_jobs: Number of parallel jobs for cross-validation.
    - progress_callback: Optional callback receiving the completed iteration number.

    Returns:
    - perm_scores: The permutation scores from the permutation test.
    '''
    perm_scores = []
    perm_test_model = clone(model)  # Clone the model to avoid altering the original
    permutation_ids = generate_permutation_ids(len(y), n_permutations) #initialize permutation ids

    for i, perm_indices in enumerate(permutation_ids, start=1):
        if covariate_exclusion:
            X_permutable = X[[feat for feat in X.columns if not feat in covariate_variables]]
            X_nonperm = X[[feat for feat in X.columns if feat in covariate_variables]]
            X_permuted = X_permutable.iloc[perm_indices].reset_index(drop=True)
            X_permuted = pd.concat([X_permuted, X_nonperm.reset_index(drop=True)], axis=1)
        else:
            X_permuted = X.iloc[perm_indices].reset_index(drop=True)
        perm_score = cross_validate(
            perm_test_model,
             X_permuted, y,
             cv=cv,
             return_train_score=False,
             n_jobs=n_jobs
             )["test_score"].mean()
        perm_scores.append(perm_score)
        if progress_callback is not None:
            progress_callback(i)

    return perm_scores

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
