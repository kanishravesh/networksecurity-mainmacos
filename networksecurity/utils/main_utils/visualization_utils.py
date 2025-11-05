import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.tree import plot_tree


sns.set(style="whitegrid", font_scale=1.1)


def plot_correlation_matrix(df, save_path="Artifacts/correlation_heatmap.png"):
    """
    Generates and saves a correlation heatmap for numeric features.
    """
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.figure(figsize=(12, 10))
        corr = df.corr(numeric_only=True)
        sns.heatmap(corr, cmap='coolwarm', annot=False)
        plt.title("Feature Correlation Heatmap", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f" Saved correlation heatmap to {save_path}")
        return save_path
    except Exception as e:
        print(f" Failed to generate correlation heatmap: {e}")



def plot_model_comparison(model_scores: dict, save_path="Artifacts/model_comparison.png"):
    """
    Plots a bar chart comparing models by F1-score or Accuracy.
    """
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.figure(figsize=(8, 5))
        models = list(model_scores.keys())
        scores = list(model_scores.values())
        sns.barplot(x=models, y=scores, palette="crest")
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("F1 Score")
        plt.title("Model Performance Comparison", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f" Saved model comparison chart to {save_path}")
        return save_path
    except Exception as e:
        print(f" Failed to generate model comparison chart: {e}")



def plot_feature_importance(model, feature_names, save_path="Artifacts/feature_importance.png"):
    """
    Plots feature importances for tree-based models.
    """
    try:
        if not hasattr(model, "feature_importances_"):
            print(" Model does not have feature_importances_ attribute.")
            return None

        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:15]  # top 15
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        plt.figure(figsize=(10, 6))
        sns.barplot(y=np.array(feature_names)[indices], x=importances[indices], orient='h', palette="viridis")
        plt.title("Top Feature Importances (Random Forest)", fontsize=14, fontweight='bold')
        plt.xlabel("Importance Score")
        plt.ylabel("Feature Name")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f" Saved feature importance chart to {save_path}")
        return save_path
    except Exception as e:
        print(f" Failed to generate feature importance plot: {e}")



def plot_decision_tree(model, feature_names, save_path="Artifacts/decision_tree.png"):
    """
    Saves a small visualization of the top part of a decision tree.
    """
    try:
        if not hasattr(model, "tree_"):
            print(" Model is not a DecisionTreeClassifier.")
            return None

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.figure(figsize=(20, 10))
        plot_tree(
            model,
            filled=True,
            feature_names=feature_names,
            class_names=["Safe", "Phishing"],
            rounded=True,
            max_depth=3,  # Only show top 3 levels for clarity
            fontsize=10
        )
        plt.title("Decision Tree Visualization (Top 3 Levels)", fontsize=14, fontweight='bold')
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f" Saved decision tree visualization to {save_path}")
        return save_path
    except Exception as e:
        print(f" Failed to plot decision tree: {e}")
