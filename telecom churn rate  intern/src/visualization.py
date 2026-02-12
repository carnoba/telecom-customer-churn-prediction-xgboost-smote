import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_confusion_matrix(cm, model_name, save_path=None):
    """
    Plot a heatmap for the confusion matrix.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix: {model_name}')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    plt.close()

def plot_roc_curves(results, save_path=None):
    """
    Plot ROC Curve for all models compared.
    """
    plt.figure(figsize=(10, 8))
    for name, metrics in results.items():
        plt.plot(metrics['fpr'], metrics['tpr'], label=f"{name} (AUC = {metrics['auc']:.2f})")
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-AUC Curves')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    plt.close()

def plot_feature_importance(model, feature_names, model_name, save_path=None):
    """
    Plot feature importance for the best model (Tree-based).
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(12, 8))
        plt.title(f'Feature Importance: {model_name}')
        sns.barplot(x=importances[indices], y=[feature_names[i] for i in indices], palette='viridis')
        plt.xlabel('Relative Importance')
        if save_path:
            plt.savefig(save_path)
            print(f"Saved: {save_path}")
        plt.close()
    else:
        print(f"Model {model_name} does not support feature_importances_ visualization.")
