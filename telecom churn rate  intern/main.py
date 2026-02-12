from src.preprocessing import load_and_preprocess_data
from src.models import apply_smote, train_models, evaluate_models
from src.visualization import plot_confusion_matrix, plot_roc_curves, plot_feature_importance
import os

def main():
    # Path to dataset
    dataset_path = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset {dataset_path} not found.")
        return

    # 1. Preprocessing
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data(dataset_path)
    
    # 2. Apply SMOTE to training data
    X_train_res, y_train_res = apply_smote(X_train, y_train)
    
    # 3. Model Training
    models = train_models(X_train_res, y_train_res)
    
    # 4. Model Evaluation
    results = evaluate_models(models, X_test, y_test)
    
    # 5. Reporting & Visualizations
    # ROC Curves for all models
    plot_roc_curves(results, save_path='roc_curves.png')
    
    # Find best model based on AUC
    best_model_name = max(results, key=lambda x: results[x]['auc'])
    print(f"\nBest Performing Model: {best_model_name}")
    
    # Confusion Matrix for best model
    plot_confusion_matrix(results[best_model_name]['cm'], best_model_name, save_path='confusion_matrix.png')
    
    # Feature Importance for best model
    plot_feature_importance(models[best_model_name], feature_names, best_model_name, save_path='feature_importance.png')

    print("\n--- Pipeline Execution Finished Successfully ---")

if __name__ == "__main__":
    main()
