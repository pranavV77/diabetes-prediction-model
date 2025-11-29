"""
Diabetes Prediction - Main Pipeline
Complete end-to-end machine learning pipeline
"""

import os
import sys
from datetime import datetime

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")

def main():
    print_header("DIABETES PREDICTION PROJECT")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Step 1: Create project structure
    print_header("STEP 1: Creating Project Structure")
    folders = ['data', 'preprocessing', 'models', 'results', 
               'results/visualizations', 'results/metrics']
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"✓ Created: {folder}/")
    
    # Check if dataset exists
    if not os.path.exists('data/diabetes_prediction_dataset.csv'):
        print("\n⚠ WARNING: Dataset not found!")
        print("Please place 'diabetes_prediction_dataset.csv' in the data/ folder")
        print("Then run this script again.")
        return
    
    # Step 2: Data Preprocessing
    print_header("STEP 2: Data Preprocessing")
    try:
        from preprocessing.data_preprocessing import DiabetesDataPreprocessor
        preprocessor = DiabetesDataPreprocessor()
        X_train, X_test, y_train, y_test = preprocessor.preprocess()
        print("\n✓ Preprocessing completed successfully!")
    except Exception as e:
        print(f"\n✗ Error in preprocessing: {str(e)}")
        return
    
    # Step 3: Model Training
    print_header("STEP 3: Training Models")
    try:
        from models.train_models import DiabetesModelTrainer
        trainer = DiabetesModelTrainer()
        trainer.load_data()
        trainer.train_all_models()
        summary_df = trainer.save_results()
        trainer.print_summary()
        print("\n✓ Model training completed successfully!")
    except Exception as e:
        print(f"\n✗ Error in training: {str(e)}")
        return
    
    # Step 4: Generate Visualizations
    print_header("STEP 4: Generating Visualizations")
    try:
        from results.generate_visualizations import VisualizationGenerator
        viz_gen = VisualizationGenerator()
        viz_gen.generate_all_visualizations()
    except Exception as e:
        print(f"\n✗ Error in visualization: {str(e)}")
        return
    
    # Final Summary
    print_header("PROJECT COMPLETED SUCCESSFULLY!")
    print("Project Structure:")
    print("├── data/")
    print("│   ├── diabetes_prediction_dataset.csv")
    print("│   ├── X_train.pkl, X_test.pkl")
    print("│   ├── y_train.pkl, y_test.pkl")
    print("│   └── scaler.pkl, label_encoders.pkl")
    print("├── models/")
    print("│   ├── logistic_regression_model.pkl")
    print("│   ├── decision_tree_model.pkl")
    print("│   ├── random_forest_model.pkl")
    print("│   ├── knn_model.pkl")
    print("│   └── naive_bayes_model.pkl")
    print("├── results/")
    print("│   ├── metrics/")
    print("│   │   ├── all_metrics.json")
    print("│   │   ├── detailed_results.pkl")
    print("│   │   └── model_comparison.csv")
    print("│   └── visualizations/")
    print("│       ├── 01_dataset_overview.png")
    print("│       ├── 02_correlation_heatmap.png")
    print("│       ├── 03_model_comparison.png")
    print("│       ├── 04_confusion_matrices.png")
    print("│       ├── 05_roc_curves.png")
    print("│       ├── 06_precision_recall_curves.png")
    print("│       └── 07_metric_radar.png")
    print("\nBest Model Performance:")
    print(summary_df.nlargest(1, 'Accuracy').to_string(index=False))
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
