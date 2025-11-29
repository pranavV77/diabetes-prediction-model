"""
Model Training Module
Train and evaluate multiple ML algorithms
"""

import pandas as pd
import numpy as np
import pickle
import json
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, 
    precision_recall_curve, average_precision_score
)
import warnings
warnings.filterwarnings('ignore')

class DiabetesModelTrainer:
    def __init__(self):
        self.models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'Naive Bayes': GaussianNB()
        }
        self.results = {}
        
    def load_data(self):
        """Load preprocessed data"""
        print("Loading preprocessed data...")
        with open('data/X_train.pkl', 'rb') as f:
            self.X_train = pickle.load(f)
        with open('data/X_test.pkl', 'rb') as f:
            self.X_test = pickle.load(f)
        with open('data/y_train.pkl', 'rb') as f:
            self.y_train = pickle.load(f)
        with open('data/y_test.pkl', 'rb') as f:
            self.y_test = pickle.load(f)
        print(f"Data loaded: Train={len(self.X_train)}, Test={len(self.X_test)}")
    
    def train_model(self, name, model):
        """Train a single model"""
        print(f"\nTraining {name}...")
        model.fit(self.X_train, self.y_train)
        print(f"✓ {name} trained successfully")
        return model
    
    def evaluate_model(self, name, model):
        """Evaluate model and compute all metrics"""
        print(f"Evaluating {name}...")
        
        # Predictions
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred),
            'recall': recall_score(self.y_test, y_pred),
            'f1_score': f1_score(self.y_test, y_pred),
            'confusion_matrix': confusion_matrix(self.y_test, y_pred).tolist(),
        }
        
        # ROC-AUC and PR-AUC (requires probability predictions)
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(self.y_test, y_pred_proba)
            metrics['pr_auc'] = average_precision_score(self.y_test, y_pred_proba)
            
            # Store curves for visualization
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
            precision, recall, _ = precision_recall_curve(self.y_test, y_pred_proba)
            
            metrics['roc_curve'] = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist()
            }
            metrics['pr_curve'] = {
                'precision': precision.tolist(),
                'recall': recall.tolist()
            }
        else:
            metrics['roc_auc'] = None
            metrics['pr_auc'] = None
        
        # Store predictions for visualization
        metrics['y_pred'] = y_pred.tolist()
        if y_pred_proba is not None:
            metrics['y_pred_proba'] = y_pred_proba.tolist()
        
        return metrics
    
    def train_all_models(self):
        """Train and evaluate all models"""
        print("\n" + "="*60)
        print("TRAINING ALL MODELS")
        print("="*60)
        
        for name, model in self.models.items():
            # Train
            trained_model = self.train_model(name, model)
            
            # Evaluate
            metrics = self.evaluate_model(name, trained_model)
            
            # Store results
            self.results[name] = metrics
            
            # Save model
            with open(f'models/{name.lower().replace(" ", "_")}_model.pkl', 'wb') as f:
                pickle.dump(trained_model, f)
            
            # Print summary
            print(f"\n{name} Results:")
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1-Score:  {metrics['f1_score']:.4f}")
            if metrics['roc_auc']:
                print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
                print(f"  PR-AUC:    {metrics['pr_auc']:.4f}")
        
        return self.results
    
    def save_results(self):
        """Save all results"""
        print("\n" + "="*60)
        print("SAVING RESULTS")
        print("="*60)
        
        # Save complete results as JSON
        results_json = {}
        for name, metrics in self.results.items():
            results_json[name] = {
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score'],
                'roc_auc': metrics['roc_auc'],
                'pr_auc': metrics['pr_auc'],
                'confusion_matrix': metrics['confusion_matrix']
            }
        
        with open('results/metrics/all_metrics.json', 'w') as f:
            json.dump(results_json, f, indent=4)
        
        # Save detailed results with curves
        with open('results/metrics/detailed_results.pkl', 'wb') as f:
            pickle.dump(self.results, f)
        
        # Create summary DataFrame
        summary_df = pd.DataFrame({
            'Model': list(results_json.keys()),
            'Accuracy': [m['accuracy'] for m in results_json.values()],
            'Precision': [m['precision'] for m in results_json.values()],
            'Recall': [m['recall'] for m in results_json.values()],
            'F1-Score': [m['f1_score'] for m in results_json.values()],
            'ROC-AUC': [m['roc_auc'] for m in results_json.values()],
            'PR-AUC': [m['pr_auc'] for m in results_json.values()]
        })
        summary_df.to_csv('results/metrics/model_comparison.csv', index=False)
        
        print("✓ Results saved successfully!")
        print(f"  - results/metrics/all_metrics.json")
        print(f"  - results/metrics/detailed_results.pkl")
        print(f"  - results/metrics/model_comparison.csv")
        
        return summary_df
    
    def print_summary(self):
        """Print comparison summary"""
        print("\n" + "="*60)
        print("MODEL COMPARISON SUMMARY")
        print("="*60)
        
        summary = pd.read_csv('results/metrics/model_comparison.csv')
        print(summary.to_string(index=False))
        
        # Find best model for each metric
        print("\n" + "="*60)
        print("BEST MODELS BY METRIC")
        print("="*60)
        for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'PR-AUC']:
            if summary[metric].notna().any():
                best_idx = summary[metric].idxmax()
                best_model = summary.loc[best_idx, 'Model']
                best_score = summary.loc[best_idx, metric]
                print(f"{metric:12s}: {best_model:20s} ({best_score:.4f})")


if __name__ == "__main__":
    trainer = DiabetesModelTrainer()
    trainer.load_data()
    trainer.train_all_models()
    trainer.save_results()
    trainer.print_summary()
    print("\n✓ Model training completed successfully!")
