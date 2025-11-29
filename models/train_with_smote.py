"""
Stage 2: Training with SMOTE
Only use this if Stage 1 doesn't give enough recall improvement
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
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
import warnings
warnings.filterwarnings('ignore')


class SMOTEModelTrainer:
    def __init__(self, use_smotetomek=True):
        """
        use_smotetomek: If True, uses SMOTETomek (SMOTE + boundary cleaning)
                       If False, uses pure SMOTE
        """
        self.use_smotetomek = use_smotetomek
        
        # Same optimized models as Stage 1
        self.models = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000, 
                random_state=42,
                class_weight='balanced',
                C=0.1,
                solver='liblinear'
            ),
            'Decision Tree': DecisionTreeClassifier(
                random_state=42,
                class_weight='balanced',
                max_depth=10,
                min_samples_split=50,
                min_samples_leaf=20
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                class_weight='balanced_subsample',
                max_depth=20,
                min_samples_split=5,
                n_jobs=-1
            ),
            'KNN': KNeighborsClassifier(
                n_neighbors=7,
                weights='distance',
                metric='manhattan',
                n_jobs=-1
            ),
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
        
        print(f"Original data loaded:")
        print(f"  Training: {len(self.X_train)} samples")
        print(f"  Class distribution: {np.bincount(self.y_train)}")
        print(f"  Imbalance ratio: {np.bincount(self.y_train)[0] / np.bincount(self.y_train)[1]:.1f}:1")
    
    def apply_smote(self):
        """
        Apply SMOTE or SMOTETomek to training data
        
        IMPORTANT: Only apply to TRAINING data, NEVER to test data!
        """
        print("\n" + "="*80)
        if self.use_smotetomek:
            print("APPLYING SMOTETomek (SMOTE + Boundary Cleaning)")
        else:
            print("APPLYING SMOTE (Synthetic Minority Oversampling)")
        print("="*80)
        
        print("\n📊 Understanding what's happening:")
        print("1. Finding nearest neighbors of diabetic patients")
        print("2. Creating synthetic patients by interpolation")
        print("3. New samples will be between existing diabetic patients")
        
        if self.use_smotetomek:
            # SMOTETomek: SMOTE + remove noisy boundary samples
            print("4. Removing noisy samples at class boundaries (Tomek links)")
            resampler = SMOTETomek(
                sampling_strategy=0.5,  # Make minority class 50% of majority
                random_state=42
            )
        else:
            # Pure SMOTE
            resampler = SMOTE(
                sampling_strategy=0.5,
                random_state=42,
                k_neighbors=5  # Use 5 nearest neighbors
            )
        
        # Apply resampling
        print("\nApplying resampling...")
        self.X_train_resampled, self.y_train_resampled = resampler.fit_resample(
            self.X_train, self.y_train
        )
        
        # Report results
        print("\n✓ Resampling complete!")
        print(f"\nBefore SMOTE:")
        print(f"  Total samples: {len(self.y_train)}")
        print(f"  Non-diabetic: {np.bincount(self.y_train)[0]} (91%)")
        print(f"  Diabetic:     {np.bincount(self.y_train)[1]} (9%)")
        
        print(f"\nAfter SMOTE:")
        print(f"  Total samples: {len(self.y_train_resampled)}")
        print(f"  Non-diabetic: {np.bincount(self.y_train_resampled)[0]} (67%)")
        print(f"  Diabetic:     {np.bincount(self.y_train_resampled)[1]} (33%)")
        
        synthetic_created = np.bincount(self.y_train_resampled)[1] - np.bincount(self.y_train)[1]
        print(f"\n✨ Created {synthetic_created} synthetic diabetic patients")
        
        # Verify synthetic samples are realistic
        self.verify_synthetic_quality()
    
    def verify_synthetic_quality(self):
        """
        Verify that synthetic samples have realistic values
        """
        print("\n" + "="*80)
        print("VERIFYING SYNTHETIC SAMPLE QUALITY")
        print("="*80)
        
        # Compare distributions of original vs all resampled
        print("\n📊 Feature Distribution Check:")
        print("Comparing original diabetic patients vs resampled (original + synthetic)")
        print("If distributions are similar, synthetic samples are realistic!\n")
        
        feature_names = self.X_train.columns if hasattr(self.X_train, 'columns') else [f'Feature_{i}' for i in range(self.X_train.shape[1])]
        
        # Convert everything to numpy arrays
        y_train_array = self.y_train.values if hasattr(self.y_train, 'values') else self.y_train
        X_train_array = self.X_train.values if hasattr(self.X_train, 'values') else self.X_train
        
        # Convert resampled to numpy if needed
        X_resampled_array = self.X_train_resampled if isinstance(self.X_train_resampled, np.ndarray) else self.X_train_resampled.values
        y_resampled_array = self.y_train_resampled if isinstance(self.y_train_resampled, np.ndarray) else self.y_train_resampled.values
        
        # Get diabetic samples only
        X_diabetic_original = X_train_array[y_train_array == 1]
        X_diabetic_resampled = X_resampled_array[y_resampled_array == 1]
        
        print(f"{'Feature':<25} {'Original Mean':<15} {'Resampled Mean':<15} {'Difference':<12} {'Status'}")
        print("-" * 85)
        
        for i, feature in enumerate(feature_names):
            orig_mean = X_diabetic_original[:, i].mean()
            resamp_mean = X_diabetic_resampled[:, i].mean()
            
            diff_pct = abs(orig_mean - resamp_mean) / orig_mean * 100 if orig_mean != 0 else 0
            status = "✓ Good" if diff_pct < 10 else "⚠ Check" if diff_pct < 20 else "✗ Large"
            
            print(f"{feature:<25} {orig_mean:<15.3f} {resamp_mean:<15.3f} {diff_pct:<12.1f}% {status}")
        
        print("\n✓ If most features show <10% difference, synthetic samples are realistic!")
    
    def train_model(self, name, model):
        """Train model on SMOTE-resampled data"""
        print(f"\nTraining {name} on resampled data...")
        model.fit(self.X_train_resampled, self.y_train_resampled)
        print(f"✓ {name} trained on {len(self.X_train_resampled)} samples")
        return model
    
    def evaluate_model(self, name, model):
        """
        Evaluate model on ORIGINAL test data (no SMOTE on test!)
        This is crucial - we only SMOTE training data!
        """
        print(f"Evaluating {name} on REAL test data...")
        
        # Predictions on ORIGINAL test set
        y_pred = model.predict(self.X_test)
        
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        else:
            y_pred_proba = None
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred),
            'recall': recall_score(self.y_test, y_pred),
            'f1_score': f1_score(self.y_test, y_pred),
            'confusion_matrix': confusion_matrix(self.y_test, y_pred).tolist(),
        }
        
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(self.y_test, y_pred_proba)
            metrics['pr_auc'] = average_precision_score(self.y_test, y_pred_proba)
            
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
            precision, recall, _ = precision_recall_curve(self.y_test, y_pred_proba)
            
            metrics['roc_curve'] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist()}
            metrics['pr_curve'] = {'precision': precision.tolist(), 'recall': recall.tolist()}
        else:
            metrics['roc_auc'] = None
            metrics['pr_auc'] = None
        
        return metrics
    
    def train_all_models(self):
        """Train and evaluate all models"""
        print("\n" + "="*80)
        print("TRAINING ALL MODELS WITH SMOTE")
        print("="*80)
        
        for name, model in self.models.items():
            # Train on SMOTE data
            trained_model = self.train_model(name, model)
            
            # Evaluate on REAL test data
            metrics = self.evaluate_model(name, trained_model)
            
            # Store results
            self.results[name] = metrics
            
            # Save model
            filename = f'models/smote_{name.lower().replace(" ", "_")}_model.pkl'
            with open(filename, 'wb') as f:
                pickle.dump(trained_model, f)
            
            # Print summary
            print(f"\n{name} Results (tested on REAL data):")
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f} ⬅ Key metric!")
            print(f"  F1-Score:  {metrics['f1_score']:.4f}")
            if metrics['roc_auc']:
                print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
                print(f"  PR-AUC:    {metrics['pr_auc']:.4f}")
        
        return self.results
    
    def compare_with_previous(self):
        """Compare SMOTE results with previous stages"""
        print("\n" + "="*80)
        print("COMPARISON: Original → Optimized → SMOTE")
        print("="*80)
        
        # Load previous results
        try:
            with open('results/metrics/all_metrics.json', 'r') as f:
                original = json.load(f)
        except:
            original = {}
        
        try:
            with open('results/metrics/optimized_metrics.json', 'r') as f:
                optimized = json.load(f)
        except:
            optimized = {}
        
        comparison = []
        for model_name in self.results.keys():
            orig = original.get(model_name, {})
            opt = optimized.get(model_name, {})
            smote = self.results[model_name]
            
            comparison.append({
                'Model': model_name,
                'Original_Recall': orig.get('recall', 0),
                'Optimized_Recall': opt.get('recall', 0),
                'SMOTE_Recall': smote['recall'],
                'Total_Gain': smote['recall'] - orig.get('recall', 0),
                'F1_Original': orig.get('f1_score', 0),
                'F1_SMOTE': smote['f1_score'],
                'F1_Gain': smote['f1_score'] - orig.get('f1_score', 0)
            })
        
        df = pd.DataFrame(comparison)
        print("\n" + df.to_string(index=False))
        
        # Save
        df.to_csv('results/metrics/smote_comparison.csv', index=False)
        print("\n✓ Saved to results/metrics/smote_comparison.csv")
        
        return df
    
    def save_results(self):
        """Save SMOTE results"""
        print("\n" + "="*80)
        print("SAVING SMOTE RESULTS")
        print("="*80)
        
        # JSON format
        results_json = {
            name: {
                'accuracy': m['accuracy'],
                'precision': m['precision'],
                'recall': m['recall'],
                'f1_score': m['f1_score'],
                'roc_auc': m['roc_auc'],
                'pr_auc': m['pr_auc'],
                'confusion_matrix': m['confusion_matrix']
            }
            for name, m in self.results.items()
        }
        
        with open('results/metrics/smote_metrics.json', 'w') as f:
            json.dump(results_json, f, indent=4)
        
        # Detailed results
        with open('results/metrics/smote_detailed_results.pkl', 'wb') as f:
            pickle.dump(self.results, f)
        
        # CSV summary
        summary_df = pd.DataFrame({
            'Model': list(self.results.keys()),
            'Accuracy': [m['accuracy'] for m in self.results.values()],
            'Precision': [m['precision'] for m in self.results.values()],
            'Recall': [m['recall'] for m in self.results.values()],
            'F1-Score': [m['f1_score'] for m in self.results.values()],
            'ROC-AUC': [m['roc_auc'] for m in self.results.values()],
            'PR-AUC': [m['pr_auc'] for m in self.results.values()]
        })
        summary_df.to_csv('results/metrics/smote_model_comparison.csv', index=False)
        
        print("✓ SMOTE results saved!")
        print(f"  - results/metrics/smote_metrics.json")
        print(f"  - results/metrics/smote_detailed_results.pkl")
        print(f"  - results/metrics/smote_model_comparison.csv")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("DIABETES PREDICTION - STAGE 2: SMOTE TRAINING")
    print("="*80)
    print("\n⚠ IMPORTANT: This creates synthetic diabetic patients")
    print("Only use if Stage 1 (class weights) didn't achieve target recall")
    print("\nTarget: Recall > 0.85, F1 > 0.83")
    print("="*80)
    
    # Use SMOTETomek (better than pure SMOTE)
    trainer = SMOTEModelTrainer(use_smotetomek=True)
    trainer.load_data()
    trainer.apply_smote()
    trainer.train_all_models()
    trainer.save_results()
    comparison = trainer.compare_with_previous()
    
    print("\n" + "="*80)
    print("✓ STAGE 2 (SMOTE) COMPLETED!")
    print("="*80)
    print("\nKey Points:")
    print("✓ Models trained on synthetic + real data")
    print("✓ Tested on 100% REAL data (no synthetic in test set)")
    print("✓ Good test performance = synthetic samples were realistic!")