"""
Calculate Specificity and Compare Models Before/After SMOTE
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE

def calculate_specificity(y_true, y_pred):
    """
    Specificity = TN / (TN + FP)
    
    Measures how well model identifies negative cases (No Diabetes)
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    return specificity

def calculate_all_metrics(y_true, y_pred):
    """Calculate all metrics including specificity"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0  # Sensitivity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall (Sensitivity)': recall,
        'Specificity': specificity,
        'F1-Score': f1
    }

# Load data
print("Loading data...")
with open('data/X_train.pkl', 'rb') as f:
    X_train = pickle.load(f)
with open('data/X_test.pkl', 'rb') as f:
    X_test = pickle.load(f)
with open('data/y_train.pkl', 'rb') as f:
    y_train = pickle.load(f)
with open('data/y_test.pkl', 'rb') as f:
    y_test = pickle.load(f)

# Apply SMOTE
print("Applying SMOTE...")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Models to compare
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Naive Bayes': GaussianNB()
}

results_comparison = []

print("\n" + "="*80)
print("MODEL COMPARISON: WITHOUT SMOTE vs WITH SMOTE")
print("="*80)

for name, model in models.items():
    print(f"\n{'='*80}")
    print(f"MODEL: {name}")
    print(f"{'='*80}")
    
    # ========== WITHOUT SMOTE ==========
    print("\n📊 WITHOUT SMOTE (Original Imbalanced Data):")
    print("-" * 80)
    model_original = model.__class__(**model.get_params())
    model_original.fit(X_train, y_train)
    y_pred_original = model_original.predict(X_test)
    metrics_original = calculate_all_metrics(y_test, y_pred_original)
    
    print(f"Confusion Matrix:")
    print(f"  TN: {metrics_original['TN']:5d}  |  FP: {metrics_original['FP']:5d}")
    print(f"  FN: {metrics_original['FN']:5d}  |  TP: {metrics_original['TP']:5d}")
    print(f"\nMetrics:")
    print(f"  Accuracy:    {metrics_original['Accuracy']:.4f} ({metrics_original['Accuracy']*100:.2f}%)")
    print(f"  Precision:   {metrics_original['Precision']:.4f} ({metrics_original['Precision']*100:.2f}%)")
    print(f"  Recall:      {metrics_original['Recall (Sensitivity)']:.4f} ({metrics_original['Recall (Sensitivity)']*100:.2f}%)")
    print(f"  Specificity: {metrics_original['Specificity']:.4f} ({metrics_original['Specificity']*100:.2f}%)")
    print(f"  F1-Score:    {metrics_original['F1-Score']:.4f} ({metrics_original['F1-Score']*100:.2f}%)")
    
    # ========== WITH SMOTE ==========
    print("\n📊 WITH SMOTE (Balanced Data):")
    print("-" * 80)
    model_smote = model.__class__(**model.get_params())
    model_smote.fit(X_train_smote, y_train_smote)
    y_pred_smote = model_smote.predict(X_test)
    metrics_smote = calculate_all_metrics(y_test, y_pred_smote)
    
    print(f"Confusion Matrix:")
    print(f"  TN: {metrics_smote['TN']:5d}  |  FP: {metrics_smote['FP']:5d}")
    print(f"  FN: {metrics_smote['FN']:5d}  |  TP: {metrics_smote['TP']:5d}")
    print(f"\nMetrics:")
    print(f"  Accuracy:    {metrics_smote['Accuracy']:.4f} ({metrics_smote['Accuracy']*100:.2f}%)")
    print(f"  Precision:   {metrics_smote['Precision']:.4f} ({metrics_smote['Precision']*100:.2f}%)")
    print(f"  Recall:      {metrics_smote['Recall (Sensitivity)']:.4f} ({metrics_smote['Recall (Sensitivity)']*100:.2f}%)")
    print(f"  Specificity: {metrics_smote['Specificity']:.4f} ({metrics_smote['Specificity']*100:.2f}%)")
    print(f"  F1-Score:    {metrics_smote['F1-Score']:.4f} ({metrics_smote['F1-Score']*100:.2f}%)")
    
    # ========== CHANGES ==========
    print("\n📈 CHANGES (SMOTE vs Original):")
    print("-" * 80)
    print(f"  Accuracy:    {metrics_smote['Accuracy'] - metrics_original['Accuracy']:+.4f}")
    print(f"  Precision:   {metrics_smote['Precision'] - metrics_original['Precision']:+.4f} {'⬇️ DECREASED' if metrics_smote['Precision'] < metrics_original['Precision'] else '⬆️ INCREASED'}")
    print(f"  Recall:      {metrics_smote['Recall (Sensitivity)'] - metrics_original['Recall (Sensitivity)']:+.4f} {'⬆️ INCREASED' if metrics_smote['Recall (Sensitivity)'] > metrics_original['Recall (Sensitivity)'] else '⬇️ DECREASED'}")
    print(f"  Specificity: {metrics_smote['Specificity'] - metrics_original['Specificity']:+.4f}")
    print(f"  F1-Score:    {metrics_smote['F1-Score'] - metrics_original['F1-Score']:+.4f}")
    
    # Store for summary
    results_comparison.append({
        'Model': name,
        'Acc_Original': metrics_original['Accuracy'],
        'Prec_Original': metrics_original['Precision'],
        'Rec_Original': metrics_original['Recall (Sensitivity)'],
        'Spec_Original': metrics_original['Specificity'],
        'F1_Original': metrics_original['F1-Score'],
        'Acc_SMOTE': metrics_smote['Accuracy'],
        'Prec_SMOTE': metrics_smote['Precision'],
        'Rec_SMOTE': metrics_smote['Recall (Sensitivity)'],
        'Spec_SMOTE': metrics_smote['Specificity'],
        'F1_SMOTE': metrics_smote['F1-Score']
    })

# ========== SUMMARY TABLE ==========
print("\n" + "="*80)
print("SUMMARY TABLE: ALL MODELS")
print("="*80)

df_comparison = pd.DataFrame(results_comparison)
print("\nWithout SMOTE:")
print(df_comparison[['Model', 'Acc_Original', 'Prec_Original', 'Rec_Original', 'Spec_Original', 'F1_Original']].to_string(index=False))

print("\nWith SMOTE:")
print(df_comparison[['Model', 'Acc_SMOTE', 'Prec_SMOTE', 'Rec_SMOTE', 'Spec_SMOTE', 'F1_SMOTE']].to_string(index=False))

# Save results
df_comparison.to_csv('results/metrics/smote_comparison.csv', index=False)
print("\n✅ Results saved to: results/metrics/smote_comparison.csv")