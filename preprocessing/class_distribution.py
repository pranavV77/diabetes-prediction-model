"""
Finding Class Distribution Before and After SMOTE
"""

import pandas as pd
import numpy as np
from collections import Counter
from imblearn.over_sampling import SMOTE
import pickle

# Load your preprocessed data
with open('data/X_train.pkl', 'rb') as f:
    X_train = pickle.load(f)
with open('data/y_train.pkl', 'rb') as f:
    y_train = pickle.load(f)

print("="*60)
print("CLASS DISTRIBUTION ANALYSIS")
print("="*60)

# ========== BEFORE SMOTE ==========
print("\n📊 BEFORE SMOTE:")
print("-" * 60)

# Count classes
class_counts_before = Counter(y_train)
total_samples_before = len(y_train)

print(f"Total Samples: {total_samples_before:,}")
print(f"\nClass 0 (No Diabetes): {class_counts_before[0]:,} samples ({class_counts_before[0]/total_samples_before*100:.2f}%)")
print(f"Class 1 (Diabetes):    {class_counts_before[1]:,} samples ({class_counts_before[1]/total_samples_before*100:.2f}%)")
print(f"\nImbalance Ratio: {class_counts_before[0]/class_counts_before[1]:.2f}:1")

# ========== APPLY SMOTE ==========
print("\n🔄 APPLYING SMOTE...")
print("-" * 60)

# Initialize SMOTE
smote = SMOTE(random_state=42)

# Apply SMOTE
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# ========== AFTER SMOTE ==========
print("\n📊 AFTER SMOTE:")
print("-" * 60)

# Count classes after SMOTE
class_counts_after = Counter(y_train_smote)
total_samples_after = len(y_train_smote)

print(f"Total Samples: {total_samples_after:,}")
print(f"\nClass 0 (No Diabetes): {class_counts_after[0]:,} samples ({class_counts_after[0]/total_samples_after*100:.2f}%)")
print(f"Class 1 (Diabetes):    {class_counts_after[1]:,} samples ({class_counts_after[1]/total_samples_after*100:.2f}%)")
print(f"\nImbalance Ratio: {class_counts_after[0]/class_counts_after[1]:.2f}:1")

# ========== SUMMARY ==========
print("\n" + "="*60)
print("SUMMARY OF CHANGES")
print("="*60)
print(f"Original Samples:     {total_samples_before:,}")
print(f"After SMOTE Samples:  {total_samples_after:,}")
print(f"New Samples Created:  {total_samples_after - total_samples_before:,}")
print(f"\nDiabetes Cases:")
print(f"  Before: {class_counts_before[1]:,}")
print(f"  After:  {class_counts_after[1]:,}")
print(f"  Increase: {class_counts_after[1] - class_counts_before[1]:,} synthetic samples")

# Save SMOTE data
with open('data/X_train_smote.pkl', 'wb') as f:
    pickle.dump(X_train_smote, f)
with open('data/y_train_smote.pkl', 'wb') as f:
    pickle.dump(y_train_smote, f)

print("\n✅ SMOTE data saved successfully!")
print("   - data/X_train_smote.pkl")
print("   - data/y_train_smote.pkl")