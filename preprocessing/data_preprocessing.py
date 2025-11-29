"""
Data Preprocessing Module
Handles data loading, cleaning, encoding, and splitting
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import os
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

class DiabetesDataPreprocessor:
    def __init__(self, data_path='data/diabetes_prediction_dataset.csv'):
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_data(self):
        """Load the dataset"""
        print("Loading dataset...")
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        return self.df
    
    def explore_data(self):
        """Basic data exploration"""
        print("\n=== Dataset Info ===")
        print(self.df.info())
        print("\n=== First 5 rows ===")
        print(self.df.head())
        print("\n=== Statistical Summary ===")
        print(self.df.describe())
        print("\n=== Missing Values ===")
        print(self.df.isnull().sum())
        print("\n=== Target Distribution ===")
        print(self.df['diabetes'].value_counts())
        print(f"Diabetes prevalence: {self.df['diabetes'].mean()*100:.2f}%")
        
    def clean_data(self):
        """Clean and prepare data"""
        print("\nCleaning data...")
        
        # Handle missing values if any
        if self.df.isnull().sum().sum() > 0:
            print("Handling missing values...")
            self.df = self.df.dropna()
        
        # Remove duplicates
        duplicates = self.df.duplicated().sum()
        if duplicates > 0:
            print(f"Removing {duplicates} duplicate rows...")
            self.df = self.df.drop_duplicates()
        
        print(f"Data after cleaning: {self.df.shape[0]} rows")
        
    def encode_categorical(self):
        """Encode categorical variables"""
        print("\nEncoding categorical variables...")
        
        categorical_cols = ['gender', 'smoking_history']
        
        for col in categorical_cols:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col])
            self.label_encoders[col] = le
            print(f"Encoded {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")


    def split_data(self, test_size=0.2, random_state=42):
        """Split data into train and test sets"""
        print("\nSplitting data...")
        
        # Separate features and target
        X = self.df.drop('diabetes', axis=1)
        y = self.df['diabetes']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        return X_train, X_test, y_train, y_test
    
    def scale_features(self, X_train, X_test):
        """Scale numerical features"""
        print("\nScaling features...")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert back to DataFrame
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
        
        return X_train_scaled, X_test_scaled
    
    def save_preprocessed_data(self, X_train, X_test, y_train, y_test):
        """Save preprocessed data"""
        print("\nSaving preprocessed data...")
        
        # Save as pickle files
        with open('data/X_train.pkl', 'wb') as f:
            pickle.dump(X_train, f)
        with open('data/X_test.pkl', 'wb') as f:
            pickle.dump(X_test, f)
        with open('data/y_train.pkl', 'wb') as f:
            pickle.dump(y_train, f)
        with open('data/y_test.pkl', 'wb') as f:
            pickle.dump(y_test, f)
        with open('data/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        with open('data/label_encoders.pkl', 'wb') as f:
            pickle.dump(self.label_encoders, f)
            
        print("Data saved successfully!")
    
    def preprocess(self):
        """Complete preprocessing pipeline"""
        self.load_data()
        self.explore_data()
        self.clean_data()
        self.encode_categorical()
        X_train, X_test, y_train, y_test = self.split_data()
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        self.save_preprocessed_data(X_train_scaled, X_test_scaled, y_train, y_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test

    def cross_validate_model(self, X_train, y_train):
        print("\nstart cross val")
        model = LogisticRegression()
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Average CV score: {np.mean(cv_scores)}")

if __name__ == "__main__":
    preprocessor = DiabetesDataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.preprocess()
    print("\n✓ Preprocessing completed successfully!")
