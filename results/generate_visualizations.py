"""
Visualization Module
Generate comprehensive visualizations for case study
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

class VisualizationGenerator:
    def __init__(self):
        self.load_data()
        self.load_results()
        
    def load_data(self):
        """Load original and preprocessed data"""
        print("Loading data for visualization...")
        self.df = pd.read_csv('data/diabetes_prediction_dataset.csv')
        with open('data/y_test.pkl', 'rb') as f:
            self.y_test = pickle.load(f)
    
    def load_results(self):
        """Load model results"""
        with open('results/metrics/detailed_results.pkl', 'rb') as f:
            self.results = pickle.load(f)
        self.summary_df = pd.read_csv('results/metrics/model_comparison.csv')
    
    def plot_dataset_overview(self):
        """Plot dataset characteristics"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Dataset Overview', fontsize=16, fontweight='bold')
        
        # 1. Target distribution
        ax = axes[0, 0]
        diabetes_counts = self.df['diabetes'].value_counts()
        colors = ['#2ecc71', '#e74c3c']
        ax.pie(diabetes_counts, labels=['No Diabetes', 'Diabetes'], 
               autopct='%1.1f%%', colors=colors, startangle=90)
        ax.set_title('Target Distribution', fontweight='bold')
        
        # 2. Age distribution
        ax = axes[0, 1]
        self.df[self.df['diabetes']==0]['age'].hist(ax=ax, bins=30, alpha=0.7, 
                                                     label='No Diabetes', color='#2ecc71')
        self.df[self.df['diabetes']==1]['age'].hist(ax=ax, bins=30, alpha=0.7, 
                                                     label='Diabetes', color='#e74c3c')
        ax.set_xlabel('Age')
        ax.set_ylabel('Frequency')
        ax.set_title('Age Distribution by Diabetes Status', fontweight='bold')
        ax.legend()
        
        # 3. BMI distribution
        ax = axes[0, 2]
        self.df[self.df['diabetes']==0]['bmi'].hist(ax=ax, bins=30, alpha=0.7, 
                                                     label='No Diabetes', color='#2ecc71')
        self.df[self.df['diabetes']==1]['bmi'].hist(ax=ax, bins=30, alpha=0.7, 
                                                     label='Diabetes', color='#e74c3c')
        ax.set_xlabel('BMI')
        ax.set_ylabel('Frequency')
        ax.set_title('BMI Distribution by Diabetes Status', fontweight='bold')
        ax.legend()
        
        # 4. Gender distribution
        ax = axes[1, 0]
        gender_diabetes = pd.crosstab(self.df['gender'], self.df['diabetes'])
        gender_diabetes.plot(kind='bar', ax=ax, color=colors)
        ax.set_xlabel('Gender')
        ax.set_ylabel('Count')
        ax.set_title('Diabetes by Gender', fontweight='bold')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.legend(['No Diabetes', 'Diabetes'])
        
        # 5. HbA1c levels
        ax = axes[1, 1]
        self.df.boxplot(column='HbA1c_level', by='diabetes', ax=ax)
        ax.set_xlabel('Diabetes Status')
        ax.set_ylabel('HbA1c Level')
        ax.set_title('HbA1c Levels by Diabetes Status', fontweight='bold')
        ax.set_xticklabels(['No Diabetes', 'Diabetes'])
        plt.suptitle('')
        
        # 6. Blood Glucose levels
        ax = axes[1, 2]
        self.df.boxplot(column='blood_glucose_level', by='diabetes', ax=ax)
        ax.set_xlabel('Diabetes Status')
        ax.set_ylabel('Blood Glucose Level')
        ax.set_title('Blood Glucose Levels by Diabetes Status', fontweight='bold')
        ax.set_xticklabels(['No Diabetes', 'Diabetes'])
        plt.suptitle('')
        
        plt.tight_layout()
        plt.savefig('results/visualizations/01_dataset_overview.png', dpi=300, bbox_inches='tight')
        print("✓ Dataset overview saved")
        plt.close()
    
    def plot_correlation_heatmap(self):
        """Plot correlation heatmap"""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Select numerical columns
        numerical_cols = ['age', 'hypertension', 'heart_disease', 'bmi', 
                         'HbA1c_level', 'blood_glucose_level', 'diabetes']
        corr_matrix = self.df[numerical_cols].corr()
        
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, ax=ax, cbar_kws={'label': 'Correlation'})
        ax.set_title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig('results/visualizations/02_correlation_heatmap.png', dpi=300, bbox_inches='tight')
        print("✓ Correlation heatmap saved")
        plt.close()
    
    def plot_model_comparison(self):
        """Plot model comparison metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        models = self.summary_df['Model']
        x = np.arange(len(models))
        width = 0.6
        
        # 1. Accuracy comparison
        ax = axes[0, 0]
        bars = ax.bar(x, self.summary_df['Accuracy'], width, color='#3498db', alpha=0.8)
        ax.set_ylabel('Accuracy', fontweight='bold')
        ax.set_title('Model Accuracy Comparison', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylim([0.9, 1.0])
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}', ha='center', va='bottom', fontsize=9)
        
        # 2. Precision, Recall, F1-Score
        ax = axes[0, 1]
        x_pos = np.arange(len(models))
        width = 0.25
        ax.bar(x_pos - width, self.summary_df['Precision'], width, label='Precision', color='#e74c3c', alpha=0.8)
        ax.bar(x_pos, self.summary_df['Recall'], width, label='Recall', color='#2ecc71', alpha=0.8)
        ax.bar(x_pos + width, self.summary_df['F1-Score'], width, label='F1-Score', color='#f39c12', alpha=0.8)
        ax.set_ylabel('Score', fontweight='bold')
        ax.set_title('Precision, Recall, F1-Score Comparison', fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim([0, 1.0])
        
        # 3. ROC-AUC comparison
        ax = axes[1, 0]
        roc_auc_scores = self.summary_df['ROC-AUC'].dropna()
        roc_models = self.summary_df[self.summary_df['ROC-AUC'].notna()]['Model']
        bars = ax.bar(range(len(roc_models)), roc_auc_scores, width, color='#9b59b6', alpha=0.8)
        ax.set_ylabel('ROC-AUC Score', fontweight='bold')
        ax.set_title('ROC-AUC Score Comparison', fontweight='bold')
        ax.set_xticks(range(len(roc_models)))
        ax.set_xticklabels(roc_models, rotation=45, ha='right')
        ax.set_ylim([0.9, 1.0])
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}', ha='center', va='bottom', fontsize=9)
        
        # 4. PR-AUC comparison
        ax = axes[1, 1]
        pr_auc_scores = self.summary_df['PR-AUC'].dropna()
        pr_models = self.summary_df[self.summary_df['PR-AUC'].notna()]['Model']
        bars = ax.bar(range(len(pr_models)), pr_auc_scores, width, color='#1abc9c', alpha=0.8)
        ax.set_ylabel('PR-AUC Score', fontweight='bold')
        ax.set_title('PR-AUC Score Comparison', fontweight='bold')
        ax.set_xticks(range(len(pr_models)))
        ax.set_xticklabels(pr_models, rotation=45, ha='right')
        ax.set_ylim([0, 1.0])
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('results/visualizations/03_model_comparison.png', dpi=300, bbox_inches='tight')
        print("✓ Model comparison saved")
        plt.close()
    
    def plot_confusion_matrices(self):
        """Plot confusion matrices for all models"""
        n_models = len(self.results)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Confusion Matrices', fontsize=16, fontweight='bold')
        axes = axes.flatten()
        
        for idx, (name, metrics) in enumerate(self.results.items()):
            cm = np.array(metrics['confusion_matrix'])
            ax = axes[idx]
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       cbar_kws={'label': 'Count'})
            ax.set_title(f'{name}', fontweight='bold')
            ax.set_ylabel('Actual')
            ax.set_xlabel('Predicted')
            ax.set_xticklabels(['No Diabetes', 'Diabetes'])
            ax.set_yticklabels(['No Diabetes', 'Diabetes'])
            
            # Add accuracy text
            accuracy = metrics['accuracy']
            ax.text(0.5, -0.15, f'Accuracy: {accuracy:.4f}', 
                   transform=ax.transAxes, ha='center', fontsize=10)
        
        # Hide extra subplot
        axes[-1].axis('off')
        
        plt.tight_layout()
        plt.savefig('results/visualizations/04_confusion_matrices.png', dpi=300, bbox_inches='tight')
        print("✓ Confusion matrices saved")
        plt.close()
    
    def plot_roc_curves(self):
        """Plot ROC curves"""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
        
        for idx, (name, metrics) in enumerate(self.results.items()):
            if 'roc_curve' in metrics and metrics['roc_curve']:
                fpr = metrics['roc_curve']['fpr']
                tpr = metrics['roc_curve']['tpr']
                roc_auc = metrics['roc_auc']
                
                ax.plot(fpr, tpr, color=colors[idx], lw=2, 
                       label=f'{name} (AUC = {roc_auc:.4f})')
        
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontweight='bold', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontweight='bold', fontsize=12)
        ax.set_title('ROC Curves - All Models', fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/visualizations/05_roc_curves.png', dpi=300, bbox_inches='tight')
        print("✓ ROC curves saved")
        plt.close()
    
    def plot_precision_recall_curves(self):
        """Plot Precision-Recall curves"""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
        
        for idx, (name, metrics) in enumerate(self.results.items()):
            if 'pr_curve' in metrics and metrics['pr_curve']:
                precision = metrics['pr_curve']['precision']
                recall = metrics['pr_curve']['recall']
                pr_auc = metrics['pr_auc']
                
                ax.plot(recall, precision, color=colors[idx], lw=2,
                       label=f'{name} (AUC = {pr_auc:.4f})')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontweight='bold', fontsize=12)
        ax.set_ylabel('Precision', fontweight='bold', fontsize=12)
        ax.set_title('Precision-Recall Curves - All Models', fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='lower left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/visualizations/06_precision_recall_curves.png', dpi=300, bbox_inches='tight')
        print("✓ Precision-Recall curves saved")
        plt.close()
    
    def plot_metric_radar(self):
        """Plot radar chart comparing all metrics"""
        from math import pi
        
        fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
        
        # Metrics to compare
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        num_metrics = len(metrics)
        
        # Compute angle for each axis
        angles = [n / float(num_metrics) * 2 * pi for n in range(num_metrics)]
        angles += angles[:1]
        
        # Plot each model
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
        
        for idx, (_, row) in enumerate(self.summary_df.iterrows()):
            values = [row['Accuracy'], row['Precision'], row['Recall'], 
                     row['F1-Score'], row['ROC-AUC'] if pd.notna(row['ROC-AUC']) else 0]
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, color=colors[idx], 
                   label=row['Model'])
            ax.fill(angles, values, alpha=0.15, color=colors[idx])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=11, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.set_title('Model Performance Radar Chart', fontsize=16, 
                    fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig('results/visualizations/07_metric_radar.png', dpi=300, bbox_inches='tight')
        print("✓ Metric radar chart saved")
        plt.close()
    
    def generate_all_visualizations(self):
        """Generate all visualizations"""
        print("\n" + "="*60)
        print("GENERATING VISUALIZATIONS")
        print("="*60 + "\n")
        
        self.plot_dataset_overview()
        self.plot_correlation_heatmap()
        self.plot_model_comparison()
        self.plot_confusion_matrices()
        self.plot_roc_curves()
        self.plot_precision_recall_curves()
        self.plot_metric_radar()
        
        print("\n" + "="*60)
        print("✓ All visualizations generated successfully!")
        print("="*60)
        print("\nGenerated files:")
        print("  1. 01_dataset_overview.png")
        print("  2. 02_correlation_heatmap.png")
        print("  3. 03_model_comparison.png")
        print("  4. 04_confusion_matrices.png")
        print("  5. 05_roc_curves.png")
        print("  6. 06_precision_recall_curves.png")
        print("  7. 07_metric_radar.png")


if __name__ == "__main__":
    viz_gen = VisualizationGenerator()
    viz_gen.generate_all_visualizations()
