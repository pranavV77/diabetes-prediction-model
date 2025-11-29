"""
SMOTE Comparison Visualizations
Generate visualizations comparing Before SMOTE vs After SMOTE
"""

import pandas as pd
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10

class SMOTEVisualizationGenerator:
    def __init__(self):
        self.load_data()
        self.load_results()
        
    def load_data(self):
        """Load original data"""
        print("Loading data...")
        self.df = pd.read_csv('data/diabetes_prediction_dataset.csv')
        with open('data/y_test.pkl', 'rb') as f:
            self.y_test = pickle.load(f)
    
    def load_results(self):
        """Load both original and SMOTE results"""
        print("Loading results...")
        
        # Original results
        with open('results/metrics/all_metrics.json') as f:
            self.original_results = json.load(f)
        
        # SMOTE results
        with open('results/metrics/smote_metrics.json') as f:
            self.smote_results = json.load(f)
        
        # Load detailed results for curves
        with open('results/metrics/detailed_results.pkl', 'rb') as f:
            self.original_detailed = pickle.load(f)
        
        with open('results/metrics/smote_detailed_results.pkl', 'rb') as f:
            self.smote_detailed = pickle.load(f)
    
    def plot_before_after_comparison(self):
        """Main comparison: Before vs After SMOTE"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Model Performance: Before vs After SMOTE', fontsize=16, fontweight='bold', y=0.995)
        
        models = list(self.original_results.keys())
        x = np.arange(len(models))
        width = 0.35
        
        # Colors
        color_before = '#e74c3c'
        color_after = '#2ecc71'
        
        # 1. Recall Comparison
        ax = axes[0, 0]
        recalls_before = [self.original_results[m]['recall'] for m in models]
        recalls_after = [self.smote_results[m]['recall'] for m in models]
        
        bars1 = ax.bar(x - width/2, recalls_before, width, label='Before SMOTE', color=color_before, alpha=0.8)
        bars2 = ax.bar(x + width/2, recalls_after, width, label='After SMOTE', color=color_after, alpha=0.8)
        
        ax.set_ylabel('Recall', fontweight='bold', fontsize=12)
        ax.set_title('Recall: Before vs After SMOTE', fontweight='bold', fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim([0.5, 1.0])
        ax.axhline(y=0.85, color='green', linestyle='--', alpha=0.5, label='Target (0.85)')
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 2. Precision Comparison
        ax = axes[0, 1]
        precisions_before = [self.original_results[m]['precision'] for m in models]
        precisions_after = [self.smote_results[m]['precision'] for m in models]
        
        bars1 = ax.bar(x - width/2, precisions_before, width, label='Before SMOTE', color=color_before, alpha=0.8)
        bars2 = ax.bar(x + width/2, precisions_after, width, label='After SMOTE', color=color_after, alpha=0.8)
        
        ax.set_ylabel('Precision', fontweight='bold', fontsize=12)
        ax.set_title('Precision: Before vs After SMOTE', fontweight='bold', fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim([0.3, 1.0])
        
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 3. F1-Score Comparison
        ax = axes[1, 0]
        f1_before = [self.original_results[m]['f1_score'] for m in models]
        f1_after = [self.smote_results[m]['f1_score'] for m in models]
        
        bars1 = ax.bar(x - width/2, f1_before, width, label='Before SMOTE', color=color_before, alpha=0.8)
        bars2 = ax.bar(x + width/2, f1_after, width, label='After SMOTE', color=color_after, alpha=0.8)
        
        ax.set_ylabel('F1-Score', fontweight='bold', fontsize=12)
        ax.set_title('F1-Score: Before vs After SMOTE', fontweight='bold', fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim([0.4, 0.9])
        ax.axhline(y=0.83, color='green', linestyle='--', alpha=0.5, label='Target (0.83)')
        
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 4. Improvement Percentage
        ax = axes[1, 1]
        recall_improvements = [(recalls_after[i] - recalls_before[i]) * 100 for i in range(len(models))]
        
        colors = [color_after if imp > 0 else color_before for imp in recall_improvements]
        bars = ax.bar(models, recall_improvements, color=colors, alpha=0.8)
        
        ax.set_ylabel('Recall Improvement (%)', fontweight='bold', fontsize=12)
        ax.set_title('Recall Improvement After SMOTE', fontweight='bold', fontsize=13)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5 if height > 0 else height - 1,
                   f'{height:.1f}%', ha='center', va='bottom' if height > 0 else 'top', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('results/visualizations/08_smote_before_after_comparison.png', dpi=300, bbox_inches='tight')
        print("✓ Before/After comparison saved")
        plt.close()
    
    def plot_confusion_matrices_comparison(self):
        """Confusion matrices: Before vs After for top 3 models"""
        top_models = ['Random Forest', 'Logistic Regression', 'Decision Tree']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Confusion Matrices: Before SMOTE (Top) vs After SMOTE (Bottom)', 
                     fontsize=16, fontweight='bold')
        
        for idx, model in enumerate(top_models):
            # Before SMOTE
            ax = axes[0, idx]
            cm_before = np.array(self.original_results[model]['confusion_matrix'])
            sns.heatmap(cm_before, annot=True, fmt='d', cmap='Reds', ax=ax,
                       cbar_kws={'label': 'Count'}, vmin=0, vmax=cm_before.max())
            ax.set_title(f'{model}\nBefore SMOTE', fontweight='bold')
            ax.set_ylabel('Actual')
            ax.set_xlabel('Predicted')
            ax.set_xticklabels(['No Diabetes', 'Diabetes'])
            ax.set_yticklabels(['No Diabetes', 'Diabetes'])
            
            # Add metrics
            recall_before = self.original_results[model]['recall']
            precision_before = self.original_results[model]['precision']
            ax.text(0.5, -0.15, f'Recall: {recall_before:.3f} | Precision: {precision_before:.3f}',
                   transform=ax.transAxes, ha='center', fontsize=10)
            
            # After SMOTE
            ax = axes[1, idx]
            cm_after = np.array(self.smote_results[model]['confusion_matrix'])
            sns.heatmap(cm_after, annot=True, fmt='d', cmap='Greens', ax=ax,
                       cbar_kws={'label': 'Count'}, vmin=0, vmax=cm_after.max())
            ax.set_title(f'{model}\nAfter SMOTE', fontweight='bold')
            ax.set_ylabel('Actual')
            ax.set_xlabel('Predicted')
            ax.set_xticklabels(['No Diabetes', 'Diabetes'])
            ax.set_yticklabels(['No Diabetes', 'Diabetes'])
            
            # Add metrics and improvement
            recall_after = self.smote_results[model]['recall']
            precision_after = self.smote_results[model]['precision']
            recall_gain = (recall_after - recall_before) * 100
            ax.text(0.5, -0.15, f'Recall: {recall_after:.3f} | Precision: {precision_after:.3f} | Gain: +{recall_gain:.1f}%',
                   transform=ax.transAxes, ha='center', fontsize=10,
                   color='green' if recall_gain > 0 else 'red', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('results/visualizations/09_smote_confusion_matrices_comparison.png', dpi=300, bbox_inches='tight')
        print("✓ Confusion matrices comparison saved")
        plt.close()
    
    def plot_roc_curves_comparison(self):
        """ROC curves: Before vs After"""
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
        
        # Before SMOTE
        ax = axes[0]
        for idx, (name, metrics) in enumerate(self.original_detailed.items()):
            if 'roc_curve' in metrics and metrics['roc_curve']:
                fpr = metrics['roc_curve']['fpr']
                tpr = metrics['roc_curve']['tpr']
                roc_auc = self.original_results[name]['roc_auc']
                ax.plot(fpr, tpr, color=colors[idx], lw=2, 
                       label=f'{name} (AUC={roc_auc:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontweight='bold', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontweight='bold', fontsize=12)
        ax.set_title('ROC Curves - Before SMOTE', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # After SMOTE
        ax = axes[1]
        for idx, (name, metrics) in enumerate(self.smote_detailed.items()):
            if 'roc_curve' in metrics and metrics['roc_curve']:
                fpr = metrics['roc_curve']['fpr']
                tpr = metrics['roc_curve']['tpr']
                roc_auc = self.smote_results[name]['roc_auc']
                ax.plot(fpr, tpr, color=colors[idx], lw=2, 
                       label=f'{name} (AUC={roc_auc:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontweight='bold', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontweight='bold', fontsize=12)
        ax.set_title('ROC Curves - After SMOTE', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/visualizations/10_smote_roc_comparison.png', dpi=300, bbox_inches='tight')
        print("✓ ROC curves comparison saved")
        plt.close()
    
    def plot_pr_curves_comparison(self):
        """Precision-Recall curves: Before vs After"""
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
        
        # Before SMOTE
        ax = axes[0]
        for idx, (name, metrics) in enumerate(self.original_detailed.items()):
            if 'pr_curve' in metrics and metrics['pr_curve']:
                precision = metrics['pr_curve']['precision']
                recall = metrics['pr_curve']['recall']
                pr_auc = self.original_results[name]['pr_auc']
                ax.plot(recall, precision, color=colors[idx], lw=2,
                       label=f'{name} (AUC={pr_auc:.3f})')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontweight='bold', fontsize=12)
        ax.set_ylabel('Precision', fontweight='bold', fontsize=12)
        ax.set_title('Precision-Recall Curves - Before SMOTE', fontsize=14, fontweight='bold')
        ax.legend(loc='lower left', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # After SMOTE
        ax = axes[1]
        for idx, (name, metrics) in enumerate(self.smote_detailed.items()):
            if 'pr_curve' in metrics and metrics['pr_curve']:
                precision = metrics['pr_curve']['precision']
                recall = metrics['pr_curve']['recall']
                pr_auc = self.smote_results[name]['pr_auc']
                ax.plot(recall, precision, color=colors[idx], lw=2,
                       label=f'{name} (AUC={pr_auc:.3f})')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontweight='bold', fontsize=12)
        ax.set_ylabel('Precision', fontweight='bold', fontsize=12)
        ax.set_title('Precision-Recall Curves - After SMOTE', fontsize=14, fontweight='bold')
        ax.legend(loc='lower left', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/visualizations/11_smote_pr_comparison.png', dpi=300, bbox_inches='tight')
        print("✓ PR curves comparison saved")
        plt.close()
    
    def plot_improvement_summary(self):
        """Summary of improvements from SMOTE"""
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Prepare data
        models = list(self.original_results.keys())
        metrics = ['Recall', 'Precision', 'F1-Score']
        
        before_data = {
            'Recall': [self.original_results[m]['recall'] for m in models],
            'Precision': [self.original_results[m]['precision'] for m in models],
            'F1-Score': [self.original_results[m]['f1_score'] for m in models]
        }
        
        after_data = {
            'Recall': [self.smote_results[m]['recall'] for m in models],
            'Precision': [self.smote_results[m]['precision'] for m in models],
            'F1-Score': [self.smote_results[m]['f1_score'] for m in models]
        }
        
        # Create grouped bar chart
        x = np.arange(len(models))
        width = 0.12
        
        colors_before = ['#e74c3c', '#e67e22', '#f39c12']
        colors_after = ['#27ae60', '#16a085', '#2ecc71']
        
        for i, metric in enumerate(metrics):
            ax.bar(x + (i-1)*width - width/2, before_data[metric], width, 
                   label=f'{metric} (Before)', color=colors_before[i], alpha=0.7)
            ax.bar(x + (i-1)*width + width/2, after_data[metric], width,
                   label=f'{metric} (After)', color=colors_after[i], alpha=0.9)
        
        ax.set_ylabel('Score', fontweight='bold', fontsize=13)
        ax.set_title('Complete Metrics Comparison: Before vs After SMOTE', 
                     fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right', fontsize=11)
        ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize=10)
        ax.set_ylim([0, 1.05])
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/visualizations/12_smote_complete_summary.png', dpi=300, bbox_inches='tight')
        print("✓ Complete summary saved")
        plt.close()
    
    def generate_all_visualizations(self):
        """Generate all SMOTE comparison visualizations"""
        print("\n" + "="*80)
        print("GENERATING SMOTE COMPARISON VISUALIZATIONS")
        print("="*80 + "\n")
        
        self.plot_before_after_comparison()
        self.plot_confusion_matrices_comparison()
        self.plot_roc_curves_comparison()
        self.plot_pr_curves_comparison()
        self.plot_improvement_summary()
        
        print("\n" + "="*80)
        print("✓ All SMOTE comparison visualizations generated!")
        print("="*80)
        print("\nGenerated files:")
        print("  08. Before/After metrics comparison")
        print("  09. Confusion matrices comparison")
        print("  10. ROC curves comparison")
        print("  11. Precision-Recall curves comparison")
        print("  12. Complete metrics summary")


if __name__ == "__main__":
    viz_gen = SMOTEVisualizationGenerator()
    viz_gen.generate_all_visualizations()