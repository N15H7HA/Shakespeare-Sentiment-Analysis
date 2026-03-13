#!/usr/bin/env python3
"""
Emotion Classification Evaluation Script
Calculates accuracy, precision, recall, and F1 score for Lady Macbeth and Ophelia
by comparing ground truth labels with model predictions.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class EmotionEvaluator:
    """
    Evaluates emotion classification performance by comparing
    ground truth labels with model predictions.
    """
    
    def __init__(self, source_csv: str, results_csv: str, character_name: str):
        """
        Initialize evaluator with source and results files.
        
        Args:
            source_csv: Path to source CSV with ground_truth column
            results_csv: Path to results CSV with dominant_emotion predictions
            character_name: Name of the character being evaluated
        """
        self.character_name = character_name
        
        # Load data
        self.source_df = pd.read_csv(source_csv)
        self.results_df = pd.read_csv(results_csv)
        
        # Normalize emotion labels (lowercase and strip whitespace)
        self.source_df['ground_truth_normalized'] = self.source_df['ground_truth'].str.lower().str.strip()
        self.results_df['prediction_normalized'] = self.results_df['dominant_emotion'].str.lower().str.strip()
        
        # Merge dataframes on Sno
        self.merged_df = pd.merge(
            self.source_df[['Sno', 'ground_truth_normalized']],
            self.results_df[['Sno', 'prediction_normalized']],
            on='Sno'
        )
        
        self.y_true = self.merged_df['ground_truth_normalized']
        self.y_pred = self.merged_df['prediction_normalized']
        
        # Get unique labels
        self.labels = sorted(list(set(self.y_true) | set(self.y_pred)))
        
    def calculate_metrics(self) -> dict:
        """
        Calculate all evaluation metrics.
        
        Returns:
            Dictionary containing all metrics
        """
        metrics = {}
        
        # Overall metrics
        metrics['accuracy'] = accuracy_score(self.y_true, self.y_pred)
        
        # Macro-averaged metrics (average across all classes)
        metrics['precision_macro'] = precision_score(self.y_true, self.y_pred, 
                                                      average='macro', zero_division=0)
        metrics['recall_macro'] = recall_score(self.y_true, self.y_pred, 
                                               average='macro', zero_division=0)
        metrics['f1_macro'] = f1_score(self.y_true, self.y_pred, 
                                       average='macro', zero_division=0)
        
        # Weighted metrics (weighted by support)
        metrics['precision_weighted'] = precision_score(self.y_true, self.y_pred, 
                                                        average='weighted', zero_division=0)
        metrics['recall_weighted'] = recall_score(self.y_true, self.y_pred, 
                                                  average='weighted', zero_division=0)
        metrics['f1_weighted'] = f1_score(self.y_true, self.y_pred, 
                                          average='weighted', zero_division=0)
        
        # Per-class metrics
        precision_per_class = precision_score(self.y_true, self.y_pred, 
                                              average=None, labels=self.labels, zero_division=0)
        recall_per_class = recall_score(self.y_true, self.y_pred, 
                                        average=None, labels=self.labels, zero_division=0)
        f1_per_class = f1_score(self.y_true, self.y_pred, 
                               average=None, labels=self.labels, zero_division=0)
        
        metrics['per_class'] = {}
        for i, label in enumerate(self.labels):
            metrics['per_class'][label] = {
                'precision': precision_per_class[i],
                'recall': recall_per_class[i],
                'f1': f1_per_class[i]
            }
        
        return metrics
    
    def print_metrics(self, metrics: dict):
        """
        Print evaluation metrics in a formatted way.
        """
        print("\n" + "="*70)
        print(f"   EVALUATION RESULTS FOR {self.character_name.upper()}")
        print("="*70)
        
        print("\n### OVERALL METRICS ###")
        print(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        
        print("\n### MACRO-AVERAGED METRICS (unweighted average across all emotions) ###")
        print(f"Precision (Macro): {metrics['precision_macro']:.4f}")
        print(f"Recall (Macro):    {metrics['recall_macro']:.4f}")
        print(f"F1 Score (Macro):  {metrics['f1_macro']:.4f}")
        
        print("\n### WEIGHTED METRICS (weighted by support) ###")
        print(f"Precision (Weighted): {metrics['precision_weighted']:.4f}")
        print(f"Recall (Weighted):    {metrics['recall_weighted']:.4f}")
        print(f"F1 Score (Weighted):  {metrics['f1_weighted']:.4f}")
        
        print("\n### PER-CLASS METRICS ###")
        print(f"{'Emotion':<15} {'Precision':<12} {'Recall':<12} {'F1 Score':<12}")
        print("-" * 55)
        for emotion, scores in sorted(metrics['per_class'].items()):
            print(f"{emotion.capitalize():<15} {scores['precision']:<12.4f} "
                  f"{scores['recall']:<12.4f} {scores['f1']:<12.4f}")
    
    def generate_classification_report(self):
        """
        Generate and print sklearn classification report.
        """
        print("\n" + "="*70)
        print(f"   DETAILED CLASSIFICATION REPORT - {self.character_name.upper()}")
        print("="*70)
        print(classification_report(self.y_true, self.y_pred, 
                                   labels=self.labels, zero_division=0))
    
    def plot_confusion_matrix(self, save_path: str = None):
        """
        Plot confusion matrix.
        """
        cm = confusion_matrix(self.y_true, self.y_pred, labels=self.labels)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=[e.capitalize() for e in self.labels],
                   yticklabels=[e.capitalize() for e in self.labels])
        plt.title(f'Confusion Matrix - {self.character_name}', fontsize=14, fontweight='bold')
        plt.ylabel('Ground Truth', fontsize=12)
        plt.xlabel('Predicted', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nConfusion matrix saved to: {save_path}")
        else:
            plt.savefig(f'/Users/nishh/Desktop/Research/results/{self.character_name}_confusion_matrix.png', 
                       dpi=300, bbox_inches='tight')
            print(f"\nConfusion matrix saved to: /Users/nishh/Desktop/Research/results/{self.character_name}_confusion_matrix.png")
        
        plt.close()
    
    def analyze_misclassifications(self):
        """
        Analyze and display misclassified examples.
        """
        misclassified = self.merged_df[self.merged_df['ground_truth_normalized'] != 
                                       self.merged_df['prediction_normalized']]
        
        print("\n" + "="*70)
        print(f"   MISCLASSIFICATION ANALYSIS - {self.character_name.upper()}")
        print("="*70)
        print(f"\nTotal samples: {len(self.merged_df)}")
        print(f"Correct predictions: {len(self.merged_df) - len(misclassified)}")
        print(f"Misclassifications: {len(misclassified)}")
        print(f"Error rate: {len(misclassified)/len(self.merged_df)*100:.2f}%")
        
        if len(misclassified) > 0:
            print("\n### Most Common Misclassifications ###")
            misclass_pairs = misclassified.groupby(['ground_truth_normalized', 
                                                    'prediction_normalized']).size().reset_index(name='count')
            misclass_pairs = misclass_pairs.sort_values('count', ascending=False).head(10)
            
            print(f"{'True Label':<15} {'Predicted As':<15} {'Count':<10}")
            print("-" * 45)
            for _, row in misclass_pairs.iterrows():
                print(f"{row['ground_truth_normalized'].capitalize():<15} "
                      f"{row['prediction_normalized'].capitalize():<15} "
                      f"{int(row['count']):<10}")


def main():
    """
    Main execution function.
    """
    print("="*70)
    print("   EMOTION CLASSIFICATION EVALUATION")
    print("   Comparing Ground Truth vs Model Predictions")
    print("="*70)
    
    # Evaluate Lady Macbeth
    print("\n" + "#"*70)
    print("   EVALUATING LADY MACBETH")
    print("#"*70)
    
    lady_macbeth_eval = EmotionEvaluator(
        source_csv='/Users/nishh/Desktop/Research/data/raw/LadyMacbeth.csv',
        results_csv='/Users/nishh/Desktop/Research/data/processed/LadyMacbeth_analysis_results.csv',
        character_name='Lady Macbeth'
    )
    
    lm_metrics = lady_macbeth_eval.calculate_metrics()
    lady_macbeth_eval.print_metrics(lm_metrics)
    lady_macbeth_eval.generate_classification_report()
    lady_macbeth_eval.analyze_misclassifications()
    lady_macbeth_eval.plot_confusion_matrix()
    
    # Evaluate Ophelia
    print("\n\n" + "#"*70)
    print("   EVALUATING OPHELIA")
    print("#"*70)
    
    ophelia_eval = EmotionEvaluator(
        source_csv='/Users/nishh/Desktop/Research/data/raw/Ophelia.csv',
        results_csv='/Users/nishh/Desktop/Research/data/processed/Ophelia_analysis_results.csv',
        character_name='Ophelia'
    )
    
    ophelia_metrics = ophelia_eval.calculate_metrics()
    ophelia_eval.print_metrics(ophelia_metrics)
    ophelia_eval.generate_classification_report()
    ophelia_eval.analyze_misclassifications()
    ophelia_eval.plot_confusion_matrix()
    
    # Comparative Summary
    print("\n\n" + "="*70)
    print("   COMPARATIVE SUMMARY")
    print("="*70)
    
    comparison_data = {
        'Character': ['Lady Macbeth', 'Ophelia'],
        'Accuracy': [lm_metrics['accuracy'], ophelia_metrics['accuracy']],
        'Precision (Macro)': [lm_metrics['precision_macro'], ophelia_metrics['precision_macro']],
        'Recall (Macro)': [lm_metrics['recall_macro'], ophelia_metrics['recall_macro']],
        'F1 Score (Macro)': [lm_metrics['f1_macro'], ophelia_metrics['f1_macro']],
        'Precision (Weighted)': [lm_metrics['precision_weighted'], ophelia_metrics['precision_weighted']],
        'Recall (Weighted)': [lm_metrics['recall_weighted'], ophelia_metrics['recall_weighted']],
        'F1 Score (Weighted)': [lm_metrics['f1_weighted'], ophelia_metrics['f1_weighted']]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    print("\n" + comparison_df.to_string(index=False))
    
    # Save comparison to CSV
    comparison_df.to_csv('/Users/nishh/Desktop/Research/results/evaluation_comparison.csv', index=False)
    print("\n" + "="*70)
    print("Comparison results saved to: /Users/nishh/Desktop/Research/results/evaluation_comparison.csv")
    print("="*70)


if __name__ == "__main__":
    main()