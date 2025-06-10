"""Evaluation metrics and printing functions for binary classification models."""

from sklearn.metrics import (
    roc_auc_score, accuracy_score, balanced_accuracy_score, 
    f1_score, precision_score, recall_score
)


def compute_evaluation_metrics(y_true, y_pred_proba):
    """Calculate our evaluation metrics"""
    y_pred_binary = (y_pred_proba >= 0.5).astype(int)

    accuracy = accuracy_score(y_true, y_pred_binary)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred_binary)
    f1 = f1_score(y_true, y_pred_binary)
    precision = precision_score(y_true, y_pred_binary, zero_division=0)
    recall = recall_score(y_true, y_pred_binary, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_pred_proba)

    return {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_accuracy,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'roc_auc': roc_auc
    }


def print_evaluation_results(results):
    """pretty print our metrics"""
    print("METRICS:")
    print(f"  Accuracy:           {results['accuracy']:.4f}")
    print(f"  Balanced Accuracy:  {results['balanced_accuracy']:.4f}")
    print(f"  F1 Score:           {results['f1_score']:.4f}")
    print(f"  Precision:          {results['precision']:.4f}")
    print(f"  Recall:             {results['recall']:.4f}")
    print(f"  ROC AUC:            {results['roc_auc']:.4f}")