from sklearn.metrics import roc_auc_score, precision_recall_curve
import numpy as np

class LinkPredictionEvaluator:
    """Evaluate link prediction performance."""
    
    @staticmethod
    def evaluate(y_true: List[int], y_scores: List[float]):
        """
        Evaluate link prediction performance.
        
        Args:
            y_true: True labels (0/1)
            y_scores: Predicted scores
            
        Returns:
            dict: Dictionary with evaluation metrics
        """
        auc_score = roc_auc_score(y_true, y_scores)
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        
        return {
            'auc_roc': auc_score,
            'precision': precision,
            'recall': recall
        }