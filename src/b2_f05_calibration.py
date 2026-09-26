import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score

def compute_macro_f05(y_true_list, y_pred_list):
    """
    Computes the Macro F0.5 score for a list of boolean lists.
    We convert the set of matches into binary vectors for scikit-learn.
    """
    if len(y_true_list) == 0:
        return 0.0
        
    f05_scores = []
    
    for y_true, y_pred in zip(y_true_list, y_pred_list):
        # We need a unified universe of candidates for this specific query
        universe = list(set(y_true).union(set(y_pred)))
        if not universe:
            # Both predicted empty and truth is empty (Singleton matched correctly)
            f05_scores.append(1.0)
            continue
            
        y_true_bin = [1 if c in y_true else 0 for c in universe]
        y_pred_bin = [1 if c in y_pred else 0 for c in universe]
        
        # calculate F0.5 for this query
        score = fbeta_score(y_true_bin, y_pred_bin, beta=0.5, zero_division=0)
        f05_scores.append(score)
        
    return np.mean(f05_scores)

def calibrate_threshold(df_val_preds, y_true_dict, s1_col='s1_id', c_col='c_id', prob_col='fused_score'):
    """
    df_val_preds: DataFrame containing predictions for candidate pairs.
    y_true_dict: dict mapping s1_id -> set(true_c_ids)
    
    Returns optimal threshold and max Macro F0.5.
    """
    thresholds = np.arange(0.01, 1.00, 0.01)
    best_thresh = 0.5
    max_f05 = -1.0
    
    # We only care about s1_ids that are in our validation set
    val_s1_ids = list(y_true_dict.keys())
    
    print(f"Calibrating threshold over {len(val_s1_ids)} validation queries...")
    
    # Group predictions by s1_id once for speed
    grouped_preds = df_val_preds.groupby(s1_col)
    
    # Pre-extract data
    s1_to_probs = {}
    for s1_id, group in grouped_preds:
        s1_to_probs[s1_id] = list(zip(group[c_col].values, group[prob_col].values))
        
    for thresh in thresholds:
        y_true_list = []
        y_pred_list = []
        
        for s1_id in val_s1_ids:
            true_cands = y_true_dict.get(s1_id, set())
            
            # Predict matches based on this threshold
            cands_probs = s1_to_probs.get(s1_id, [])
            pred_cands = {c for c, p in cands_probs if p > thresh}
            
            y_true_list.append(true_cands)
            y_pred_list.append(pred_cands)
            
        f05 = compute_macro_f05(y_true_list, y_pred_list)
        
        if f05 > max_f05:
            max_f05 = f05
            best_thresh = thresh
            
    print(f"Calibration Complete!")
    print(f"Optimal Threshold: {best_thresh:.2f}")
    print(f"Max Validation Macro F0.5: {max_f05:.4f}")
    
    return best_thresh, max_f05

if __name__ == '__main__':
    print("This script is imported by the Phase 5 decision engine to dynamically find the optimal F0.5 threshold.")
