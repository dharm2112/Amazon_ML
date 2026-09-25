import pandas as pd
import numpy as np
import os
import json
from sklearn.model_selection import train_test_split

def compute_macro_f05(y_true_dict, y_pred_dict):
    """
    Computes macro F0.5 score as used in the evaluation.
    y_true_dict: dict mapping source1_entity_id to set of true matched_entity_ids
    y_pred_dict: dict mapping source1_entity_id to set of predicted matched_entity_ids
    """
    f05_scores = []
    
    for s1_id in y_true_dict:
        true_matches = set(y_true_dict[s1_id])
        pred_matches = set(y_pred_dict.get(s1_id, set()))
        
        if len(true_matches) == 0 and len(pred_matches) == 0:
            f05_scores.append(1.0)
            continue
        elif len(true_matches) == 0 or len(pred_matches) == 0:
            f05_scores.append(0.0)
            continue
            
        tp = len(true_matches.intersection(pred_matches))
        fp = len(pred_matches - true_matches)
        fn = len(true_matches - pred_matches)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        if precision + recall == 0:
            f05_scores.append(0.0)
        else:
            f05 = (1.25 * precision * recall) / ((0.25 * precision) + recall)
            f05_scores.append(f05)
            
    return np.mean(f05_scores) if f05_scores else 0.0

def load_and_profile():
    base_path = 'DATA/student_resource/dataset'
    
    # Load all files
    print("Loading data...")
    train_s1 = pd.read_csv(f'{base_path}/train/train_source1.tsv', sep='\t', dtype=str).fillna('')
    train_s2 = pd.read_csv(f'{base_path}/train/train_source2.tsv', sep='\t', dtype=str).fillna('')
    train_s3 = pd.read_csv(f'{base_path}/train/train_source3.tsv', sep='\t', dtype=str).fillna('')
    train_gt = pd.read_csv(f'{base_path}/train/train_ground_truth.tsv', sep='\t', dtype=str).fillna('')
    
    test_s1 = pd.read_csv(f'{base_path}/test/test_source1.tsv', sep='\t', dtype=str).fillna('')
    test_s2 = pd.read_csv(f'{base_path}/test/test_source2.tsv', sep='\t', dtype=str).fillna('')
    test_s3 = pd.read_csv(f'{base_path}/test/test_source3.tsv', sep='\t', dtype=str).fillna('')
    
    report = []
    def log(msg):
        print(msg)
        report.append(msg)
    
    log("=== DATASET PROFILING ===")
    
    # 1. Row counts
    log(f"Row counts:")
    log(f"  train_source1: {len(train_s1):,}")
    log(f"  train_source2: {len(train_s2):,}")
    log(f"  train_source3: {len(train_s3):,}")
    log(f"  train_ground_truth: {len(train_gt):,}")
    log(f"  test_source1: {len(test_s1):,}")
    log(f"  test_source2: {len(test_s2):,}")
    log(f"  test_source3: {len(test_s3):,}")
    
    # 2. Missing values (since we fillna with '', we check empty strings)
    log("\nMissing values (empty string count):")
    for name, df in zip(
        ['train_source1', 'train_source2', 'train_source3', 'test_source1'], 
        [train_s1, train_s2, train_s3, test_s1]
    ):
        missing = (df == '').sum()
        log(f"  {name}:")
        for col in missing.index:
            log(f"    {col}: {missing[col]:,}")

    # 3. Country distribution
    log("\nCountry distribution (train_source1):")
    country_counts = train_s1['country'].value_counts()
    for c, count in country_counts.items():
        log(f"  {c}: {count:,}")
        
    log("\nCountry distribution (test_source1):")
    test_country_counts = test_s1['country'].value_counts()
    for c, count in test_country_counts.items():
        log(f"  {c}: {count:,}")

    # 4. Match-count distribution & Singleton statistics
    log("\nMatch-count distribution & Singleton stats:")
    gt_dict = {}
    match_counts = []
    
    for _, row in train_gt.iterrows():
        s1_id = row['source1_entity_id']
        matches_str = row['matched_entity_ids']
        if matches_str == '':
            matches = []
        else:
            matches = matches_str.split(',')
        gt_dict[s1_id] = set(matches)
        match_counts.append(len(matches))
        
    match_counts = np.array(match_counts)
    singletons = (match_counts == 0).sum()
    log(f"  Total Source 1 entities in GT: {len(gt_dict):,}")
    log(f"  Singletons (no matches): {singletons:,} ({(singletons/len(gt_dict))*100:.2f}%)")
    log(f"  One-to-one matches (1 match): {(match_counts == 1).sum():,} ({(np.sum(match_counts == 1)/len(gt_dict))*100:.2f}%)")
    log(f"  One-to-many matches (>1 match): {(match_counts > 1).sum():,} ({(np.sum(match_counts > 1)/len(gt_dict))*100:.2f}%)")
    log(f"  Max matches for a single entity: {match_counts.max()}")
    
    # 5. Data quality analysis
    log("\nData quality analysis:")
    # Are there any Source 1 IDs in GT not in train_source1?
    s1_ids_gt = set(train_gt['source1_entity_id'])
    s1_ids_train = set(train_s1['entity_id'])
    diff1 = s1_ids_gt - s1_ids_train
    log(f"  Source1 IDs in GT but not in train_source1: {len(diff1)}")
    
    # 6. Reproducible validation split (split at Source 1 entity level)
    log("\nCreating reproducible validation split...")
    s1_ids = train_s1['entity_id'].values
    train_ids, val_ids = train_test_split(s1_ids, test_size=0.2, random_state=42)
    
    # Save the splits
    split_dir = 'experiments/splits'
    os.makedirs(split_dir, exist_ok=True)
    np.save(f'{split_dir}/train_ids.npy', train_ids)
    np.save(f'{split_dir}/val_ids.npy', val_ids)
    
    log(f"  Train split size: {len(train_ids):,}")
    log(f"  Val split size: {len(val_ids):,}")
    
    # 7. Simple Baseline (B0)
    log("\nRunning simple baseline (B0: exact match on business_name and country) on validation split...")
    
    val_s1 = train_s1[train_s1['entity_id'].isin(val_ids)]
    
    log("  Collecting valid (name, country) keys from val_s1...")
    val_keys = set()
    for row in val_s1[['business_name', 'country']].itertuples(index=False):
        key = (str(row.business_name).lower().strip(), str(row.country).lower().strip())
        if key[0] != '':
            val_keys.add(key)
    
    log("  Indexing S2 and S3 for baseline...")
    name_country_idx = {}
    for df_source in [train_s2, train_s3]:
        for row in df_source[['entity_id', 'business_name', 'country']].itertuples(index=False):
            key = (str(row.business_name).lower().strip(), str(row.country).lower().strip())
            if key in val_keys:
                if key not in name_country_idx:
                    name_country_idx[key] = []
                name_country_idx[key].append(row.entity_id)
        
    log("  Predicting on validation set...")
    y_pred_val = {}
    for row in val_s1[['entity_id', 'business_name', 'country']].itertuples(index=False):
        s1_id = row.entity_id
        key = (str(row.business_name).lower().strip(), str(row.country).lower().strip())
        matches = name_country_idx.get(key, [])
        y_pred_val[s1_id] = set(matches)
        
    y_true_val = {s1_id: gt_dict[s1_id] for s1_id in val_ids}
    
    f05_val = compute_macro_f05(y_true_val, y_pred_val)
    log(f"  B0 Validation Macro F0.5: {f05_val:.5f}")
    
    with open('artifacts/profiling_report.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

if __name__ == '__main__':
    load_and_profile()
