import pandas as pd
import numpy as np
import os
import gc
from phase3_features import compute_pairwise_features
from phase3_model import train_pair_matcher, load_and_predict
from normalization import normalize_dataframe

def create_master_pipeline():
    print("=== STARTING PHASE 3 END-TO-END PIPELINE ===")
    
    # 1. Load the generated candidates from Phase 2
    print("\n1. Loading Candidates...")
    candidates_dict = np.load('experiments/val_candidates.npy', allow_pickle=True).item()
    
    # Convert dict to flat list of pairs
    pairs = []
    for s1_id, cands in candidates_dict.items():
        for c_id, score in cands.items():
            pairs.append({'s1_id': s1_id, 'c_id': c_id, 'retrieval_score': score})
            
    df_pairs = pd.DataFrame(pairs)
    print(f"Total candidate pairs to process: {len(df_pairs):,}")
    
    # 2. Get Ground Truth for labels
    print("\n2. Mapping labels from Ground Truth...")
    train_gt = pd.read_csv('DATA/student_resource/dataset/train/train_ground_truth.tsv', sep='\t', dtype=str).fillna('')
    y_true_dict = {}
    for _, row in train_gt.iterrows():
        matches = row['matched_entity_ids']
        y_true_dict[row['source1_entity_id']] = set(matches.split(',')) if matches else set()
        
    df_pairs['label'] = df_pairs.apply(
        lambda x: 1 if x['c_id'] in y_true_dict.get(x['s1_id'], set()) else 0, 
        axis=1
    )
    print(f"True Matches found in candidates: {df_pairs['label'].sum():,}")
    
    # 3. Load Features for S1 and S2/3
    print("\n3. Loading raw text for Feature Engineering...")
    # S1 Features
    train_s1 = pd.read_csv('DATA/student_resource/dataset/train/train_source1.tsv', sep='\t', dtype=str).fillna('')
    s1_needed = train_s1[train_s1['entity_id'].isin(df_pairs['s1_id'].unique())]
    s1_norm = normalize_dataframe(s1_needed).set_index('entity_id')
    del train_s1; gc.collect()
    
    # S23 Features
    s23_needed_ids = set(df_pairs['c_id'].unique())
    s23_list = []
    for source in ['train_source2.tsv', 'train_source3.tsv']:
        chunk_iter = pd.read_csv(f'DATA/student_resource/dataset/train/{source}', sep='\t', dtype=str, chunksize=100000)
        for chunk in chunk_iter:
            chunk = chunk.fillna('')
            needed_chunk = chunk[chunk['entity_id'].isin(s23_needed_ids)]
            if len(needed_chunk) > 0:
                s23_list.append(normalize_dataframe(needed_chunk))
                
    s23_norm = pd.concat(s23_list, ignore_index=True).set_index('entity_id')
    del s23_list; gc.collect()
    
    # 4. Compute Linguistic Features
    print("\n4. Computing NLP & Semantic Features (RapidFuzz)...")
    df_features = compute_pairwise_features(df_pairs, s1_norm, s23_norm)
    df_features['s1_id'] = df_pairs['s1_id'].values
    
    feature_cols = ['retrieval_score', 'name_jaro', 'name_ratio', 'name_jaccard', 
                   'addr_jaro', 'addr_ratio', 'addr_jaccard', 'num_overlap', 'num_conflict']
    
    # 5. Train LightGBM
    print("\n5. Splitting and Training LightGBM Model...")
    # Group split by s1_id to prevent data leakage
    unique_s1 = df_features['s1_id'].unique()
    np.random.seed(42)
    np.random.shuffle(unique_s1)
    split_idx = int(len(unique_s1) * 0.8)
    
    train_s1_ids = unique_s1[:split_idx]
    val_s1_ids = unique_s1[split_idx:]
    
    df_train = df_features[df_features['s1_id'].isin(train_s1_ids)]
    df_val = df_features[df_features['s1_id'].isin(val_s1_ids)]
    
    model = train_pair_matcher(df_train, df_val, feature_cols)
    
    print("\n=== PHASE 3 COMPLETE ===")
    print("Model is saved and ready for B1 Submission Generation!")

if __name__ == '__main__':
    create_master_pipeline()
