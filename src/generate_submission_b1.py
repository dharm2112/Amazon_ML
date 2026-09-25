import pandas as pd
import numpy as np
import os
import gc
from collections import defaultdict
from tqdm import tqdm
import joblib

from normalization import normalize_dataframe
from phase3_features import compute_pairwise_features

def main():
    print("=== STARTING ML SUBMISSION (B1) PIPELINE ===")
    
    # 1. Load S1 Test Data
    print("\n1. Loading and normalizing test_source1...")
    test_s1 = pd.read_csv('DATA/student_resource/dataset/test/test_source1.tsv', sep='\t', dtype=str).fillna('')
    s1_norm = normalize_dataframe(test_s1)
    
    # 2. Build Fast Lookup Dictionaries (O(1) matching instead of heavy TF-IDF matrix math)
    print("\n2. Building Fast Structural Lookup Maps...")
    # These dictionaries will map: (country, feature_value) -> set of s1_entity_ids
    exact_map = defaultdict(set)
    acronym_map = defaultdict(set)
    rare_tok_map = defaultdict(set)
    
    # We will compute frequency of tokens in S1 to define "rare"
    tok_freq = defaultdict(int)
    for tokens in s1_norm['name_tokens']:
        for tok in tokens:
            tok_freq[tok] += 1
            
    for row in tqdm(s1_norm.itertuples(index=False), total=len(s1_norm)):
        s1_id = row.entity_id
        country = row.country_normalized
        if not country: continue
            
        exact_map[(country, row.name_basic)].add(s1_id)
        
        if len(row.name_acronym) >= 2:
            acronym_map[(country, row.name_acronym)].add(s1_id)
            
        for tok in row.name_tokens:
            if tok_freq[tok] <= 50: # Rare token in test set
                rare_tok_map[(country, tok)].add(s1_id)
                
    print(f"  Purging overly generic terms to prevent memory crash...")
    for d in [exact_map, acronym_map, rare_tok_map]:
        keys_to_delete = [k for k, v in d.items() if len(v) > 200]
        for k in keys_to_delete:
            del d[k]
                
    # Load model before streaming
    model_path = 'experiments/models/lgbm_matcher.pkl'
    if not os.path.exists(model_path):
        raise FileNotFoundError("LightGBM model not found! Run src/run_phase3.py first.")
    model = joblib.load(model_path)
    feature_cols = ['retrieval_score', 'name_jaro', 'name_ratio', 'name_jaccard', 
                   'addr_jaro', 'addr_ratio', 'addr_jaccard', 'num_overlap', 'num_conflict']

    # 3. Stream S2/S3, generate candidates, extract features, predict, and save
    print("\n3. Streaming S2 & S3: Matching, Feature Engineering, and Predicting on-the-fly...")
    final_matches_dict = defaultdict(list)
    final_candidates_dict = defaultdict(set)
    s1_norm_idx = s1_norm.set_index('entity_id')
    match_count = 0
    
    for source in ['test_source2.tsv', 'test_source3.tsv']:
        file_path = f'DATA/student_resource/dataset/test/{source}'
        print(f"  Streaming {file_path}...")
        
        # Lowered chunksize to 50k to completely prevent C parser out-of-memory errors
        chunk_iter = pd.read_csv(file_path, sep='\t', dtype=str, chunksize=50000)
        for chunk in chunk_iter:
            chunk = chunk.fillna('')
            chunk_norm = normalize_dataframe(chunk)
            
            chunk_pairs = []
            
            for i, row in enumerate(chunk_norm.itertuples(index=False)):
                c_id = row.entity_id
                country = row.country_normalized
                if not country: continue
                    
                matches = set()
                matches.update(exact_map.get((country, row.name_basic), set()))
                if len(row.name_acronym) >= 2:
                    matches.update(acronym_map.get((country, row.name_acronym), set()))
                for tok in row.name_tokens:
                    matches.update(rare_tok_map.get((country, tok), set()))
                    
                if matches:
                    for s1_id in matches:
                        # Cap at 50 candidates per S1 to prevent explosion
                        if len(final_candidates_dict[s1_id]) < 50:
                            final_candidates_dict[s1_id].add(c_id)
                            chunk_pairs.append({'s1_id': s1_id, 'c_id': c_id, 'retrieval_score': 1.0})
                            
            if chunk_pairs:
                df_chunk_pairs = pd.DataFrame(chunk_pairs)
                chunk_norm_idx = chunk_norm.set_index('entity_id')
                
                # Compute ML features for this chunk
                df_features = compute_pairwise_features(df_chunk_pairs, s1_norm_idx, chunk_norm_idx)
                
                # Predict
                preds = model.predict(df_features[feature_cols])
                
                # Keep matches > 0.50
                for pair_idx, prob in enumerate(preds):
                    if prob > 0.50:
                        s1_id = df_chunk_pairs.iloc[pair_idx]['s1_id']
                        c_id = df_chunk_pairs.iloc[pair_idx]['c_id']
                        final_matches_dict[s1_id].append(c_id)
                        match_count += 1
                        
            del chunk, chunk_norm, chunk_pairs
            gc.collect()
            
    print(f"\nModel identified {match_count:,} high-confidence matches.")
    
    # 7. Generate Submission Files
    print("\n7. Writing submission files...")
    os.makedirs('output', exist_ok=True)
    
    with open('output/matching_results_b1.tsv', 'w', encoding='utf-8') as f_match, \
         open('output/candidate_pairs_b1.tsv', 'w', encoding='utf-8') as f_cand:
             
        f_match.write('source1_entity_id\tmatched_entity_ids\n')
        f_cand.write('source1_entity_id\tcandidate_entity_ids\n')
        
        for s1_id in s1_norm_idx.index:
            matches = final_matches_dict.get(s1_id, [])
            cands = list(final_candidates_dict.get(s1_id, set()))
            
            f_match.write(f"{s1_id}\t{','.join(matches)}\n")
            f_cand.write(f"{s1_id}\t{','.join(cands)}\n")
                
    print("Done! Saved to output/matching_results_b1.tsv")

if __name__ == '__main__':
    main()
