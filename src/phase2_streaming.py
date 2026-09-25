"""
100% Streaming Candidate Generation (Phase 2c)
Uses constant O(1) memory by streaming test_source2 and test_source3 from disk.
Computes TF-IDF and Structural matches on-the-fly.
"""
import pandas as pd
import numpy as np
import os
import gc
from collections import defaultdict
from sklearn.feature_extraction.text import HashingVectorizer
from tqdm import tqdm
from normalization import normalize_dataframe

def update_top_k(candidates_dict, s1_id, c_id, score, k=100):
    """Maintain top-k candidates for an S1 entity."""
    cands = candidates_dict[s1_id]
    cands[c_id] = max(cands.get(c_id, 0.0), float(score))
    
def get_top_k(candidates_dict, s1_id, k=100):
    cands = candidates_dict.get(s1_id, {})
    return sorted(cands.items(), key=lambda x: x[1], reverse=True)[:k]

def compute_recall_metrics(y_true_dict, candidates_dict, k_list=[10, 25, 50, 100]):
    metrics = {f"Recall@{k}": [] for k in k_list}
    candidate_counts = []
    one_to_one_recalls = []
    one_to_many_recalls = []
    
    for s1_id, true_matches in y_true_dict.items():
        cands = get_top_k(candidates_dict, s1_id, k=max(k_list))
        candidate_counts.append(len(cands))
        
        if len(true_matches) > 0:
            for k in k_list:
                cands_k = set([c for c, score in cands[:k]])
                found = len(true_matches.intersection(cands_k))
                recall_k = found / len(true_matches)
                metrics[f"Recall@{k}"].append(recall_k)
                
            recall_50 = metrics["Recall@50"][-1]
            if len(true_matches) == 1:
                one_to_one_recalls.append(recall_50)
            else:
                one_to_many_recalls.append(recall_50)
                
    results = {k: np.mean(v) if v else 0.0 for k, v in metrics.items()}
    results["Average_Candidates"] = np.mean(candidate_counts) if candidate_counts else 0
    results["95th_Percentile_Candidates"] = np.percentile(candidate_counts, 95) if candidate_counts else 0
    results["OneToOne_Recall@50"] = np.mean(one_to_one_recalls) if one_to_one_recalls else 0.0
    results["OneToMany_Recall@50"] = np.mean(one_to_many_recalls) if one_to_many_recalls else 0.0
    return results

def main():
    base_path = 'DATA/student_resource/dataset'
    
    print("Loading validation splits...")
    val_ids = np.load('experiments/splits/val_ids.npy', allow_pickle=True)
    np.random.seed(42)
    val_ids_sample = np.random.choice(val_ids, size=5000, replace=False)
    
    print("Loading S1 validation sample...")
    train_s1 = pd.read_csv(
        f'{base_path}/train/train_source1.tsv', sep='\t', dtype=str, 
        usecols=['entity_id', 'business_name', 'business_address', 'country']
    ).fillna('')
    val_s1 = train_s1[train_s1['entity_id'].isin(val_ids_sample)].copy()
    del train_s1; gc.collect()
    
    print("Normalizing S1 (val sample)...")
    val_s1_norm = normalize_dataframe(val_s1)
    val_s1_norm['search_text'] = val_s1_norm['name_basic'] + " " + val_s1_norm['address_basic']
    
    # Precompute structural lookups from S1
    print("Building structural query sets...")
    s1_rare_toks = defaultdict(list)
    s1_nums = defaultdict(list)
    s1_acronyms = defaultdict(list)
    s1_prefixes = defaultdict(list)
    
    for row in val_s1_norm.itertuples(index=False):
        eid = row.entity_id
        for tok in row.name_tokens: s1_rare_toks[tok].append(eid)
        for num in row.numeric_tokens: 
            if num and len(num) >= 3: s1_nums[num].append(eid)
        if len(row.name_acronym) >= 2: s1_acronyms[row.name_acronym].append(eid)
        if len(row.name_core) >= 5: s1_prefixes[row.name_core[:5]].append(eid)
            
    val_countries = [c for c in val_s1_norm['country_normalized'].unique() if c != '']
    final_candidates = defaultdict(dict)  # {s1_id: {c_id: score}}
    
    # Vectorizers
    char_vec = HashingVectorizer(analyzer='char_wb', ngram_range=(3,4), n_features=2**21, norm='l2', dtype=np.float32)
    tok_vec = HashingVectorizer(analyzer='word', ngram_range=(1,2), n_features=2**21, norm='l2', dtype=np.float32)
    
    for country in val_countries:
        print(f"\n=== Processing Country: {country} ===")
        df_query = val_s1_norm[val_s1_norm['country_normalized'] == country].copy()
        if len(df_query) == 0: continue
            
        q_ids = df_query['entity_id'].values
        X_query_char = char_vec.transform(df_query['search_text'])
        X_query_tok = tok_vec.transform(df_query['search_text'])
        
        # Track frequency of S1 tokens to enforce rarity
        tok_freq = defaultdict(int)
        
        for source_file in ['train_source2.tsv', 'train_source3.tsv']:
            file_path = f'{base_path}/train/{source_file}'
            print(f"  Streaming {source_file}...")
            
            chunk_iter = pd.read_csv(file_path, sep='\t', dtype=str, chunksize=100000, 
                                     usecols=['entity_id', 'business_name', 'business_address', 'country'])
            
            for chunk_idx, chunk in enumerate(chunk_iter):
                chunk = chunk.fillna('')
                chunk_country = chunk[chunk['country'].str.lower().str.strip() == country]
                if len(chunk_country) == 0: continue
                    
                chunk_norm = normalize_dataframe(chunk_country)
                chunk_norm['search_text'] = chunk_norm['name_basic'] + " " + chunk_norm['address_basic']
                idx_ids = chunk_norm['entity_id'].values
                
                # --- TF-IDF Char ---
                X_idx_char = char_vec.transform(chunk_norm['search_text']).T.tocsr()
                sim_char = X_query_char.dot(X_idx_char)
                for q_i in range(sim_char.shape[0]):
                    row = sim_char.getrow(q_i)
                    if row.nnz == 0: continue
                    s1_id = q_ids[q_i]
                    for j, score in zip(row.indices, row.data):
                        if score > 0.3:  # Threshold to avoid massive candidate sets
                            update_top_k(final_candidates, s1_id, idx_ids[j], score)
                del X_idx_char, sim_char
                
                # --- TF-IDF Token ---
                X_idx_tok = tok_vec.transform(chunk_norm['search_text']).T.tocsr()
                sim_tok = X_query_tok.dot(X_idx_tok)
                for q_i in range(sim_tok.shape[0]):
                    row = sim_tok.getrow(q_i)
                    if row.nnz == 0: continue
                    s1_id = q_ids[q_i]
                    for j, score in zip(row.indices, row.data):
                        if score > 0.2:  # Increased threshold to prevent dict explosion
                            update_top_k(final_candidates, s1_id, idx_ids[j], score)
                del X_idx_tok, sim_tok
                
                # --- Structural Matching ---
                for row in chunk_norm.itertuples(index=False):
                    c_id = row.entity_id
                    
                    # Rarity tracking
                    for tok in row.name_tokens: tok_freq[tok] += 1
                    
                    cands_to_update = set()
                    
                    for tok in row.name_tokens:
                        if tok_freq[tok] <= 500:  # Rare
                            cands_to_update.update(s1_rare_toks.get(tok, []))
                            
                    for num in row.numeric_tokens:
                        if num and len(num) >= 3:
                            cands_to_update.update(s1_nums.get(num, []))
                            
                    if len(row.name_acronym) >= 2:
                        cands_to_update.update(s1_acronyms.get(row.name_acronym, []))
                        
                    if len(row.name_core) >= 5:
                        cands_to_update.update(s1_prefixes.get(row.name_core[:5], []))
                        
                    for s1_id in cands_to_update:
                        update_top_k(final_candidates, s1_id, c_id, 0.15)
                        
                # Memory safeguard: prune dictionary to top 200 per query after each chunk
                for s1_id, cands in final_candidates.items():
                    if len(cands) > 200:
                        top_cands = sorted(cands.items(), key=lambda x: x[1], reverse=True)[:200]
                        final_candidates[s1_id] = dict(top_cands)

                del chunk, chunk_country, chunk_norm
                gc.collect()

    print("\nEvaluating Candidate Recall...")
    train_gt = pd.read_csv(f'{base_path}/train/train_ground_truth.tsv', sep='\t', dtype=str).fillna('')
    y_true_dict = {}
    for _, row in train_gt.iterrows():
        s1_id = row['source1_entity_id']
        if s1_id in val_ids_sample:
            matches_str = row['matched_entity_ids']
            y_true_dict[s1_id] = set(matches_str.split(',')) if matches_str else set()
            
    # Final candidates list - MUST BE A DICT for get_top_k to work later
    candidates_dict = {s1_id: dict(get_top_k(final_candidates, s1_id, 100)) for s1_id in val_ids_sample}
            
    print("\nSaving candidates to experiments/...")
    os.makedirs('experiments', exist_ok=True)
    np.save('experiments/val_candidates.npy', dict(candidates_dict))
    print("Done! Candidates saved to experiments/val_candidates.npy")
            
    metrics = compute_recall_metrics(y_true_dict, candidates_dict)
    
    print("\n--- PHASE 2c (Streaming) CANDIDATE RECALL METRICS ---")
    report_lines = ["=== PHASE 2c CANDIDATE GENERATION METRICS ==="]
    for k, v in metrics.items():
        line = f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v:.1f}"
        print(line)
        report_lines.append(line)
        
    os.makedirs('artifacts', exist_ok=True)
    with open('artifacts/phase2c_results.txt', 'w') as f:
        f.write('\n'.join(report_lines))

if __name__ == '__main__':
    main()
