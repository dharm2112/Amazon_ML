import pandas as pd
import numpy as np
import os
import gc
from collections import defaultdict
from sklearn.feature_extraction.text import HashingVectorizer
from tqdm import tqdm
from normalization import normalize_dataframe

def compute_recall_metrics(y_true_dict, candidates_dict, k_list=[10, 25, 50, 100]):
    metrics = {f"Recall@{k}": [] for k in k_list}
    candidate_counts = []
    
    one_to_one_recalls = []
    one_to_many_recalls = []
    
    for s1_id, true_matches in y_true_dict.items():
        cands = candidates_dict.get(s1_id, [])
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

def get_candidates(df_query, df_index, text_col, k=50, index_chunk_size=100000, query_chunk_size=50, analyzer='char_wb', ngram_range=(3,4)):
    """Retrieve top-k candidates using HashingVectorizer and chunked index processing to save memory."""
    print(f"  Vectorizing queries and processing {len(df_index)} index docs in chunks...")
    vectorizer = HashingVectorizer(analyzer=analyzer, ngram_range=ngram_range, n_features=2**21, norm='l2', alternate_sign=False, dtype=np.float32)
    
    X_query = vectorizer.transform(df_query[text_col])
    query_ids = df_query['entity_id'].values
    index_ids_all = df_index['entity_id'].values
    
    candidates = defaultdict(list)
    
    for i in tqdm(range(0, len(df_index), index_chunk_size)):
        batch_df = df_index.iloc[i:i+index_chunk_size]
        X_index_chunk = vectorizer.transform(batch_df[text_col]).T.tocsr()
        index_ids_chunk = index_ids_all[i:i+index_chunk_size]
        
        for j in range(0, X_query.shape[0], query_chunk_size):
            X_query_chunk = X_query[j:j+query_chunk_size]
            similarities = X_query_chunk.dot(X_index_chunk)
            
            for row_idx in range(similarities.shape[0]):
                global_q_idx = j + row_idx
                s1_id = query_ids[global_q_idx]
                row = similarities.getrow(row_idx)
                
                if row.nnz == 0: continue
                    
                indices = row.indices
                data = row.data
                
                if len(indices) > k:
                    top_k_idx = np.argpartition(-data, k)[:k]
                    top_idx_chunk = indices[top_k_idx]
                    top_data_chunk = data[top_k_idx]
                else:
                    top_idx_chunk = indices
                    top_data_chunk = data
                    
                current_cands = candidates[s1_id]
                for idx, score in zip(top_idx_chunk, top_data_chunk):
                    current_cands.append((index_ids_chunk[idx], float(score)))
                    
                current_cands.sort(key=lambda x: x[1], reverse=True)
                candidates[s1_id] = current_cands[:k]
            
            del similarities
            gc.collect()
            
    return candidates

def load_source_for_country(file_path, country):
    """Streams a TSV file and extracts only rows matching the country to save memory."""
    print(f"  Streaming {file_path} for country: {country}")
    chunks = []
    for chunk in pd.read_csv(file_path, sep='\t', dtype=str, usecols=['entity_id', 'business_name', 'business_address', 'country'], chunksize=250000):
        chunk = chunk.fillna('')
        chunk_country = chunk[chunk['country'].str.lower().str.strip() == country]
        if len(chunk_country) > 0:
            chunks.append(chunk_country)
    
    if not chunks:
        return pd.DataFrame(columns=['entity_id', 'business_name', 'business_address', 'country'])
    
    return pd.concat(chunks, ignore_index=True)

def main():
    base_path = 'DATA/student_resource/dataset'
    
    print("Loading validation splits...")
    val_ids = np.load('experiments/splits/val_ids.npy', allow_pickle=True)
    
    np.random.seed(42)
    val_ids_sample = np.random.choice(val_ids, size=5000, replace=False)
    
    print("Loading S1 validation sample...")
    train_s1 = pd.read_csv(f'{base_path}/train/train_source1.tsv', sep='\t', dtype=str, usecols=['entity_id', 'business_name', 'business_address', 'country']).fillna('')
    val_s1 = train_s1[train_s1['entity_id'].isin(val_ids_sample)].copy()
    
    del train_s1
    gc.collect()
    
    print("Normalizing S1 (val sample)...")
    val_s1_norm = normalize_dataframe(val_s1)
    val_s1_norm['search_text'] = val_s1_norm['name_basic'] + " " + val_s1_norm['address_basic']
    
    val_countries = [c for c in val_s1_norm['country_normalized'].unique() if c != '']
    candidates_dict = defaultdict(list)
    
    for country in val_countries:
        print(f"\nProcessing Country: {country}")
        df_query = val_s1_norm[val_s1_norm['country_normalized'] == country]
        if len(df_query) == 0: continue
            
        # Stream load S2 and S3 for this country ONLY
        s2_country = load_source_for_country(f'{base_path}/train/train_source2.tsv', country)
        s3_country = load_source_for_country(f'{base_path}/train/train_source3.tsv', country)
        
        s2_norm = normalize_dataframe(s2_country)
        s3_norm = normalize_dataframe(s3_country)
        
        df_index = pd.concat([s2_norm, s3_norm], ignore_index=True)
        df_index['search_text'] = df_index['name_basic'] + " " + df_index['address_basic']
        
        # Free memory immediately
        del s2_country, s3_country, s2_norm, s3_norm
        gc.collect()
        
        print("  Running Char TF-IDF Retrieval...")
        char_cands = get_candidates(df_query, df_index, 'search_text', k=50, analyzer='char_wb', ngram_range=(3,4))
        
        print("  Running Token TF-IDF Retrieval...")
        token_cands = get_candidates(df_query, df_index, 'search_text', k=50, analyzer='word', ngram_range=(1,2))
        
        print("  Merging candidates...")
        for s1_id in df_query['entity_id']:
            merged = {}
            for c_id, score in char_cands.get(s1_id, []):
                merged[c_id] = score
            for c_id, score in token_cands.get(s1_id, []):
                merged[c_id] = max(merged.get(c_id, 0.0), score)
                
            sorted_merged = sorted(merged.items(), key=lambda x: x[1], reverse=True)
            candidates_dict[s1_id] = sorted_merged[:100]
            
        del df_index, char_cands, token_cands
        gc.collect()
        
    print("\nEvaluating Candidate Recall...")
    train_gt = pd.read_csv(f'{base_path}/train/train_ground_truth.tsv', sep='\t', dtype=str).fillna('')
    y_true_dict = {}
    for _, row in train_gt.iterrows():
        s1_id = row['source1_entity_id']
        if s1_id in val_ids_sample:
            matches_str = row['matched_entity_ids']
            matches = matches_str.split(',') if matches_str != '' else []
            y_true_dict[s1_id] = set(matches)
            
    metrics = compute_recall_metrics(y_true_dict, candidates_dict)
    
    print("\n--- PHASE 2 CANDIDATE RECALL METRICS ---")
    report_lines = ["=== PHASE 2 CANDIDATE GENERATION METRICS ==="]
    for k, v in metrics.items():
        line = f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
        print(line)
        report_lines.append(line)
        
    # Save the output explicitly to artifacts folder
    os.makedirs('artifacts', exist_ok=True)
    with open('artifacts/phase2_results.txt', 'w') as f:
        f.write('\n'.join(report_lines))

if __name__ == '__main__':
    main()
