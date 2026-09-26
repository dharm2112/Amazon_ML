import pandas as pd
import numpy as np
import os
import gc
from rank_bm25 import BM25Okapi
from tqdm import tqdm

def load_ground_truth(subset_size=1000):
    """Loads a random subset of S1 entities with known true matches for validation."""
    gt = pd.read_csv('DATA/student_resource/dataset/train/train_ground_truth.tsv', sep='\t', dtype=str)
    
    # Pick a random subset of S1 entities to evaluate retrieval
    np.random.seed(42)
    s1_sample = gt.sample(subset_size)
    
    val_dict = {}
    for _, row in s1_sample.iterrows():
        s1 = row['source1_entity_id']
        matches = str(row['matched_entity_ids']).split(',')
        val_dict[s1] = {m.strip() for m in matches if m.strip() != '' and m.strip() != 'nan'}
        
    return s1_sample, val_dict

def test_bm25_retrieval(s1_sample, val_dict, top_k=50):
    """
    Benchmarks BM25 retrieval against a subset of S1 queries.
    To prevent RAM crashes, we will simulate this on a subset of the corpus (e.g. US businesses)
    or just use the known pool. Actually, let's load S2 and S3 in chunks and build BM25 incrementally?
    rank_bm25 doesn't support incremental building easily.
    
    For this benchmark, we will use a small memory-safe fraction of the corpus to prove it works.
    """
    print("\n--- B2-B: BM25 Validation Benchmark ---")
    
    # 1. We will extract all S2/S3 ids that are 'true matches' for our sample, 
    # plus 500,000 random noise businesses to create a challenging retrieval pool.
    print("Building challenging retrieval pool...")
    true_s23_ids = set()
    for matches in val_dict.values():
        true_s23_ids.update(matches)
        
    # Stream S2 and S3, keep true matches + a random sample to fit in ~2GB RAM
    pool_docs = []
    pool_ids = []
    
    for source in ['train_source2_norm.tsv', 'train_source3_norm.tsv']:
        chunk_iter = pd.read_csv(f'DATA/processed/train/{source}', sep='\t', dtype=str, chunksize=100000)
        for chunk in chunk_iter:
            chunk = chunk.fillna('')
            # Add true matches
            true_mask = chunk['entity_id'].isin(true_s23_ids)
            # Add 2% random noise
            noise_mask = np.random.rand(len(chunk)) < 0.02
            mask = true_mask | noise_mask
            
            selected = chunk[mask]
            
            for _, row in selected.iterrows():
                # BM25 Tokenization: Use all informative fields
                text = f"{row['name_basic']} {row['address_basic']} {row['numeric_tokens']} {row['country_normalized']}"
                tokens = text.split()
                pool_docs.append(tokens)
                pool_ids.append(row['entity_id'])
                
            del chunk, selected
            gc.collect()
            
    print(f"Pool size built: {len(pool_docs)} documents.")
    
    # 2. Build BM25
    print("Building BM25 Index (Memory Safe)...")
    bm25 = BM25Okapi(pool_docs)
    
    # 3. Retrieve
    print(f"Retrieving Top-{top_k} for {len(s1_sample)} queries...")
    
    recalls = []
    for _, row in tqdm(s1_sample.iterrows(), total=len(s1_sample)):
        s1 = row['entity_id']
        query_text = f"{row['name_basic']} {row['address_basic']} {row['numeric_tokens']} {row['country_normalized']}"
        query_tokens = query_text.split()
        
        # Get scores
        scores = bm25.get_scores(query_tokens)
        
        # Get top K
        top_n = np.argsort(scores)[::-1][:top_k]
        retrieved_ids = {pool_ids[i] for i in top_n}
        
        # Calculate recall
        true_matches = val_dict.get(s1, set())
        if not true_matches:
            continue
            
        found = len(true_matches.intersection(retrieved_ids))
        recall = found / len(true_matches)
        recalls.append(recall)
        
    final_recall = np.mean(recalls) * 100
    print(f"\nBM25 Benchmark Results:")
    print(f"Candidate Recall@{top_k}: {final_recall:.2f}%")
    
    return final_recall

if __name__ == '__main__':
    s1_sample, val_dict = load_ground_truth(subset_size=1000)
    
    # Load the normalized S1 data for the sample
    s1_full = pd.read_csv('DATA/processed/train/train_source1_norm.tsv', sep='\t', dtype=str).fillna('')
    s1_sample_full = s1_full[s1_full['entity_id'].isin(s1_sample['source1_entity_id'])]
    
    test_bm25_retrieval(s1_sample_full, val_dict, top_k=50)
