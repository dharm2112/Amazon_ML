import pandas as pd
import numpy as np
import os
import gc
import torch
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def load_ground_truth(subset_size=1000):
    gt = pd.read_csv('DATA/student_resource/dataset/train/train_ground_truth.tsv', sep='\t', dtype=str)
    np.random.seed(42)
    s1_sample = gt.sample(subset_size)
    
    val_dict = {}
    for _, row in s1_sample.iterrows():
        s1 = row['source1_entity_id']
        matches = str(row['matched_entity_ids']).split(',')
        val_dict[s1] = {m.strip() for m in matches if m.strip() != '' and m.strip() != 'nan'}
        
    return s1_sample, val_dict

def test_dense_retrieval(s1_sample, val_dict, top_k=50):
    print("\n--- B2-C: Dense FAISS Validation Benchmark ---")
    
    print("Building challenging retrieval pool...")
    true_s23_ids = set()
    for matches in val_dict.values():
        true_s23_ids.update(matches)
        
    pool_docs = []
    pool_ids = []
    
    # We use exactly the same logic to keep the test fair
    np.random.seed(42)
    for source in ['train_source2_norm.tsv', 'train_source3_norm.tsv']:
        chunk_iter = pd.read_csv(f'DATA/processed/train/{source}', sep='\t', dtype=str, chunksize=100000)
        for chunk in chunk_iter:
            chunk = chunk.fillna('')
            true_mask = chunk['entity_id'].isin(true_s23_ids)
            noise_mask = np.random.rand(len(chunk)) < 0.02
            mask = true_mask | noise_mask
            
            selected = chunk[mask]
            
            for _, row in selected.iterrows():
                # For Dense Retrieval, natural language sentences are better than raw tokens
                text = f"{row['name_basic']} {row['address_basic']} {row['country_normalized']}"
                pool_docs.append(text)
                pool_ids.append(row['entity_id'])
                
            del chunk, selected
            gc.collect()
            
    print(f"Pool size built: {len(pool_docs)} documents.")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading SentenceTransformer on {device}...")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device=device)
    
    dim = model.get_sentence_embedding_dimension()
    index = faiss.IndexFlatIP(dim)
    
    print("Encoding pool (this tests GPU embedding speed)...")
    embeddings = model.encode(pool_docs, batch_size=256, show_progress_bar=True, convert_to_numpy=True)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    
    print(f"Retrieving Top-{top_k} for {len(s1_sample)} queries...")
    s1_queries = []
    for _, row in s1_sample.iterrows():
        text = f"{row['name_basic']} {row['address_basic']} {row['country_normalized']}"
        s1_queries.append(text)
        
    query_embeddings = model.encode(s1_queries, batch_size=256, show_progress_bar=False, convert_to_numpy=True)
    faiss.normalize_L2(query_embeddings)
    
    distances, indices = index.search(query_embeddings, top_k)
    
    recalls = []
    for i, (_, row) in enumerate(s1_sample.iterrows()):
        s1 = row['entity_id']
        true_matches = val_dict.get(s1, set())
        
        if not true_matches:
            continue
            
        retrieved_ids = {pool_ids[idx] for idx in indices[i] if idx != -1}
        found = len(true_matches.intersection(retrieved_ids))
        recall = found / len(true_matches)
        recalls.append(recall)
        
    final_recall = np.mean(recalls) * 100
    print(f"\nDense FAISS Benchmark Results:")
    print(f"Candidate Recall@{top_k}: {final_recall:.2f}%")
    
    return final_recall

if __name__ == '__main__':
    s1_sample, val_dict = load_ground_truth(subset_size=1000)
    s1_full = pd.read_csv('DATA/processed/train/train_source1_norm.tsv', sep='\t', dtype=str).fillna('')
    s1_sample_full = s1_full[s1_full['entity_id'].isin(s1_sample['source1_entity_id'])]
    test_dense_retrieval(s1_sample_full, val_dict, top_k=50)
