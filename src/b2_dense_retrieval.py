import pandas as pd
import numpy as np
import os
import gc
import torch
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm
import pickle

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def build_and_save_faiss_index(split='train', model_name='paraphrase-multilingual-MiniLM-L12-v2', chunksize=100000):
    device = get_device()
    print(f"Loading SentenceTransformer '{model_name}' on {device}...")
    model = SentenceTransformer(model_name, device=device)
    
    # We will embed name + address for max semantic coverage
    dim = model.get_sentence_embedding_dimension()
    
    # FAISS Index on CPU RAM (9GB for 6M rows, easily fits)
    # Using L2 normalized Inner Product for Cosine Similarity
    index = faiss.IndexFlatIP(dim)
    
    entity_id_map = []
    
    # We stream the pre-processed data to save time!
    for source in [f'{split}_source2_norm.tsv', f'{split}_source3_norm.tsv']:
        file_path = f'DATA/processed/{split}/{source}'
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found. Are you sure preparation is done?")
            continue
            
        print(f"\nEmbedding {file_path} in chunks of {chunksize}...")
        chunk_iter = pd.read_csv(file_path, sep='\t', dtype=str, chunksize=chunksize)
        
        for chunk in chunk_iter:
            chunk = chunk.fillna('')
            # Combine name and address for dense semantic meaning
            texts = (chunk['name_basic'] + " " + chunk['address_basic']).tolist()
            
            # Encode on GPU
            embeddings = model.encode(texts, batch_size=256, show_progress_bar=False, convert_to_numpy=True)
            faiss.normalize_L2(embeddings)
            
            # Add to FAISS index
            index.add(embeddings)
            entity_id_map.extend(chunk['entity_id'].tolist())
            
            print(f"  Added {len(texts)} vectors. Total FAISS size: {index.ntotal}")
            
            del chunk, texts, embeddings
            gc.collect()
            
    # Save the index and the map
    os.makedirs('experiments/faiss', exist_ok=True)
    index_path = f'experiments/faiss/{split}_s23_index.faiss'
    map_path = f'experiments/faiss/{split}_s23_map.pkl'
    
    print(f"\nSaving FAISS index to {index_path}...")
    faiss.write_index(index, index_path)
    
    print(f"Saving ID mapping to {map_path}...")
    with open(map_path, 'wb') as f:
        pickle.dump(entity_id_map, f)
        
    print("Done building FAISS Dense Index!")
    return model, index, entity_id_map

def retrieve_dense_candidates(model, index, entity_id_map, split='train', top_k=50):
    print(f"\nRetrieving top {top_k} Dense Semantic candidates for {split} S1...")
    
    s1_path = f'DATA/processed/{split}/{split}_source1_norm.tsv'
    s1 = pd.read_csv(s1_path, sep='\t', dtype=str).fillna('')
    
    queries = (s1['name_basic'] + " " + s1['address_basic']).tolist()
    
    print(f"Encoding {len(queries)} S1 queries on GPU...")
    query_embeddings = model.encode(queries, batch_size=256, show_progress_bar=True, convert_to_numpy=True)
    faiss.normalize_L2(query_embeddings)
    
    print(f"Searching FAISS Index for Top-{top_k} matches...")
    distances, indices = index.search(query_embeddings, top_k)
    
    dense_candidates = {}
    
    for i, s1_id in enumerate(tqdm(s1['entity_id'])):
        cands = set()
        for j in range(top_k):
            match_idx = indices[i][j]
            if match_idx != -1:
                c_id = entity_id_map[match_idx]
                cands.add(c_id)
        dense_candidates[s1_id] = cands
        
    save_path = f'experiments/faiss/{split}_dense_candidates.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(dense_candidates, f)
        
    print(f"Dense candidates saved to {save_path}")

def main():
    print("=== B2 DENSE MULTILINGUAL RETRIEVER ===")
    
    if get_device() == 'cpu':
        print("WARNING: Running Dense Retrieval on CPU will take days. Aborting.")
        return
        
    # Wait for the user's data preparation script to finish before running this
    if not os.path.exists('DATA/processed/train/train_source2_norm.tsv'):
        print("Waiting for prepare_team_data.py to finish processing DATA/processed/...")
        return
        
    # Build for train (to train Cross-Encoder)
    model, index, entity_map = build_and_save_faiss_index(split='train')
    retrieve_dense_candidates(model, index, entity_map, split='train', top_k=50)

if __name__ == '__main__':
    main()
