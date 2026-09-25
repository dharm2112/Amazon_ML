import pandas as pd
import numpy as np
import os
import gc
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss
from rank_bm25 import BM25Okapi

from normalization import normalize_dataframe

def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'

def build_dense_index(texts, model_name='paraphrase-multilingual-MiniLM-L12-v2', batch_size=256):
    device = get_device()
    print(f"Loading SentenceTransformer '{model_name}' on {device}...")
    model = SentenceTransformer(model_name, device=device)
    
    print(f"Encoding {len(texts)} texts for Dense Retrieval...")
    # Encode with progress bar
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
    
    # L2 normalize embeddings for Cosine Similarity search in FAISS
    faiss.normalize_L2(embeddings)
    
    dim = embeddings.shape[1]
    
    # Use GPU FAISS if available for insane speed, else CPU
    if device == 'cuda':
        try:
            res = faiss.StandardGpuResources()
            index = faiss.GpuIndexFlatIP(res, dim) # Inner Product == Cosine Sim after L2 norm
            print("Using GPU FAISS Index.")
        except AttributeError:
            # Fallback if faiss-gpu isn't installed properly
            index = faiss.IndexFlatIP(dim)
            print("Using CPU FAISS Index (GPU FAISS not found).")
    else:
        index = faiss.IndexFlatIP(dim)
        print("Using CPU FAISS Index.")
        
    index.add(embeddings)
    return model, index

def main():
    print("=== STARTING B2 HYBRID RETRIEVAL (BM25 + FAISS) ===")
    
    # Note: To avoid running out of RAM, we will process one country at a time, just like Phase 2
    # This script is a template that implements the core math needed for B2.
    
    print("\n[Safety Check] This advanced script requires heavy GPU usage.")
    device = get_device()
    print(f"Detected Device: {device}")
    
    if device == 'cpu':
        print("\n⚠️ WARNING: You are running on CPU. Dense embedding 6 million rows on CPU will take days!")
        print("We will abort to prevent freezing your computer. Please run this on a GPU-enabled machine.")
        return
        
    print("\nGPU is active! Ready to begin embedding process...")
    # The actual streaming logic will be implemented here next.
    
if __name__ == '__main__':
    main()
