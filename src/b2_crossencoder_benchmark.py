import pandas as pd
import numpy as np
import torch
from sentence_transformers import CrossEncoder
from sklearn.metrics import roc_auc_score
import gc

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def test_cross_encoder():
    print("\n--- B2-C: Cross-Encoder Validation Benchmark ---")
    
    # Load ground truth to create match=1 or match=0 labels
    gt = pd.read_csv('DATA/student_resource/dataset/train/train_ground_truth.tsv', sep='\t', dtype=str)
    
    # We will load the text dictionaries first to quickly build pairs
    print("Loading text dictionaries...")
    s1_full = pd.read_csv('DATA/processed/train/train_source1_norm.tsv', sep='\t', dtype=str, usecols=['entity_id', 'name_basic', 'address_basic']).fillna('')
    s2_full = pd.read_csv('DATA/processed/train/train_source2_norm.tsv', sep='\t', dtype=str, usecols=['entity_id', 'name_basic', 'address_basic']).fillna('')
    s3_full = pd.read_csv('DATA/processed/train/train_source3_norm.tsv', sep='\t', dtype=str, usecols=['entity_id', 'name_basic', 'address_basic']).fillna('')
    s23_full = pd.concat([s2_full, s3_full], ignore_index=True)
    
    s1_dict = s1_full.set_index('entity_id').to_dict('index')
    s23_dict = s23_full.set_index('entity_id').to_dict('index')
    
    del s1_full, s2_full, s3_full, s23_full
    gc.collect()
    
    # Build text pairs
    print("Building text pairs for Cross-Encoder...")
    text_pairs = []
    y_true_valid = []
    
    # Sample a subset of GT to test
    gt_sample = gt.sample(2500, random_state=42)
    s23_ids = list(s23_dict.keys())
    
    for _, row in gt_sample.iterrows():
        s1 = row['source1_entity_id']
        if s1 not in s1_dict:
            continue
            
        matches = str(row['matched_entity_ids']).split(',')
        matches = [m.strip() for m in matches if m.strip() != '' and m.strip() != 'nan']
        
        t1 = f"{s1_dict[s1]['name_basic']} {s1_dict[s1]['address_basic']}"
        
        # Add positives
        for m in matches:
            if m in s23_dict:
                t2 = f"{s23_dict[m]['name_basic']} {s23_dict[m]['address_basic']}"
                text_pairs.append([t1, t2])
                y_true_valid.append(1)
                
        # Add a random negative
        random_s2 = np.random.choice(s23_ids)
        if random_s2 not in matches:
            t2 = f"{s23_dict[random_s2]['name_basic']} {s23_dict[random_s2]['address_basic']}"
            text_pairs.append([t1, t2])
            y_true_valid.append(0)
            
    y_true_valid = np.array(y_true_valid)
    print(f"Generated {len(text_pairs)} valid text pairs ({sum(y_true_valid)} positive, {len(y_true_valid)-sum(y_true_valid)} negative).")
    

    print(f"Generated {len(text_pairs)} valid text pairs.")
    
    # 3. Load Cross-Encoder
    device = get_device()
    # MS MARCO is highly trained on query-document matching, perfect for this.
    model_name = 'cross-encoder/ms-marco-MiniLM-L-6-v2' 
    print(f"Loading Cross-Encoder '{model_name}' on {device}...")
    model = CrossEncoder(model_name, device=device)
    
    # 4. Predict
    print("Running pairs through Deep Learning Cross-Encoder...")
    # This will output logits/scores. 
    scores = model.predict(text_pairs, batch_size=128, show_progress_bar=True)
    
    # 5. Evaluate
    auc = roc_auc_score(y_true_valid, scores)
    
    print("\n--- Cross-Encoder Benchmark Results ---")
    print(f"ROC-AUC Score: {auc:.4f}")
    
    if auc < 0.995:
        print("WARNING: Cross-Encoder underperformed our LightGBM (0.995 AUC) on this validation set!")
    else:
        print("SUCCESS: Cross-Encoder beats or matches LightGBM!")

if __name__ == '__main__':
    test_cross_encoder()
