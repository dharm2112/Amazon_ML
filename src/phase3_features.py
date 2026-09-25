import pandas as pd
import numpy as np
from rapidfuzz import fuzz

def jaccard_similarity(list1, list2):
    if not isinstance(list1, list) or not isinstance(list2, list):
        return 0.0
    set1, set2 = set(list1), set(list2)
    if len(set1) == 0 and len(set2) == 0:
        return 1.0 # both empty means they agree
    if len(set1) == 0 or len(set2) == 0:
        return 0.0
    return len(set1.intersection(set2)) / len(set1.union(set2))

def compute_pairwise_features(df_pairs, s1_features, s23_features):
    """
    Computes numerical features for pairs of entities.
    df_pairs: DataFrame with 's1_id', 'c_id', 'retrieval_score'
    s1_features: DataFrame of normalized Source 1 data, indexed by entity_id
    s23_features: DataFrame of normalized Source 2 & 3 data, indexed by entity_id
    """
    print("Extracting feature columns...")
    s1_names = df_pairs['s1_id'].map(s1_features['name_basic']).fillna('').values
    c_names = df_pairs['c_id'].map(s23_features['name_basic']).fillna('').values
    
    s1_addrs = df_pairs['s1_id'].map(s1_features['address_basic']).fillna('').values
    c_addrs = df_pairs['c_id'].map(s23_features['address_basic']).fillna('').values
    
    print("Computing RapidFuzz string distances...")
    name_jaro = [fuzz.jaro_winkler(s1, c) / 100.0 for s1, c in zip(s1_names, c_names)]
    name_ratio = [fuzz.ratio(s1, c) / 100.0 for s1, c in zip(s1_names, c_names)]
    
    addr_jaro = [fuzz.jaro_winkler(s1, c) / 100.0 for s1, c in zip(s1_addrs, c_addrs)]
    addr_ratio = [fuzz.ratio(s1, c) / 100.0 for s1, c in zip(s1_addrs, c_addrs)]
    
    print("Computing Jaccard similarities...")
    s1_name_toks = df_pairs['s1_id'].map(s1_features['name_tokens']).values
    c_name_toks = df_pairs['c_id'].map(s23_features['name_tokens']).values
    name_jaccard = [jaccard_similarity(s1, c) for s1, c in zip(s1_name_toks, c_name_toks)]
    
    s1_addr_toks = df_pairs['s1_id'].map(s1_features['address_tokens']).values
    c_addr_toks = df_pairs['c_id'].map(s23_features['address_tokens']).values
    addr_jaccard = [jaccard_similarity(s1, c) for s1, c in zip(s1_addr_toks, c_addr_toks)]
    
    print("Computing structural & numeric features...")
    s1_nums = df_pairs['s1_id'].map(s1_features['numeric_tokens']).values
    c_nums = df_pairs['c_id'].map(s23_features['numeric_tokens']).values
    
    num_overlap = []
    num_conflict = []
    
    for n1, n2 in zip(s1_nums, c_nums):
        if not isinstance(n1, list) or not isinstance(n2, list):
            num_overlap.append(0)
            num_conflict.append(0)
            continue
            
        set1, set2 = set(n1), set(n2)
        intersect = len(set1.intersection(set2))
        num_overlap.append(intersect)
        
        # Conflict: both have numbers, but NO numbers overlap
        if len(set1) > 0 and len(set2) > 0 and intersect == 0:
            num_conflict.append(1)
        else:
            num_conflict.append(0)
            
    print("Assembling final feature DataFrame...")
    features = pd.DataFrame({
        'retrieval_score': df_pairs['retrieval_score'].values,
        'name_jaro': name_jaro,
        'name_ratio': name_ratio,
        'name_jaccard': name_jaccard,
        'addr_jaro': addr_jaro,
        'addr_ratio': addr_ratio,
        'addr_jaccard': addr_jaccard,
        'num_overlap': num_overlap,
        'num_conflict': num_conflict,
    })
    
    if 'label' in df_pairs.columns:
        features['label'] = df_pairs['label'].values
        
    return features
