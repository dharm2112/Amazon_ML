"""
Phase 2b: Structural Retrieval Boost
Adds recall via rare-token exact lookup, prefix lookup, acronym lookup, and numeric-token lookup.
Union with TF-IDF candidates to push Recall@50 toward 98-99%.
"""
import pandas as pd
import numpy as np
import os
import gc
import json
from collections import defaultdict
from tqdm import tqdm
from normalization import normalize_dataframe
from phase2_candidates import (
    compute_recall_metrics, get_candidates, load_source_for_country
)

def build_inverted_index(df_index, key_fn, val_col='entity_id'):
    """Build {key -> [list of entity_ids]} from a dataframe using key_fn."""
    idx = defaultdict(list)
    for row in df_index.itertuples(index=False):
        keys = key_fn(row)
        eid = getattr(row, val_col)
        for k in keys:
            if k:
                idx[k].append(eid)
    return idx

def structural_lookup(df_query, df_index, k=50):
    """
    Fast structural retrieval using:
      1. Rare token lookup (tokens appearing in <=100 docs in the index)
      2. Discriminative numeric token lookup (length>=3, appears in <=500 docs)
      3. Acronym lookup (length >= 2)
      4. 5-char prefix lookup on name_core
    Returns dict {s1_id: set(candidate_ids)}
    """
    print(f"    Building token frequency maps over {len(df_index)} docs...")

    RARE_TOK_THRESHOLD = 100
    RARE_NUM_THRESHOLD = 500
    MAX_INDEX_SET = 200  # cap any index bucket to prevent giant sets

    # --- Pass 1: compute frequencies ---
    tok_freq = defaultdict(int)
    num_freq = defaultdict(int)
    for row in df_index[['name_tokens', 'numeric_tokens']].itertuples(index=False):
        for tok in row.name_tokens:
            tok_freq[tok] += 1
        for num in row.numeric_tokens:
            if num and len(num) >= 3:  # skip single/double digit numbers
                num_freq[num] += 1

    # --- Pass 2: build indexes only for discriminative tokens ---
    print(f"    Building inverted indexes...")
    rare_tok_idx = defaultdict(list)
    num_tok_idx = defaultdict(list)
    acronym_idx = defaultdict(list)
    prefix_idx = defaultdict(list)

    for row in df_index[['entity_id', 'name_tokens', 'numeric_tokens', 'name_acronym', 'name_core']].itertuples(index=False):
        eid = row.entity_id

        # Rare tokens
        for tok in row.name_tokens:
            if tok_freq.get(tok, 9999) <= RARE_TOK_THRESHOLD:
                bucket = rare_tok_idx[tok]
                if len(bucket) < MAX_INDEX_SET:
                    bucket.append(eid)

        # Discriminative numerics (length>=3 and rare enough)
        for num in row.numeric_tokens:
            if num and len(num) >= 3 and num_freq.get(num, 9999) <= RARE_NUM_THRESHOLD:
                bucket = num_tok_idx[num]
                if len(bucket) < MAX_INDEX_SET:
                    bucket.append(eid)

        # Acronym
        acr = row.name_acronym
        if len(acr) >= 2:
            bucket = acronym_idx[acr]
            if len(bucket) < MAX_INDEX_SET:
                bucket.append(eid)

        # 5-char prefix
        core = row.name_core
        if len(core) >= 5:
            bucket = prefix_idx[core[:5]]
            if len(bucket) < MAX_INDEX_SET:
                bucket.append(eid)

    # --- Lookup queries ---
    print(f"    Looking up {len(df_query)} query entities...")
    candidates = {}
    for row in df_query[['entity_id', 'name_tokens', 'numeric_tokens', 'name_acronym', 'name_core']].itertuples(index=False):
        s1_id = row.entity_id
        cands = set()

        for tok in row.name_tokens:
            if tok_freq.get(tok, 9999) <= RARE_TOK_THRESHOLD:
                cands.update(rare_tok_idx.get(tok, []))

        for num in row.numeric_tokens:
            if num and len(num) >= 3 and num_freq.get(num, 9999) <= RARE_NUM_THRESHOLD:
                cands.update(num_tok_idx.get(num, []))

        acr = row.name_acronym
        if len(acr) >= 2:
            cands.update(acronym_idx.get(acr, []))

        core = row.name_core
        if len(core) >= 5:
            cands.update(prefix_idx.get(core[:5], []))

        candidates[s1_id] = cands

    # Cleanup
    del tok_freq, num_freq, rare_tok_idx, num_tok_idx, acronym_idx, prefix_idx
    gc.collect()

    return candidates


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

    val_countries = [c for c in val_s1_norm['country_normalized'].unique() if c != '']
    merged_candidates = defaultdict(dict)  # {s1_id: {c_id: score}}

    for country in val_countries:
        print(f"\n=== Processing Country: {country} ===")
        df_query = val_s1_norm[val_s1_norm['country_normalized'] == country]
        if len(df_query) == 0: continue

        s2_country = load_source_for_country(f'{base_path}/train/train_source2.tsv', country)
        s3_country = load_source_for_country(f'{base_path}/train/train_source3.tsv', country)
        s2_norm = normalize_dataframe(s2_country)
        s3_norm = normalize_dataframe(s3_country)
        df_index = pd.concat([s2_norm, s3_norm], ignore_index=True)
        df_index['search_text'] = df_index['name_basic'] + " " + df_index['address_basic']
        del s2_country, s3_country, s2_norm, s3_norm; gc.collect()

        # --- B1b: Structural Retrieval ---
        print("  Running Structural Retrieval (rare tokens, numerics, acronyms, prefixes)...")
        struct_cands = structural_lookup(df_query, df_index, k=50)

        print("  Freeing up memory by dropping structural columns from index...")
        df_index.drop(columns=['name_tokens', 'numeric_tokens', 'name_acronym', 'name_core', 'name_basic', 'address_basic', 'business_name', 'business_address', 'country'], inplace=True, errors='ignore')
        gc.collect()

        # --- B1a: Char TF-IDF ---
        print("  Running Char TF-IDF Retrieval...")
        char_cands = get_candidates(df_query, df_index, 'search_text', k=50, analyzer='char_wb', ngram_range=(3, 4))

        # --- B1a: Token TF-IDF ---
        print("  Running Token TF-IDF Retrieval...")
        token_cands = get_candidates(df_query, df_index, 'search_text', k=50, analyzer='word', ngram_range=(1, 2))

        # --- Merge all retrievers ---
        print("  Merging candidates from all retrievers...")
        for s1_id in df_query['entity_id']:
            merged = {}
            for c_id, score in char_cands.get(s1_id, []):
                merged[c_id] = max(merged.get(c_id, 0.0), float(score))
            for c_id, score in token_cands.get(s1_id, []):
                merged[c_id] = max(merged.get(c_id, 0.0), float(score))
            # Structural candidates get a base score of 0.1 (ensures they are included)
            for c_id in struct_cands.get(s1_id, set()):
                if c_id not in merged:
                    merged[c_id] = 0.1

            merged_candidates[s1_id] = merged

        del df_index, char_cands, token_cands, struct_cands; gc.collect()

    # Convert to sorted list of (c_id, score) for recall computation
    candidates_dict = {}
    for s1_id, merged in merged_candidates.items():
        sorted_cands = sorted(merged.items(), key=lambda x: x[1], reverse=True)
        candidates_dict[s1_id] = sorted_cands[:100]

    # --- Evaluate ---
    print("\nEvaluating Candidate Recall...")
    train_gt = pd.read_csv(f'{base_path}/train/train_ground_truth.tsv', sep='\t', dtype=str).fillna('')
    y_true_dict = {}
    for _, row in train_gt.iterrows():
        s1_id = row['source1_entity_id']
        if s1_id in val_ids_sample:
            matches_str = row['matched_entity_ids']
            y_true_dict[s1_id] = set(matches_str.split(',')) if matches_str else set()

    metrics = compute_recall_metrics(y_true_dict, candidates_dict)

    print("\n--- PHASE 2b (B1b) CANDIDATE RECALL METRICS ---")
    report_lines = ["=== PHASE 2b CANDIDATE GENERATION METRICS (TF-IDF + Structural) ==="]
    for k, v in metrics.items():
        line = f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v:.1f}"
        print(line)
        report_lines.append(line)

    os.makedirs('artifacts', exist_ok=True)
    with open('artifacts/phase2b_results.txt', 'w') as f:
        f.write('\n'.join(report_lines))

    # Save candidate dict for Phase 3
    print("\nSaving candidates to experiments/...")
    os.makedirs('experiments', exist_ok=True)
    np.save('experiments/val_candidates.npy', dict(candidates_dict))
    print("Done! Candidates saved to experiments/val_candidates.npy")


if __name__ == '__main__':
    main()
