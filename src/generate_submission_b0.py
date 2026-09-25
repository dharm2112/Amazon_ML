"""
Submission Generator - Phase 0 Baseline (B0)
Strategy: Exact match on (business_name, country) between test_source1 and test_source2/3
Streams directly to output files — no large lists held in memory.
"""
import pandas as pd
import numpy as np
import os
import gc
from collections import defaultdict
from tqdm import tqdm

def main():
    base_path = 'DATA/student_resource/dataset'
    os.makedirs('output', exist_ok=True)

    print("Loading test_source1...")
    test_s1 = pd.read_csv(
        f'{base_path}/test/test_source1.tsv', sep='\t', dtype=str,
        usecols=['entity_id', 'business_name', 'country']
    ).fillna('')
    print(f"  test_source1 rows: {len(test_s1):,}")

    print("Building S1 lookup keys...")
    test_s1['lookup_key'] = test_s1['business_name'].str.lower().str.strip() + \
                             '|||' + test_s1['country'].str.lower().str.strip()
    s1_keys = set(test_s1['lookup_key'].values)
    print(f"  Unique S1 (name, country) keys: {len(s1_keys):,}")

    # Index test_source2 and test_source3 in streaming fashion
    print("\nStreaming test_source2 and test_source3 to build index...")
    name_country_idx = defaultdict(list)

    for source_path in [
        f'{base_path}/test/test_source2.tsv',
        f'{base_path}/test/test_source3.tsv',
    ]:
        print(f"  Streaming {source_path}...")
        for chunk in pd.read_csv(
            source_path, sep='\t', dtype=str,
            usecols=['entity_id', 'business_name', 'country'],
            chunksize=200000
        ):
            chunk = chunk.fillna('')
            chunk['lookup_key'] = chunk['business_name'].str.lower().str.strip() + \
                                   '|||' + chunk['country'].str.lower().str.strip()
            matched = chunk[chunk['lookup_key'].isin(s1_keys)]
            for row in matched[['entity_id', 'lookup_key']].itertuples(index=False):
                name_country_idx[row.lookup_key].append(row.entity_id)
            del chunk, matched
        gc.collect()

    print(f"  Indexed {len(name_country_idx):,} unique matching keys")

    # Stream write predictions directly to output files
    print("\nGenerating and writing output files...")

    match_count = 0
    singleton_count = 0
    candidate_count = 0

    with open('output/matching_results.tsv', 'w', encoding='utf-8') as f_match, \
         open('output/candidate_pairs.tsv', 'w', encoding='utf-8') as f_cand:

        # Write headers — must match validator exactly
        f_match.write('source1_entity_id\tmatched_entity_ids\n')
        f_cand.write('source1_entity_id\tcandidate_entity_ids\n')  # plural, comma-separated list

        for row in tqdm(test_s1.itertuples(index=False), total=len(test_s1)):
            s1_id = row.entity_id
            key = row.lookup_key
            matches = name_country_idx.get(key, [])

            # Write matching_results line
            matched_ids_str = ','.join(matches) if matches else ''
            f_match.write(f'{s1_id}\t{matched_ids_str}\n')

            # Write candidate_pairs line — one row per S1, comma-separated candidate IDs
            # For B0, candidates == matches (no separate blocking stage)
            f_cand.write(f'{s1_id}\t{matched_ids_str}\n')

            candidate_count += len(matches)
            if matches:
                match_count += 1
            else:
                singleton_count += 1

    print(f"\n  Total S1 test entities written: {match_count + singleton_count:,}")
    print(f"  Entities with at least one match: {match_count:,}")
    print(f"  Singleton entities (no match): {singleton_count:,}")
    print(f"  Total candidate pair rows: {candidate_count:,}")
    print("\nOutput files saved:")
    print("  output/matching_results.tsv")
    print("  output/candidate_pairs.tsv")
    print("\nNow validating submission...")

if __name__ == '__main__':
    main()
