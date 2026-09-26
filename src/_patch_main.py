"""New main() for generate_submission_b2.py — country-partitioned full run."""

NEW_MAIN = r'''

def main():
    global CE_THRESHOLD, TOP_K_LGBM

    parser = argparse.ArgumentParser(description='B2: B1-Candidates + LightGBM + Cross-Encoder')
    parser.add_argument('--mode',    choices=['validate', 'full'], default='validate',
                        help='validate=50K entities; full=all 1.73M')
    parser.add_argument('--n-val',   type=int, default=50_000)
    parser.add_argument('--ce-threshold', type=float, default=CE_THRESHOLD)
    parser.add_argument('--top-k',   type=int, default=TOP_K_LGBM)
    args = parser.parse_args()

    CE_THRESHOLD = args.ce_threshold
    TOP_K_LGBM  = args.top_k
    is_validate  = (args.mode == 'validate')

    print("=" * 60)
    print("  TriMatch-ER B2 -- LightGBM + Cross-Encoder Pipeline")
    print(f"  Mode       : {'VALIDATION (' + str(args.n_val) + ' entities)' if is_validate else 'FULL RUN (1.73M entities)'}")
    print(f"  Top-K LGBM : {TOP_K_LGBM}")
    print(f"  CE logit   : >= {CE_THRESHOLD}")
    print(f"  CE batch   : {CE_BATCH_SIZE}")
    print("=" * 60)

    candidates_file = 'output/candidate_pairs_b1.tsv'
    b1_results_file = 'output/matching_results_b1.tsv'
    s1_norm_file    = 'DATA/processed/test/test_source1_norm.tsv'
    s2_norm_file    = 'DATA/processed/test/test_source2_norm.tsv'
    s3_norm_file    = 'DATA/processed/test/test_source3_norm.tsv'
    lgbm_model_file = 'experiments/models/lgbm_matcher.pkl'
    suffix      = '_val' if is_validate else ''
    output_file = f'output/matching_results_b2{suffix}.tsv'
    os.makedirs('output', exist_ok=True)

    print(f"\n[MODEL] Loading LightGBM...")
    lgbm_model = joblib.load(lgbm_model_file)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[MODEL] Loading Cross-Encoder on {device.upper()}...")
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=device)
    print(f"        Device: {device.upper()} {'(GPU!)' if device=='cuda' else '(CPU)'}")

    # --------------------------------------------------------
    # VALIDATION MODE: single pass on n_val entities
    # --------------------------------------------------------
    if is_validate:
        candidates = load_candidates(candidates_file, max_entities=args.n_val)
        all_s1_ids = list(candidates.keys())
        needed_s23 = set(cid for cids in candidates.values() for cid in cids)
        print(f"    Unique S23 IDs needed : {len(needed_s23):,}")
        s1_idx  = load_norm_data(s1_norm_file, needed_ids=set(all_s1_ids), desc='S1 norm')
        s2_idx  = load_norm_data(s2_norm_file, needed_ids=needed_s23, desc='S2 norm')
        s3_idx  = load_norm_data(s3_norm_file, needed_ids=needed_s23, desc='S3 norm')
        s23_idx = pd.concat([s2_idx, s3_idx]); del s2_idx, s3_idx, needed_s23; gc.collect()
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("source1_entity_id\tmatched_entity_ids\n")
        q     = queue.Queue(maxsize=Q_MAXSIZE)
        t_cpu = threading.Thread(target=cpu_producer,
                                 args=(all_s1_ids, candidates, s1_idx, s23_idx, lgbm_model, q),
                                 daemon=True)
        t_gpu = threading.Thread(target=gpu_consumer,
                                 args=(cross_encoder, s1_idx, s23_idx, q, output_file,
                                       all_s1_ids, args.mode),
                                 daemon=True)
        t0 = time.time()
        t_gpu.start(); t_cpu.start(); t_cpu.join(); t_gpu.join()
        print(f"\n[TIMING] {(time.time()-t0)/60:.1f} min")
        errors, _ = validate_output(output_file, all_s1_ids)
        if os.path.exists(b1_results_file):
            compare_with_b1(output_file, b1_results_file, all_s1_ids)
        print("\n" + "=" * 60)
        print("  [VALIDATE DONE] Run full with: python src/generate_submission_b2.py --mode full")
        print("=" * 60)
        return

    # --------------------------------------------------------
    # FULL MODE: country-partitioned to avoid OOM
    #
    # Problem: 50.5M pairs + 3.9M S23 rows = OOM (4.5GB+)
    # Fix: process india/us/france one at a time
    # Peak RAM per partition: ~600MB S1 + ~850MB candidates + ~500MB S23 = ~2GB
    # --------------------------------------------------------

    # Step A: Build S1->country map (2 columns only = ~80MB)
    print("\n[PARTITION] Building S1 country map (2-col scan)...")
    country_map = {}
    for chunk in pd.read_csv(s1_norm_file, sep='\t', dtype=str,
                             usecols=['entity_id', 'country_normalized'],
                             chunksize=500_000):
        chunk = chunk.fillna('')
        for eid, c in zip(chunk['entity_id'].values, chunk['country_normalized'].values):
            country_map[eid] = c.strip().lower()

    countries      = sorted(set(country_map.values()) - {''})
    country_s1_ids = {c: [] for c in countries}
    for eid, c in country_map.items():
        if c:
            country_s1_ids[c].append(eid)
    del country_map; gc.collect()
    for c, ids in country_s1_ids.items():
        print(f"    {c:10s}: {len(ids):,} S1 entities")

    all_s1_ids_ordered = [eid for c in countries for eid in country_s1_ids[c]]

    # Write output header
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")

    # Step B: Process each country
    t_total = time.time()
    for ci, country in enumerate(countries, 1):
        s1_ids_country = country_s1_ids[country]
        s1_id_set      = set(s1_ids_country)

        print(f"\n{'='*55}")
        print(f"  PARTITION {ci}/{len(countries)}: {country.upper()}  ({len(s1_ids_country):,} S1 entities)")
        print(f"{'='*55}")

        # Load candidates for this country (stream & filter)
        print(f"  [DATA] Loading candidates for {country}...")
        t0 = time.time()
        candidates = {}
        needed_s23 = set()
        for chunk in pd.read_csv(candidates_file, sep='\t', dtype=str, chunksize=200_000):
            chunk = chunk.fillna('')
            chunk = chunk[chunk['source1_entity_id'].isin(s1_id_set)]
            for s1_id, raw in zip(chunk['source1_entity_id'].values,
                                  chunk['candidate_entity_ids'].values):
                cids = [x.strip() for x in raw.split(',') if x.strip()] if raw else []
                candidates[s1_id] = cids
                needed_s23.update(cids)
        for s1_id in s1_ids_country:       # ensure singletons present
            candidates.setdefault(s1_id, [])
        total_pairs = sum(len(v) for v in candidates.values())
        print(f"    {total_pairs:,} pairs | {len(needed_s23):,} unique S23 IDs | {time.time()-t0:.1f}s")

        # Load features
        s1_idx  = load_norm_data(s1_norm_file, needed_ids=s1_id_set,
                                 desc=f'S1 norm ({country})')
        s2_idx  = load_norm_data(s2_norm_file, needed_ids=needed_s23, desc='S2 norm')
        s3_idx  = load_norm_data(s3_norm_file, needed_ids=needed_s23, desc='S3 norm')
        s23_idx = pd.concat([s2_idx, s3_idx])
        del s2_idx, s3_idx, needed_s23; gc.collect()
        print(f"    S23 entities loaded: {len(s23_idx):,}")

        # Run parallel pipeline
        print(f"  [PIPELINE] CPU (LGBM) + GPU (CE) ...")
        q     = queue.Queue(maxsize=Q_MAXSIZE)
        t_cpu = threading.Thread(target=cpu_producer,
                                 args=(s1_ids_country, candidates, s1_idx, s23_idx,
                                       lgbm_model, q),
                                 daemon=True)
        t_gpu = threading.Thread(target=gpu_consumer,
                                 args=(cross_encoder, s1_idx, s23_idx, q, output_file,
                                       s1_ids_country, 'full'),
                                 daemon=True)
        t0 = time.time()
        t_gpu.start(); t_cpu.start(); t_cpu.join(); t_gpu.join()
        print(f"  [TIMING] Partition {country}: {(time.time()-t0)/60:.1f} min")

        del candidates, s1_idx, s23_idx; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_elapsed = time.time() - t_total
    print(f"\n[TIMING] Total full run: {total_elapsed/60:.1f} min")

    errors, _ = validate_output(output_file, all_s1_ids_ordered)

    if not errors:
        zip_path = 'output/submission_b2.zip'
        print(f"\n[ZIP] Creating {zip_path}...")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.write(output_file, arcname='matching_results.tsv')
        print(f"    Zip size: {os.path.getsize(zip_path)/1e6:.1f} MB")

    print("\n" + "=" * 60)
    if errors:
        print("  [FAILED] Validation errors -- check above.")
    else:
        print("  [SUCCESS] B2 submission ready!")
        print(f"  Output : {output_file}")
        print(f"  Submit : output/submission_b2.zip")
    print("=" * 60)


if __name__ == '__main__':
    main()
'''

if __name__ == '__main__':
    # Read the current script, cut at main(), append new main
    import sys
    src = open('src/generate_submission_b2.py', encoding='utf-8').read()
    # Find where the old main section begins (the comment block + def main)
    cut_marker = '\ndef main():'
    idx = src.find(cut_marker)
    if idx == -1:
        print("ERROR: could not find main()", file=sys.stderr)
        sys.exit(1)
    # Also cut the preceding comment block (3 lines: blank, comment, comment, comment, blank)
    # Walk backwards to find the blank line before the # section
    pre = src[:idx].rstrip()
    # Remove the trailing comment block if present
    lines = pre.split('\n')
    while lines and (lines[-1].startswith('#') or lines[-1].strip() == ''):
        lines.pop()
    header = '\n'.join(lines)
    new_content = header + NEW_MAIN
    open('src/generate_submission_b2.py', 'w', encoding='utf-8').write(new_content)
    print(f"Patched successfully. Total lines: {new_content.count(chr(10))}")
