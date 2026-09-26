"""
TriMatch-ER B2 — Kaggle Production Pipeline
===========================================
Designed to run on Kaggle GPU Notebooks (T4 / P100 / L4).

Features:
- Dynamic dataset auto-discovery under /kaggle/input/ or local directory
- CPU (LightGBM pruning) + GPU (Cross-Encoder reranking) parallel pipeline
- Automatic submission zip creation (`submission_b2.zip`) in /kaggle/working/

Usage in Kaggle Notebook / Terminal:
    python generate_submission_b2_kaggle.py --mode full
"""

import sys, os, gc, time, argparse, warnings, queue, threading, zipfile, glob
import builtins

import numpy as np
import pandas as pd
import torch
import joblib
from rapidfuzz import fuzz
from sentence_transformers import CrossEncoder
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ── Force UTF-8 + always-flush prints ──────────────────────
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
_orig_print = builtins.print
def _fprint(*a, **kw):
    kw.setdefault('flush', True)
    _orig_print(*a, **kw)
builtins.print = _fprint

# ── Config Defaults ─────────────────────────────────────────
FEATURE_COLS = [
    'retrieval_score', 'name_jaro', 'name_ratio', 'name_jaccard',
    'addr_jaro', 'addr_ratio', 'addr_jaccard', 'num_overlap', 'num_conflict'
]
TOP_K_LGBM      = 10    # LightGBM top-K sent to Cross-Encoder
LGBM_THRESHOLD  = 0.0   # Keep all top-K regardless of LightGBM score
CE_THRESHOLD    = 0.45  # Cross-Encoder raw logit threshold
CE_BATCH_SIZE   = 512   # GPU batch size for Cross-Encoder
PROC_BATCH_SIZE = 2000  # Number of S1 entities per processing batch
Q_MAXSIZE       = 10    # Queue depth between CPU→GPU threads

_DONE = object()        # Thread sentinel


def find_file(filename, search_dirs=None):
    """Auto-locate file in current dir, ./output, ./models, or /kaggle/input/**/"""
    if search_dirs is None:
        search_dirs = [
            '.',
            'output',
            'models',
            'experiments/models',
            'DATA/processed/test',
            '/kaggle/working',
            '/kaggle/input'
        ]
    
    # 1. Direct path check
    if os.path.exists(filename):
        return filename

    basename = os.path.basename(filename)

    # 2. Search defined directories by exact basename
    for d in search_dirs:
        if not os.path.exists(d):
            continue
        for root, _, files in os.walk(d):
            if basename in files:
                target_path = os.path.join(root, basename)
                print(f"[PATH RESOLVER] Found '{basename}' at: {target_path}")
                return target_path

    # 3. Candidate pairs wildcard fallback (candidate_pairs.tsv, candidate_pairs_b1.tsv, etc.)
    if 'candidate' in basename.lower():
        for d in search_dirs:
            if not os.path.exists(d):
                continue
            for root, _, files in os.walk(d):
                for f in files:
                    if 'candidate' in f.lower() and f.endswith('.tsv'):
                        target_path = os.path.join(root, f)
                        print(f"[PATH RESOLVER] Candidate fallback found '{f}' at: {target_path}")
                        return target_path

    # 4. If file is a test TSV and missing, look for test.zip and auto-extract it
    test_zips = glob.glob('/kaggle/input/**/test.zip', recursive=True) + glob.glob('**/test.zip', recursive=True)
    if test_zips:
        print(f"[PATH RESOLVER] '{basename}' not found directly. Extracting '{test_zips[0]}'...")
        dest_dir = '/kaggle/working' if os.path.exists('/kaggle/working') else '.'
        with zipfile.ZipFile(test_zips[0], 'r') as zf:
            zf.extractall(dest_dir)
        for d in search_dirs:
            if not os.path.exists(d):
                continue
            for root, _, files in os.walk(d):
                if basename in files:
                    target_path = os.path.join(root, basename)
                    print(f"[PATH RESOLVER] Found '{basename}' after extraction at: {target_path}")
                    return target_path

    raise FileNotFoundError(
        f"Could not locate file '{filename}' (basename: '{basename}'). "
        f"Searched in: {search_dirs}"
    )


# ═══════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════

def load_candidates(candidates_file, max_entities=None):
    print(f"[DATA] Loading candidate pairs from {candidates_file}...")
    t0 = time.time()

    dfs = []
    for chunk in pd.read_csv(candidates_file, sep='\t', dtype=str, chunksize=500_000):
        chunk = chunk.fillna('')
        dfs.append(chunk)

    df = pd.concat(dfs, ignore_index=True)
    del dfs
    gc.collect()

    if max_entities:
        df = df.iloc[:max_entities]
        print(f"    [VAL] Limiting to first {max_entities:,} S1 entities.")

    s1_ids    = df['source1_entity_id'].values
    cand_strs = df['candidate_entity_ids'].values

    candidates = {}
    for s1_id, raw in zip(s1_ids, cand_strs):
        if raw:
            candidates[s1_id] = [c.strip() for c in raw.split(',') if c.strip()]
        else:
            candidates[s1_id] = []

    elapsed = time.time() - t0
    total_pairs = sum(len(v) for v in candidates.values())
    print(f"    S1 entities   : {len(candidates):,}")
    print(f"    Total pairs   : {total_pairs:,}")
    print(f"    Avg cands/S1  : {total_pairs/max(1,len(candidates)):.1f}")
    print(f"    Loaded in     : {elapsed:.1f}s")
    return candidates


def load_norm_data(norm_file, needed_ids=None, desc=''):
    print(f"[DATA] Loading {desc or os.path.basename(norm_file)}...")
    t0 = time.time()

    needed = set(needed_ids) if needed_ids is not None else None
    dfs = []

    for chunk in pd.read_csv(norm_file, sep='\t', dtype=str, chunksize=500_000):
        chunk = chunk.fillna('')
        if needed is not None:
            chunk = chunk[chunk['entity_id'].isin(needed)]
            if chunk.empty:
                if needed and len(dfs) > 0:
                    found = set(pd.concat(dfs, ignore_index=True)['entity_id'])
                    if needed <= found:
                        break
            else:
                dfs.append(chunk)
        else:
            dfs.append(chunk)

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)
    keep_cols = [c for c in [
        'entity_id', 'name_basic', 'address_basic',
        'name_tokens', 'address_tokens', 'numeric_tokens'
    ] if c in df.columns]
    df = df[keep_cols].set_index('entity_id')

    for col in ['name_tokens', 'address_tokens', 'numeric_tokens']:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: x.split() if isinstance(x, str) else []
            )

    elapsed = time.time() - t0
    print(f"    Rows loaded   : {len(df):,}  in {elapsed:.1f}s")
    return df


# ═══════════════════════════════════════════════════════════
#  FEATURE COMPUTATION
# ═══════════════════════════════════════════════════════════

def _jaccard(a, b):
    if not isinstance(a, list): a = []
    if not isinstance(b, list): b = []
    sa, sb = set(a), set(b)
    if not sa and not sb: return 1.0
    if not sa or  not sb: return 0.0
    return len(sa & sb) / len(sa | sb)


def compute_features_batch(pairs_df, s1_idx, s23_idx):
    s1_ids = pairs_df['s1_id'].values
    c_ids  = pairs_df['c_id'].values

    s1_name = s1_idx['name_basic'].reindex(s1_ids).fillna('').values
    c_name  = s23_idx['name_basic'].reindex(c_ids).fillna('').values
    s1_addr = s1_idx['address_basic'].reindex(s1_ids).fillna('').values
    c_addr  = s23_idx['address_basic'].reindex(c_ids).fillna('').values
    s1_ntok = s1_idx['name_tokens'].reindex(s1_ids).values
    c_ntok  = s23_idx['name_tokens'].reindex(c_ids).values
    s1_atok = s1_idx['address_tokens'].reindex(s1_ids).values
    c_atok  = s23_idx['address_tokens'].reindex(c_ids).values
    s1_num  = s1_idx['numeric_tokens'].reindex(s1_ids).values
    c_num   = s23_idx['numeric_tokens'].reindex(c_ids).values

    n = len(s1_ids)
    name_jaro   = np.empty(n); name_ratio   = np.empty(n)
    addr_jaro   = np.empty(n); addr_ratio   = np.empty(n)
    name_jac    = np.empty(n); addr_jac     = np.empty(n)
    num_overlap = np.zeros(n, dtype=int)
    num_conflict = np.zeros(n, dtype=int)

    for i in range(n):
        name_jaro[i]  = fuzz.partial_ratio(s1_name[i], c_name[i]) / 100.0
        name_ratio[i] = fuzz.ratio(s1_name[i], c_name[i]) / 100.0
        addr_jaro[i]  = fuzz.partial_ratio(s1_addr[i], c_addr[i]) / 100.0
        addr_ratio[i] = fuzz.ratio(s1_addr[i], c_addr[i]) / 100.0
        name_jac[i]   = _jaccard(s1_ntok[i], c_ntok[i])
        addr_jac[i]   = _jaccard(s1_atok[i], c_atok[i])

        n1 = set(s1_num[i]) if isinstance(s1_num[i], list) else set()
        n2 = set(c_num[i])  if isinstance(c_num[i],  list) else set()
        intersect = len(n1 & n2)
        num_overlap[i]  = intersect
        num_conflict[i] = 1 if n1 and n2 and intersect == 0 else 0

    return pd.DataFrame({
        'retrieval_score': pairs_df['retrieval_score'].values,
        'name_jaro':   name_jaro,
        'name_ratio':  name_ratio,
        'name_jaccard': name_jac,
        'addr_jaro':   addr_jaro,
        'addr_ratio':  addr_ratio,
        'addr_jaccard': addr_jac,
        'num_overlap':  num_overlap,
        'num_conflict': num_conflict,
    })


# ═══════════════════════════════════════════════════════════
#  CPU PRODUCER & GPU CONSUMER
# ═══════════════════════════════════════════════════════════

def cpu_producer(s1_ids_list, candidates, s1_idx, s23_idx, lgbm_model, q_out):
    try:
        for batch_start in range(0, len(s1_ids_list), PROC_BATCH_SIZE):
            batch_ids = s1_ids_list[batch_start:batch_start + PROC_BATCH_SIZE]

            rows = []
            for s1_id in batch_ids:
                cids = candidates.get(s1_id, [])
                for c_id in cids:
                    if c_id in s23_idx.index:
                        rows.append({'s1_id': s1_id, 'c_id': c_id, 'retrieval_score': 1.0})

            if not rows:
                q_out.put(('SINGLETONS', batch_ids))
                continue

            pairs_df = pd.DataFrame(rows)
            feat_df  = compute_features_batch(pairs_df, s1_idx, s23_idx)
            lgbm_scores = lgbm_model.predict(feat_df[FEATURE_COLS])
            pairs_df['lgbm_score'] = lgbm_scores

            pairs_df = (
                pairs_df
                .sort_values('lgbm_score', ascending=False)
                .groupby('s1_id', sort=False)
                .head(TOP_K_LGBM)
                .reset_index(drop=True)
            )

            q_out.put(('BATCH', pairs_df))

    except Exception as e:
        import traceback
        q_out.put(('ERROR', f"{e}\n{traceback.format_exc()}"))
    finally:
        q_out.put(_DONE)


def gpu_consumer(cross_encoder, s1_idx, s23_idx, q_in, output_file,
                 all_s1_ids, mode='full'):
    matched   = {}
    n_pairs_scored = 0
    n_errors  = 0

    pbar = tqdm(total=len(all_s1_ids), desc="  Entities processed",
                unit="S1", mininterval=5, leave=True)

    try:
        while True:
            item = q_in.get()
            if item is _DONE:
                break

            tag, payload = item

            if tag == 'ERROR':
                print(f"\n[CPU-ERROR] {payload}")
                n_errors += 1
                continue

            if tag == 'SINGLETONS':
                for s1_id in payload:
                    matched[s1_id] = []
                pbar.update(len(payload))
                continue

            pairs_df = payload
            s1_batch_ids = pairs_df['s1_id'].unique()

            s1_names = s1_idx['name_basic'].reindex(pairs_df['s1_id']).fillna('').values
            s1_addrs = s1_idx['address_basic'].reindex(pairs_df['s1_id']).fillna('').values
            c_names  = s23_idx['name_basic'].reindex(pairs_df['c_id']).fillna('').values
            c_addrs  = s23_idx['address_basic'].reindex(pairs_df['c_id']).fillna('').values

            text_pairs = [
                [f"{n} {a}".strip(), f"{cn} {ca}".strip()]
                for n, a, cn, ca in zip(s1_names, s1_addrs, c_names, c_addrs)
            ]

            ce_logits = cross_encoder.predict(
                text_pairs,
                batch_size=CE_BATCH_SIZE,
                show_progress_bar=False
            )
            if isinstance(ce_logits, (int, float, np.floating)):
                ce_logits = [ce_logits]
            ce_logits = np.array(ce_logits)

            lgbm_norm = pairs_df['lgbm_score'].values
            fused     = 0.3 * lgbm_norm + 0.7 * ce_logits
            pairs_df  = pairs_df.copy()
            pairs_df['ce_logit'] = ce_logits
            pairs_df['fused']    = fused

            hits = pairs_df[pairs_df['ce_logit'] >= CE_THRESHOLD]

            for s1_id in s1_batch_ids:
                s1_hits = hits[hits['s1_id'] == s1_id]['c_id'].tolist()
                matched[s1_id] = s1_hits

            n_pairs_scored += len(pairs_df)
            pbar.update(len(s1_batch_ids))

    finally:
        pbar.close()

    print(f"\n[WRITE] Writing {len(all_s1_ids):,} rows to {output_file}...")
    n_with_match = 0
    file_mode = 'a' if mode == 'full' else 'w'
    with open(output_file, file_mode, encoding='utf-8') as f:
        if file_mode == 'w':
            f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in tqdm(all_s1_ids, desc="  Writing", unit="row", mininterval=5):
            hits = matched.get(s1_id, [])
            if hits:
                n_with_match += 1
            f.write(f"{s1_id}\t{','.join(hits)}\n")

    print("\n" + "=" * 50)
    print("  PIPELINE METRICS")
    print("=" * 50)
    print(f"  Total S1 entities     : {len(all_s1_ids):,}")
    print(f"  Entities with match   : {n_with_match:,}  ({n_with_match/max(1,len(all_s1_ids))*100:.1f}%)")
    print(f"  Singletons            : {len(all_s1_ids)-n_with_match:,}  ({(len(all_s1_ids)-n_with_match)/max(1,len(all_s1_ids))*100:.1f}%)")
    print(f"  Total CE pairs scored : {n_pairs_scored:,}")
    print(f"  CPU errors            : {n_errors}")
    print("=" * 50)

    return matched


# ═══════════════════════════════════════════════════════════
#  VALIDATION & MAIN
# ═══════════════════════════════════════════════════════════

def validate_output(output_file, all_s1_ids):
    print(f"\n[VALIDATE] Checking {output_file}...")
    errors = []

    if not os.path.exists(output_file):
        errors.append("Output file does not exist!")
        return errors

    df = pd.read_csv(output_file, sep='\t', dtype=str).fillna('')
    expected_cols = ['source1_entity_id', 'matched_entity_ids']
    if list(df.columns) != expected_cols:
        errors.append(f"Wrong columns: {list(df.columns)}")

    if len(df) != len(all_s1_ids):
        errors.append(f"Row count mismatch: got {len(df):,}, expected {len(all_s1_ids):,}")

    dupes = df['source1_entity_id'].duplicated().sum()
    if dupes:
        errors.append(f"Found {dupes:,} duplicate source1_entity_id values!")

    if errors:
        print(f"\n  [FAIL] {len(errors)} validation error(s):")
        for e in errors:
            print(f"    ERROR: {e}")
    else:
        print(f"\n  [PASS] All validation checks passed! Total rows: {len(df):,}")

    return errors


def main():
    global CE_THRESHOLD, TOP_K_LGBM, CE_BATCH_SIZE

    parser = argparse.ArgumentParser(description='B2 Kaggle Execution Pipeline')
    parser.add_argument('--mode', choices=['validate', 'full'], default='full')
    parser.add_argument('--n-val', type=int, default=50_000)
    parser.add_argument('--ce-threshold', type=float, default=CE_THRESHOLD)
    parser.add_argument('--top-k', type=int, default=TOP_K_LGBM)
    parser.add_argument('--batch-size', type=int, default=CE_BATCH_SIZE)
    args = parser.parse_args()

    CE_THRESHOLD = args.ce_threshold
    TOP_K_LGBM   = args.top_k
    CE_BATCH_SIZE = args.batch_size
    is_validate  = (args.mode == 'validate')

    print("=" * 60)
    print("  TriMatch-ER B2 -- Kaggle High-Performance Pipeline")
    print(f"  Mode       : {'VALIDATION (' + str(args.n_val) + ')' if is_validate else 'FULL RUN (1.73M)'}")
    print(f"  Top-K LGBM : {TOP_K_LGBM}")
    print(f"  CE Threshold: {CE_THRESHOLD}")
    print(f"  CE Batch   : {CE_BATCH_SIZE}")
    print("=" * 60)

    # Auto-resolve file locations
    try:
        candidates_file = find_file('candidate_pairs_b1.tsv')
    except FileNotFoundError:
        candidates_file = find_file('candidate_pairs.tsv')

    s1_norm_file    = find_file('test_source1_norm.tsv')
    s2_norm_file    = find_file('test_source2_norm.tsv')
    s3_norm_file    = find_file('test_source3_norm.tsv')
    lgbm_model_file = find_file('lgbm_matcher.pkl')

    out_dir = '/kaggle/working' if os.path.exists('/kaggle/working') else '.'
    output_file = os.path.join(out_dir, 'matching_results.tsv')
    zip_path    = os.path.join(out_dir, 'submission_b2.zip')

    print(f"\n[MODEL] Loading LightGBM from {lgbm_model_file}...")
    lgbm_model = joblib.load(lgbm_model_file)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[MODEL] Loading Cross-Encoder on {device.upper()}...")
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=device)

    if is_validate:
        candidates = load_candidates(candidates_file, max_entities=args.n_val)
        all_s1_ids = list(candidates.keys())
        needed_s23 = set(cid for cids in candidates.values() for cid in cids)
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
                                       all_s1_ids, 'validate'),
                                 daemon=True)
        t0 = time.time()
        t_gpu.start(); t_cpu.start(); t_cpu.join(); t_gpu.join()
        print(f"\n[TIMING] Validation run: {(time.time()-t0)/60:.1f} min")
        validate_output(output_file, all_s1_ids)
        return

    # FULL MODE: Partitioned by country
    print("\n[PARTITION] Scanning S1 countries...")
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

    all_s1_ids_ordered = [eid for c in countries for eid in country_s1_ids[c]]

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")

    t_total = time.time()
    for ci, country in enumerate(countries, 1):
        s1_ids_country = country_s1_ids[country]
        s1_id_set      = set(s1_ids_country)

        print(f"\n{'='*55}")
        print(f"  PARTITION {ci}/{len(countries)}: {country.upper()}  ({len(s1_ids_country):,} entities)")
        print(f"{'='*55}")

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
        for s1_id in s1_ids_country:
            candidates.setdefault(s1_id, [])

        s1_idx  = load_norm_data(s1_norm_file, needed_ids=s1_id_set, desc=f'S1 ({country})')
        s2_idx  = load_norm_data(s2_norm_file, needed_ids=needed_s23, desc='S2')
        s3_idx  = load_norm_data(s3_norm_file, needed_ids=needed_s23, desc='S3')
        s23_idx = pd.concat([s2_idx, s3_idx])
        del s2_idx, s3_idx, needed_s23; gc.collect()

        q     = queue.Queue(maxsize=Q_MAXSIZE)
        t_cpu = threading.Thread(target=cpu_producer,
                                 args=(s1_ids_country, candidates, s1_idx, s23_idx, lgbm_model, q),
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
    print(f"\n[TIMING] Total run time: {total_elapsed/60:.1f} min")

    errors = validate_output(output_file, all_s1_ids_ordered)

    if not errors:
        print(f"\n[ZIP] Creating Kaggle submission archive: {zip_path}...")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.write(output_file, arcname='matching_results.tsv')
        print(f"    Zip Size: {os.path.getsize(zip_path)/1e6:.1f} MB")
        print("\n" + "=" * 60)
        print(f"  [SUCCESS] B2 Kaggle Submission created at: {zip_path}")
        print("=" * 60)


if __name__ == '__main__':
    main()
