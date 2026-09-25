import pandas as pd
import os
import gc
from tqdm import tqdm
from normalization import normalize_dataframe

def process_and_save_dataset(input_path, output_path, chunksize=100000):
    """
    Reads a raw TSV dataset, normalizes it, and saves it to a clean TSV.
    This saves the rest of the team from having to re-run preprocessing every time!
    """
    if not os.path.exists(input_path):
        print(f"Skipping {input_path} (File not found)")
        return
        
    print(f"\nProcessing {input_path} -> {output_path}")
    
    # If the file is small enough, do it in memory
    file_size_mb = os.path.getsize(input_path) / (1024 * 1024)
    
    if file_size_mb < 500: # Small files like source1
        df = pd.read_csv(input_path, sep='\t', dtype=str).fillna('')
        df_norm = normalize_dataframe(df)
        df_norm.to_csv(output_path, sep='\t', index=False)
        print(f"Saved {len(df_norm)} normalized rows.")
    else:
        # Large files like source2/3 need streaming
        chunk_iter = pd.read_csv(input_path, sep='\t', dtype=str, chunksize=chunksize)
        
        first_chunk = True
        total_rows = 0
        for chunk in tqdm(chunk_iter, desc="Streaming and Normalizing"):
            chunk = chunk.fillna('')
            chunk_norm = normalize_dataframe(chunk)
            
            mode = 'w' if first_chunk else 'a'
            header = first_chunk
            
            # Convert lists to string so they save properly in CSV
            for col in chunk_norm.columns:
                if chunk_norm[col].apply(type).eq(list).any():
                    chunk_norm[col] = chunk_norm[col].apply(lambda x: ','.join(x) if isinstance(x, list) else x)
            
            chunk_norm.to_csv(output_path, sep='\t', index=False, mode=mode, header=header)
            
            total_rows += len(chunk_norm)
            first_chunk = False
            del chunk, chunk_norm
            gc.collect()
            
        print(f"Saved {total_rows} normalized rows.")

def main():
    print("=== TEAM DATA PREPROCESSING PIPELINE ===")
    print("Normalizing all datasets ONE TIME and saving to DATA/processed/")
    
    os.makedirs('DATA/processed/train', exist_ok=True)
    os.makedirs('DATA/processed/test', exist_ok=True)
    
    # Train datasets
    process_and_save_dataset('DATA/student_resource/dataset/train/train_source1.tsv', 'DATA/processed/train/train_source1_norm.tsv')
    process_and_save_dataset('DATA/student_resource/dataset/train/train_source2.tsv', 'DATA/processed/train/train_source2_norm.tsv')
    process_and_save_dataset('DATA/student_resource/dataset/train/train_source3.tsv', 'DATA/processed/train/train_source3_norm.tsv')
    
    # Test datasets
    process_and_save_dataset('DATA/student_resource/dataset/test/test_source1.tsv', 'DATA/processed/test/test_source1_norm.tsv')
    process_and_save_dataset('DATA/student_resource/dataset/test/test_source2.tsv', 'DATA/processed/test/test_source2_norm.tsv')
    process_and_save_dataset('DATA/student_resource/dataset/test/test_source3.tsv', 'DATA/processed/test/test_source3_norm.tsv')
    
    print("\n✅ All datasets normalized and saved! Team can now load them instantly.")

if __name__ == '__main__':
    main()
