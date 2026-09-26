# TriMatch-ER B2 — Kaggle Setup & Run Guide

Everything needed to run the B2 pipeline on Kaggle GPU (T4 / P100 / L4) is prepared inside this folder: `kaggle_b2_bundle`.

---

## Direct Dataset Upload Instructions

**YES, you can upload `kaggle_b2_bundle` directly as a Kaggle Dataset!**

### Step 1: Create Dataset on Kaggle
1. Open Kaggle -> Click **+ Create** -> **New Dataset**.
2. Drag and drop the **`kaggle_b2_bundle`** folder.
3. Also drag and drop the candidate pairs file & test dataset files into the same upload window:
   - `output/candidate_pairs_b1.tsv`
   - `DATA/processed/test/test_source1_norm.tsv`
   - `DATA/processed/test/test_source2_norm.tsv`
   - `DATA/processed/test/test_source3_norm.tsv`
4. Name your dataset (e.g. `amazon-ml-b2-dataset`) and click **Create**.

---

### Step 2: Create Kaggle Notebook & Enable GPU
1. Go to **Code -> New Notebook**.
2. Under **Settings** (right sidebar):
   - **Accelerator**: Select **GPU T4 x2** or **GPU P100**.
   - **Internet**: Set to **ON** (required to auto-download Cross-Encoder weights).
3. Under **Input** (right sidebar), click **+ Add Input**, search for your dataset `amazon-ml-b2-dataset`, and click **Add**.

---

### Step 3: Run the Notebook / Script
In cell 1 of your notebook:
```python
!pip install -q rapidfuzz sentence-transformers lightgbm joblib tqdm
import glob, shutil
script = glob.glob('/kaggle/input/**/generate_submission_b2_kaggle.py', recursive=True)[0]
shutil.copy(script, './generate_submission_b2_kaggle.py')
```

In cell 2 of your notebook:
```bash
!python generate_submission_b2_kaggle.py --mode full
```

---

## Expected Output
When completed, the output zip file will automatically be saved at:
`/kaggle/working/submission_b2.zip`

You can download `submission_b2.zip` directly from the Kaggle Notebook output panel!
