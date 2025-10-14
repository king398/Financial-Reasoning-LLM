
import pandas as pd
from datasets import Dataset, concatenate_datasets
from joblib import Parallel, delayed
from tqdm import tqdm
import os
# === CONFIGURATION ===
CSV_PATH = "nasdaq_exteral_data.csv"
CHUNK_SIZE = 5000  # tune based on available RAM (start small)
N_JOBS = os.cpu_count()  # number of parallel workers (adjust to CPU cores)
HF_DATASET_REPO = "Mithilss/finance-data"


# === FUNCTION TO PROCESS ONE CHUNK ===
def process_chunk(chunk_df, idx):
    """Convert one pandas chunk to a Hugging Face Dataset."""
    try:
        # Convert everything to string to avoid Arrow type issues
        chunk_df = chunk_df.astype(str)
        ds = Dataset.from_pandas(chunk_df, preserve_index=False)
        return ds
    except Exception as e:
        print(f"Error in chunk {idx}: {e}")
        return None


# === READ AND PARALLELIZE ===
reader = pd.read_csv(CSV_PATH, chunksize=CHUNK_SIZE, low_memory=False)
chunks = list(reader)  # load all chunk DataFrames into a list of manageable size

# Parallel conversion
datasets_list = Parallel(n_jobs=N_JOBS, backend="loky", prefer="processes")(
    delayed(process_chunk)(chunk, i) for i, chunk in enumerate((chunks))
)

# Filter out failed chunks
datasets_list = [d for d in datasets_list if d is not None]

# === CONCATENATE AND PUSH ===
if len(datasets_list) == 0:
    raise RuntimeError("No chunks were successfully processed.")

big_dataset = concatenate_datasets(datasets_list)
big_dataset.push_to_hub(HF_DATASET_REPO, token=HF_TOKEN)
