import os
import pandas as pd
from huggingface_hub import login, snapshot_download

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(BASE, "data", "hest1k")

token = os.environ.get("HF_TOKEN")
if not token:
    token_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".hf_token")
    if os.path.exists(token_file):
        token = open(token_file, encoding="utf-8").read().strip()
if token:
    login(token=token, add_to_git_credential=False)

meta_df = pd.read_csv("hf://datasets/MahmoodLab/hest/HEST_v1_1_0.csv")

SUBSETS = {
    "her2st": (
        "Spatial deconvolution of",
        os.path.join(DATA_ROOT, "her2st"),
    ),
    "kidney": (
        "Spatial localization with Spatial Transcriptomics for an atlas of healthy and injured cell states",
        os.path.join(DATA_ROOT, "kidney"),
    ),
}

for name, (prefix, local_dir) in SUBSETS.items():
    ids = meta_df.loc[meta_df["dataset_title"].str.startswith(prefix), "id"].tolist()
    print(f"[{name}] {len(ids)} samples: {ids}")
    os.makedirs(local_dir, exist_ok=True)
    patterns = []
    for i in ids:
        patterns += [f"wsis/{i}*", f"st/{i}*"]
    snapshot_download(
        repo_id="MahmoodLab/hest",
        repo_type="dataset",
        allow_patterns=patterns,
        local_dir=local_dir,
    )
    print(f"[{name}] done -> {local_dir}")
