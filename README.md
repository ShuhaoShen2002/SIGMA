# SIGMA: Spatial Gene Expression Prediction from H&E Images

SIGMA is a deep learning method for predicting spatial gene expression from H&E-stained histopathology images. The model uses UNI2-h foundation-model embeddings to predict the expression of 300 selected genes at each spatial transcriptomics spot.

All commands in this README are executed from the `SIGMA/` directory (the scripts and notebooks rely on relative paths).

## File Organization

```
SIGMA/
├── README.md                     # This file
├── requirements.txt              # Python dependencies
├── main_her2st.py                # SIGMA training on HER2ST (SPA119–SPA154)
├── main_kidney.py                # SIGMA training on Kidney (NCBI692–NCBI714)
├── inference.py                  # Load a checkpoint and evaluate on the HER2ST test slide
├── model/                        # Model implementation
│   ├── __init__.py
│   ├── model.py                  # SIGMA: tile transformer + gene-wise attention pooling + patch-context conditioning + MLP head
│   └── vit.py                    # Reference ViT implementation (not used by the default SIGMA model)
├── dataset/                      # Dataset loaders
│   ├── __init__.py
│   ├── HER2STDataset.py          # Loads preprocessed HER2ST embeddings + expression labels
│   └── KidneyDataset.py          # Loads preprocessed Kidney embeddings + expression labels
├── utils/                        # Data download & preprocessing
│   ├── download_hest1k.py        # Downloads the HER2ST/Kidney subsets of HEST-1k from Hugging Face
│   ├── preprocess_her2st.ipynb   # HER2ST: gene selection + UNI2-h patch/tile embedding extraction
│   └── preprocess_kidney.ipynb   # Kidney: gene selection + UNI2-h patch/tile embedding extraction
├── logs/                         # Training/eval logs (generated, git-ignored)
└── data/hest1k/                  # Raw + preprocessed data (git-ignored except the .txt lists)
```

## Environment Setup

1. **Create a conda environment** (recommended):
   ```bash
   conda create -n sigma python=3.10
   conda activate sigma
   ```

2. **Install PyTorch with CUDA support** (pick the command matching your CUDA version at [pytorch.org](https://pytorch.org), e.g. for CUDA 12.1):
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```
   Note: `torch` in `requirements.txt` is a fallback. Training and embedding extraction require a GPU, so installing the CUDA build first is strongly recommended.

3. **Install the remaining dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install the UNI2-h encoder** (only needed for data preprocessing):
   ```bash
   pip install "uni @ git+https://github.com/mahmoodlab/UNI.git"
   ```
   Then download the UNI2-h weights from Hugging Face (`mahmoodlab/UNI2-h`) and point `assets_dir` in the preprocessing notebooks to the checkpoint directory.

## Data

Datasets are downloaded from **HEST-1k** ([mahmoodlab/hest](https://github.com/mahmoodlab/hest), hosted on Hugging Face as `MahmoodLab/hest`). The HER2ST (breast cancer) and Kidney subsets are organized as:

```
data/hest1k/
├── her2st/
│   ├── wsis/                     # H&E whole-slide images (*.tif)
│   ├── st/                       # Spatial transcriptomics (*.h5ad)
│   └── processed_data/           # Gene/slide lists + preprocessed embeddings
└── kidney/
    ├── wsis/
    ├── st/
    └── processed_data/
```

The HER2ST subset spans slides **SPA119–SPA154** and the Kidney subset spans **NCBI692–NCBI714**; these are the slide IDs used by the preprocessing notebooks and the train/test lists.

### Downloading the data

1. Create a Hugging Face account and accept the terms of use of the [MahmoodLab/hest](https://huggingface.co/datasets/MahmoodLab/hest) dataset.
2. Provide your access token either as an environment variable or in a `utils/.hf_token` file:
   ```bash
   HF_TOKEN=hf_xxx python utils/download_hest1k.py
   ```
3. The script reads `HEST_v1_1_0.csv`, selects the slides whose `dataset_title` matches the HER2ST / Kidney subset names, and downloads only their `wsis/` and `st/` files into `data/hest1k/{her2st,kidney}/`.

## Data Preprocessing

Preprocessing converts the raw H&E images and `.h5ad` files into (a) the 300-gene list and (b) UNI2-h patch/tile embeddings for every spot. Both datasets have a dedicated notebook under `utils/` — run `preprocess_her2st.ipynb` and `preprocess_kidney.ipynb` in Jupyter, or execute them headlessly:

```bash
# optional: headless execution requires Jupyter
pip install jupyter

jupyter nbconvert --to notebook --inplace --execute utils/preprocess_her2st.ipynb
jupyter nbconvert --to notebook --inplace --execute utils/preprocess_kidney.ipynb
```

Before running, edit the configuration cell of each notebook:
- `device` — GPU ID (e.g., `"cuda:0"`)
- `assets_dir` — directory containing the UNI2-h checkpoint
- `data_path` — defaults to `../data/hest1k/{her2st,kidney}/`; keep the default when running from the `SIGMA/` root

Each notebook selects the top-300 genes by mean expression (after standard HVG analysis and removal of ribosomal/mitochondrial genes) and extracts UNI2-h embeddings for every spot — a 224×224 px patch embedding plus 16 tile embeddings (56×56 px with 3×3 context) — saving the results under `processed_data/`:

```
processed_data/
├── selected_gene_list_300.txt        # generated gene list
├── all_slide_list.txt                # generated slide list
├── 1spot_ebd/{slide}.pt              # patch embeddings, (n_spots, 1536)
└── 1spot_tiles_ebd_56_3x3/{slide}.pt # tile embeddings, (n_spots, 16, 1536)
```

This step requires a GPU and writes roughly **1.5 GB per slide** (slides SPA119–SPA154 for HER2ST, NCBI692–NCBI714 for Kidney).

At load time, the dataset classes keep only the 300 selected genes, drop spots with zero expression across all selected genes, apply `log2(x + 1)` normalization to the expression labels, and return `(patch_embed, tiles_embed, label)` per spot.

## Running the Model

Make sure the preprocessing step is complete (the dataset loaders raise `FileNotFoundError` if a slide's embeddings are missing).

```bash
# HER2ST dataset
python main_her2st.py

# Kidney dataset
python main_kidney.py
```

Each script trains SIGMA for 200 epochs with MSE loss, evaluates on the held-out test slide every epoch, and saves:
- `SIGMA_{HER2ST|Kidney}_best.pth` — best checkpoint (lowest test MSE)
- `SIGMA_{HER2ST|Kidney}_eval.pt` — test-slide predictions from the best epoch
- `logs/SIGMA_{HER2ST|Kidney}.log` — training/evaluation log

**Before running**, configure the following in each script:
- `device` — GPU ID
- `log_file` — output log filename
- `gene_list` — gene selection file (default `selected_gene_list_300.txt`)
- `BATCH_SIZE`, `LR`, `EPOCH`, `DECAY_RATE` — hyperparameters

### Inference only

```bash
python inference.py
```

`inference.py` loads `SIGMA_HER2ST_best.pth` (produced by `main_her2st.py`), evaluates on the HER2ST test slide, saves the predictions to `SIGMA_HER2ST_eval.pt`, and logs metrics to `logs/SIGMA_HER2ST_eval.log`. It is hardcoded to HER2ST — for the Kidney model, adapt the dataset class, checkpoint name, and log path accordingly (or simply use the per-epoch evaluation that `main_kidney.py` already performs).

## Model Configuration

Default hyperparameters:

| Parameter | Value |
|---|---|
| Epochs | 200 |
| Learning rate | 1e-4 |
| Batch size | 64 |
| Weight decay | 1e-5 |
| Dropout | 0.3 |
| Gene count | 300 |
| Embedding dimension | 1536 (UNI2-h) |
| MLP dimension | 1536 |
| Transformer depth | 3 |
| Attention heads | 8 |

## SIGMA Model Architecture

The model takes two inputs per spatial transcriptomics spot:
- **Patch embedding**: UNI2-h embedding of the 224×224 px patch — `(B, 1536)`
- **Tiles embedding**: UNI2-h embeddings of the 16 contextual tiles — `(B, 16, 1536)`

Architecture flow:
1. Add a learned position embedding to the 16 tiles, then encode them with a **TransformerEncoder** (self-attention over tiles).
2. **GeneWiseAttentionPooling** — per-gene attention weights over the encoded tiles, producing a `(B, 300, 1536)` representation.
3. **Patch-context conditioning** (implemented by `CrossEncoder`) — the patch embedding is projected into a context vector that is added to every gene token through a residual connection, then transformed by a position-wise feed-forward network.
4. Concatenate the pooled gene representation with the patch-conditioned one and apply the **MLP head** → `(B, 300)` per-gene expression predictions.

The model is trained with MSE loss on `log2(x+1)`-normalized expression values; the per-gene attention weights are also returned and can be used for interpretability.

## Evaluation Metrics

- **MSE** — Mean Squared Error
- **ASE (MAE)** — Average Absolute Error
- **RMSE** — Root Mean Squared Error
- **PCC-10** — Mean Pearson correlation of the 10 best-predicted genes
- **PCC-50** — Mean Pearson correlation of the 50 best-predicted genes
- **PCC-All** — Mean Pearson correlation across all predicted genes

The best checkpoint is selected by the lowest MSE on the held-out test slide.
