import numpy as np
import pandas as pd
import torch
import anndata
import os


class HER2STDataset(torch.utils.data.Dataset):
    def __init__(self, path='./data/hest1k/her2st/', mode='train', gene_list='selected_gene_list_300.txt',
                 patch_size=224, tiles_size=56, filter_zero_spots=True):
        self.processed_path = os.path.join(path, 'processed_data')
        self.gene_list = pd.read_csv(os.path.join(self.processed_path, gene_list), header=None).values.flatten().tolist()
        self.slide_list = pd.read_csv(os.path.join(self.processed_path, f'{mode}_slide_list.txt'), header=None).values.flatten().tolist()

        self.patch_embed_list = torch.tensor([])
        self.tiles_embed_list = torch.tensor([])
        self.gene_exp_list = torch.tensor([])

        for slide in self.slide_list:
            patch_ebd_path = os.path.join(self.processed_path, '1spot_ebd', f'{slide}.pt')
            tiles_ebd_path = os.path.join(self.processed_path, f'1spot_tiles_ebd_{tiles_size}_3x3', f'{slide}.pt')

            if not (os.path.exists(patch_ebd_path) and os.path.exists(tiles_ebd_path)):
                raise FileNotFoundError(f"Missing data for slide {slide}")

            patch_embed = torch.load(patch_ebd_path)
            tiles_embed = torch.load(tiles_ebd_path)

            gene_exp_path = os.path.join(path, 'st', f'{slide}.h5ad')
            if not os.path.exists(gene_exp_path):
                raise FileNotFoundError(f"Missing gene expression data for slide {slide}")

            gene_exp = anndata.read_h5ad(gene_exp_path)
            selected_gene_exp = gene_exp[:, self.gene_list].copy()

            if filter_zero_spots:
                zero_spot_idx = selected_gene_exp.X.sum(axis=1) == 0
                zero_spot_idx = np.asarray(zero_spot_idx).flatten()
                not_zero_spot_idx = ~zero_spot_idx
                selected_gene_exp = selected_gene_exp[not_zero_spot_idx, :].copy()
                patch_embed = patch_embed[not_zero_spot_idx]
                tiles_embed = tiles_embed[not_zero_spot_idx]

            gene_exp_df = pd.DataFrame(
                selected_gene_exp.X,
                columns=self.gene_list,
                index=[f"{slide}_{i}" for i in range(len(selected_gene_exp))]
            ).T
            gene_exp_df = gene_exp_df.astype(float)
            gene_exp_df = np.log2(gene_exp_df + 1)

            gene_exp_tensor = torch.tensor(gene_exp_df.T.values, dtype=torch.float32)

            self.gene_exp_list = torch.cat((self.gene_exp_list, gene_exp_tensor), dim=0)
            self.patch_embed_list = torch.cat((self.patch_embed_list, patch_embed), dim=0)
            self.tiles_embed_list = torch.cat((self.tiles_embed_list, tiles_embed), dim=0)
            print(f"Processed slide {slide}: patch_embed {patch_embed.shape}, tiles_embed {tiles_embed.shape}, gene_exp {gene_exp_tensor.shape}")

    def __len__(self):
        return len(self.patch_embed_list)

    def __getitem__(self, index):
        return self.patch_embed_list[index], self.tiles_embed_list[index], self.gene_exp_list[index]
