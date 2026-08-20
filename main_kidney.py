import os
import torch
import numpy as np
import random
import math
import time
from dataset.KidneyDataset import KidneyDataset
from model.model import SIGMA
import logging

# Hyperparameters
EPOCH = 200
LR = 1e-4
BATCH_SIZE = 64
DECAY_RATE = 1e-5

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda:0")

os.makedirs('logs', exist_ok=True)
log_file = os.path.join('logs', 'SIGMA_Kidney.log')

logging.basicConfig(
    filename=log_file,
    filemode='a',
    format='%(asctime)s %(levelname)s: %(message)s',
    level=logging.INFO
)
logger = logging.getLogger()

# Also print to console
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
console.setFormatter(formatter)
logger.addHandler(console)

model = SIGMA(dropout=0.3, gene_nums=300, emb_dim=1536, mlp_dim=1536).to(device)

trainset = KidneyDataset(mode='train', gene_list='selected_gene_list_300.txt', tiles_size=56)
testset = KidneyDataset(mode='test', gene_list='selected_gene_list_300.txt', tiles_size=56)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=DECAY_RATE)
loss_mse = torch.nn.MSELoss()


def test_model(valloader):
    time0 = time.time()
    model.eval()
    all_outputs = []
    all_labels = []
    all_corrs = []
    for data in valloader:
        patch_embed, tiles_embed, label = data
        patch_embed, tiles_embed, label = patch_embed.to(device), tiles_embed.to(device), label.to(device)
        with torch.no_grad():
            output, _ = model(patch_embed, tiles_embed)
        all_outputs.append(output.cpu())
        all_labels.append(label.cpu())
    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    mse = torch.mean((all_outputs - all_labels) ** 2).item()
    ase = torch.mean(torch.abs(all_outputs - all_labels)).item()
    rmse = math.sqrt(mse)
    for i in range(all_outputs.shape[1]):
        x = all_outputs[:, i].cpu().numpy()
        y = all_labels[:, i].cpu().numpy()
        cor = np.corrcoef(x, y)[0][1]
        all_corrs.append(cor)
    logger.info('MSE: {:.6f}'.format(mse))
    logger.info('ASE: {:.6f}'.format(ase))
    logger.info('RMSE: {:.6f}'.format(rmse))
    logger.info('PCC-10: {:.6f}'.format(np.mean(sorted(all_corrs)[::-1][:10])))
    logger.info('PCC-50: {:.6f}'.format(np.mean(sorted(all_corrs)[::-1][:50])))
    logger.info('PCC-All: {:.6f}'.format(np.mean(sorted(all_corrs)[::-1])))
    return rmse, all_outputs


best_rmse = float('inf')
for epoch in range(EPOCH):
    model.train()
    train_loss = 0.0
    for i, data in enumerate(trainloader):
        patch_embed, tiles_embed, label = data
        patch_embed, tiles_embed, label = patch_embed.to(device), tiles_embed.to(device), label.to(device)

        optimizer.zero_grad()
        output, _ = model(patch_embed, tiles_embed)
        loss = loss_mse(output, label)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if i % 20 == 0:
            logger.info('[TRAIN ITER {}/{}] Loss: {:.4f}'.format(i, len(trainloader), loss.item()))

    logger.info('EPOCH: {}'.format(epoch))
    rmse, all_outputs = test_model(testloader)
    if rmse < best_rmse:
        best_rmse = rmse
        logger.info('=========================>> New best RMSE: {:.6f}'.format(rmse))
        torch.save(model.state_dict(), 'SIGMA_Kidney_best.pth')
        torch.save(all_outputs, 'SIGMA_Kidney_eval.pt')
