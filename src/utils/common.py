import os
import collections
import random
import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def how_many_nas(df):
    ctr = collections.Counter()
    for col in df.columns:
        ctr.update({col: df[col].isnull().sum()})
    print(f"Total columns: {df.shape[0]}")
    for el in ctr.most_common():
        print(el)


def threshed_sigmoid(logits, threshold=0.5):
    return torch.where(
        torch.sigmoid(logits) > threshold,
        torch.ones_like(logits),
        torch.zeros_like(logits)
    )


def fetch_best_model_filename(model_save_path):
    checkpoint_files = os.listdir(model_save_path)
    best_checkpoint_files = [f for f in checkpoint_files if 'best_' in f]
    best_checkpoint_val_loss = [
        float(".".join(x.split('=')[1].split('.')[0:2]))
        for x in best_checkpoint_files]
    best_idx = np.array(best_checkpoint_val_loss).argmax()
    return os.path.join(model_save_path, best_checkpoint_files[best_idx])
