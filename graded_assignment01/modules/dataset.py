import os
import torch
from torch.utils.data import Dataset
from skimage import io
import pandas as pd
import numpy as np

from utils import generate_phoc_vector, generate_phos_vector


class phosc_dataset(Dataset):
    """
    Minimal dataset that:
      - reads a CSV with columns ["Image","Word"]
      - builds PHOS (165-dim), PHOC (604-dim) and concatenated PHOSC (769-dim)
      - returns (image_tensor, phosc_vector_tensor, word_string)
    """
    def __init__(self, csvfile, root_dir, transform=None, calc_phosc=True):
        """
        Args:
            csvfile (str): path to CSV with at least columns: Image, Word
            root_dir (str): folder where images live
            transform: torchvision transform (e.g., transforms.ToTensor())
            calc_phosc (bool): if True, store the concatenated PHOSC vector
        """
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.calc_phosc = calc_phosc

        # Read CSV (only two columns are required)
        df = pd.read_csv(csvfile)
        if 'Image' not in df.columns or 'Word' not in df.columns:
            raise ValueError("CSV must contain 'Image' and 'Word' columns")

        # Build unique word list, then look up vectors once
        words = list(set(df['Word'].tolist()))
        # Generators are pure-Python/NumPy (no TF/PyTorch deps)
        phos_map = {w: np.asarray(generate_phos_vector(w), dtype=np.float32) for w in words}
        phoc_map = {w: np.asarray(generate_phoc_vector(w), dtype=np.float32) for w in words}

        # Attach vectors and full paths
        phos_vecs, phoc_vecs, phosc_vecs = [], [], []
        img_names = []
        for _, row in df.iterrows():
            w = row['Word']
            phos = phos_map[w]
            phoc = phoc_map[w]
            phos_vecs.append(phos)
            phoc_vecs.append(phoc)
            phosc_vecs.append(np.concatenate([phos, phoc], axis=0))
            img_names.append(row['Image'])

        self.df_all = pd.DataFrame({
            "Image": img_names,
            "Word": df['Word'].tolist(),
            "phos": phos_vecs,
            "phoc": phoc_vecs,
            "phosc": phosc_vecs
        })

    def __getitem__(self, index):
        # Load image (H,W,C) in [0..255]
        img_path = os.path.join(self.root_dir, self.df_all.iloc[index, 0])
        image = io.imread(img_path)

        # Choose what to feed as target: here we use concatenated PHOSC (165+604 = 769)
        y_np = self.df_all.iloc[index, self.df_all.columns.get_loc("phosc")]
        y = torch.tensor(y_np, dtype=torch.float32)

        # Apply transform -> (C,H,W) float in [0,1]
        if self.transform:
            image = self.transform(image)

        return image.float(), y.float(), self.df_all.iloc[index, 1]

    def __len__(self):
        return len(self.df_all)


if __name__ == '__main__':
    from torchvision.transforms import transforms
    dataset = phosc_dataset('image_data/IAM_test_unseen.csv', '../image_data/IAM_test',
                            transform=transforms.ToTensor())
    print(dataset.df_all.head())
    print(dataset.__getitem__(0))
