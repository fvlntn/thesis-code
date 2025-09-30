import os
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import torch
from monai.transforms import AddChannel, Compose, RandRotate, RandFlip, Resize, EnsureType
from monai.transforms import apply_transform, LoadImage, Randomizable
from monai.utils import MAX_SEED, get_seed
from torch.utils.data import DataLoader, Dataset

from utils import get_batch_size


def getQTransforms(augment=False):
    if augment:
        print('=> Using augmented transforms for train set')
        train_trans = Compose(
            [
                AddChannel(),
                Resize((128, 128, 128)),
                RandRotate(180, 180, 180, prob=0.8),
                RandFlip(prob=0.5, spatial_axis=(0, 1)),
                EnsureType(),
            ]
        )
    else:
        train_trans = Compose(
            [
                AddChannel(),
                Resize((128, 128, 128)),
                EnsureType(),
            ]
        )
    val_trans = Compose(
        [
            AddChannel(),
            Resize((128, 128, 128)),
            EnsureType()
        ]
    )
    return train_trans, val_trans


class MaskDataset(Dataset, Randomizable):
    def __init__(
            self,
            csv,
            transform: Optional[Callable] = None,
    ) -> None:
        self.csv = csv
        self.transform = transform
        self.set_random_state(seed=get_seed())
        self._seed = 0

    def __len__(self) -> int:
        return len(self.csv)

    def randomize(self, data: Optional[Any] = None) -> None:
        self._seed = self.R.randint(MAX_SEED, dtype="uint32")

    def __getitem__(self, index: int):
        self.randomize()
        img_loader = LoadImage(
            reader="NibabelReader",
            image_only=True,
            dtype=np.float32,
            as_closest_canonical=True,
        )

        img = img_loader(self.csv['MaskFilename'][index])
        img = img.squeeze()

        if self.transform is not None:
            if isinstance(self.transform, Randomizable):
                self.transform.set_random_state(seed=self._seed)
            img = apply_transform(self.transform, img)

        truth = img_loader(self.csv['TruthFilename'][index])
        truth = truth.squeeze()

        if self.transform is not None:
            if isinstance(self.transform, Randomizable):
                self.transform.set_random_state(seed=self._seed)
            truth = apply_transform(self.transform, truth)

        data = [img, self.csv['Dice'][index], self.csv['Sensitivity'][index], self.csv['Specificity'][index], truth,
                self.csv['MaskFilename'][index], self.csv['TruthFilename'][index]]

        return tuple(data)


def getMaskDataset(batch=1, augment=False, training=True, data_dir=None):
    # MaskDataset = seuillage sur IRM
    # MaskDatasetDeux = seuillage sur IRM + masques UNet
    # MaskDataset3 = seuillage sur IRM dans le masque
    # MaskDataset4 = seulement les masques UNet
    # MaskDataset5 = seuillage sur UNet
    # MaskDataset6 = seuillage sur UNet seulement si dice > 50%

    if data_dir is None:
        data_dir = os.path.join(os.getcwd(), 'dataset', 'QNet', 'MaskDataset6')
    print('=> Using 6th Mask dataset')

    csv = os.path.join(data_dir, "results.csv")
    df = pd.read_csv(csv, sep=';')
    df = df.iloc[np.random.permutation(len(df))]
    df = df.reset_index(drop=True)
    for array in np.where(df['Dice'] == 'Dice'):
        df = df.drop(array)
    df['Dice'] = df['Dice'].astype(float)
    df['Sensitivity'] = df['Sensitivity'].astype(float)
    df['Specificity'] = df['Specificity'].astype(float)

    train_trans, val_trans = getQTransforms(augment)

    batch_size = get_batch_size(batch, training)

    train_ds = MaskDataset(df[:1536].reset_index(drop=True), transform=train_trans)
    val_ds = MaskDataset(df[1536:][:514].reset_index(drop=True), transform=val_trans)
    test_ds = MaskDataset(df[2050:].reset_index(drop=True), transform=val_trans)

    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=4, shuffle=True,
                              pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=4, shuffle=True,
                            pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=1, pin_memory=torch.cuda.is_available())

    dataloader = {'train': train_loader, 'valid': val_loader, 'test': test_loader}
    size = {'train': len(train_ds), 'valid': len(val_ds), 'test': len(test_ds)}

    return dataloader, size
