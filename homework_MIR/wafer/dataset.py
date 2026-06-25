"""Datasets and augmentation for cached WM-811K wafer images."""
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2


def random_rotate_flip(img, rng):
    """img: (3,S,S) float32 in [0,1].  Arbitrary rotation + flips.

    Wafer failure patterns are invariant to rotation/reflection (a centre stays
    a centre, a ring stays a ring, a scratch stays a scratch), so this is a
    strong, label-preserving augmentation."""
    c, h, w = img.shape
    # arbitrary-angle rotation about the centre
    ang = rng.uniform(0, 360)
    M = cv2.getRotationMatrix2D((w / 2 - 0.5, h / 2 - 0.5), ang, 1.0)
    out = np.empty_like(img)
    for k in range(c):
        out[k] = cv2.warpAffine(img[k], M, (w, h),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if rng.random() < 0.5:
        out = out[:, :, ::-1]
    if rng.random() < 0.5:
        out = out[:, ::-1, :]
    return np.ascontiguousarray(out)


class WaferDataset(Dataset):
    def __init__(self, X, y=None, train=False, seed=0):
        self.X = X                      # (N,3,S,S) uint8
        self.y = y
        self.train = train
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        img = self.X[i].astype(np.float32) / 255.0
        if self.train:
            img = random_rotate_flip(img, self.rng)
        t = torch.from_numpy(img)
        if self.y is None:
            return t
        return t, int(self.y[i])


class TwoViewDataset(Dataset):
    """Returns (weak, strong) views of an unlabeled image for FixMatch-style
    pseudo-labeling.  Weak = light aug, strong = aug + cutout-style erasing."""
    def __init__(self, X, seed=0):
        self.X = X
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.X)

    def _erase(self, img):
        c, h, w = img.shape
        for _ in range(self.rng.integers(1, 3)):
            eh, ew = self.rng.integers(h // 8, h // 3), self.rng.integers(w // 8, w // 3)
            y0, x0 = self.rng.integers(0, h - eh), self.rng.integers(0, w - ew)
            img[:, y0:y0 + eh, x0:x0 + ew] = 0
        return img

    def __getitem__(self, i):
        base = self.X[i].astype(np.float32) / 255.0
        weak = random_rotate_flip(base, self.rng)
        strong = self._erase(random_rotate_flip(base, self.rng).copy())
        return torch.from_numpy(weak), torch.from_numpy(strong)
