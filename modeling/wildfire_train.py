"""산불 탐지 프로젝트 모델링 단계 초안: 학습 때의 처리 (전처리 단계에서 옮겨 둔 코드, 튜닝 R3에서 N·가중치를 고른다).

영상당 최대 N장 추출(에폭마다 다시 뽑음), 한 에폭 장수 기준 클래스 가중치, 좌우 뒤집기·작은 확대/이동 증강, torch Dataset.
사진 처리 자체는 ../wildfire_data.py의 preprocess()를 그대로 쓴다.

    import wildfire_data as W, wildfire_train as WT
    m = W.load_manifest(); tr = m[m.split == 'train']
    sampler = WT.VideoSampler(tr, n_per_video=8)
    ds = WT.WildfireDataset(tr, train=True)
    loader = DataLoader(ds, batch_size=16, sampler=sampler, num_workers=12, persistent_workers=True)
    for epoch in range(E): sampler.set_epoch(epoch); ...
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from wildfire_data import IMAGENET_MEAN, IMAGENET_STD, ZipReader, preprocess  # noqa: E402

# ---------------------------------------------------------------- 학습 세트: 영상당 N장 추출, 클래스 가중치

def video_epoch_indices(df, n_per_video, epoch, seed=0):
    """영상(vid)·클래스(y)마다 최대 n장을 뽑은 행 위치(0..len-1). 에폭마다 다른 프레임을 뽑는다.
    n_per_video=None이면 전부."""
    pos = np.arange(len(df))
    if n_per_video is None:
        return pos
    rng = np.random.default_rng([seed, epoch])
    out = []
    for g in pd.Series(pos).groupby([df.vid.to_numpy(), df.y.to_numpy()]).indices.values():
        g = pos[g]
        out.append(g if len(g) <= n_per_video else rng.choice(g, n_per_video, replace=False))
    out = np.concatenate(out)
    rng.shuffle(out)
    return out


def expected_counts(df, n_per_video):
    """한 에폭에 뽑히는 클래스별 사진 수 (영상마다 min(프레임 수, n))."""
    sizes = df.groupby(['vid', 'y']).size()
    k = sizes if n_per_video is None else sizes.clip(upper=n_per_video)
    return k.groupby(level='y').sum()


def class_weights(counts):
    """전체 ÷ (클래스 수 × 클래스 장수). counts: y → 장수 (한 에폭에 실제로 보는 장수를 넣는다)."""
    counts = pd.Series(counts).sort_index()
    return (counts.sum() / (len(counts) * counts)).to_dict()


# ---------------------------------------------------------------- torch 부분 (학습 환경에서만 import)

def _torch():
    import torch
    import torchvision.transforms.v2 as T
    return torch, T


def make_transform(train, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """학습: 좌우 뒤집기 + 작은 확대(1.0~1.1)·이동(±2%). 밝기·대비·색은 평가용 손상과 겹쳐 기준선에 넣지 않는다."""
    torch, T = _torch()
    ops = [T.PILToTensor()]
    if train:
        ops += [T.RandomHorizontalFlip(), T.RandomAffine(degrees=0, translate=(0.02, 0.02), scale=(1.0, 1.1))]
    ops += [T.ToDtype(torch.float32, scale=True), T.Normalize(mean, std)]
    return T.Compose(ops)


def WildfireDataset(df, train=False, transform=None):
    """(사진 텐서 3×972×1920, y) 를 돌려주는 torch Dataset."""
    torch, _ = _torch()
    tf = transform or make_transform(train)
    rows = df[['src', 'member', 'y']].reset_index(drop=True)
    for s in rows.src.unique():
        ZipReader.open(s).close()             # 파일 목록은 작업 프로세스를 만들기 전에 읽어 둔다

    class _DS(torch.utils.data.Dataset):
        def __len__(self):
            return len(rows)

        def __getitem__(self, i):
            r = rows.iloc[i]
            return tf(preprocess(ZipReader.read(r.src, r.member))), int(r.y)

    return _DS()


def VideoSampler(df, n_per_video=8, seed=0):
    """에폭마다 영상당 최대 n장을 새로 뽑는 Sampler. 학습 루프에서 sampler.set_epoch(epoch)를 부른다."""
    torch, _ = _torch()

    class _S(torch.utils.data.Sampler):
        def __init__(self):
            self.epoch = 0
            self.n = len(video_epoch_indices(df, n_per_video, 0, seed))

        def set_epoch(self, e):
            self.epoch = e

        def __len__(self):
            return self.n

        def __iter__(self):
            return iter(video_epoch_indices(df, n_per_video, self.epoch, seed).tolist())

    return _S()
