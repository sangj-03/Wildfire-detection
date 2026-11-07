"""Aggregates for the README's overview figures: histogram counts and shares only, no frame-level rows.

Reads the outputs of the data stage on the machine that has AI Hub 71330 (they are not in Git):
prep/manifest.csv.gz (notebooks/preprocess.ipynb), the label caches of notebooks/eda.ipynb, and the leakage
checks report/aihub/leak_pairs.csv (check_aihub_leak.py) and report/aihub/split_v2_check.csv (check_aihub_split.py).

    python figures/make_figure_data.py   →  figures/data/{leak,flame_size,season,dark_by_time,distractor_area}.csv
"""
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))  # the project folder (this file is in figures/)
OUT = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
SUB = {'불꽃': 'fire', '연기': 'smoke', '정상': 'negative'}
NEAR = 6   # thumbnail difference below which two frames count as near-identical (check_aihub_leak.py)


def leak():
    """Nearest training frame for val/test frames: official split (same video) vs split_v2 (same place)."""
    off = pd.read_csv(f'{ROOT}/report/aihub/leak_pairs.csv', usecols=['d_near']).d_near
    v2 = pd.read_csv(f'{ROOT}/report/aihub/split_v2_check.csv', usecols=['d']).d
    rows = []
    for name, d in (('official', off), ('place_date', v2)):
        finite = d[np.isfinite(d)]
        counts = np.histogram(finite.clip(upper=40), bins=np.arange(0, 42, 1))[0]
        for b, c in zip(range(41), counts):
            rows.append({'split': name, 'bin': b, 'count': int(c), 'n': len(d), 'n_compared': len(finite),
                         'n_near': int((finite < NEAR).sum())})
    return pd.DataFrame(rows)


def flame_size(m):
    """Short side of the largest flame box per fire frame: at the model input (frame width 1920, 4K halved) and if
    the frame were shrunk to 224 px on its short side (the usual ImageNet input), as in eda.ipynb section 8."""
    img = pd.read_parquet(f'{ROOT}/notebooks/cache/aihub_images.parquet', columns=['idx', 'split', 'path', 'w', 'h'])
    img['member'] = img.path.str.replace(r'\.json$', '.jpg', regex=True)
    fire = m[m['sub'] == '불꽃'][['src', 'member']].merge(img, left_on=['src', 'member'], right_on=['split', 'member'])
    ann = pd.read_parquet(f'{ROOT}/notebooks/cache/aihub_ann.parquet', columns=['idx', 'cat', 'bw', 'bh'])
    ann = ann[(ann.cat == 3) & ann.idx.isin(fire.idx)]   # 3 = 화염 (flame)
    short = ann.assign(s=ann[['bw', 'bh']].min(axis=1)).groupby('idx').s.max()
    f = fire.set_index('idx').join(short, how='inner')
    sizes = {'model_input': f.s * 1920 / f.w, 'short_224': f.s * 224 / f[['w', 'h']].min(axis=1)}
    edges = np.logspace(0, 3, 46)   # 1 px to 1,000 px
    rows = []
    for name, v in sizes.items():
        counts = np.histogram(v.clip(1, 999), bins=edges)[0]
        for lo, hi, c in zip(edges[:-1], edges[1:], counts):
            rows.append({'size': name, 'lo': lo, 'hi': hi, 'count': int(c), 'n': len(v),
                         'median': round(float(v.median()), 1), 'pct_under_8': round(100 * float((v < 8).mean()), 1)})
    return pd.DataFrame(rows)


def season(m):
    """Share of each class's frames by month."""
    t = pd.crosstab(m.month, m['sub'], normalize='columns') * 100
    return t.rename(columns=SUB).round(2).rename_axis('month').reset_index()


def dark_by_time(m):
    """Share of frames that are actually dark (mean brightness < 50) by the time-of-day label."""
    g = m.groupby(['timezone', 'sub']).dark.agg(['mean', 'size']).reset_index()
    g['dark_pct'] = (100 * g['mean']).round(2)
    g['sub'] = g['sub'].map(SUB)
    g['timezone'] = g.timezone.map({'오전': 'morning', '정오': 'noon', '오후': 'afternoon', '야간': 'night'})
    return g[['timezone', 'sub', 'dark_pct', 'size']].rename(columns={'size': 'frames'})


def distractor_area(m):
    """Area share of each distractor in the cropped frame, over the negatives that hold it."""
    neg = m[m.y == 0]
    rows = []
    for col, name in (('crop_cloud', 'cloud'), ('crop_fog', 'fog_haze'), ('crop_chimney', 'chimney_smoke')):
        v = neg.loc[neg[col] > 0, col] * 100
        counts = np.histogram(v, bins=np.arange(0, 102.5, 2.5))[0]
        for lo, c in zip(np.arange(0, 100, 2.5), counts):
            rows.append({'distractor': name, 'lo': lo, 'count': int(c), 'n': len(v),
                         'pct_ge_20': round(100 * float((v >= 20).mean()), 1)})
    return pd.DataFrame(rows)


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    m = pd.read_csv(f'{ROOT}/prep/manifest.csv.gz', dtype={'date': str, 'area_bin': str, 'combo': str},
                    usecols=['split', 'y', 'sub', 'src', 'member', 'month', 'timezone', 'dark',
                             'crop_cloud', 'crop_fog', 'crop_chimney'])
    for name, df in (('leak', leak()), ('flame_size', flame_size(m)), ('season', season(m)),
                     ('dark_by_time', dark_by_time(m)), ('distractor_area', distractor_area(m))):
        df.to_csv(f'{OUT}/{name}.csv', index=False)
        print(name, len(df), 'rows')
