"""같은 장소의 서로 다른 촬영일 묶음 사이에 거의 같은(또는 똑같은) 사진이 있는지 모든 쌍을 잰다.

split_v1 점검에서 날짜가 다른 묶음 사이에 썸네일 차이 0.0인 사진(같은 영상이 다른 날짜로 다시 들어간 것)이 나왔다.
묶음 A에서 최대 1,000장을 뽑아 묶음 B 전체와 비교하고(양방향), 차이 < 6인 사진 수를 센다.
사용: python3 -I aihub_group_links.py  →  report/aihub/group_links.csv (make_aihub_split.py가 읽어 묶음을 합친다)
"""
import os
import itertools, re

import numpy as np
import pandas as pd

OUT = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'report', 'aihub')
lab = pd.read_csv(f'{OUT}/labels.csv.gz', usecols=['path', 'split', 'db', 'use'])
meta = pd.read_csv(f'{OUT}/images.csv.gz', low_memory=False, usecols=['path', 'address', 'datetime'])
df = lab.merge(meta, on='path').rename(columns={'split': 'src'})
df = df[(df.db != 'PositiveSyntheticDB') & df.use].copy()


def norm_date(s):
    s = str(s)
    m = re.match(r'(20\d{6})_', s) or re.match(r'(\d{6})_', s)
    return m.group(1) if len(m.group(1)) == 8 else '20' + m.group(1)


df['group'] = df.address.str.replace('강원도강릉시', '강릉 ') + '/' + df.datetime.map(norm_date)
df['jpg'] = df.path.str.rsplit('/', n=1).str[-1].str.replace('.json', '.jpg')
th, key = [], []
for src in ('train', 'val'):
    img = pd.read_csv(f'{OUT}/img_{src}.csv.gz', usecols=['jpg'])
    t = np.load(f'{OUT}/img_{src}_thumbs.npy').reshape(len(img), -1).astype(np.float32)
    th.append((t - t.mean(1, keepdims=True)) / (t.std(1, keepdims=True) + 1) * 40); key.append(img.assign(src=src))
th = np.concatenate(th); key = pd.concat(key, ignore_index=True); key['k'] = np.arange(len(key))
df = df.merge(key, on=['src', 'jpg'])

rows = []
for addr, g in df.groupby('address'):
    groups = sorted(g.group.unique())
    for a, b in itertools.combinations(groups, 2):
        for x, y in ((a, b), (b, a)):
            X = g[g.group == x]; X = X.sample(min(1000, len(X)), random_state=0); ty = th[g[g.group == y].k.values]
            d = np.array([np.abs(ty - th[k]).mean(1).min() for k in X.k.values])
            rows.append(dict(address=addr, group=x, other=y, checked=len(X), near=int((d < 6).sum()), same=int((d < 0.5).sum())))
        print(addr, a, b, rows[-2]['near'], rows[-1]['near'], flush=True)
r = pd.DataFrame(rows)
r['near_pct'] = (100 * r.near / r.checked).round(2)
r.to_csv(f'{OUT}/group_links.csv', index=False)
print(r[r.near > 0].sort_values('near_pct', ascending=False).to_string())
