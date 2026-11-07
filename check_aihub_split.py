"""새 분할(split_v1, split_v2)에서 검증·시험 사진이 학습에 거의 같은 사진을 갖는지 확인한다.

같은 장소(주소)의 학습 사진 전부와 썸네일 차이를 재고 가장 작은 값을 본다(check_aihub_leak.py와 같은 척도, 차이 < 6이면 거의 같음).
다른 날짜라도 같은 카메라 위치에서 찍은 정지 풍경은 거의 같을 수 있어, 묶음끼리 이어지는지도 센다.
사용: python3 -I check_aihub_split.py [분할·클래스당 사진 수] [v1|v2]  →  report/aihub/split_<판>_check.csv, split_<판>_cross_blocks.csv
"""
import sys

import numpy as np
import pandas as pd

OUT = '/data/home/seame/dl-team-project/team6/report/aihub'
N = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
VER = sys.argv[2] if len(sys.argv) > 2 else 'v1'
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 100)

sp = pd.read_csv(f'{OUT}/split_{VER}.csv.gz')
th, key = [], []
for src in ('train', 'val'):
    img = pd.read_csv(f'{OUT}/img_{src}.csv.gz', usecols=['jpg'])
    t = np.load(f'{OUT}/img_{src}_thumbs.npy').reshape(len(img), -1).astype(np.float32)
    t = (t - t.mean(1, keepdims=True)) / (t.std(1, keepdims=True) + 1) * 40
    th.append(t); key.append(pd.DataFrame({'src': src, 'jpg': img.jpg}))
th = np.concatenate(th); key = pd.concat(key, ignore_index=True); key['k'] = np.arange(len(key))
sp = sp.merge(key, on=['src', 'jpg'], how='left')
print('썸네일 없는 사진', sp.k.isna().sum())
sp = sp.dropna(subset=['k']); sp['k'] = sp.k.astype(int)
tr = sp[sp.split == 'train']
tr_by_addr = {a: g for a, g in tr.groupby('address')}

rows = []
for (s, lb), g in sp[(sp.split != 'train') & sp.label.isin(['fire', 'smoke', 'normal'])].groupby(['split', 'label']):
    for r in g.sample(min(N, len(g)), random_state=0).itertuples():
        t = tr_by_addr.get(r.address)
        if t is None:
            rows.append(dict(split=s, label=lb, address=r.address, block=r.block, d=np.inf, near_block=None)); continue
        d = np.abs(th[t.k.values] - th[r.k]).mean(1); i = d.argmin()
        rows.append(dict(split=s, label=lb, address=r.address, block=r.block, d=float(d[i]), near_block=t.block.values[i]))
res = pd.DataFrame(rows); res.to_csv(f'{OUT}/split_{VER}_check.csv', index=False)
res['same_place_in_train'] = np.isfinite(res.d)
print('\n[검증·시험 사진: 같은 장소 학습 사진과의 가장 작은 차이]')
print(res.groupby(['split', 'label']).agg(사진=('d', 'size'), 같은장소가학습에있음pct=('same_place_in_train', lambda x: round(100 * x.mean(), 1)),
                                          거의같음pct=('d', lambda x: round(100 * (x < 6).mean(), 1)),
                                          차이중앙값=('d', lambda x: round(float(np.median(x[np.isfinite(x)])) if np.isfinite(x).any() else np.nan, 1))))
cross = res[res.d < 6].groupby(['block', 'near_block']).size().sort_values(ascending=False).rename('거의같은사진').reset_index()
cross['검사한사진'] = cross.block.map(res.groupby('block').size())
cross.to_csv(f'{OUT}/split_{VER}_cross_blocks.csv', index=False)
print('\n[거의 같은 사진이 이어진 묶음 (검증·시험 묶음 → 학습 묶음)]'); print(cross.head(30))
