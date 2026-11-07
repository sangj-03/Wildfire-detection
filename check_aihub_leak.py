"""공식 학습/검증 분할 누수를 사진으로 확인: 검증 사진마다 같은 영상의 가장 가까운 학습 프레임을 꺼내 얼마나 같은지 잰다.

비교 기준: 같은 클래스지만 다른 장소+촬영일 묶음의 학습 사진 1장.
차이 = 32×18 회색 썸네일을 밝기 평균·대비로 맞춘 뒤 평균 절대차(summarize_aihub_images.py의 근접 중복과 같은 척도).
사용: python3 -I check_aihub_leak.py [클래스당 사진 수]  →  report/aihub/leak_pairs.csv, report/aihub/samples/leak_pairs.jpg
"""
import os
import io, sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from aihub_split_zip import SplitZip

BASE = ('/data/aihub/71330/265.지역안전재난(산불)_방재의_고도화를_위한_대규모_인공지능_데이터베이스_구축/'
        '01-1.정식개방데이터')
TS = f'{BASE}/Training/01.원천데이터/TS.zip'
VS = f'{BASE}/Validation/01.원천데이터/VS.zip'
OUT = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'report', 'aihub')
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000

lab = pd.read_csv(f'{OUT}/labels.csv.gz', usecols=['path', 'split', 'db', 'label', 'use', 'vid'])
meta = pd.read_csv(f'{OUT}/images.csv.gz', usecols=['path', 'frame', 'address', 'datetime'], low_memory=False)
df = lab.merge(meta, on='path')
df = df[(df.db != 'PositiveSyntheticDB') & df.use & df.label.isin(['fire', 'smoke', 'normal'])].copy()
df['jpg'] = df.path.str.rsplit('/', n=1).str[-1].str.replace('.json', '.jpg')
df['group'] = df.address + '/' + df.datetime.astype(str).str[:8]
tr, va = df[df.split == 'train'], df[df.split == 'val']
by_vid = {v: g.sort_values('frame') for v, g in tr.groupby('vid')}
tr_lb = {lb: g for lb, g in tr.groupby('label')}
rng = np.random.default_rng(0)
pairs = []
for lb, g in va.groupby('label'):
    for r in g.sample(min(N, len(g)), random_state=0).itertuples():
        t = by_vid.get(r.vid)
        if t is None:
            continue
        near = t.iloc[np.abs(t.frame.values - r.frame).argmin()]
        cand = tr_lb[lb][tr_lb[lb].group != r.group]
        other = cand.iloc[rng.integers(0, len(cand))]
        pairs.append(dict(label=lb, val=r.jpg, near=near.jpg, gap=abs(near.frame - r.frame), other=other.jpg))
pairs = pd.DataFrame(pairs)
ts, vs = SplitZip(TS), SplitZip(VS)          # 볼륨을 다 받기 전이면 받은 부분의 파일만 (fork 전에 한 번만 훑는다)
tsn = {n.rsplit('/', 1)[-1]: n for n in ts.namelist()}; vsn = {n.rsplit('/', 1)[-1]: n for n in vs.namelist()}
pairs = pairs[pairs.near.isin(tsn) & pairs.other.isin(tsn) & pairs.val.isin(vsn)].reset_index(drop=True)
print('학습 zip 전체 받음:', ts.complete, '/ 쌍', len(pairs), pairs.groupby('label').size().to_dict(), flush=True)


def thumb(b):
    im = Image.open(io.BytesIO(b)).convert('L')
    a = np.asarray(im.resize((32, 18), Image.BILINEAR), dtype=np.float32).ravel()
    return (a - a.mean()) / (a.std() + 1) * 40


def work(chunk):
    out = []
    for r in chunk.itertuples():
        v = thumb(vs.read(vsn[r.val])); a = thumb(ts.read(tsn[r.near])); o = thumb(ts.read(tsn[r.other]))
        out.append((np.abs(v - a).mean(), np.abs(v - o).mean()))
    return out


ts.close(); vs.close()
with Pool(16) as pool:
    res = [x for part in pool.map(work, [pairs.iloc[i:i + 50] for i in range(0, len(pairs), 50)]) for x in part]
pairs['d_near'], pairs['d_other'] = zip(*res)
pairs.to_csv(f'{OUT}/leak_pairs.csv', index=False)
print(pairs.groupby('label')[['gap', 'd_near', 'd_other']].median().round(2))
for t in (4, 6, 8):
    print(f'차이 < {t}: 같은 영상 학습 프레임', pairs.groupby('label').d_near.apply(lambda s: round(100 * (s < t).mean(), 1)).to_dict(),
          '/ 다른 묶음 학습 사진', pairs.groupby('label').d_other.apply(lambda s: round(100 * (s < t).mean(), 1)).to_dict())

# 그림: 클래스마다 무작위 3쌍 (왼쪽 검증, 오른쪽 학습의 가장 가까운 프레임)
ts.close(); vs.close()
FONT = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 16, index=3)
pick = pairs.groupby('label').sample(3, random_state=1)
W = 480; rows = []
for r in pick.itertuples():
    a = Image.open(io.BytesIO(vs.read(vsn[r.val]))).convert('RGB'); b = Image.open(io.BytesIO(ts.read(tsn[r.near]))).convert('RGB')
    a = a.resize((W, round(a.height * W / a.width))); b = b.resize((W, round(b.height * W / b.width)))
    rows.append((a, b, f'{r.label} · 검증(왼쪽)과 학습(오른쪽) 프레임 번호 차이 {r.gap} · 차이 {r.d_near:.1f}'))
H = max(a.height for a, _, _ in rows) + 26
S = Image.new('RGB', (2 * W + 10, H * len(rows)), 'white'); dr = ImageDraw.Draw(S)
for k, (a, b, cap) in enumerate(rows):
    S.paste(a, (0, k * H)); S.paste(b, (W + 10, k * H)); dr.text((4, k * H + a.height + 2), cap, fill='black', font=FONT)
S.save(f'{OUT}/samples/leak_pairs.jpg', quality=85)
print('saved leak_pairs.jpg')
