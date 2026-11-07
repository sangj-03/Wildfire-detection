"""SmokeFire(HF EdBianchi/SmokeFire) 점검: 클래스·분할 개수, 이미지 크기·형식, 중복·근접 중복(분할 간 누수, 클래스 간 충돌).

사용: python3 analyze_smokefire.py  →  team6/report/ 에 CSV·요약·샘플 그림 저장
"""
import glob, hashlib, io, json, os, random
from collections import Counter, defaultdict

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw

ROOT = '/data/home/seame/dl-team-project/team6'
DATA = f'{ROOT}/data/SmokeFire'
OUT = f'{ROOT}/report'
NEAR = 20         # 16×16 dHash(256비트) 해밍 거리 이하이고
MAE = 8.0         # 32×32 흑백 축소본의 평균 절대 차이(0~255)도 이 값 미만이면 근접 중복으로 본다
NAMES = ['Fire', 'Normal', 'Smoke']   # README의 class_label 순서
os.makedirs(OUT, exist_ok=True)


def dhash(img, n=16):
    g = np.asarray(img.convert('L').resize((n + 1, n), Image.BILINEAR), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return np.packbits(bits)


def label_names(path):
    meta = pq.read_schema(path).metadata or {}
    hf = json.loads(meta.get(b'huggingface', b'{}'))
    feats = hf.get('info', {}).get('features', {})
    for k, v in feats.items():
        if v.get('_type') == 'ClassLabel':
            return k, v['names']
    return None, None


rows, thumbs = [], []
files = sorted(glob.glob(f'{DATA}/**/*.parquet', recursive=True))
print('parquet files:', [os.path.relpath(f, DATA) for f in files])
for f in files:
    split = next((s for s in ('train', 'validation', 'test') if s in os.path.basename(f)), os.path.basename(f))
    lab_col, names = label_names(f)
    if lab_col is None:
        lab_col, names = 'label', NAMES
    t = pq.read_table(f)
    cols = t.column_names
    img_col = 'image' if 'image' in cols else cols[0]
    for i, rec in enumerate(t.to_pylist()):
        im = rec[img_col]
        b = im['bytes'] if isinstance(im, dict) else im
        lab = rec[lab_col] if lab_col else None
        img = Image.open(io.BytesIO(b))
        w, h = img.size
        thumbs.append(np.asarray(img.convert('L').resize((32, 32), Image.BILINEAR), dtype=np.float32))
        rows.append(dict(split=split, idx=len(rows), cls=names[lab] if names else lab, w=w, h=h, fmt=img.format,
                         mode=img.mode, nbytes=len(b), md5=hashlib.md5(b).hexdigest(), dhash=dhash(img),
                         src=(im.get('path') if isinstance(im, dict) else None), file=os.path.basename(f), row=i))

import pandas as pd
df = pd.DataFrame(rows)
df.drop(columns=['dhash']).to_csv(f'{OUT}/images.csv', index=False)
print('\n[개수] 분할 × 클래스')
print(pd.crosstab(df.split, df.cls, margins=True))
print('\n[형식·모드]', dict(Counter(zip(df.fmt, df['mode']))))
print('\n[크기] 짧은 변 분위수 (px):', df[['w', 'h']].min(axis=1).quantile([0, .05, .25, .5, .75, .95, 1]).round().to_dict())
print('[크기] 긴 변 분위수 (px):', df[['w', 'h']].max(axis=1).quantile([0, .05, .25, .5, .75, .95, 1]).round().to_dict())
print('[크기] 짧은 변 < 224 비율: %.1f%%' % (100 * (df[['w', 'h']].min(axis=1) < 224).mean()))
print('[비율] 가로/세로 분위수:', (df.w / df.h).quantile([.05, .5, .95]).round(2).to_dict())
print('[크기] 클래스별 짧은 변 중앙값:', df.assign(s=df[['w', 'h']].min(axis=1)).groupby('cls').s.median().to_dict())
print('[크기] 가장 흔한 해상도:', Counter(zip(df.w, df.h)).most_common(8))

# 완전 중복(같은 바이트)
g = df.groupby('md5')
exact = [list(v) for v in g.groups.values() if len(v) > 1]
print('\n[완전 중복] 묶음 %d개, 관련 이미지 %d장' % (len(exact), sum(len(v) for v in exact)))

# 근접 중복: 256비트 dHash로 후보를 고르고, 32×32 축소본 평균 절대 차이로 확인
H = np.stack(df.dhash.tolist())                       # (N, 32) uint8
T = np.stack(thumbs).reshape(len(df), -1)             # (N, 1024)
pairs = []
for s in range(0, len(H), 300):
    x = np.bitwise_xor(H[s:s + 300, None, :], H[None, :, :])
    d = np.unpackbits(x, axis=2).sum(axis=2)
    ii, jj = np.nonzero(d <= NEAR)
    for i, j in zip(ii + s, jj):
        if i < j:
            mae = float(np.abs(T[i] - T[j]).mean())
            if mae < MAE:
                pairs.append((int(i), int(j), int(d[i - s, j]), round(mae, 2)))
pp = pd.DataFrame(pairs, columns=['i', 'j', 'ham', 'mae'])
pp['split_i'] = df.split.values[pp.i]; pp['split_j'] = df.split.values[pp.j]
pp['cls_i'] = df.cls.values[pp.i]; pp['cls_j'] = df.cls.values[pp.j]
pp.to_csv(f'{OUT}/near_dup_pairs.csv', index=False)
print('\n[근접 중복] dHash256 거리 <= %d 이고 축소본 차이 < %.0f 인 쌍: %d (완전 중복 포함)' % (NEAR, MAE, len(pp)))
cross = pp[pp.split_i != pp.split_j]
print('  분할이 다른 쌍(누수):', len(cross), dict(Counter(tuple(sorted(t)) for t in zip(cross.split_i, cross.split_j))))
conf = pp[pp.cls_i != pp.cls_j]
print('  클래스가 다른 쌍(라벨 충돌):', len(conf), dict(Counter(tuple(sorted(t)) for t in zip(conf.cls_i, conf.cls_j))))

# 근접 중복 묶음(연결 요소)과 평가 세트 오염 정도
parent = list(range(len(df)))
def find(a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]; a = parent[a]
    return a
for i, j in zip(pp.i, pp.j):
    parent[find(i)] = find(j)
comp = defaultdict(list)
for k in range(len(df)):
    comp[find(k)].append(k)
groups = [v for v in comp.values() if len(v) > 1]
print('  근접 중복 묶음 %d개, 묶음에 속한 이미지 %d장, 가장 큰 묶음 %d장' % (len(groups), sum(map(len, groups)), max(map(len, groups)) if groups else 0))
in_train = set(np.nonzero((df.split == 'train').values)[0])
for sp in ('validation', 'test'):
    idx = np.nonzero((df.split == sp).values)[0]
    leak = sum(1 for k in idx if any(m in in_train for m in comp[find(k)] if m != k))
    print('  %s %d장 중 학습 세트에 근접 중복이 있는 이미지: %d장 (%.1f%%)' % (sp, len(idx), leak, 100 * leak / len(idx)))

summary = dict(n=len(df), exact_groups=len(exact), near_pairs=len(pp), cross_split_pairs=len(cross), cross_class_pairs=len(conf))
json.dump(summary, open(f'{OUT}/summary.json', 'w'), ensure_ascii=False, indent=1)

# 샘플 그림: 클래스별 무작위 24장 / 분할 간 근접 중복 쌍 / 클래스 간 충돌 쌍
def load(k):
    r = df.iloc[k]
    t = pq.read_table(f'{DATA}/' + next(os.path.relpath(f, DATA) for f in files if os.path.basename(f) == r.file))
    im = t.column('image')[int(r.row)].as_py()
    return Image.open(io.BytesIO(im['bytes'])).convert('RGB')

def sheet(ids, path, cols=8, cell=150, labels=None):
    rws = (len(ids) + cols - 1) // cols
    S = Image.new('RGB', (cols * cell, rws * (cell + 16)), 'white')
    dr = ImageDraw.Draw(S)
    for n, k in enumerate(ids):
        im = load(k); im.thumbnail((cell, cell))
        x, y = (n % cols) * cell, (n // cols) * (cell + 16)
        S.paste(im, (x, y))
        dr.text((x + 2, y + cell + 2), labels[n] if labels else f'{df.cls[k]}/{df.split[k]}', fill='black')
    S.save(path)

random.seed(0)
for c in sorted(df.cls.unique()):
    ids = random.sample(list(np.nonzero((df.cls == c).values)[0]), 32)
    sheet(ids, f'{OUT}/sample_{c}.png', labels=[f'{df.w[k]}x{df.h[k]}' for k in ids])
for name, sub in (('cross_split', cross), ('cross_class', conf)):
    if len(sub):
        s2 = sub.sample(min(12, len(sub)), random_state=0)
        ids = [k for t in zip(s2.i, s2.j) for k in t]
        sheet(ids, f'{OUT}/pairs_{name}.png', cols=8, labels=[f'{df.cls[k]}/{df.split[k][:5]}' for k in ids])
print('\n그림 저장:', sorted(os.listdir(OUT)))
