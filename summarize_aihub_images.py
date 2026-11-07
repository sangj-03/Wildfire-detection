"""analyze_aihub_images.py 결과(report/aihub/img_<split>.csv.gz)로 예전에 걱정한 점들을 사진 전체에서 다시 잰다.

사용: python3 -I summarize_aihub_images.py val
 1) 시간대 라벨별 실제 밝기 — '야간' 정상이 정말 낮 사진인지, 어두운 정상 사진이 있는지
 2) 날짜·GPS 글자가 찍힌 사진 비율 — 클래스마다 다르면 글자만 보고 맞힐 수 있다
 3) 두 단계 숫자 규칙(깊이 2 결정 트리)으로 클래스를 얼마나 맞히는지 — 장소+촬영일 묶음 절반으로 규칙을 찾고 나머지 절반에서 잰다
 4) 같은 카메라 안 거의 같은 사진 묶음 수 — 사진 수보다 실제로 다른 장면이 얼마나 되는지
쓰는 사진: 실제 사진(합성 양성 제외), 쓰기로 한 장소(use), 음성 중 백색/회색연기 라벨 사진(neg_with_smoke)은 따로 센다.
"""
import os
import sys

import numpy as np
import pandas as pd

OUT = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'report', 'aihub')
split = sys.argv[1] if len(sys.argv) > 1 else 'val'
pd.set_option('display.width', 220); pd.set_option('display.max_columns', 30)

img = pd.read_csv(f'{OUT}/img_{split}.csv.gz')
th = np.load(f'{OUT}/img_{split}_thumbs.npy').astype(np.float32)
img['k'] = np.arange(len(img))
lab = pd.read_csv(f'{OUT}/labels.csv.gz', usecols=['path', 'split', 'db', 'label', 'use', 'vid'])
meta = pd.read_csv(f'{OUT}/images.csv.gz', low_memory=False,
                   usecols=['path', 'timezone', 'weather', 'device', 'address', 'place_cam', 'folder_obj', 'datetime', 'frame'])
lab = lab[lab.split == split].merge(meta, on='path')
lab['jpg'] = lab.path.str.rsplit('/', n=1).str[-1].str.replace('.json', '.jpg')
df = img.merge(lab, on='jpg', how='inner')
print(f'사진 {len(img)}장 중 라벨과 이어진 사진 {len(df)}장')
df['bpp'] = df.nbytes / (df.w * df.h)
df['short'] = df[['w', 'h']].min(axis=1)
df['has_text'] = df.text > 60
df['dark'] = df.bright < 50                       # SmokeFire 점검과 같은 기준
df['group'] = df.address + '/' + df.datetime.astype(str).str[:8]
real = df[(df.db != 'PositiveSyntheticDB') & df.use]
cls = real[real.label.isin(['fire', 'smoke', 'normal'])].copy()


def show(title, t, name=None):
    print(f'\n[{title}]'); print(t)
    if name:
        t.to_csv(f'{OUT}/img_{split}_{name}.csv')


# 1) 시간대 라벨과 실제 밝기
t = real.groupby(['label', 'timezone']).agg(사진=('bright', 'size'), 평균밝기=('bright', 'mean'),
                                           어두운사진pct=('dark', lambda s: 100 * s.mean())).round(1)
show('시간대 라벨별 실제 밝기 (어두운 사진 = 평균 밝기 50 미만)', t, 'timezone_bright')
n = real[real.label == 'normal']
print('정상 사진 중 어두운 사진: %d장 / %d장' % (n.dark.sum(), len(n)))
print('정상 사진 밝기 하위 10장:', n.nsmallest(10, 'bright')[['bright', 'timezone', 'folder_obj', 'address']].values.tolist())

# 2) 날짜·GPS 글자
t = real.groupby('label').agg(사진=('has_text', 'size'), 글자pct=('has_text', lambda s: 100 * s.mean())).round(1)
show('왼쪽 위 날짜·GPS 글자가 있는 사진 비율', t, 'text_label')
t = real[real.has_text].groupby(['label', 'device', 'address']).size().unstack(0).fillna(0).astype(int)
show('글자 있는 사진: 기기·장소별 수', t)
print('글자 점수 분포(사진 수): ', np.histogram(real.text, bins=[-300, 0, 20, 40, 60, 80, 120, 300])[0].tolist())

# 3) 깊이 2 결정 트리 (오분류 수 최소, 각 노드 임계값 후보는 분위수 50개)
FEATS = {'시각만': ['bright', 'contrast', 'sat', 'sharp', 'dark_px'],
         '시각+파일': ['bright', 'contrast', 'sat', 'sharp', 'dark_px', 'bpp', 'jpeg_qmean', 'short', 'text']}
C = np.array(['fire', 'normal', 'smoke'])


def best_split(X, y, feats):
    best = (-1, None, None, None, None)
    for f in feats:
        x = X[f].values
        for t in np.unique(np.quantile(x, np.linspace(.02, .98, 50))):
            l = x < t
            if l.all() or (~l).all():
                continue
            cl, cr = np.bincount(y[l], minlength=3), np.bincount(y[~l], minlength=3)
            score = cl.max() + cr.max()
            if score > best[0]:
                best = (score, f, t, cl.argmax(), cr.argmax())
    return best


def fit(X, y, feats):
    _, f, t, _, _ = best_split(X, y, feats)
    l = X[f].values < t
    kids = [best_split(X[m], y[m], feats) for m in (l, ~l)]
    return f, t, kids


def predict(tree, X):
    f, t, kids = tree
    l = X[f].values < t
    out = np.empty(len(X), int)
    for m, (_, f2, t2, a, b) in zip((l, ~l), kids):
        out[m] = np.where(X[m][f2].values < t2, a, b)
    return out


def describe(tree):
    f, t, kids = tree
    s = []
    for side, (_, f2, t2, a, b) in zip((f'{f}<{t:.3g}', f'{f}≥{t:.3g}'), kids):
        s.append(f'{side} → ({f2}<{t2:.3g} → {C[a]}, 아니면 {C[b]})')
    return ' / '.join(s)


y_all = pd.Categorical(cls.label, categories=C).codes
groups = cls.group.unique()
res = []
for seed in range(5):
    rng = np.random.default_rng(seed)
    g_tr = set(rng.choice(groups, len(groups) // 2, replace=False))
    m = cls.group.isin(g_tr).values
    for name, feats in FEATS.items():
        tree = fit(cls[m], y_all[m], feats)
        p = predict(tree, cls[~m]); yt = y_all[~m]
        acc = (p == yt).mean(); bal = np.mean([(p[yt == c] == c).mean() for c in range(3) if (yt == c).any()])
        res.append(dict(seed=seed, feats=name, acc=round(100 * acc, 1), bal_acc=round(100 * bal, 1), rule=describe(tree)))
r = pd.DataFrame(res)
show('깊이 2 숫자 규칙 정확도 (%, 장소+촬영일 묶음을 반씩 나눠 5번)', r.groupby('feats')[['acc', 'bal_acc']].agg(['mean', 'min', 'max']), 'rule_acc')
for x in r[r.seed == 0].itertuples():
    print(f'  규칙 예({x.feats}): {x.rule}')
print('클래스 비율(다수 클래스만 찍으면):', (cls.label.value_counts(normalize=True) * 100).round(1).to_dict())
show('클래스별 지표 중앙값', cls.groupby('label')[['bright', 'contrast', 'sat', 'sharp', 'dark_px', 'bpp', 'jpeg_qmean', 'short']].median().round(3), 'feat_median')

# 4) 같은 카메라 안 거의 같은 사진 묶음 (32×18 회색 썸네일, 밝기 평균·대비를 맞춘 뒤 평균 절대차 < 6)
def groups_in(idx):
    a = th[idx].reshape(len(idx), -1)
    a = (a - a.mean(1, keepdims=True)) / (a.std(1, keepdims=True) + 1) * 40
    par = list(range(len(idx)))
    def find(i):
        while par[i] != i:
            par[i] = par[par[i]]; i = par[i]
        return i
    for i in range(len(idx)):
        d = np.abs(a[i + 1:] - a[i]).mean(1)
        for j in np.nonzero(d < 6)[0] + i + 1:
            par[find(i)] = find(j)
    return len({find(i) for i in range(len(idx))})

rows = []
for (lb, cam), g in real.groupby(['label', 'place_cam']):
    rows.append(dict(label=lb, cam=cam, n=len(g), groups=groups_in(g.k.values)))
d = pd.DataFrame(rows).groupby('label')[['n', 'groups']].sum()
d['사진당_다른장면'] = (d.groups / d.n).round(3)
show('같은 카메라 안 거의 같은 사진 묶음 수 (실제 사진, 쓰는 장소)', d, 'near_dup')
