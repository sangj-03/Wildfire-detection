"""DeepFire(Khan·Hassan 2020, Mendeley Data gjmr63rz2r v1, CC BY 4.0) 점검.

 1) 분할·클래스 개수, 크기·형식·JPEG 품질
 2) 완전 중복·근접 중복(분할 간 누수, 클래스 간 충돌) — SmokeFire 점검과 같은 기준(dHash256 ≤ 20, 32×32 흑백 차이 < 8)
 3) 어두운 사진 비율 — 짧은 변 256으로 줄인 뒤 평균 밝기 50 미만 (AI Hub·D-Fire 점검과 같은 기준)
 4) 깊이 2 숫자 규칙 정확도 — 학습·검증 폴더로 규칙을 찾고 시험 폴더에서 잰다
 5) 받아 둔 다른 데이터(SmokeFire·Kaggle 2종·D-Fire)와 겹치는 사진 — 사진 전체 해시라 잘라 낸 사진은 못 찾는다
사용: python3 -I analyze_deepfire.py  →  team6/report/deepfire/
"""
import glob, hashlib, json, os, random
from collections import Counter, defaultdict
from multiprocessing import Pool

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = '/data/home/seame/dl-team-project/team6'
DATA = f'{ROOT}/data/DeepFire/extract/Dataset'
OUT = f'{ROOT}/report/deepfire'
NEAR, MAE = 20, 8.0
SPLITS = {'Training and Validation': 'trainval', 'Testing': 'test'}
EXT = {'SmokeFire': f'{ROOT}/data/SmokeFire_img',
       'Kaggle_mohnish': f'{ROOT}/data/kaggle_cache/datasets/mohnishsaiprasad/forest-fire-images/versions/1/Data',
       'Kaggle_kutaykutlu': f'{ROOT}/data/kaggle_cache/datasets/kutaykutlu/forest-fire/versions/1',
       'D-Fire': f'{ROOT}/data/DFire'}
FONT = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 12, index=3)
POP = np.array([bin(i).count('1') for i in range(256)], np.uint8)
os.makedirs(OUT, exist_ok=True)
pd.set_option('display.width', 200)


def dhash(im, n=16):
    g = np.asarray(im.convert('L').resize((n + 1, n), Image.BILINEAR), dtype=np.int16)
    return np.packbits((g[:, 1:] > g[:, :-1]).flatten())


def thumb(im):
    return np.asarray(im.convert('L').resize((32, 32), Image.BILINEAR), dtype=np.uint8)


def measure(p):
    b = open(p, 'rb').read()
    im = Image.open(p)
    q = im.quantization[0] if getattr(im, 'quantization', None) else None
    fmt = im.format; im = im.convert('RGB'); w, h = im.size
    s = 256 / min(w, h)
    sm = im.resize((max(1, round(w * s)), max(1, round(h * s))))
    a = np.asarray(sm, dtype=np.float32); g = a.mean(2)
    hsv = np.asarray(sm.convert('HSV'), dtype=np.float32)
    hue, sat, val = hsv[..., 0] * 360 / 255, hsv[..., 1], hsv[..., 2]
    lap = g[1:-1, 1:-1] * 4 - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    r = dict(path=p, w=w, h=h, fmt=fmt, nbytes=len(b), md5=hashlib.md5(b).hexdigest(),
             jpeg_q0=(q[0] if q else -1), jpeg_qmean=(float(np.mean(q)) if q else -1),
             bright=g.mean(), contrast=g.std(), sat=sat.mean(), sharp=lap.var(), dark_px=(g < 30).mean(),
             fire_px=(((hue < 60) | (hue > 340)) & (sat > 100) & (val > 150)).mean(),   # 불꽃 색(빨강~노랑, 진하고 밝음)
             green_px=((hue > 70) & (hue < 170) & (sat > 60) & (val > 40)).mean())     # 숲 색
    return r, dhash(im), thumb(im)


def hash_only(p):
    try:
        im = Image.open(p); b = open(p, 'rb').read()
        return dict(path=p, w=im.size[0], h=im.size[1], md5=hashlib.md5(b).hexdigest()), dhash(im), thumb(im)
    except Exception as e:
        return dict(path=p, error=repr(e)[:120]), np.zeros(32, np.uint8), np.zeros((32, 32), np.uint8)


def near_pairs(HA, TA, HB, TB, same=False, step=64):
    """A의 각 사진과 B의 사진 중 dHash 거리 ≤ NEAR 이고 축소본 차이 < MAE 인 쌍."""
    out = []
    TA = TA.reshape(len(TA), -1).astype(np.float32); TB = TB.reshape(len(TB), -1).astype(np.float32)
    for s in range(0, len(HA), step):
        d = POP[np.bitwise_xor(HA[s:s + step, None, :], HB[None, :, :])].sum(axis=2, dtype=np.int32)
        for i, j in zip(*np.nonzero(d <= NEAR)):
            i2 = i + s
            if same and i2 >= j:
                continue
            mae = float(np.abs(TA[i2] - TB[j]).mean())
            if mae < MAE:
                out.append((int(i2), int(j), int(d[i, j]), round(mae, 2)))
    return pd.DataFrame(out, columns=['i', 'j', 'ham', 'mae'])


def tile(p, cell, text):
    im = Image.open(p).convert('RGB'); im.thumbnail((cell, cell))
    t = Image.new('RGB', (cell, cell + 16), 'white'); t.paste(im, ((cell - im.width) // 2, (cell - im.height) // 2))
    ImageDraw.Draw(t).text((2, cell), text, fill='black', font=FONT)
    return t


def sheet(items, path, cols=8, cell=160):
    """items: [(사진 경로, 아래 글자)]"""
    if not items:
        return
    S = Image.new('RGB', (cols * cell, ((len(items) + cols - 1) // cols) * (cell + 16)), 'white')
    for k, (p, text) in enumerate(items):
        S.paste(tile(p, cell, text), ((k % cols) * cell, (k // cols) * (cell + 16)))
    S.save(path, quality=88)


# ---- 깊이 2 결정 트리 (summarize_aihub_images.py와 같은 방식, 두 클래스) ----
def best_split(X, y, feats):
    best = (-1, None, None, None, None)
    for f in feats:
        x = X[f].values
        for t in np.unique(np.quantile(x, np.linspace(.02, .98, 50))):
            l = x < t
            if l.all() or (~l).all():
                continue
            cl, cr = np.bincount(y[l], minlength=2), np.bincount(y[~l], minlength=2)
            score = cl.max() + cr.max()
            if score > best[0]:
                best = (score, f, t, cl.argmax(), cr.argmax())
    return best


def fit(X, y, feats, depth):
    top = best_split(X, y, feats)
    if depth == 1:
        return top
    l = X[top[1]].values < top[2]
    return top[1], top[2], [best_split(X[m], y[m], feats) for m in (l, ~l)]


def predict(tree, X):
    if len(tree) == 5:
        _, f, t, a, b = tree
        return np.where(X[f].values < t, a, b)
    f, t, kids = tree
    l = X[f].values < t
    out = np.empty(len(X), int)
    for m, (_, f2, t2, a, b) in zip((l, ~l), kids):
        out[m] = np.where(X[m][f2].values < t2, a, b)
    return out


def describe(tree, C):
    if len(tree) == 5:
        _, f, t, a, b = tree
        return f'{f}<{t:.3g} → {C[a]}, 아니면 {C[b]}'
    f, t, kids = tree
    return ' / '.join(f'{side} → ({f2}<{t2:.3g} → {C[a]}, 아니면 {C[b]})'
                      for side, (_, f2, t2, a, b) in zip((f'{f}<{t:.3g}', f'{f}≥{t:.3g}'), kids))


if __name__ == '__main__':
    paths = sorted(glob.glob(f'{DATA}/*/*/*.jpg'))
    with Pool(16) as pool:
        res = pool.map(measure, paths, chunksize=16)
    df = pd.DataFrame([r for r, _, _ in res])
    H = np.stack([h for _, h, _ in res]); T = np.stack([t for _, _, t in res])
    df['split'] = df.path.str.split('/').str[-3].map(SPLITS)
    df['cls'] = df.path.str.split('/').str[-2]
    df['file'] = df.path.str.rsplit('/', n=1).str[-1]
    df['bpp'] = df.nbytes / (df.w * df.h)
    df['dark'] = df.bright < 50
    summary = {}

    print('[1] 개수 (분할 × 클래스)'); print(pd.crosstab(df.split, df.cls, margins=True))
    print('형식·크기:', dict(Counter(zip(df.fmt, df.w, df.h))))
    t = df.groupby(['split', 'cls']).agg(사진=('file', 'size'), 파일KB중앙=('nbytes', lambda s: s.median() / 1024),
                                          JPEG양자화q0중앙=('jpeg_q0', 'median'), q0종류=('jpeg_q0', 'nunique'),
                                          선명도중앙=('sharp', 'median')).round(1)
    print(t); t.to_csv(f'{OUT}/split_class_file.csv')
    print('JPEG 양자화 q0 분포 (분할·클래스별 상위 5):')
    for k, g in df.groupby(['split', 'cls']):
        print('  ', k, Counter(g.jpeg_q0).most_common(5))

    # 2) 중복
    exact = [list(v) for v in df.groupby('md5').groups.values() if len(v) > 1]
    pp = near_pairs(H, T, H, T, same=True)
    for c in ('split', 'cls', 'file'):
        pp[f'{c}_i'] = df[c].values[pp.i]; pp[f'{c}_j'] = df[c].values[pp.j]
    pp.to_csv(f'{OUT}/near_dup_pairs.csv', index=False)
    cross = pp[pp.split_i != pp.split_j]; conf = pp[pp.cls_i != pp.cls_j]
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
    tv = set(np.nonzero((df.split == 'trainval').values)[0])
    te = np.nonzero((df.split == 'test').values)[0]
    leak = [k for k in te if any(m in tv for m in comp[find(k)] if m != k)]
    df['near_group_size'] = [len(comp[find(k)]) for k in range(len(df))]
    print(f'\n[2] 완전 중복 묶음 {len(exact)}개 (사진 {sum(map(len, exact))}장)')
    print(f'근접 중복 쌍 {len(pp)}개 (완전 중복 포함), 묶음 {len(groups)}개·사진 {sum(map(len, groups))}장, '
          f'가장 큰 묶음 {max(map(len, groups)) if groups else 0}장')
    print('  분할이 다른 쌍(누수):', len(cross), ' 클래스가 다른 쌍(라벨 충돌):', len(conf))
    print(f'  시험 {len(te)}장 중 학습·검증에 근접 중복이 있는 사진: {len(leak)}장 ({100 * len(leak) / len(te):.1f}%)',
          dict(Counter(df.cls.values[leak])))
    print('  클래스별 근접 중복 묶음에 든 사진:', df[df.near_group_size > 1].groupby('cls').size().to_dict())
    summary.update(n=len(df), exact_groups=len(exact), exact_images=sum(map(len, exact)), near_pairs=len(pp),
                   near_groups=len(groups), near_images=sum(map(len, groups)), cross_split_pairs=len(cross),
                   cross_class_pairs=len(conf), test_leak=len(leak))

    # 3) 어두운 사진
    t = df.groupby('cls').agg(사진=('dark', 'size'), 어두운사진=('dark', 'sum'), 어두운pct=('dark', lambda s: 100 * s.mean()),
                             평균밝기=('bright', 'mean')).round(1)
    print('\n[3] 어두운 사진 (평균 밝기 50 미만)'); print(t); t.to_csv(f'{OUT}/dark_by_class.csv')
    print('  어두운 사진 중 불 비율: %.1f%%' % (100 * (df[df.dark].cls == 'fire').mean() if df.dark.any() else 0))
    summary['dark'] = t.to_dict()

    # 4) 숫자 규칙
    C = np.array(['fire', 'nofire'])
    FEATS = {'시각만': ['bright', 'contrast', 'sat', 'sharp', 'dark_px', 'fire_px', 'green_px'],
             '시각+파일': ['bright', 'contrast', 'sat', 'sharp', 'dark_px', 'fire_px', 'green_px', 'bpp', 'jpeg_q0'],
             '파일만': ['bpp', 'jpeg_q0', 'jpeg_qmean']}
    tr, ts = df[df.split == 'trainval'], df[df.split == 'test']
    ytr, yts = pd.Categorical(tr.cls, categories=C).codes, pd.Categorical(ts.cls, categories=C).codes
    print('\n[4] 숫자 규칙 정확도 (학습·검증으로 찾고 시험 380장에서 잼, 한 클래스만 찍으면 50%)')
    rules = []
    for name, feats in FEATS.items():
        for depth in (1, 2):
            tree = fit(tr, ytr, feats, depth)
            acc = 100 * (predict(tree, ts) == yts).mean()
            rules.append(dict(feats=name, depth=depth, acc=round(acc, 1), rule=describe(tree, C)))
            print(f'  {name} 깊이{depth}: {acc:.1f}%  ({describe(tree, C)})')
    pd.DataFrame(rules).to_csv(f'{OUT}/rule_acc.csv', index=False)
    summary['rules'] = rules
    print('  특성 중앙값:'); print(df.groupby('cls')[FEATS['시각+파일']].median().round(3).T)

    # 5) 다른 데이터와 겹침
    ext_rows, ext_H, ext_T = [], [], []
    cache = f'{OUT}/ext_hash.npz'
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        ext = pd.DataFrame(z['rows'].tolist()); ext_H, ext_T = z['H'], z['T']
    else:
        jobs = []
        for name, root in EXT.items():
            for p in sorted(glob.glob(f'{root}/**/*', recursive=True)):
                if p.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                    jobs.append((name, p))
        with Pool(16) as pool:
            out = pool.map(hash_only, [p for _, p in jobs], chunksize=64)
        for (name, p), (r, h, t) in zip(jobs, out):
            r['dataset'] = name; r['sub'] = os.path.relpath(os.path.dirname(p), EXT[name])
            ext_rows.append(r); ext_H.append(h); ext_T.append(t)
        ext = pd.DataFrame(ext_rows); ext_H, ext_T = np.stack(ext_H), np.stack(ext_T)
        np.savez(cache, rows=np.array(ext.to_dict('records'), dtype=object), H=ext_H, T=ext_T)
    print(f'\n[5] 다른 데이터와 겹침 — 비교한 사진 {len(ext)}장', ext.groupby('dataset').size().to_dict(),
          '열기 실패', int(ext.get('error', pd.Series(dtype=object)).notna().sum()))
    md5_ext = ext.groupby('md5').dataset.apply(lambda s: sorted(set(s))).to_dict()
    df['same_bytes_in'] = df.md5.map(lambda m: ','.join(md5_ext.get(m, [])))
    xp = near_pairs(H, T, ext_H, ext_T)
    xp['cls'] = df.cls.values[xp.i]; xp['split'] = df.split.values[xp.i]; xp['file'] = df.file.values[xp.i]
    xp['dataset'] = ext.dataset.values[xp.j]; xp['sub'] = ext['sub'].values[xp.j]; xp['ext_path'] = ext.path.values[xp.j]
    xp['ext_wh'] = [f'{w}x{h}' for w, h in zip(ext.w.values[xp.j], ext.h.values[xp.j])]
    xp.to_csv(f'{OUT}/cross_dataset_pairs.csv', index=False)
    hit = xp.groupby('i').dataset.apply(lambda s: ','.join(sorted(set(s))))
    df['near_in'] = df.index.map(hit).fillna('')
    t = pd.crosstab(df.cls, df.near_in.replace('', '(없음)'))
    print('DeepFire 사진별로 근접 중복이 있는 다른 데이터:'); print(t); t.to_csv(f'{OUT}/cross_dataset_by_class.csv')
    print('같은 바이트(완전 중복)도 있는 사진:', (df.same_bytes_in != '').sum(), dict(Counter(df.same_bytes_in[df.same_bytes_in != ''])))
    if len(xp):
        print('겹친 상대 폴더:'); print(xp.groupby(['dataset', 'sub', 'cls']).i.nunique().rename('DeepFire 사진 수'))
        print('상대 사진 크기 상위:', Counter(xp.ext_wh).most_common(6))
    summary['cross_dataset'] = {k: int(v) for k, v in Counter(df.near_in[df.near_in != '']).items()}
    df.to_csv(f'{OUT}/images.csv', index=False)
    json.dump(summary, open(f'{OUT}/summary.json', 'w'), ensure_ascii=False, indent=1, default=str)

    # 그림
    random.seed(0)
    for c in ('fire', 'nofire'):
        ids = random.sample(list(np.nonzero((df.cls == c).values)[0]), 48)
        sheet([(df.path[k], f'{df.split[k]} {df.file[k]}') for k in ids], f'{OUT}/sample_{c}.jpg')
        d = df[(df.cls == c) & df.dark]
        sheet([(r.path, f'{r.file} 밝기{r.bright:.0f}') for r in d.sort_values('bright').head(48).itertuples()],
              f'{OUT}/dark_{c}.jpg')
    for name, sub in (('cross_split', cross), ('cross_class', conf), ('within', pp[(pp.split_i == pp.split_j) & (pp.cls_i == pp.cls_j)])):
        s2 = sub.sample(min(16, len(sub)), random_state=0)
        sheet([(df.path[k], f'{df.cls[k]} {df.split[k]} {df.file[k]}') for t in zip(s2.i, s2.j) for k in t],
              f'{OUT}/pairs_{name}.jpg')
    if len(xp):
        s2 = xp.drop_duplicates('i').sample(min(16, xp.i.nunique()), random_state=0)
        sheet([it for r in s2.itertuples() for it in ((df.path[r.i], f'DeepFire {r.cls} {r.file}'),
                                                       (r.ext_path, f'{r.dataset} {r.sub[-14:]} {r.ext_wh}'))],
              f'{OUT}/pairs_cross_dataset.jpg')
    print('\n저장:', sorted(os.listdir(OUT)))
