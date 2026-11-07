"""산불 탐지 프로젝트 다른 출처 시험 데이터(D-Fire, DeepFire) 노트북(external.ipynb)에서 쓰는 무거운 계산.

사진 처리 함수 자체는 학습·평가 코드와 같이 쓰는 ../external_data.py에 있다(overlay_boxes = find_overlays + 직접 지정한 자리,
mask_overlays, preprocess_external).
여기서는 (1) 라벨·사진을 전부 읽어 표로 만들고(감시 카메라 사진은 화면 글자 자리까지), (2) 사진을 전부 실제로 처리해
크기·오류·픽셀 통계와 비교용 썸네일을 잰다. 결과는 cache/에 parquet·npy로 남긴다.
"""
import glob, hashlib, io, json, os, time

import numpy as np
import pandas as pd
from PIL import Image

import eda_lib as E
import external_data as X

ROOT = E.ROOT
DFIRE = f'{ROOT}/data/DFire'
DEEPFIRE = f'{ROOT}/data/DeepFire/extract/Dataset'
CACHE = E.CACHE


def _pool(fn, jobs, procs=16):
    import multiprocessing as mp
    with mp.get_context('fork').Pool(procs) as pool:
        return pool.map(fn, jobs, chunksize=1)


# ---------------------------------------------------------------- D-Fire 라벨·사진 표

def _dfire_job(rows):
    out, boxes = [], []
    for sp, name in rows:
        ip, lp = f'{DFIRE}/{sp}/images/{name}.jpg', f'{DFIRE}/{sp}/labels/{name}.txt'
        b = open(ip, 'rb').read()
        im = Image.open(io.BytesIO(b))
        w, h, mode = im.width, im.height, im.mode
        r = dict(name=name, orig_split=sp, path=os.path.relpath(ip, ROOT), w=w, h=h, mode=mode,
                 md5=hashlib.md5(b).hexdigest(), nbytes=len(b), n_fire=0, n_smoke=0, n_bad_box=0, n_bad_line=0)
        for line in open(lp).read().strip().splitlines():
            t = line.split()
            if not t:
                continue
            if len(t) != 5:
                r['n_bad_line'] += 1
                continue
            c, cx, cy, bw, bh = int(float(t[0])), *map(float, t[1:])
            ok = c in (0, 1) and bw > 0 and bh > 0
            if not ok:
                r['n_bad_box'] += 1
            else:
                r['n_fire' if c == 1 else 'n_smoke'] += 1
            boxes.append(dict(name=name, cls='fire' if c == 1 else 'smoke', cx=cx, cy=cy, bw=bw, bh=bh, valid=ok))
        r['overlays'] = json.dumps(X.overlay_boxes(name, np.asarray(im.convert('RGB')))) if name.startswith('AoF') else '[]'
        out.append(r)
    return out, boxes


def dfire_table(rebuild=False):
    """D-Fire 사진 한 장에 한 줄(크기, md5, 박스 수, 감시 카메라 화면 글자 자리) + 박스 하나에 한 줄."""
    pi, pb = f'{CACHE}/ext_dfire_images.parquet', f'{CACHE}/ext_dfire_boxes.parquet'
    if os.path.exists(pi) and os.path.exists(pb) and not rebuild:
        return pd.read_parquet(pi), pd.read_parquet(pb)
    items = sorted((sp, os.path.basename(p)[:-4]) for sp in ('train', 'test') for p in glob.glob(f'{DFIRE}/{sp}/images/*.jpg'))
    res = _pool(_dfire_job, [items[i:i + 200] for i in range(0, len(items), 200)])
    img = pd.DataFrame([r for a, _ in res for r in a])
    box = pd.DataFrame([r for _, b in res for r in b])
    img.to_parquet(pi, index=False)
    box.to_parquet(pb, index=False)
    return img, box


def dfire_pairing():
    """사진·라벨 파일 짝 맞추기 (이름 목록만)."""
    rows = []
    for sp in ('train', 'test'):
        ims = {os.path.basename(p)[:-4] for p in glob.glob(f'{DFIRE}/{sp}/images/*')}
        lbs = {os.path.basename(p)[:-4] for p in glob.glob(f'{DFIRE}/{sp}/labels/*')}
        rows.append(dict(분할=sp, 사진=len(ims), 라벨=len(lbs), 라벨_없는_사진=len(ims - lbs), 사진_없는_라벨=len(lbs - ims)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- DeepFire 표

def deepfire_table():
    rows = []
    for sp_dir, sp in (('Training and Validation', 'trainval'), ('Testing', 'test')):
        for cls in ('fire', 'nofire'):
            for p in sorted(glob.glob(f'{DEEPFIRE}/{sp_dir}/{cls}/*')):
                b = open(p, 'rb').read()
                im = Image.open(io.BytesIO(b))
                rows.append(dict(name=os.path.basename(p), orig_split=sp, folder=cls, path=os.path.relpath(p, ROOT),
                                 w=im.width, h=im.height, mode=im.mode, md5=hashlib.md5(b).hexdigest(), nbytes=len(b)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 전부 처리해 보기

def _process_job(rows):
    out, ths = [], []
    for key, path, ovj in rows:
        r = {'key': key}
        th = np.zeros((18, 32), np.uint8)
        try:
            b = open(f'{ROOT}/{path}', 'rb').read()
            boxes = json.loads(ovj)
            t0 = time.perf_counter()
            fit = X.preprocess_external(b, boxes, 'fit')
            native = X.preprocess_external(b, boxes, 'native')
            r.update(out_w=fit.width, out_h=fit.height, t_prep=time.perf_counter() - t0)
            st = E.pixel_stats(native)                      # 가린 뒤, 회색 여백 없이 사진 내용만 (짧은 변 256px)
            r.update({('p_' + k if k not in ('w', 'h') else 'nat_' + k): v for k, v in st.items()})
            th = np.asarray(native.convert('L').resize((32, 18), Image.BILINEAR), dtype=np.uint8)
        except Exception as e:
            r['error'] = repr(e)[:200]
        out.append(r)
        ths.append(th)
    return out, np.stack(ths)


def process_all(items, name, rebuild=False):
    """items: DataFrame[key, path, overlays] → 처리 결과 표 + 32×18 회색 썸네일(npy)."""
    pp, pt = f'{CACHE}/{name}.parquet', f'{CACHE}/{name}_thumbs.npy'
    if os.path.exists(pp) and os.path.exists(pt) and not rebuild:
        return pd.read_parquet(pp), np.load(pt)
    it = list(items[['key', 'path', 'overlays']].itertuples(index=False, name=None))
    res = _pool(_process_job, [it[i:i + 100] for i in range(0, len(it), 100)], procs=18)
    df = pd.DataFrame([r for a, _ in res for r in a])
    th = np.concatenate([t for _, t in res])
    df.to_parquet(pp, index=False)
    np.save(pt, th)
    return df, th


# ---------------------------------------------------------------- 거의 같은 사진 묶음 (썸네일)

def near_dup_groups(th, thr=6.0, k=8, chunk=4000):
    """32×18 썸네일(밝기 평균·대비를 맞춤, check_aihub_leak.py와 같은 척도)의 평균 절대차가 thr 미만인 사진끼리 묶는다.
    사진마다 BLAS L2 거리 상위 k개만 정확히 잰다. 반환: 묶음 번호 배열, 이은 쌍 수."""
    x = th.reshape(len(th), -1).astype(np.float32)
    x = (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1) * 40
    n = len(x)
    par = np.arange(n)

    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a

    sq = (x * x).sum(1)
    pairs = 0
    for i0 in range(0, n, chunk):
        q = x[i0:i0 + chunk]
        d2 = sq[i0:i0 + chunk, None] + sq[None, :] - 2 * q @ x.T
        d2[np.arange(len(q)), np.arange(i0, i0 + len(q))] = np.inf
        top = np.argpartition(d2, k, axis=1)[:, :k]
        exact = np.abs(q[:, None, :] - x[top]).mean(2)
        for a, (js, ds) in enumerate(zip(top, exact)):
            for j, d in zip(js, ds):
                if d < thr:
                    ra, rb = find(i0 + a), find(int(j))
                    if ra != rb:
                        par[ra] = rb
                    pairs += 1
    return np.array([find(i) for i in range(n)]), pairs
