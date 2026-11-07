"""산불 탐지 프로젝트 전처리 노트북(preprocess.ipynb)에서 쓰는 무거운 계산.

사진 처리 함수 자체는 학습 코드와 같이 쓰는 ../wildfire_data.py에 있다(preprocess, to_processed_xy).
여기서는 (1) 음성 사진의 방해 요소 다각형을 래스터로 그려 처리 전·후 화면 면적을 재고,
(2) 본 실험 사진 전부를 실제로 처리해 크기·오류·픽셀 통계·채널 평균을 잰다. 결과는 cache/에 parquet으로 남긴다.
"""
import io, json, time, zipfile

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

import eda_lib as E
import wildfire_data as W

GRID_W, GRID_H = 480, 270             # 1920×1080의 1/4 격자에 다각형을 그린다
GRID_TOP = W.CROP_TOP * GRID_W // W.BASE_W   # 잘린 위 10% = 격자 27줄
DIST_CATS = {4: '구름', 5: '안개/연무', 6: '굴뚝연기'}


# ---------------------------------------------------------------- 방해 요소 면적 (다각형 래스터)

def _masks(d, w):
    """JSON 하나 → 종류별 0/1 격자(겹친 다각형·같은 박스 중복은 합집합으로 한 번만 센다). segmentation이 비면 bbox를 쓴다."""
    g = GRID_W / w
    out = {}
    for a in d.get('annotations') or []:
        c = a.get('category_id')
        if c not in DIST_CATS:
            continue
        m = out.setdefault(c, Image.new('L', (GRID_W, GRID_H), 0))
        dr = ImageDraw.Draw(m)
        polys = [p for p in a.get('segmentation') or [] if isinstance(p, list) and len(p) >= 6 and len(p) % 2 == 0]
        if polys:
            for p in polys:
                dr.polygon([(p[i] * g, p[i + 1] * g) for i in range(0, len(p), 2)], fill=1)
        else:
            x, y, bw, bh = a['bbox']
            dr.rectangle([x * g, y * g, (x + bw) * g, (y + bh) * g], fill=1)
    return {c: np.asarray(m, dtype=bool) for c, m in out.items()}


def _area_job(job):
    split, items = job
    if split not in E._LZ:
        E._LZ[split] = zipfile.ZipFile(E.LABEL_ZIP[split])
    z = E._LZ[split]
    rows = []
    for idx, path, w in items:
        ms = _masks(json.loads(z.read(path)), w)
        r = {'idx': idx}
        empty = np.zeros((GRID_H, GRID_W), bool)
        for c, name in DIST_CATS.items():
            m = ms.get(c, empty)
            r[f'full_{name}'] = float(m.mean())
            r[f'crop_{name}'] = float(m[GRID_TOP:].mean())
        cf = ms.get(4, empty) | ms.get(5, empty)
        r['full_구름안개'] = float(cf.mean())
        r['crop_구름안개'] = float(cf[GRID_TOP:].mean())
        rows.append(r)
    return rows


def distractor_area(img, rebuild=False):
    """img: DataFrame[idx, official(train/val), path(JSON), w] → 종류별 화면 면적 비율(처리 전 full_*, 처리 후 crop_*)."""
    def run():
        jobs = []
        for split, g in img.groupby('official'):
            it = list(g[['idx', 'path', 'w']].itertuples(index=False, name=None))
            jobs += [(split, it[i:i + 2000]) for i in range(0, len(it), 2000)]
        res = E.pmap(_area_job, jobs, procs=18)
        return pd.DataFrame([r for part in res for r in part])
    return E.cached('prep_distractor_area', run, rebuild)


# ---------------------------------------------------------------- 사진 전부 처리해 보기

def _process_job(job):
    out = []
    for key, src, member in job:
        r = {'key': key}
        try:
            t0 = time.perf_counter()
            b = E.image_zip(src).read(member)
            t1 = time.perf_counter()
            im = W.preprocess(b)
            t2 = time.perf_counter()
            a = np.asarray(im, dtype=np.float32) / 255
            r.update(out_w=im.width, out_h=im.height, t_read=t1 - t0, t_prep=t2 - t1,
                     **{f'm_{c}': float(a[..., i].mean()) for i, c in enumerate('rgb')},
                     **{f'q_{c}': float((a[..., i] ** 2).mean()) for i, c in enumerate('rgb')})
            r.update(E.pixel_stats(im))
        except Exception as e:
            r['error'] = repr(e)[:200]
        out.append(r)
    return out


def process_all(items, rebuild=False):
    """items: DataFrame[key, src, member] → 사진마다 처리 결과 크기, 걸린 시간, 채널 평균·제곱 평균, pixel_stats(처리 후)."""
    def run():
        for s in items.src.unique():
            E.image_zip(s)
        it = list(items[['key', 'src', 'member']].itertuples(index=False, name=None))
        res = E.pmap(_process_job, [it[i:i + 300] for i in range(0, len(it), 300)], procs=18)
        return pd.DataFrame([r for part in res for r in part])
    return E.cached('prep_processed_stats', run, rebuild)


# ---------------------------------------------------------------- 처리 전·후 그림

def before_after(row, boxes, tw=560):
    """원본(빨간 선 = 자르는 위치)과 처리 후 사진을 나란히. boxes: 처리 후 좌표 (x0, y0, x1, y1, 이름)."""
    b = E.image_zip(row.src).read(row.member)
    orig = Image.open(io.BytesIO(b)).convert('RGB')
    proc = W.preprocess(b)
    s0 = tw / orig.width
    a = orig.resize((tw, round(orig.height * s0)))
    yc = round(W.CROP_TOP * orig.width / W.BASE_W * s0)
    band = a.crop((0, 0, tw, yc))
    a.paste(Image.blend(band, Image.new('RGB', band.size, '#000000'), 0.6), (0, 0))   # 잘려 나가는 띠를 어둡게
    ImageDraw.Draw(a).line([(0, yc), (tw, yc)], fill='#e0301e', width=2)
    s1 = tw / proc.width
    p = proc.resize((tw, round(proc.height * s1)))
    dp = ImageDraw.Draw(p)
    for x0, y0, x1, y1, nm in boxes:
        dp.rectangle([x0 * s1, y0 * s1, max(x1 * s1, x0 * s1 + 5), max(y1 * s1, y0 * s1 + 5)],
                     outline=E.BOX_COLOR.get(nm, '#ffffff'), width=2)
    return a, p, orig.size


def top_strips(rows, band=150, tw=480, cols=4):
    """카메라마다 한 장의 맨 위 band px(1920 폭 기준)를 모은 격자. 빨간 선 = 자르는 위치(108px).
    rows: DataFrame[src, member, label] → (JPEG 바이트, 띠 사진 목록)."""
    strips = []
    s = tw / W.BASE_W
    for r in rows.itertuples():
        im = Image.open(io.BytesIO(E.image_zip(r.src).read(r.member))).convert('RGB')
        if im.width != W.BASE_W:
            im = im.resize((W.BASE_W, round(im.height * W.BASE_W / im.width)))
        st = im.crop((0, 0, W.BASE_W, band)).resize((tw, round(band * s)))
        ImageDraw.Draw(st).line([(0, W.CROP_TOP * s), (tw, W.CROP_TOP * s)], fill='#e0301e', width=1)
        strips.append(st)
    h = strips[0].height
    rws = (len(strips) + cols - 1) // cols
    S = Image.new('RGB', (cols * (tw + 3), rws * (h + 3)), '#ffffff')
    for i, st in enumerate(strips):
        S.paste(st, ((i % cols) * (tw + 3), (i // cols) * (h + 3)))
    buf = io.BytesIO()
    S.save(buf, 'JPEG', quality=80)
    return buf.getvalue(), strips


# ---------------------------------------------------------------- 선택지 시험: 4K 출신 사진 선명도 맞추기

def _blur_job(job):
    from PIL import ImageFilter
    radius, rows = job
    out = []
    for key, src, member in rows:
        im = W.preprocess(E.image_zip(src).read(member)).filter(ImageFilter.GaussianBlur(radius))
        r = E.pixel_stats(im)
        r['key'] = key
        out.append(r)
    return out


def blur_stats(items, radius=1.25, rebuild=False):
    """items: DataFrame[key, src, member] → 처리 후 가우시안 블러(radius)를 더한 사진의 pixel_stats (선택지 효과 재기용)."""
    def run():
        for s in items.src.unique():
            E.image_zip(s)
        it = list(items[['key', 'src', 'member']].itertuples(index=False, name=None))
        res = E.pmap(_blur_job, [(radius, it[i:i + 100]) for i in range(0, len(it), 100)], procs=18)
        return pd.DataFrame([r for part in res for r in part])
    return E.cached(f'prep_blur{radius}_sample', run, rebuild)
