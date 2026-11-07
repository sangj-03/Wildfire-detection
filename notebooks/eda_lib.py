"""산불 탐지 프로젝트 EDA 노트북(eda.ipynb)에서 쓰는 함수.

무거운 계산(AI Hub 라벨 63만 개 읽기, 사진을 열어 픽셀 재기)은 여기서 여러 프로세스로 돌리고, 결과는 cache/에 parquet으로 남긴다.
AI Hub 학습 사진(TS.z01~TS.zip 분할 zip)은 합치지 않고 aihub_split_zip.SplitZip으로 읽는다(검증 VS.zip도 같은 방법).
"""
import io, json, os, re, sys, zipfile
import multiprocessing as mp

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))  # the project folder (this file is in notebooks/)
sys.path.insert(0, ROOT)
from aihub_split_zip import SplitZip  # noqa: E402

AIHUB = ('/data/aihub/71330/265.지역안전재난(산불)_방재의_고도화를_위한_대규모_인공지능_데이터베이스_구축/'
         '01-1.정식개방데이터')
LABEL_ZIP = {'train': f'{AIHUB}/Training/02.라벨링데이터/TL.zip', 'val': f'{AIHUB}/Validation/02.라벨링데이터/VL.zip'}
IMAGE_ZIP = {'train': f'{AIHUB}/Training/01.원천데이터/TS.zip', 'val': f'{AIHUB}/Validation/01.원천데이터/VS.zip'}
REPORT = f'{ROOT}/report/aihub'
CACHE = f'{ROOT}/notebooks/cache'
FONT_PATH = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
CAT = {1: '흑색연기', 2: '백색/회색연기', 3: '화염', 4: '구름', 5: '안개/연무', 6: '굴뚝연기'}
STD_CATEGORIES = list(CAT.items())
ENV_KEYS = ('address', 'datetime', 'timezone', 'device', 'collection', 'weather', 'humidity', 'distance', 'gps', 'elevation')
# 실제 양성 16곳 중 쓰지 않는 12곳 (2026-10-15 결정, summarize_aihub_labels.py와 같은 목록)
DROP_POS_PLACES = ['강원도강릉시' + p for p in ['경포대공터', '경포대입구', '모래고개입구', '춘갑봉등산로', '춘갑봉민가',
                   '옥천면비닐하우스', '옥천면교각밑', '춘갑봉샛길', '옥천면계곡', '사천면', '죽헌저수지방둑', '죽헌저수지공터']]
# 사진 위 박스 색: 화염·연기는 차트의 불꽃·연기 색을 밝게, 방해 요소(구름·안개·굴뚝)는 흰색
BOX_COLOR = {'화염': '#ff7b42', '흑색연기': '#4d9bff', '백색/회색연기': '#4d9bff',
             '구름': '#ffffff', '안개/연무': '#ffffff', '굴뚝연기': '#ffffff'}
os.makedirs(CACHE, exist_ok=True)


# ---------------------------------------------------------------- 공통

def cached(name, fn, rebuild=False):
    """cache/<name>.parquet이 있으면 읽고, 없으면 fn()을 돌려 저장한다."""
    p = f'{CACHE}/{name}.parquet'
    if os.path.exists(p) and not rebuild:
        return pd.read_parquet(p)
    df = fn()
    df.to_parquet(p, index=False)
    return df


_ZIPS = {}


def image_zip(src):
    """AI Hub 원천 zip('train' = TS 분할 zip, 'val' = VS.zip). 중앙 디렉터리는 한 번만 읽는다."""
    if src not in _ZIPS:
        _ZIPS[src] = SplitZip(IMAGE_ZIP[src])
    return _ZIPS[src]


def pmap(fn, jobs, procs=16):
    """fork로 여러 프로세스에 나눠 돌린다. 열린 zip 파일은 먼저 닫는다(프로세스끼리 파일 위치를 같이 쓰지 않게)."""
    for z in _ZIPS.values():
        z.close()
    with mp.get_context('fork').Pool(procs) as pool:
        return pool.map(fn, jobs, chunksize=1)


def short_place(addr):
    """'강원도강릉시솔향수목원' → '솔향수목원', '경기도파주시적성면' → '파주시적성면'."""
    s = str(addr)
    s = re.sub(r'^강원도강릉시', '', s)
    return re.sub(r'^(서울특별시|부산광역시|대구광역시|인천광역시|광주광역시|대전광역시|울산광역시|세종특별자치시|경기도|강원도|'
                  r'강원특별자치도|충청북도|충청남도|전라북도|전북특별자치도|전라남도|경상북도|경상남도|제주특별자치도)', '', s)


def norm_date(s):
    """촬영 시각 앞부분을 YYYYMMDD로 (20220914_..., 221027_... 두 형식). make_aihub_split.py와 같은 규칙."""
    s = str(s)
    if m := re.match(r'(20\d{6})_', s):
        return m.group(1)
    if m := re.match(r'(\d{6})_', s):
        return '20' + m.group(1)
    return None


# ---------------------------------------------------------------- AI Hub 라벨 전수 읽기

def _shoelace(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


_LZ = {}


def _parse_labels(job):
    split, start, names = job
    if split not in _LZ:                       # 프로세스마다 라벨 zip을 한 번만 연다
        _LZ[split] = zipfile.ZipFile(LABEL_ZIP[split])
    z = _LZ[split]
    imgs, anns = [], []
    for k, n in enumerate(names):
        idx = start + k
        r = dict(idx=idx, split=split, path=n, parse_error=None)
        try:
            d = json.loads(z.read(n))
        except Exception as e:
            r['parse_error'] = repr(e)[:200]
            imgs.append(r)
            continue
        parts = n.split('/')
        db, obj = parts[0].split('_', 1)
        m = re.search(r'_(\d+)\.json$', n)
        info = d.get('info') or {}
        ims = d.get('images') or []
        im0 = ims[0] if ims else {}
        env = d.get('environment') or {}
        r.update(db=db, folder_obj=obj, place_cam=parts[1] if len(parts) > 3 else None,
                 video_fuel=parts[2] if len(parts) > 3 else None, depth=len(parts), frame=int(m.group(1)) if m else -1,
                 top_keys='|'.join(d.keys()), info_name=info.get('name'), license=info.get('license'),
                 date_created=info.get('date_created'), camera_id=info.get('camera_id'), video_id=info.get('video_id'),
                 n_images=len(ims), w=im0.get('width'), h=im0.get('height'), file_name=im0.get('file_name'),
                 image_id=im0.get('id'),
                 cats_std=[(c.get('id'), c.get('name')) for c in d.get('categories') or []] == STD_CATEGORIES)
        for key in ENV_KEYS:
            v = env.get(key)
            r[key] = None if v is None else str(v)
        r['env_missing'] = sum(key not in env for key in ENV_KEYS)
        annl = d.get('annotations') or []
        r['n_ann'] = len(annl)
        imgs.append(r)
        for a in annl:
            bb = list(a.get('bbox') or [])
            x, y, bw, bh = (bb + [np.nan] * 4)[:4]
            seg = a.get('segmentation') or []
            polys = [np.asarray(p, dtype=float).reshape(-1, 2) for p in seg
                     if isinstance(p, list) and len(p) >= 6 and len(p) % 2 == 0]
            row = dict(idx=idx, ann_id=a.get('id'), image_id=a.get('image_id'), cat=a.get('category_id'),
                       x=float(x), y=float(y), bw=float(bw), bh=float(bh), n_bbox=len(bb),
                       area=float(a.get('area') if a.get('area') is not None else np.nan), iscrowd=a.get('iscrowd'),
                       n_poly=len(seg), bad_seg=(not seg) or len(polys) != len(seg), n_pts=sum(len(p) for p in polys),
                       px0=np.nan, py0=np.nan, px1=np.nan, py1=np.nan, poly_area=np.nan)
            if polys:
                allp = np.concatenate(polys)
                row.update(px0=allp[:, 0].min(), py0=allp[:, 1].min(), px1=allp[:, 0].max(), py1=allp[:, 1].max(),
                           poly_area=float(sum(_shoelace(p) for p in polys)))
            anns.append(row)
    return imgs, anns


def aihub_labels(rebuild=False):
    """TL.zip·VL.zip의 JSON 전부 → (사진 표, 물체 표). 사진 한 장 = 한 줄, 물체(annotation) 하나 = 한 줄."""
    pi, pa = f'{CACHE}/aihub_images.parquet', f'{CACHE}/aihub_ann.parquet'
    if os.path.exists(pi) and os.path.exists(pa) and not rebuild:
        return pd.read_parquet(pi), pd.read_parquet(pa)
    jobs, start = [], 0
    for split in ('train', 'val'):
        names = sorted(n for n in zipfile.ZipFile(LABEL_ZIP[split]).namelist() if n.endswith('.json'))
        jobs += [(split, start + i, names[i:i + 4000]) for i in range(0, len(names), 4000)]
        start += len(names)
    res = pmap(_parse_labels, jobs)
    img = pd.DataFrame([r for a, _ in res for r in a]).sort_values('idx').reset_index(drop=True)
    ann = pd.DataFrame([r for _, b in res for r in b])
    img.to_parquet(pi, index=False)
    ann.to_parquet(pa, index=False)
    return img, ann


def aihub_zip_listing():
    """라벨 zip과 원천 zip에 든 파일 이름 전부 (짝 맞추기 검사용)."""
    rows = []
    for split, p in LABEL_ZIP.items():
        rows += [('label', split, n) for n in zipfile.ZipFile(p).namelist() if not n.endswith('/')]
    for split in IMAGE_ZIP:
        rows += [('image', split, n) for n in image_zip(split).namelist() if not n.endswith('/')]
    return pd.DataFrame(rows, columns=['kind', 'split', 'name'])


def aihub_full_pixels():
    """analyze_aihub_images.py가 원천 사진 63만 장을 전부 열어 잰 값(실제 가로·세로, 밝기 등)과 32×18 썸네일."""
    tabs, ths = [], []
    for src in ('train', 'val'):
        t = pd.read_csv(f'{REPORT}/img_{src}.csv.gz')
        t.insert(0, 'src', src)
        tabs.append(t)
        ths.append(np.load(f'{REPORT}/img_{src}_thumbs.npy'))
    tab = pd.concat(tabs, ignore_index=True)
    th = np.concatenate(ths).reshape(len(tab), -1).astype(np.float32)
    th = (th - th.mean(1, keepdims=True)) / (th.std(1, keepdims=True) + 1) * 40   # check_aihub_leak.py와 같은 척도
    return tab, th


# ---------------------------------------------------------------- 픽셀 측정

HUE_BINS = 12   # 30도 간격


def pixel_stats(im):
    """짧은 변 256px로 줄여 잰다(analyze_aihub_images.py와 같은 정의).
    불꽃색 = 색상 <60° 또는 >340°, 채도 >100, 명도 >150. 회색 = 채도 <40. 어두운 화소 = 회색조 <30."""
    q = getattr(im, 'quantization', None)
    im = im.convert('RGB')
    w, h = im.size
    s = 256 / min(w, h)
    sm = im.resize((max(1, round(w * s)), max(1, round(h * s))))
    g = np.asarray(sm, dtype=np.float32).mean(2)
    hsv = np.asarray(sm.convert('HSV'), dtype=np.float32)
    hue, sat, val = hsv[..., 0] * 360 / 255, hsv[..., 1], hsv[..., 2]
    lap = g[1:-1, 1:-1] * 4 - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    fire = ((hue < 60) | (hue > 340)) & (sat > 100) & (val > 150)
    hh = np.histogram(hue[sat > 60], bins=HUE_BINS, range=(0, 360))[0] / g.size
    out = dict(w=w, h=h, jpeg_q0=(q[0][0] if q else -1), bright=float(g.mean()), contrast=float(g.std()),
               sat=float(sat.mean()), sharp=float(lap.var()), dark_px=float((g < 30).mean()), fire_px=float(fire.mean()),
               gray_px=float((sat < 40).mean()))
    out.update({f'hue{i:02d}': float(v) for i, v in enumerate(hh)})
    return out


def _pixels_aihub(job):
    out = []
    for src, jpg in job:
        try:
            b = image_zip(src).read(jpg)
            r = pixel_stats(Image.open(io.BytesIO(b)))
            r['nbytes'] = len(b)
        except Exception as e:
            r = dict(error=repr(e)[:200])
        r.update(src=src, jpg=jpg)
        out.append(r)
    return out


def aihub_pixels(items, rebuild=False):
    """items: DataFrame[src, jpg(zip 안 경로)] → 사진마다 pixel_stats."""
    def run():
        for s in items.src.unique():
            image_zip(s)                       # 중앙 디렉터리는 fork 전에 한 번만 읽는다
        pairs = list(items[['src', 'jpg']].itertuples(index=False, name=None))
        res = pmap(_pixels_aihub, [pairs[i:i + 400] for i in range(0, len(pairs), 400)])
        return pd.DataFrame([r for part in res for r in part])
    return cached('aihub_pixels_sample', run, rebuild)


# ---------------------------------------------------------------- 근접 중복 (썸네일)

def nearest_diff(th, q, cand, chunk=20000):
    """q·cand: 썸네일 행 번호. 질문 사진마다 후보 중 가장 비슷한 사진의 (평균 절대차, 행 번호).
    BLAS로 L2 거리 상위 5개를 고른 뒤 그 5개만 평균 절대차를 정확히 잰다."""
    Q = th[q]
    best_d = np.full(len(q), np.inf, dtype=np.float32)
    best_j = np.full(len(q), -1, dtype=np.int64)
    qq = (Q * Q).sum(1)[:, None]
    for i0 in range(0, len(cand), chunk):
        cj = np.asarray(cand[i0:i0 + chunk])
        C = th[cj]
        d2 = qq + (C * C).sum(1)[None, :] - 2 * Q @ C.T
        k = min(5, len(cj))
        top = np.argpartition(d2, k - 1, axis=1)[:, :k]
        exact = np.abs(Q[:, None, :] - C[top]).mean(2)
        a = exact.argmin(1)
        d = exact[np.arange(len(q)), a]
        better = d < best_d
        best_d[better] = d[better]
        best_j[better] = cj[top[np.arange(len(q)), a]][better]
    return best_d, best_j


# ---------------------------------------------------------------- 지름길 점검: 깊이 2 결정 나무

def _gini_split(x, y, w, n_cls, n_thr=48):
    """특성 하나에서 가중 지니가 가장 작아지는 임계값."""
    best = (np.inf, None)
    for t in np.unique(np.quantile(x, np.linspace(0.02, 0.98, n_thr))):
        L = x < t
        g = 0.0
        for side in (L, ~L):
            ws = w[side].sum()
            if ws == 0:
                continue
            p = np.bincount(y[side], weights=w[side], minlength=n_cls) / ws
            g += ws * (1 - (p * p).sum())
        if g < best[0]:
            best = (g, t)
    return best


def _fit_node(X, y, w, n_cls, depth):
    leaf = int(np.argmax(np.bincount(y, weights=w, minlength=n_cls)))
    if depth == 0 or len(np.unique(y)) == 1:
        return ('leaf', leaf)
    best = (np.inf, None, None)
    for j in range(X.shape[1]):
        g, t = _gini_split(X[:, j], y, w, n_cls)
        if t is not None and g < best[0]:
            best = (g, j, t)
    if best[1] is None:
        return ('leaf', leaf)
    _, j, t = best
    L = X[:, j] < t
    return ('split', j, t, _fit_node(X[L], y[L], w[L], n_cls, depth - 1), _fit_node(X[~L], y[~L], w[~L], n_cls, depth - 1))


def _predict(node, X):
    if node[0] == 'leaf':
        return np.full(len(X), node[1])
    _, j, t, a, b = node
    out = np.empty(len(X), dtype=int)
    L = X[:, j] < t
    out[L] = _predict(a, X[L])
    out[~L] = _predict(b, X[~L])
    return out


def _describe(node, feats, classes, ind=''):
    if node[0] == 'leaf':
        return f'{ind}→ {classes[node[1]]}'
    _, j, t, a, b = node
    return (f'{ind}{feats[j]} < {t:.4g}\n{_describe(a, feats, classes, ind + "   ")}\n'
            f'{ind}{feats[j]} ≥ {t:.4g}\n{_describe(b, feats, classes, ind + "   ")}')


def shortcut_tree(train, test, feats, label='label', depth=2):
    """클래스 균형 가중치로 깊이 2 나무를 학습 쪽에서 만들고, 시험 쪽 균형 정확도(클래스별 재현율 평균)를 잰다.
    사진 내용을 보지 않는 숫자 몇 개로 얼마나 맞히는지 = 모델이 쓸 수 있는 지름길의 크기."""
    classes = sorted(train[label].unique())
    ci = {c: i for i, c in enumerate(classes)}
    ytr = train[label].map(ci).to_numpy()
    yte = test[label].map(ci).to_numpy()
    w = 1.0 / np.bincount(ytr, minlength=len(classes))[ytr]
    node = _fit_node(train[feats].to_numpy(float), ytr, w, len(classes), depth)
    pred = _predict(node, test[feats].to_numpy(float))
    rec = [float((pred[yte == i] == i).mean()) for i in range(len(classes)) if (yte == i).any()]
    return float(np.mean(rec)), _describe(node, feats, classes)


# ---------------------------------------------------------------- 사진 격자

_FONTS = {}


def _font(size):
    if size not in _FONTS:
        _FONTS[size] = ImageFont.truetype(FONT_PATH, size, index=1)   # 1 = Noto Sans CJK KR
    return _FONTS[size]


def load_image(jpg, src):
    """AI Hub 원천 사진 한 장 (jpg = zip 안 경로, src = 'train'/'val' 공식 분할의 zip)."""
    return Image.open(io.BytesIO(image_zip(src).read(jpg))).convert('RGB')


def grid(tiles, cols=8, tw=240, th=None, title=None, cap_lines=1):
    """tiles: dict(img=PIL, caption=str, boxes=[(x, y, w, h, 이름)] 원본 픽셀 좌표) 목록 → JPEG 바이트.
    th를 주면 칸 높이를 고정하고 사진을 그 안에 맞춘다(비율 유지, 남는 곳은 회색)."""
    th = th or round(tw * 9 / 16)
    cap_h = 15 * cap_lines + 4 if any(t.get('caption') for t in tiles) else 0
    title_h = 26 if title else 0
    rows = (len(tiles) + cols - 1) // cols
    W, H = cols * (tw + 4) + 4, title_h + rows * (th + cap_h + 4) + 4
    S = Image.new('RGB', (W, H), '#f4f4f2')
    dr = ImageDraw.Draw(S)
    if title:
        dr.text((6, 4), title, fill='#1a1a19', font=_font(16))
    for k, t in enumerate(tiles):
        im = t['img']
        s = min(tw / im.width, th / im.height)
        sm = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))))
        x0 = 4 + (k % cols) * (tw + 4)
        y0 = title_h + 4 + (k // cols) * (th + cap_h + 4)
        cell = Image.new('RGB', (tw, th), '#b9b8b3')
        ox, oy = (tw - sm.width) // 2, (th - sm.height) // 2
        cell.paste(sm, (ox, oy))
        cd = ImageDraw.Draw(cell)
        for (bx, by, bw, bh, nm) in t.get('boxes') or []:
            x1, y1 = ox + bx * s, oy + by * s
            x2, y2 = ox + (bx + bw) * s, oy + (by + bh) * s
            if x2 - x1 < 6:                     # 아주 작은 불꽃도 보이도록 최소 6px 상자
                c = (x1 + x2) / 2; x1, x2 = c - 3, c + 3
            if y2 - y1 < 6:
                c = (y1 + y2) / 2; y1, y2 = c - 3, c + 3
            cd.rectangle([x1, y1, x2, y2], outline=BOX_COLOR.get(nm, '#ffffff'), width=2)
        S.paste(cell, (x0, y0))
        if t.get('caption'):
            dr.multiline_text((x0 + 1, y0 + th + 2), t['caption'], fill='#3d3c39', font=_font(12), spacing=2)
    buf = io.BytesIO()
    S.save(buf, 'JPEG', quality=82)
    return buf.getvalue()


def crop_box(im, box, scale=3.0, min_side=64):
    """박스 주변을 넓게 잘라 낸다(작은 불꽃을 크게 보기)."""
    x, y, w, h = box
    side = max(min_side, scale * max(w, h))
    cx, cy = x + w / 2, y + h / 2
    l, t = max(0, cx - side / 2), max(0, cy - side / 2)
    r, b = min(im.width, l + side), min(im.height, t + side)
    return im.crop((int(l), int(t), int(r), int(b))), (x - int(l), y - int(t), w, h)
