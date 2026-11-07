"""받아 둔 산불 데이터의 '음성(산불 아님)' 사진에서 불꽃처럼 보이는 색 영역이 있는 사진을 센다.

라벨에 불꽃놀이·노을·조명 같은 분류가 없으므로 색으로 후보를 고른다.
불꽃색 = 빨강~노랑(색상 <60° 또는 >340°), 진함(채도 >100), 밝음(명도 >150) — DeepFire 점검과 같은 기준.
짧은 변 256으로 줄여 잰다. 같은 기준으로 불꽃 사진(양성)도 재서 '불꽃 사진이면 보통 얼마나 나오는지'와 비교한다.
사용: python3 -I check_fire_lookalike.py  →  report/lookalike/measure.csv.gz, 그 뒤 격자 그림은 같은 파일을 sheets 인자로
"""
import io, os, random, sys, zipfile
from multiprocessing import Pool

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, '/data/home/seame/dl-team-project/team6')
from aihub_split_zip import SplitZip

ROOT = '/data/home/seame/dl-team-project/team6'
OUT = f'{ROOT}/report/lookalike'
BASE = ('/data/aihub/71330/265.지역안전재난(산불)_방재의_고도화를_위한_대규모_인공지능_데이터베이스_구축/'
        '01-1.정식개방데이터')
AZ = {'val': f'{BASE}/Validation/01.원천데이터/VS.zip', 'train': f'{BASE}/Training/01.원천데이터/TS.zip'}
FONT = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 12, index=3)
os.makedirs(OUT, exist_ok=True)
ZF = {}


def opener(src):
    if src not in ZF:
        ZF[src] = SplitZip(AZ[src]) if src == 'train' else zipfile.ZipFile(AZ[src])
    return ZF[src]


def load(item):
    src, key = item
    if src in AZ:
        return Image.open(io.BytesIO(opener(src).read(key)))
    return Image.open(key)


def measure(job):
    out = []
    for r in job:
        try:
            im = load((r['src'], r['key'])).convert('RGB'); w, h = im.size
            s = 256 / min(w, h); sm = im.resize((max(1, round(w * s)), max(1, round(h * s))))
            hsv = np.asarray(sm.convert('HSV'), dtype=np.float32)
            hue, sat, val = hsv[..., 0] * 360 / 255, hsv[..., 1], hsv[..., 2]
            fire = ((hue < 60) | (hue > 340)) & (sat > 100) & (val > 150)
            hot = fire & (val > 230)                                   # 아주 밝은 불꽃색 점(조명·불꽃놀이·해)
            g = np.asarray(sm, dtype=np.float32).mean(2)
            top = fire[: fire.shape[0] // 2]
            out.append(dict(r, w=w, h=h, fire_px=fire.mean(), hot_px=hot.mean(), fire_top=top.mean(), bright=g.mean()))
        except Exception as e:
            out.append(dict(r, error=repr(e)[:120]))
    return out


def items():
    rows = []
    sp = pd.read_csv(f'{ROOT}/report/aihub/split_v1.csv.gz', usecols=['jpg', 'src', 'db', 'label', 'path'])
    names = {}
    for src in ('val', 'train'):
        z = opener(src); names.update({n.rsplit('/', 1)[-1]: (src, n) for n in z.namelist() if n.lower().endswith('.jpg')})
    ZF['train'].close()          # 중앙 디렉터리는 한 번만 읽고, 파일은 fork 뒤 프로세스마다 따로 연다
    del ZF['val']                # zipfile 핸들은 프로세스끼리 나눠 쓰면 안 되므로 각자 새로 연다
    neg = sp[sp.db == 'NegativeDB']
    random.seed(0)
    fire = sp[sp.label == 'fire'].sample(5000, random_state=0)      # 비교용 불꽃 사진
    for t, kind in ((neg, 'neg'), (fire, 'pos_fire')):
        for x in t.itertuples():
            src, key = names[x.jpg]
            rows.append(dict(dataset='AIHub', kind=kind, sub=x.path.split('/')[0].replace('NegativeDB_', '') + ('(연기 박스)' if x.label == 'neg_with_smoke' else ''),
                             name=x.jpg, src=src, key=key))
    d = pd.read_csv(f'{ROOT}/report/night_normal/dfire_all.csv')
    for x in d[d.cat.isin(['none', 'fire'])].itertuples():
        rows.append(dict(dataset='D-Fire', kind='neg' if x.cat == 'none' else 'pos_fire', sub=x.src, name=x.base, src='file', key=x.path))
    for root, ds, sub_of in ((f'{ROOT}/data/SmokeFire_img', 'SmokeFire', lambda p: p.split('/')[-2]),
                             (f'{ROOT}/data/DeepFire/extract/Dataset', 'DeepFire', lambda p: p.split('/')[-2])):
        for dp, _, fs in os.walk(root):
            for f in fs:
                p = f'{dp}/{f}'; sub = sub_of(p)
                if sub in ('Normal', 'nofire', 'Fire', 'fire'):
                    rows.append(dict(dataset=ds, kind='neg' if sub in ('Normal', 'nofire') else 'pos_fire', sub=sub, name=f, src='file', key=p))
    return rows


def sheet(df, path, title_col, n=48, cols=8, cell=200):
    if not len(df):
        return
    pick = df.head(n)
    S = Image.new('RGB', (cols * cell, ((len(pick) + cols - 1) // cols) * (cell + 16)), 'white')
    for k, r in enumerate(pick.itertuples()):
        im = load((r.src, r.key)).convert('RGB'); im.thumbnail((cell, cell))
        t = Image.new('RGB', (cell, cell + 16), 'white'); t.paste(im, ((cell - im.width) // 2, (cell - im.height) // 2))
        ImageDraw.Draw(t).text((2, cell), f'{k} {getattr(r, title_col)} {100 * r.fire_px:.1f}%', fill='black', font=FONT)
        S.paste(t, ((k % cols) * cell, (k // cols) * (cell + 16)))
    S.save(path, quality=85)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'sheets':
        df = pd.read_csv(f'{OUT}/measure.csv.gz')
        neg = df[df.kind == 'neg']
        for ds, g in neg.groupby('dataset'):
            g = g.sort_values('fire_px', ascending=False)
            sheet(g, f'{OUT}/top_{ds}.jpg', 'sub')
            c = g[g.fire_px >= 0.005]
            sheet(c.sample(min(48, len(c)), random_state=0), f'{OUT}/rand_{ds}.jpg', 'sub')    # 후보 중 무작위
        sys.exit()
    rows = items()
    print('사진', len(rows), pd.DataFrame(rows).groupby(['dataset', 'kind']).size().to_dict(), flush=True)
    random.seed(1); random.shuffle(rows)
    jobs = [rows[i:i + 400] for i in range(0, len(rows), 400)]
    res = []
    with Pool(18) as pool:
        for k, r in enumerate(pool.imap_unordered(measure, jobs)):
            res += r
            if k % 50 == 0:
                print(k * 400, flush=True)
    df = pd.DataFrame(res)
    df.to_csv(f'{OUT}/measure.csv.gz', index=False)
    print('saved', len(df), 'errors', int(df['error'].notna().sum()) if 'error' in df else 0)
