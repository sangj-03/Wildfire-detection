"""AI Hub 71330 검증 사진(VS.zip) 샘플 격자: 그룹마다 서로 다른 영상에서 한 장씩 골라 라벨 박스를 그린다.

summarize_aihub_labels.py가 만든 report/aihub/labels.csv.gz를 읽는다.
사용: python3 -I aihub_sample_sheets.py  →  report/aihub/sheet_*.jpg
"""
import os
import io, json, random, zipfile

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

BASE = ('/data/aihub/71330/265.지역안전재난(산불)_방재의_고도화를_위한_대규모_인공지능_데이터베이스_구축/'
        '01-1.정식개방데이터/Validation')
OUT = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'report', 'aihub')
FONT = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 15, index=3)   # KR
COLOR = {1: (40, 40, 40), 2: (255, 40, 40), 3: (255, 160, 0), 4: (60, 140, 255), 5: (0, 200, 200), 6: (200, 0, 200)}
NAME = {1: '흑연', 2: '백연', 3: '화염', 4: '구름', 5: '안개', 6: '굴뚝'}

vl = zipfile.ZipFile(f'{BASE}/02.라벨링데이터/VL.zip')
vs = zipfile.ZipFile(f'{BASE}/01.원천데이터/VS.zip')
jpgs = {n.rsplit('/', 1)[-1]: n for n in vs.namelist() if n.lower().endswith('.jpg')}
lab = pd.read_csv(f'{OUT}/labels.csv.gz')
meta = pd.read_csv(f'{OUT}/images.csv.gz', usecols=['path', 'weather', 'timezone', 'distance', 'folder_obj', 'address'])
lab = lab.merge(meta, on='path')
lab = lab[lab.split == 'val']


def tile(path, cell_w=480):
    d = json.loads(vl.read(path))
    fn = d['images'][0]['file_name']
    im = Image.open(io.BytesIO(vs.read(jpgs[fn]))).convert('RGB')
    s = cell_w / im.width
    im = im.resize((cell_w, round(im.height * s)))
    dr = ImageDraw.Draw(im)
    for a in d['annotations']:
        x, y, w, h = [v * s for v in a['bbox']]
        c = a['category_id']
        dr.rectangle([x, y, x + w, y + h], outline=COLOR[c], width=2)
        dr.text((x + 2, max(0, y - 18)), NAME[c], fill=COLOR[c], font=FONT)
    return im


def sheet(rows, name, cols=4, cell_w=480):
    rows = list(rows.itertuples())
    tiles = []
    for r in rows:
        try:
            tiles.append((tile(r.path, cell_w), f'{r.folder_obj} · {r.weather} · {r.timezone} · {r.distance}m · {r.address[:10]}'))
        except KeyError:
            print('사진 없음:', r.path)
    cell_h = max(t.height for t, _ in tiles) + 24
    S = Image.new('RGB', (cols * cell_w, ((len(tiles) + cols - 1) // cols) * cell_h), 'white')
    dr = ImageDraw.Draw(S)
    for k, (t, cap) in enumerate(tiles):
        x, y = (k % cols) * cell_w, (k // cols) * cell_h
        S.paste(t, (x, y))
        dr.text((x + 4, y + t.height + 2), cap, fill='black', font=FONT)
    S.save(f'{OUT}/sheet_{name}.jpg', quality=85)
    print('saved', name, len(tiles))


def pick(d, n, seed=0):
    """영상마다 한 장씩, 무작위 n개 영상."""
    one = d.groupby('vid').sample(1, random_state=seed)
    return one.sample(min(n, len(one)), random_state=seed)


random.seed(0)
sheet(pick(lab[lab.label == 'neg_with_smoke'], 12), 'neg_with_smoke')
for f in ('구름', '굴뚝연기', '안개연무'):
    sheet(pick(lab[(lab.label == 'normal') & (lab.folder_obj == f)], 8), f'normal_{f}')
sheet(pick(lab[(lab.label == 'smoke') & (lab.db == 'PositiveRealDB')], 12), 'smoke_real')
sheet(pick(lab[(lab.label == 'fire') & (lab.db == 'PositiveRealDB')], 12), 'fire_real')
sheet(pick(lab[lab.db == 'PositiveSyntheticDB'], 8), 'synthetic')
