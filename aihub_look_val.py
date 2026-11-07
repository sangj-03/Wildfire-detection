"""검증 사진 전체(VS.zip)에서 조건에 맞는 사진을 영상마다 1장씩 골라 박스를 그린 격자로 저장.
python3 -I look_full.py <tag> <pandas query> <n> [crop|corner] [strat열]
"""
import io, json, sys, zipfile
import numpy as np, pandas as pd
from PIL import Image, ImageDraw, ImageFont

SP = '/data/home/seame/dl-team-project/team6/report/aihub/samples'
T = '/data/home/seame/dl-team-project/team6/report/aihub'
B = ('/data/aihub/71330/265.지역안전재난(산불)_방재의_고도화를_위한_대규모_인공지능_데이터베이스_구축/01-1.정식개방데이터/Validation')
FONT = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 15, index=3)
COLOR = {1: (40, 40, 40), 2: (255, 40, 40), 3: (255, 160, 0), 4: (60, 140, 255), 5: (0, 200, 200), 6: (200, 0, 200)}
NAME = {1: '흑연', 2: '백연', 3: '화염', 4: '구름', 5: '안개', 6: '굴뚝'}
vl = zipfile.ZipFile(f'{B}/02.라벨링데이터/VL.zip'); vs = zipfile.ZipFile(f'{B}/01.원천데이터/VS.zip')
jp = {n.rsplit('/', 1)[-1]: n for n in vs.namelist()}
tag, query, n = sys.argv[1], sys.argv[2], int(sys.argv[3]); mode = sys.argv[4] if len(sys.argv) > 4 else ''
strat = sys.argv[5] if len(sys.argv) > 5 else None
lab = pd.read_csv(f'{T}/labels.csv.gz'); meta = pd.read_csv(f'{T}/images.csv.gz', low_memory=False)
lab = lab.merge(meta.drop(columns=['split', 'db']), on='path'); lab = lab[lab.split == 'val']
lab['jpg'] = lab.path.str.rsplit('/', n=1).str[-1].str.replace('.json', '.jpg')
img = pd.read_csv(f'{T}/img_val.csv.gz', usecols=['jpg', 'bright', 'sharp', 'text']); lab = lab.merge(img, on='jpg')
d = lab.query(query); print(tag, '후보', len(d), '영상', d.vid.nunique())
one = d.groupby('vid').sample(1, random_state=0)
if strat:
    one = one.groupby(strat).sample(1, random_state=0) if one[strat].nunique() >= n else one
one = one.sample(min(n, len(one)), random_state=0)
tiles = []
for r in one.itertuples():
    js = json.loads(vl.read(r.path)); im = Image.open(io.BytesIO(vs.read(jp[r.jpg]))).convert('RGB')
    anns = js['annotations']; ox = oy = 0
    if mode == 'crop':
        a = max([a for a in anns if a['category_id'] in (1, 2, 3)] or anns, key=lambda a: a['bbox'][2] * a['bbox'][3])
        x, y, w, h = a['bbox']; m = max(w, h) * 1.2 + 80; cx, cy = x + w / 2, y + h / 2
        im = im.crop((cx - m, cy - m, cx + m, cy + m)); ox, oy = cx - m, cy - m
    elif mode == 'corner':
        s0 = 1920 / im.width; im = im.resize((1920, round(im.height * s0))).crop((0, 0, 700, 140)); anns = []
    s = 480 / im.width; im = im.resize((480, round(im.height * s))); dr = ImageDraw.Draw(im)
    for a in anns:
        x, y, w, h = a['bbox']; x, y = (x - ox) * s, (y - oy) * s; w, h = w * s, h * s; c = a['category_id']
        dr.rectangle([x, y, x + w, y + h], outline=COLOR[c], width=2); dr.text((x + 2, max(0, y - 18)), NAME[c], fill=COLOR[c], font=FONT)
    tiles.append((im, f'{r.label}·{r.weather}·{r.timezone}·{r.device}·밝기{r.bright:.0f}·{r.address.replace("강원도강릉시", "강릉 ")[:12]}'))
cols = 4; ch = max(t.height for t, _ in tiles) + 24
S = Image.new('RGB', (cols * 480, ((len(tiles) + cols - 1) // cols) * ch), 'white'); dr = ImageDraw.Draw(S)
for k, (t, cap) in enumerate(tiles):
    x, y = (k % cols) * 480, (k // cols) * ch; S.paste(t, (x, y)); dr.text((x + 4, y + t.height + 2), cap, fill='black', font=FONT)
S.save(f'{SP}/sheet_{tag}.jpg', quality=85); print('saved', tag, S.size)
