"""계획서에 나온 데이터에서 '밤 정상'(불·연기가 없고 어두운 사진)으로 쓸 만한 사진을 찾는다.

어두운 사진 = 짧은 변 256으로 줄인 뒤 평균 밝기 50 미만 (AI Hub·SmokeFire 점검과 같은 기준).
사용: python3 -I check_night_normal.py <이름> <정상 사진 목록 txt>  →  report/night_normal/<이름>.csv, <이름>_dark.jpg(어두운 사진 격자)
"""
import os, random, sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

OUT = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'report', 'night_normal')
FONT = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 13, index=3)


def measure(p):
    try:
        im = Image.open(p); w, h = im.size
        im = im.convert('RGB'); s = 256 / min(w, h)
        a = np.asarray(im.resize((max(1, round(w * s)), max(1, round(h * s)))), dtype=np.float32)
        return dict(path=p, w=w, h=h, bright=float(a.mean()))
    except Exception as e:
        return dict(path=p, error=repr(e)[:120])


if __name__ == '__main__':
    name, listing = sys.argv[1], sys.argv[2]
    paths = [l for l in open(listing).read().split('\n') if l]
    os.makedirs(OUT, exist_ok=True)
    with Pool(16) as pool:
        df = pd.DataFrame(pool.map(measure, paths, chunksize=64))
    df['dark'] = df.bright < 50
    df.to_csv(f'{OUT}/{name}.csv', index=False)
    d = df[df.dark]
    print(f'{name}: 정상 {len(df)}장, 어두운 사진 {len(d)}장 ({100 * len(d) / max(1, len(df)):.1f}%), 열기 실패 {df.bright.isna().sum()}장')
    if len(d):
        print('어두운 사진 짧은 변 분위수', d[['w', 'h']].min(axis=1).quantile([.1, .5, .9]).to_dict())
        random.seed(0); pick = random.sample(list(d.itertuples()), min(40, len(d)))
        W = 240; tiles = []
        for r in pick:
            im = Image.open(r.path).convert('RGB'); im.thumbnail((W, W))
            t = Image.new('RGB', (W, W + 18), 'white'); t.paste(im, ((W - im.width) // 2, (W - im.height) // 2))
            ImageDraw.Draw(t).text((3, W + 1), f'{os.path.basename(r.path)[:22]} {r.w}×{r.h} 밝기{r.bright:.0f}', fill='black', font=FONT)
            tiles.append(t)
        cols = 8
        S = Image.new('RGB', (cols * W, ((len(tiles) + cols - 1) // cols) * (W + 18)), 'white')
        for k, t in enumerate(tiles):
            S.paste(t, ((k % cols) * W, (k // cols) * (W + 18)))
        S.save(f'{OUT}/{name}_dark.jpg', quality=85)
        print('saved', f'{OUT}/{name}_dark.jpg')
