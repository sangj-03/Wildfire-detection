"""AI Hub 71330 사진 화질·지름길 점검: 사진마다 밝기·대비·채도·선명도·JPEG 품질·날짜 글자 점수와 작은 썸네일을 잰다.

밝기·대비·채도·선명도는 SmokeFire(report/images_quality.csv)와 같은 방법(짧은 변 256px로 줄인 뒤)으로 잰다.
날짜 글자 점수: 일부 양성 사진 왼쪽 위(1920 폭 기준 x 90~310, y 45~110)에 찍힌 '날짜 시각 / GPS' 글자의
변하지 않는 획(text_stroke.npy) 평균 밝기 - 획 둘레(text_ring.npy) 평균 밝기. 글자가 있으면 140 이상, 없으면 0 근처.
사용: python3 -I analyze_aihub_images.py val|train   →  report/aihub/img_<split>.csv.gz, img_<split>_thumbs.npy (32×18 회색, 근접 중복용)
학습 사진(TS.z01~TS.zip 분할 zip)은 합치지 않고 aihub_split_zip.SplitZip으로 읽는다.
"""
import os
import io, sys, zipfile
from multiprocessing import Pool

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from aihub_split_zip import SplitZip

BASE = ('/data/aihub/71330/265.지역안전재난(산불)_방재의_고도화를_위한_대규모_인공지능_데이터베이스_구축/'
        '01-1.정식개방데이터')
ZIPS = {'val': f'{BASE}/Validation/01.원천데이터/VS.zip', 'train': f'{BASE}/Training/01.원천데이터/TS.zip'}
OUT = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'report', 'aihub')
STROKE = np.load(f'{OUT}/text_stroke.npy'); RING = np.load(f'{OUT}/text_ring.npy')


def measure(b):
    im = Image.open(io.BytesIO(b))
    q = im.quantization[0] if getattr(im, 'quantization', None) else None
    im = im.convert('RGB'); w, h = im.size
    s = 1920 / w
    c = np.asarray(im.convert('L').resize((1920, round(h * s))).crop((90, 45, 310, 110)), dtype=np.float32)
    s = 256 / min(w, h)
    sm = im.resize((round(w * s), round(h * s)))
    a = np.asarray(sm, dtype=np.float32); g = a.mean(2)
    hsv = np.asarray(sm.convert('HSV'), dtype=np.float32)
    lap = g[1:-1, 1:-1] * 4 - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    th = np.asarray(im.convert('L').resize((32, 18), Image.BILINEAR), dtype=np.uint8)
    return dict(w=w, h=h, jpeg_q0=(q[0] if q else -1), jpeg_qmean=(float(np.mean(q)) if q else -1),
                bright=g.mean(), contrast=g.std(), sat=hsv[..., 1].mean(), sharp=lap.var(),
                dark_px=(g < 30).mean(), text=float(c[STROKE].mean() - c[RING].mean())), th


SPLIT_ZIP = None      # 학습: fork 전에 중앙 디렉터리를 한 번만 읽어 둔다 (파일은 프로세스마다 따로 연다)


def work(args):
    zpath, names = args
    z = SPLIT_ZIP or zipfile.ZipFile(zpath); rows, ths = [], []
    for n in names:
        b = z.read(n)
        try:
            r, th = measure(b)
        except Exception as e:
            r, th = dict(error=repr(e)[:200]), np.zeros((18, 32), np.uint8)
        r.update(jpg=n.rsplit('/', 1)[-1], nbytes=len(b)); rows.append(r); ths.append(th)
    return rows, np.stack(ths)


if __name__ == '__main__':
    split = sys.argv[1]
    if split == 'train':
        SPLIT_ZIP = SplitZip(ZIPS[split]); listing = SPLIT_ZIP.namelist(); SPLIT_ZIP.close()
    else:
        listing = zipfile.ZipFile(ZIPS[split]).namelist()
    names = sorted(n for n in listing if n.lower().endswith('.jpg'))
    print(split, len(names), 'jpg', flush=True)
    jobs = [(ZIPS[split], names[i:i + 500]) for i in range(0, len(names), 500)]
    rows, ths = [], []
    with Pool(18) as pool:
        for k, (r, t) in enumerate(pool.imap(work, jobs)):      # imap: 순서 유지 → 썸네일과 줄 순서가 같다
            rows += r; ths.append(t)
            if k % 20 == 0:
                print(k * 500, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f'{OUT}/img_{split}.csv.gz', index=False)
    np.save(f'{OUT}/img_{split}_thumbs.npy', np.concatenate(ths))
    print('saved', len(df), 'errors', df['error'].notna().sum() if 'error' in df else 0)
