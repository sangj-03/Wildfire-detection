"""AI Hub 71330(지역안전재난(산불) 방재의 고도화를 위한 대규모 인공지능 데이터베이스) 라벨 점검.

TL.zip(학습)·VL.zip(검증)의 JSON을 전부 읽어 사진 한 장에 한 줄인 표를 만든다.
사용: python3 -I analyze_aihub_labels.py  →  report/aihub/images.csv.gz
"""
import json, os, re, zipfile
from multiprocessing import Pool

import pandas as pd

BASE = ('/data/aihub/71330/265.지역안전재난(산불)_방재의_고도화를_위한_대규모_인공지능_데이터베이스_구축/'
        '01-1.정식개방데이터')
ZIPS = {'train': f'{BASE}/Training/02.라벨링데이터/TL.zip', 'val': f'{BASE}/Validation/02.라벨링데이터/VL.zip'}
OUT = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'report', 'aihub')
CAT = {1: 'black', 2: 'white', 3: 'flame', 4: 'cloud', 5: 'fog', 6: 'chimney'}   # 흑색연기, 백색/회색연기, 화염, 구름, 안개/연무, 굴뚝연기
os.makedirs(OUT, exist_ok=True)


def parse(args):
    split, names = args
    z = zipfile.ZipFile(ZIPS[split])
    rows = []
    for n in names:
        try:
            d = json.loads(z.read(n))
        except Exception as e:
            rows.append(dict(split=split, path=n, error=repr(e)[:200]))
            continue
        folder, place_cam, video_fuel = n.split('/')[:3]
        db, obj = folder.split('_', 1)
        img = (d.get('images') or [{}])[0]
        env = d.get('environment') or {}
        info = d.get('info') or {}
        r = dict(split=split, path=n, db=db, folder_obj=obj, place_cam=place_cam, video_fuel=video_fuel,
                 frame=(int(m.group(1)) if (m := re.search(r'_(\d+)\.json$', n)) else None),
                 w=img.get('width'), h=img.get('height'), file_name=img.get('file_name'),
                 camera_id=info.get('camera_id'), video_id=info.get('video_id'), license=info.get('license'),
                 date_created=info.get('date_created'), n_img=len(d.get('images') or []))
        for k in ('address', 'datetime', 'timezone', 'device', 'collection', 'weather', 'humidity', 'distance',
                  'gps', 'elevation'):
            r[k] = env.get(k)
        for c in CAT.values():
            r[f'n_{c}'] = 0; r[f'area_{c}'] = 0.0; r[f'box_{c}'] = 0.0
        bad = 0
        for a in d.get('annotations') or []:
            c = CAT.get(a.get('category_id'))
            if c is None:
                bad += 1; continue
            x, y, bw, bh = (a.get('bbox') or [0, 0, 0, 0])[:4]
            r[f'n_{c}'] += 1
            r[f'area_{c}'] = max(r[f'area_{c}'], float(a.get('area') or 0))
            r[f'box_{c}'] = max(r[f'box_{c}'], float(min(bw, bh)))     # 가장 큰 물체의 박스 짧은 변(px)
        r['bad_cat'] = bad
        rows.append(r)
    return rows


if __name__ == '__main__':
    jobs = []
    for split, p in ZIPS.items():
        names = sorted(n for n in zipfile.ZipFile(p).namelist() if n.endswith('.json'))
        print(split, len(names), 'JSON')
        jobs += [(split, names[i:i + 5000]) for i in range(0, len(names), 5000)]
    with Pool(16) as pool:
        rows = [r for chunk in pool.imap_unordered(parse, jobs) for r in chunk]
    df = pd.DataFrame(rows).sort_values(['split', 'path']).reset_index(drop=True)
    df.to_csv(f'{OUT}/images.csv.gz', index=False)
    print('saved', len(df), 'rows,', df['error'].notna().sum() if 'error' in df else 0, 'errors')
