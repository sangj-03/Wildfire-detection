"""eda.ipynb(AI Hub 71330 EDA)를 만들고 실행한다. 사용: ../.venv_eda/bin/python build_eda.py [--no-exec]
해석 문장은 eda_obs.json에 있다(분할별 수치가 든 문장은 분할을 바꾸면 다시 고쳐야 한다). 10/24 첫판(split_v1)은 export/eda_split_v1.ipynb."""
import json, os, sys, time

import nbformat
from nbclient import NotebookClient

NB_DIR = '/data/home/seame/dl-team-project/team6/notebooks'
OUT = f'{NB_DIR}/eda.ipynb'
OBS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'eda_obs.json')
OBS = json.load(open(OBS_PATH)) if os.path.exists(OBS_PATH) else {}

cells = []


def md(s):
    cells.append(nbformat.v4.new_markdown_cell(s.strip('\n')))


def code(s):
    cells.append(nbformat.v4.new_code_cell(s.strip('\n')))


def obs(key):
    txt = OBS.get(key)
    if txt:
        md(txt)


# ============================================================ 머리말
md(r'''
# 산불 탐지 프로젝트 · AI Hub 산불 데이터 탐색(EDA)

**데이터** AI Hub 71330 「지역안전재난(산불) 방재의 고도화를 위한 대규모 인공지능 데이터베이스」 공개분 (학습 560,365장 · 검증 69,635장, 라벨 JSON 같은 수)
**목적** 라벨·사진이 믿을 만한지 확인하고, 클래스·촬영 조건·물체 크기·방해 요소(구름·안개/연무·굴뚝연기)·픽셀 특성을 살펴, 전처리와 실험 설계(구름·안개가 오경보를 늘린다는 가설 검증)를 정한다.

### 분석 구성과 데이터 범위

| 절 | 분석 | 처음 안 | 이 노트북의 범위 | 이유 |
|---|---|---|---|---|
| 3 | JSON 라벨과 사진 매칭 · 무결성 (짝, 구조, 열림, 크기 일치, 좌표, 환경 값 형식) | 전체 100% 권장 | **100%** (JSON 63만 개, 사진 63만 장) | 한 장이라도 어긋나면 학습 중 오류나 틀린 정답이 된다. 라벨 읽기와 이미 전수로 열어 둔 결과로 충분해 비용이 작다 |
| 4 | 클래스 개수 · 비율 | 전체 100% | **100%** | 라벨만 읽으면 된다 |
| 5 | 실제 사진 보기 | 1~5% 또는 클래스별 50~100장 | **클래스·조건별 격자 약 180장 (영상 하나에 1장)** | 1~5%는 6천~3만 장이라 노트북에 담을 수 없고, 같은 영상의 이웃 프레임이라 많이 봐도 새로 보이는 것이 적다. 대신 영상·장소가 겹치지 않게 뽑는다 |
| 6 | 이미지 크기 · 해상도 · 파일 크기 | 전체 100% 권장 | **100%** | 실제로 사진을 열어 잰 크기로 본다 |
| 7 | 촬영 조건 · 장소 · 날짜 *(추가)* | – | **100%** | 조건이 클래스와 묶여 있으면 모델이 조건으로 답을 맞힌다 |
| 8 | 물체 박스 크기 · 위치 *(추가)* | – | **박스 100%** | 불꽃이 얼마나 작은지가 입력 크기를 정한다 |
| 9 | 방해 요소 구름 · 안개/연무 · 굴뚝연기 *(추가)* | – | **정상 사진 100%** | 가설의 핵심 변수 |
| 10 | 밝기 · 색상 등 픽셀 | 우선 10% | **10% 표본 (영상마다 10장 중 1장) + 기본 5개 값은 전수 결과와 비교** | 연속 프레임이라 10%로 충분하다. 10.1에서 전수와 비교해 확인한다 |
| 11 | 지름길 점검 *(추가)* | – | 10과 같은 표본 | 사진 내용 없이 숫자 몇 개로 클래스를 맞히는 정도 |
| 12 | 중복 · 누수 *(추가)* | – | **검증·시험 클래스당 1,000장 × 같은 카메라·장소의 학습 사진 전부** | 공식 분할의 누수를 다시 재고, 장소+촬영일 분할을 확인한다 |

### 실행 방법
- 커널: `Python (wildfire EDA)` (`/data/home/seame/dl-team-project/team6/.venv_eda`). 무거운 계산은 `eda_lib.py`가 여러 프로세스로 돌리고 결과를 `cache/`에 저장한다.
- 처음 실행(캐시 없음)은 약 4분, 캐시가 있으면 약 1분. `REBUILD = True`로 바꾸면 처음부터 다시 잰다.
- 그대로 읽어 오는 이전 결과: 원천 사진 63만 장 전수 측정과 32×18 썸네일(`analyze_aihub_images.py`), 장소+촬영일 분할 `split_v2`(`make_aihub_split_v2.py`).

> **분할: `split_v2` (2026-10-25 다시 실행).** 10/24 첫판은 `split_v1`을 썼다. split_v1은 비율을 맞출 층에 기기·방해 요소가 없어 시험 음성의 63%가 드론이었고 검증에는 안개 음성이 한 장도 없었다.
> 전처리 노트북(`preprocess.ipynb`) 3.1에서 음성 묶음을 다시 배정해 split_v2를 만들었다(양성 묶음은 그대로, 누수 검사 0장). 분할과 상관없는 결과(무결성, 박스 크기, 픽셀 통계 등)는 첫판과 같다.

> 사진 출처: AI Hub(aihub.or.kr) 71330. 팀 내부 분석에만 쓰고 재배포하지 않는다(이 노트북을 공개 저장소나 웹에 올리지 않는다).
''')

md(r'''
## 0. 준비
''')
code(r'''
import os, sys, time, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from IPython.display import Image as IImage, display

NB = '/data/home/seame/dl-team-project/team6/notebooks'
sys.path.insert(0, NB)
import eda_lib as E

REBUILD = False          # True면 cache/를 무시하고 처음부터 다시 잰다
pd.set_option('display.width', 220); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 120)
pd.set_option('display.max_colwidth', None)

# 색: 클래스 3개는 노트북 전체에서 고정(불꽃 주황 · 연기 파랑 · 정상 청록), 클래스와 상관없는 단일 계열은 회색
INK, INK2, SURFACE, GRID, NEUTRAL = '#1a1a19', '#52514e', '#fcfcfb', '#e8e7e3', '#6b6a65'
CLS = ['불꽃', '연기', '정상']
CLS_COLOR = {'불꽃': '#eb6834', '연기': '#2a78d6', '정상': '#1baf7a'}
CLS_EN = {'fire': '불꽃', 'smoke': '연기', 'normal': '정상', 'neg_with_smoke': '제외: 음성 속 연기'}
SPLITS = ['train', 'val', 'test']
SPLIT_KO = {'train': '학습', 'val': '검증', 'test': '시험'}
BLUES = LinearSegmentedColormap.from_list('blues', ['#fcfcfb', '#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b'])
plt.rcParams.update({
    'font.family': ['Noto Sans CJK KR'], 'axes.unicode_minus': False, 'figure.dpi': 110,
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'axes.edgecolor': '#c3c2bc', 'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': False, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.axisbelow': True,
    'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK,
    'axes.titlesize': 11, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left', 'axes.titlepad': 10,
    'font.size': 9.5, 'legend.frameon': False, 'legend.fontsize': 9, 'lines.linewidth': 2})


def show_jpeg(b):
    display(IImage(data=b, format='jpeg'))


def pct(s):
    """불리언 배열 → 비율(%)"""
    return round(100 * float(np.mean(s)), 1) if len(s) else np.nan


def grouped_bars(ax, tab, colors, ylabel='', fmt=None):
    """tab: 행 = x축 범주, 열 = 계열. 막대 사이 2px 간격."""
    n = tab.shape[1]; x = np.arange(len(tab)); w = 0.8 / n
    for i, c in enumerate(tab.columns):
        xs = x + (i - (n - 1) / 2) * w
        ax.bar(xs, tab[c].values, width=w, color=colors[c], label=c, edgecolor=SURFACE, linewidth=1.5)
        if fmt:
            for xi, v in zip(xs, tab[c].values):
                if np.isfinite(v):
                    ax.text(xi, v, fmt.format(v), ha='center', va='bottom', fontsize=8, color=INK2)
    ax.set_xticks(x, tab.index); ax.grid(axis='y'); ax.set_ylabel(ylabel)


def stacked_100(ax, tab, colors):
    """tab: 행 = 막대, 열 = 계열(사진 수) → 가로 100% 누적 막대. 6% 이상 칸에만 비율을 적는다."""
    share = tab.div(tab.sum(axis=1), axis=0) * 100
    left = np.zeros(len(tab)); y = np.arange(len(tab))
    for c in tab.columns:
        ax.barh(y, share[c], left=left, color=colors[c], height=0.6, label=c, edgecolor=SURFACE, linewidth=2)
        for yi, l, v in zip(y, left, share[c]):
            if v >= 6:
                ax.text(l + v / 2, yi, f'{v:.0f}%', ha='center', va='center', fontsize=8.5, color=INK)
        left += share[c].values
    ax.set_yticks(y, tab.index); ax.invert_yaxis(); ax.set_xlim(0, 100); ax.set_xlabel('%'); ax.grid(axis='x')


def step_hist(ax, groups, colors, bins):
    """groups: {이름: 값 배열}. 계열별 계단 히스토그램, 세로축 = 그 계열 사진 중 구간에 든 비율(%)."""
    for k, v in groups.items():
        v = np.asarray(v, float); v = v[np.isfinite(v)]
        if len(v):
            ax.hist(v, bins=bins, histtype='step', weights=np.full(len(v), 100 / len(v)), color=colors[k], linewidth=1.8,
                    label=f'{k} ({len(v):,})')
    ax.grid(axis='y')
''')

# ============================================================ 1. 불러오기
md(r'''
## 1. 데이터 불러오기

| 이름 | 내용 |
|---|---|
| `img` | 사진 한 장에 한 줄 (라벨 JSON 63만 개 전부: 폴더 이름, info, images, environment 값) |
| `ann` | 물체(annotation) 하나에 한 줄 (bbox, 다각형의 범위·넓이) |
| `listing` | 라벨 zip(TL·VL)과 원천 zip(TS 분할 zip·VS)에 든 파일 이름 전부 |
| `full`, `TH` | 원천 사진 63만 장을 전부 열어 잰 값(실제 가로·세로, 밝기·대비·채도·선명도, 파일 크기)과 32×18 썸네일 |
| `sp` | 장소+촬영일 분할 split_v2 (학습·검증·시험 60:20:20) |
''')
code(r'''
t0 = time.time()
img, ann = E.aihub_labels(rebuild=REBUILD)
img = img.rename(columns={'split': 'official'})            # official = AI Hub 공식 분할(train/val)
listing = E.cached('aihub_listing', E.aihub_zip_listing, REBUILD)
full, TH = E.aihub_full_pixels()
sp = pd.read_csv(f'{E.REPORT}/split_v2.csv.gz', usecols=['path', 'label', 'split', 'group', 'block', 'date'])
print(f'사진(JSON) {len(img):,} · 물체 {len(ann):,} · zip 안 파일 {len(listing):,} · 전수 측정 {len(full):,} · split_v2 {len(sp):,}')
print(f'걸린 시간 {time.time() - t0:.0f}초')
''')

md(r'''
### 1.1 우리 클래스 붙이기

AI Hub에는 사진 단위 클래스가 없다. 양성/음성은 `info.name`(= 폴더 이름 앞부분)에 있고, 불꽃·연기는 물체 박스로 알 수 있다. 아래 규칙으로 바꾼다(라벨 변환 — 사람이 새로 붙인 라벨이 아니다).

| 우리 클래스 | 규칙 |
|---|---|
| 불꽃 | 화염 박스가 하나라도 있음 (연기가 함께 있어도 불꽃) |
| 연기 | 흑색연기·백색/회색연기 박스만 있음 |
| 정상 | 음성 DB(`NegativeDB`)이고 연기·화염 박스가 없음 |
| 제외: 음성 속 연기 | 음성 DB인데 백색/회색연기 박스가 있음 (산불인지 판별 불확실) |
| 제외: 합성 양성 | `PositiveSyntheticDB` (CCTV 화면에 불꽃·연기를 붙인 사진) |
| 제외: 안 쓰는 양성 장소 | 실제 양성 16곳 중 팀이 쓰지 않기로 한 12곳 (10/15 팀 결정) |

본 실험 범위는 장소+촬영일 분할 `split_v2`에 든 사진이다(split_v1과 같은 사진, 분할만 다름).
''')
code(r'''
CATS = list(E.CAT.values())
cnt = ann.assign(c=ann.cat.map(E.CAT)).groupby(['idx', 'c']).size().unstack(fill_value=0).reindex(columns=CATS, fill_value=0)
img = img.merge(cnt, left_on='idx', right_index=True, how='left')
img[CATS] = img[CATS].fillna(0).astype(int)
img['jpg'] = img.path.str.replace(r'\.json$', '.jpg', regex=True)
img['jpg_name'] = img.jpg.str.rsplit('/', n=1).str[-1]
img['place'] = img.address.map(E.short_place)
img['date'] = img.datetime.map(E.norm_date)
img['vid'] = img.place_cam + '/' + img.video_fuel
img['fuel'] = img.video_fuel.str.split('_', n=1).str[1]
img['dist_m'] = pd.to_numeric(img.distance, errors='coerce')
img['dist_bin'] = pd.cut(img.dist_m, [-1, 49.999, 299.999, 1e9], labels=['근거리(<50m)', '중거리(50~299m)', '원거리(300m+)'])
smoke = (img['흑색연기'] + img['백색/회색연기']) > 0
flame = img['화염'] > 0
img['cls'] = np.select(
    [img.db.eq('PositiveSyntheticDB'), img.db.eq('PositiveRealDB') & img.address.isin(E.DROP_POS_PLACES),
     img.db.eq('NegativeDB') & smoke, flame, smoke, img.db.eq('NegativeDB')],
    ['제외: 합성 양성', '제외: 안 쓰는 양성 장소', '제외: 음성 속 연기', '불꽃', '연기', '정상'], default='?')
img = img.merge(sp.rename(columns={'label': 'sp_label', 'date': 'sp_date'}), on='path', how='left')
in_sp = img.split.notna()
print('split_v2의 클래스와 같은가:', bool((img.loc[in_sp, 'sp_label'].map(CLS_EN) == img.loc[in_sp, 'cls']).all()))
print('split_v2 밖의 사진:', img.loc[~in_sp, 'cls'].value_counts().to_dict())
print('규칙에 안 걸린 사진:', int((img.cls == '?').sum()))

use = img[in_sp].copy()                       # 본 실험 범위 (제외: 음성 속 연기 포함)
main = use[use.cls.isin(CLS)].copy()          # 불꽃·연기·정상
print(f'본 실험 범위 {len(use):,}장, 그중 불꽃·연기·정상 {len(main):,}장')
''')

# ============================================================ 2. 한눈에
md(r'''
## 2. 데이터 구성 한눈에

파일 구조: `{DB}_{대상}/{장소}_{카메라ID}/{영상번호}_{연소물}/{DB}_{대상}_{장소}_{카메라ID}_{영상번호}_{연소물}_{프레임}.json` (사진은 같은 경로의 `.jpg`)
''')
code(r'''
db_tab = img.groupby('db').agg(
    사진=('idx', 'size'), 공식_학습=('official', lambda s: int((s == 'train').sum())), 공식_검증=('official', lambda s: int((s == 'val').sum())),
    장소=('address', 'nunique'), 카메라=('place_cam', 'nunique'), 영상=('vid', 'nunique'),
    제작사=('license', lambda s: ', '.join(s.unique())),
    수집_방식=('collection', lambda s: ', '.join(f'{k} {v:,}' for k, v in s.value_counts().items())),
    기기=('device', lambda s: ', '.join(f'{k} {v:,}' for k, v in s.value_counts().items())))
db_tab.loc['합계'] = [len(img), int((img.official == 'train').sum()), int((img.official == 'val').sum()),
                     img.address.nunique(), img.place_cam.nunique(), img.vid.nunique(), '', '', '']
display(db_tab)
fuel = pd.crosstab(img.fuel, img.db)
display(fuel.loc[fuel.sum(axis=1).sort_values(ascending=False).index].rename_axis('연소물(폴더 이름)'))
''')

# ============================================================ 3. 무결성
md(r'''
## 3. 라벨·사진 무결성 검사 (100%)

검사마다 대상 수와 문제 수를 센다. 문제가 0이 아닌 항목은 비고에 처리 방법을 적었다.
''')
code(r'''
checks = []
def chk(table, group, name, target, bad, note='', scope=None):
    """bad: 문제 수 또는 문제 여부(불리언 배열). scope를 주면 본 실험 범위 안의 문제 수도 센다."""
    if np.ndim(bad) == 0:
        n, inside = int(bad), '–'
    else:
        bad = pd.Series(np.asarray(bad)).fillna(False).to_numpy(bool)
        n = int(bad.sum())
        inside = f'{int((bad & np.asarray(scope, dtype=bool)).sum()):,}' if scope is not None else '–'
    table.append({'구분': group, '검사': name, '대상': f'{int(target):,}', '문제': f'{n:,}', '그중 본 실험 범위': inside, '비고': note})

lab = listing[listing.kind == 'label']; ims = listing[listing.kind == 'image']
lab_json = lab[lab.name.str.endswith('.json')]
img_jpg = ims[ims.name.str.lower().str.endswith('.jpg')]
js = set(zip(lab_json.split, lab_json.name.str[:-5]))
jp = set(zip(img_jpg.split, img_jpg.name.str[:-4]))
chk(checks, '짝', '라벨 zip 안의 JSON이 아닌 파일', len(lab), len(lab) - len(lab_json))
chk(checks, '짝', '원천 zip 안의 JPG가 아닌 파일', len(ims), len(ims) - len(img_jpg))
chk(checks, '짝', 'JSON은 있는데 같은 이름의 사진이 없음', len(js), len(js - jp))
chk(checks, '짝', '사진은 있는데 같은 이름의 JSON이 없음', len(jp), len(jp - js))
bad_fn = img.file_name != img.jpg_name
chk(checks, '짝', 'JSON의 images.file_name ≠ 사진 파일 이름', len(img), bad_fn,
    f"전부 {', '.join(img.loc[bad_fn, 'db'].unique())} — file_name에 '백색백색회색연기' 오타", scope=in_sp)
chk(checks, '짝', '같은 파일 이름이 두 번 이상 (학습·검증 사이 포함)', len(img), img.jpg_name.duplicated(keep=False), scope=in_sp)
chk(checks, '구조', 'JSON 읽기 실패', len(img), img.parse_error.notna(), scope=in_sp)
chk(checks, '구조', '최상위 키가 info·categories·images·annotations·environment가 아님', len(img),
    img.top_keys != 'info|categories|images|annotations|environment', scope=in_sp)
chk(checks, '구조', 'images가 1장이 아님', len(img), img.n_images != 1, scope=in_sp)
chk(checks, '구조', 'categories 6종 목록이 표준과 다름', len(img), ~img.cats_std, scope=in_sp)
chk(checks, '구조', 'info.name ≠ 폴더 이름의 DB (양성/음성)', len(img), img.info_name != img.db, scope=in_sp)
chk(checks, '구조', 'environment 항목이 빠짐', len(img), img.env_missing > 0, scope=in_sp)
chk(checks, '구조', '물체(annotation)가 0개', len(img), img.n_ann == 0, scope=in_sp)

f = img[['official', 'jpg_name', 'w', 'h']].merge(
    full[['src', 'jpg', 'w', 'h']].rename(columns={'src': 'official', 'jpg': 'jpg_name', 'w': 'rw', 'h': 'rh'}),
    on=['official', 'jpg_name'], how='left')
chk(checks, '사진', '사진 열기 실패 (63만 장 전수로 열어 본 결과)', len(full), full['error'].notna().sum() if 'error' in full else 0)
chk(checks, '사진', 'JSON의 가로·세로 ≠ 실제 사진 크기', f.rw.notna().sum(), ((f.w != f.rw) | (f.h != f.rh)).to_numpy(), scope=in_sp)

A = ann.merge(img[['idx', 'w', 'h', 'image_id', 'db']].rename(columns={'image_id': 'img_id'}).assign(insp=in_sp.to_numpy()), on='idx')
ext = pd.DataFrame({'a': A.px0 - A.x, 'b': A.py0 - A.y, 'c': A.px1 - (A.x + A.bw), 'd': A.py1 - (A.y + A.bh)}).abs().max(axis=1)
ratio = A.poly_area / A.area
bad_area = (ratio < 0.9) | (ratio > 1.1)
dup_box = A.duplicated(['idx', 'cat', 'x', 'y', 'bw', 'bh'])
chk(checks, '물체', 'category_id가 1~6이 아님', len(A), ~A.cat.isin(list(E.CAT)), scope=A.insp)
chk(checks, '물체', 'annotation.image_id ≠ 사진 id', len(A), A.image_id != A.img_id, scope=A.insp)
chk(checks, '물체', 'bbox 숫자가 4개가 아님', len(A), A.n_bbox != 4, scope=A.insp)
chk(checks, '물체', 'bbox 가로·세로가 0 이하', len(A), (A.bw <= 0) | (A.bh <= 0), scope=A.insp)
chk(checks, '물체', 'bbox가 사진 밖으로 1px 넘게 나감', len(A),
    (A.x < -1) | (A.y < -1) | (A.x + A.bw > A.w + 1) | (A.y + A.bh > A.h + 1), scope=A.insp)
chk(checks, '물체', 'segmentation이 비었음 (bbox만 있음)', len(A), A.bad_seg, '박스(bbox)는 정상이라 클래스 변환·크기 분석에는 영향 없음', scope=A.insp)
chk(checks, '물체', '다각형의 범위와 bbox가 2px 넘게 다름', len(A), ext > 2, scope=A.insp)
chk(checks, '물체', 'area가 다각형 넓이와 10% 넘게 다름', len(A), bad_area,
    f"{pct(A.loc[bad_area, 'db'] == 'PositiveSyntheticDB')}%가 합성 양성 (area = 마스크 화소 수라 다각형보다 큼) → 면적은 다각형 넓이로 계산", scope=A.insp)
chk(checks, '물체', 'iscrowd ≠ 0', len(A), A.iscrowd != 0, scope=A.insp)
chk(checks, '물체', '같은 사진에 똑같은 박스가 두 번', len(A), dup_box, '면적을 더할 때 한 번만 센다', scope=A.insp)
pd.DataFrame(checks)
''')

md(r'''
### 3.1 촬영 환경 값의 형식

`environment`는 사람이 입력한 값이라 형식이 섞여 있을 수 있다. 촬영 시각·거리·GPS를 확인한다.
''')
code(r'''
dt = img.datetime.astype(str)
fmt = np.select([dt.str.startswith('1980'), dt.str.contains('MP4'), dt.str.fullmatch(r'20\d{6}_\d{6}'),
                 dt.str.fullmatch(r'20\d{6}_\d{5}'), dt.str.fullmatch(r'\d{6}_\d{5,6}')],
                ['1980-01-01 (기기 기본값)', '뒤에 MP4가 붙음', 'YYYYMMDD_HHMMSS (정상)', '시(時) 앞 0이 빠짐', '연도가 두 자리 (YYMMDD_)'],
                default='그 밖')
dt_tab = (pd.DataFrame({'형식': fmt, 'insp': in_sp, '예': dt})
          .groupby('형식').agg(사진=('예', 'size'), 본실험범위=('insp', 'sum'), 예=('예', 'first')).sort_values('사진', ascending=False))
display(dt_tab)

gps_addr = img.groupby('gps').address.nunique()
shared = img.gps.map(gps_addr) > 1
top_gps = img.loc[shared, 'gps'].value_counts().index[0]
outside = ~img.address.str.startswith('강원도강릉시')
env_checks = []
chk(env_checks, '환경값', '촬영 시각을 YYYYMMDD로 바꿀 수 없음', len(img), img.date.isna(), '1980년 값', scope=in_sp)
chk(env_checks, '환경값', '촬영 시각이 YYYYMMDD_HHMMSS 형식이 아님', len(img), fmt != 'YYYYMMDD_HHMMSS (정상)',
    '날짜는 두 형식 모두 YYYYMMDD로 바꿀 수 있음', scope=in_sp)
chk(env_checks, '환경값', '거리(distance)가 숫자가 아님', len(img), img.dist_m.isna(), scope=in_sp)
chk(env_checks, '환경값', '여러 주소가 같은 GPS를 씀', len(img), shared,
    f'{top_gps} 한 값이 {img.loc[img.gps == top_gps, "address"].nunique()}개 주소에 들어감', scope=in_sp)
chk(env_checks, '환경값', '강릉 밖 주소인데 GPS가 위 좌표 (강릉 솔향수목원)', outside.sum(), outside & (img.gps == top_gps),
    'GPS·고도(elevation)는 분석에 쓰지 않음', scope=in_sp)
pd.DataFrame(env_checks)
''')
obs('integrity')

# ============================================================ 4. 클래스
md(r'''
## 4. 클래스 분포 (100%)
### 4.1 원래 구성(DB × 폴더 대상)과 우리 클래스
''')
code(r'''
t = pd.crosstab([img.db, img.folder_obj], img.cls, margins=True, margins_name='합계')
t[[c for c in CLS + ['제외: 음성 속 연기', '제외: 합성 양성', '제외: 안 쓰는 양성 장소'] if c in t.columns] + ['합계']]
''')
md(r'''
### 4.2 분할별 비율: 공식 분할과 장소+촬영일 분할
''')
code(r'''
cls_split = pd.crosstab(use.split.map(SPLIT_KO), use.cls).reindex(['학습', '검증', '시험'])[CLS + ['제외: 음성 속 연기']]
display(cls_split.assign(합계=cls_split.sum(axis=1)))
both = pd.concat([pd.crosstab('공식 ' + main.official.map(SPLIT_KO), main.cls),
                  pd.crosstab('장소+촬영일 ' + main.split.map(SPLIT_KO), main.cls)])[CLS]
both = both.reindex(['공식 학습', '공식 검증', '장소+촬영일 학습', '장소+촬영일 검증', '장소+촬영일 시험'])
fig, ax = plt.subplots(figsize=(8, 3))
stacked_100(ax, both, CLS_COLOR)
ax.set_title('본 실험 범위(불꽃·연기·정상)의 분할별 클래스 비율')
ax.legend(ncol=3, loc='lower right', bbox_to_anchor=(1, 1.0))
plt.show()
both.assign(합계=both.sum(axis=1))
''')
obs('classes')

# ============================================================ 5. 사진 보기
md(r'''
## 5. 실제 사진 보기 (표본)

같은 영상의 이웃 프레임은 거의 같으므로 **영상 하나에서 1장씩** 무작위로 뽑는다. 박스 색: 주황 = 화염, 파랑 = 연기, 흰색 = 구름·안개/연무·굴뚝연기.
사진 아래 글자는 `장소 월/일 시간대·날씨`다. 더 많이 보려면 AI Hub 보기 페이지(`aihub_view_server.py`, 포트 8797)를 쓴다.
''')
code(r'''
ann_i = ann.set_index('idx').sort_index()

def per_video(df, n, seed=0):
    return df.sample(frac=1, random_state=seed).drop_duplicates('vid').head(n)

def cap(r):
    d = f'{int(r.date[4:6])}/{int(r.date[6:])}' if isinstance(r.date, str) else '?'
    return f'{r.place[:9]} {d} {r.timezone}·{r.weather}'

def aihub_tiles(rows, cats=None, caption=cap):
    tiles = []
    for r in rows.itertuples():
        a = ann_i.loc[[r.idx]]
        boxes = [(q.x, q.y, q.bw, q.bh, E.CAT[q.cat]) for q in a.itertuples() if cats is None or E.CAT[q.cat] in cats]
        tiles.append(dict(img=E.load_image(r.jpg, r.official), caption=caption(r), boxes=boxes))
    return tiles

for c in CLS:
    rows = per_video(main[main.cls == c], 32, seed=1)
    show_jpeg(E.grid(aihub_tiles(rows), cols=8, tw=240, title=f'{c} · 영상 {len(rows)}개에서 1장씩'))
''')
md(r'''
### 5.1 불꽃 확대: 화염 박스 주변만 잘라 보기
''')
code(r'''
rows = per_video(main[main.cls == '불꽃'], 24, seed=2)
big = ann[(ann.cat == 3) & ann.idx.isin(rows.idx)].sort_values('area').drop_duplicates('idx', keep='last').set_index('idx')
tiles = []
for r in rows.itertuples():
    q = big.loc[r.idx]
    crop, bb = E.crop_box(E.load_image(r.jpg, r.official), (q.x, q.y, q.bw, q.bh), scale=4, min_side=96)
    tiles.append(dict(img=crop, caption=f'{r.place[:8]} 짧은 변 {min(q.bw, q.bh):.0f}px', boxes=[(*bb, '화염')]))
show_jpeg(E.grid(tiles, cols=8, tw=180, th=180, title='화염 박스 주변 (박스의 4배 넓이로 자름, 최소 96px)'))
''')
md(r'''
### 5.2 정상 사진: 방해 요소 종류별 (폴더 이름 기준)
''')
code(r'''
tiles = []
for obj in ['구름', '안개연무', '굴뚝연기']:
    rows = per_video(main[(main.cls == '정상') & (main.folder_obj == obj)], 8, seed=3)
    tiles += aihub_tiles(rows)
show_jpeg(E.grid(tiles, cols=8, tw=240, title='정상: 1줄 구름 · 2줄 안개/연무 · 3줄 굴뚝연기 폴더 (흰 상자 = 방해 요소 박스)'))
''')
md(r'''
### 5.3 뺀 사진과 밤 사진
''')
code(r'''
tiles = aihub_tiles(per_video(img[img.cls == '제외: 음성 속 연기'], 8, seed=4))
tiles += aihub_tiles(per_video(img[img.cls == '제외: 합성 양성'], 8, seed=4))
show_jpeg(E.grid(tiles, cols=8, tw=240, title='제외: 1줄 음성 폴더인데 연기 박스가 있는 사진 · 2줄 합성 양성'))
tiles = aihub_tiles(per_video(main[(main.cls == '불꽃') & (main.timezone == '야간')], 8, seed=5))
tiles += aihub_tiles(per_video(main[(main.cls == '정상') & (main.timezone == '야간')], 8, seed=5))
show_jpeg(E.grid(tiles, cols=8, tw=240, title="시간대 라벨 '야간': 1줄 불꽃 · 2줄 정상"))
''')
obs('samples')

# ============================================================ 6. 해상도
md(r'''
## 6. 해상도·파일 크기 (100%)

크기는 JSON 값이 아니라 사진을 열어 잰 값이다(3절에서 둘이 모두 같음을 확인).
''')
code(r'''
def res(df):
    return df.w.astype(int).astype(str) + '×' + df.h.astype(int).astype(str)
r_cls = pd.crosstab(res(main), main.cls)[CLS]
r_cls['합계'] = r_cls.sum(axis=1)
display(r_cls.sort_values('합계', ascending=False))
k4 = main.assign(k4=main.w >= 3840).pivot_table(index='split', columns='cls', values='k4', aggfunc='mean').reindex(SPLITS)[CLS] * 100
k4.index = k4.index.map(SPLIT_KO)
fig, ax = plt.subplots(figsize=(7, 3))
grouped_bars(ax, k4, CLS_COLOR, ylabel='4K(3840×2160) 사진 비율 (%)', fmt='{:.0f}')
ax.set_title('분할·클래스별 4K 사진 비율')
ax.legend(ncol=3, loc='upper left')
plt.show()
k4.round(1)
''')
code(r'''
# 파일 크기: 화소당 비트(bpp). 압축 정도가 클래스마다 다르면 '파일 크기'가 지름길이 된다
fa = main[['official', 'jpg_name', 'cls', 'w', 'h']].merge(
    full[['src', 'jpg', 'nbytes', 'jpeg_q0']].rename(columns={'src': 'official', 'jpg': 'jpg_name'}), on=['official', 'jpg_name'])
fa['bpp'] = fa.nbytes * 8 / (fa.w * fa.h)
fig, ax = plt.subplots(figsize=(7, 2.8))
step_hist(ax, {c: fa.loc[fa.cls == c, 'bpp'] for c in CLS}, CLS_COLOR, bins=np.linspace(0, 6, 61))
ax.set_title('화소당 비트(bpp) 분포'); ax.set_xlabel('bpp'); ax.set_ylabel('사진 비율 (%)'); ax.legend()
plt.show()
pd.DataFrame({'bpp 중앙값': fa.groupby('cls').bpp.median(), '파일 크기 중앙값 (KB)': fa.groupby('cls').nbytes.median() / 1024,
              'JPEG 양자화 첫 값 중앙값': fa.groupby('cls').jpeg_q0.median()}).reindex(CLS).round(2)
''')
obs('resolution')

# ============================================================ 7. 메타데이터
md(r'''
## 7. 촬영 조건·장소·날짜 (100%)
### 7.1 클래스별 조건 분포
''')
code(r'''
fig, axes = plt.subplots(2, 2, figsize=(11, 6.2))
for ax, (col, title) in zip(axes.flat, [('weather', '날씨'), ('timezone', '시간대'), ('device', '기기'), ('dist_bin', '거리')]):
    t = pd.crosstab(main[col], main.cls, normalize='columns')[CLS] * 100
    if col == 'timezone':
        t = t.reindex(['오전', '정오', '오후', '야간'])
    grouped_bars(ax, t, CLS_COLOR, ylabel='클래스 안 비율 (%)')
    ax.set_title(title)
axes[0, 0].legend(ncol=3, loc='upper right')
plt.tight_layout(); plt.show()
pd.concat({col: (pd.crosstab(main[col], main.cls, normalize='columns')[CLS] * 100).round(1)
           for col in ['weather', 'timezone', 'device', 'dist_bin']})
''')
md(r'''
### 7.2 장소와 날짜
''')
code(r'''
pl = main.groupby('cls').agg(장소=('address', 'nunique'), 카메라=('place_cam', 'nunique'), 영상=('vid', 'nunique'),
                             장소_촬영일_묶음=('group', 'nunique'), 사진=('idx', 'size')).reindex(CLS)
top = main.groupby(['cls', 'place']).size().rename('n').reset_index()
pl['가장 많은 장소'] = [f"{g.sort_values('n').iloc[-1].place} {100 * g.n.max() / g.n.sum():.0f}%" for c in CLS for g in [top[top.cls == c]]]
display(pl)
pos_places = pd.crosstab(main[main.cls != '정상'].place, main[main.cls != '정상'].cls)
display(pos_places.assign(합계=pos_places.sum(axis=1)).sort_values('합계', ascending=False))
gn = main[main.cls == '정상']
print(f"정상 장소 {gn.address.nunique()}곳 중 강릉 {gn[gn.address.str.startswith('강원도강릉시')].address.nunique()}곳, "
      f"강릉 정상 사진 비율 {pct(gn.address.str.startswith('강원도강릉시'))}%, 양성과 장소가 겹치는 정상 장소: "
      f"{sorted(set(gn.place) & set(main[main.cls != '정상'].place))}")
''')
code(r'''
day = main.assign(d=pd.to_datetime(main.date, format='%Y%m%d')).groupby(['d', 'cls']).size().unstack(fill_value=0).reindex(columns=CLS)
fig, axes = plt.subplots(3, 1, figsize=(11, 4.8), sharex=True)
for ax, c in zip(axes, CLS):
    ax.bar(day.index, day[c] / 1000, width=0.9, color=CLS_COLOR[c])
    ax.set_ylabel('천 장'); ax.grid(axis='y')
    ax.text(0.005, 0.82, c, transform=ax.transAxes, fontweight='bold', color=INK)
axes[0].set_xlim(pd.Timestamp('2022-08-01'), pd.Timestamp('2022-12-20'))
axes[0].set_title('촬영일별 사진 수 (2022-08-01 ~ 12-20)')
plt.tight_layout(); plt.show()
month = pd.crosstab(main.date.str[:6].map(lambda m: f'{m[:4]}-{m[4:]}'), main.cls, normalize='columns')[CLS] * 100
display(month.round(1).rename_axis('촬영 월 (클래스 안 %)'))
summer = {c: pct(main.loc[main.cls == c, 'date'].str[4:6].isin(['08', '09'])) for c in CLS}
autumn = {c: pct(main.loc[main.cls == c, 'date'].str[4:6].isin(['10', '11', '12'])) for c in CLS}
print('8~9월 비율:', summer, '/ 10~12월 비율:', autumn)
print('그림 밖(2022-08-01 이전) 사진:', main[main.date < '20220801'].groupby('cls').size().to_dict(), '/ 장소:',
      sorted(main[main.date < '20220801'].place.unique()))
print('촬영 기간:', {c: f"{main[main.cls == c].date.min()} ~ {main[main.cls == c].date.max()}, {main[main.cls == c].date.nunique()}일" for c in CLS})
''')
code(r'''
# 날씨가 '장소+촬영일' 묶음에 묶여 있는가: 묶음 안의 날씨 값 종류 수
wg = main.groupby(['cls', 'group']).weather.nunique()
print('날씨가 한 가지뿐인 장소+촬영일 묶음 비율:', {c: f'{pct(wg[c] == 1)}% ({len(wg[c])}묶음)' for c in CLS})
tg = main.groupby(['cls', 'vid']).timezone.nunique()
print('시간대 값이 한 가지뿐인 영상 비율:', {c: f'{pct(tg[c] == 1)}%' for c in CLS})
''')
md(r'''
### 7.3 시간대 라벨과 실제 밝기 (전수 측정값)

`어두운 사진` = 짧은 변 256px로 줄였을 때 평균 밝기가 50 미만(0~255). 라벨의 '야간'이 실제로 어두운지 본다.
''')
code(r'''
mf = main.merge(full[['src', 'jpg', 'bright', 'dark_px', 'text']].rename(columns={'src': 'official', 'jpg': 'jpg_name'}),
                on=['official', 'jpg_name'], how='left')
mf['dark'] = mf.bright < 50
TZ = ['오전', '정오', '오후', '야간']
dark = mf.pivot_table(index='timezone', columns='cls', values='dark', aggfunc='mean').reindex(TZ)[CLS] * 100
n_tz = pd.crosstab(mf.timezone, mf.cls).reindex(TZ)[CLS]
fig, ax = plt.subplots(figsize=(7, 3))
grouped_bars(ax, dark, CLS_COLOR, ylabel='어두운 사진 비율 (%)', fmt='{:.0f}')
ax.set_title('시간대 라벨별 실제로 어두운 사진 비율')
ax.legend(ncol=3, loc='upper left')
plt.show()
display(pd.concat({'어두운 사진 %': dark.round(1), '사진 수': n_tz}, axis=1))
print('어두운 사진 중 불꽃 비율:', pct(mf[mf.dark].cls == '불꽃'), '% / 정상 사진 밝기의 하위 1%:',
      round(float(mf[mf.cls == '정상'].bright.quantile(0.01)), 1))
''')
obs('meta')

# ============================================================ 8. 박스
md(r'''
## 8. 물체(박스) 크기·위치·개수 (박스 100%)

사진마다 **가장 큰** 화염·연기 박스의 짧은 변을 본다(모델이 최소한 이 크기는 봐야 한다). 세 가지 입력 크기에서 잰다:
원본, 학습 입력(가로 1920으로 맞춘 크기 — 4K는 절반), 짧은 변 224로 줄였을 때(ImageNet 기본 입력).
''')
code(r'''
B = ann.merge(main[['idx', 'cls', 'w', 'h', 'split', 'dist_m']], on='idx')
B['name'] = B.cat.map(E.CAT).replace({'흑색연기': '연기', '백색/회색연기': '연기'})
B['short'] = B[['bw', 'bh']].min(axis=1)
Bl = B[B.name.isin(['화염', '연기'])].groupby(['idx', 'name']).agg(
    short=('short', 'max'), w=('w', 'first'), h=('h', 'first'), dist_m=('dist_m', 'first'), cls=('cls', 'first')).reset_index()
Bl['원본'] = Bl.short
Bl['학습 입력(가로 1920)'] = Bl.short * 1920 / Bl.w
Bl['224'] = Bl.short * 224 / Bl[['w', 'h']].min(axis=1)
SIZES = ['원본', '학습 입력(가로 1920)', '224']
qt = {}
for nm, g in Bl.groupby('name'):
    for c in SIZES:
        s = g[c]
        qt[(nm, c)] = {'사진 수': len(s), '10%': s.quantile(.1), '중앙값': s.median(), '90%': s.quantile(.9), '8px 미만 %': pct(s < 8)}
display(pd.DataFrame(qt).T.round(1))

NAME_COLOR = {'화염': CLS_COLOR['불꽃'], '연기': CLS_COLOR['연기']}
fig, axes = plt.subplots(1, 3, figsize=(12, 2.8), sharey=True)
for ax, col in zip(axes, SIZES):
    step_hist(ax, {n: Bl.loc[Bl.name == n, col] for n in ['화염', '연기']}, NAME_COLOR, bins=np.logspace(0, 3.3, 50))
    ax.set_xscale('log'); ax.axvline(8, color=INK2, linewidth=0.8); ax.text(8.5, ax.get_ylim()[1] * 0.92, '8px', fontsize=8, color=INK2)
    ax.set_title(col); ax.set_xlabel('가장 큰 박스의 짧은 변 (px)')
axes[0].set_ylabel('사진 비율 (%)'); axes[2].legend(loc='upper right')
plt.show()
''')
code(r'''
# 박스 중심 위치: 화면 어디에 불·연기가 있는가
fig, axes = plt.subplots(1, 2, figsize=(10, 2.9))
for ax, nm in zip(axes, ['화염', '연기']):
    g = B[B.name == nm]
    Hm, _, _ = np.histogram2d(((g.y + g.bh / 2) / g.h).clip(0, .999), ((g.x + g.bw / 2) / g.w).clip(0, .999),
                              bins=[18, 32], range=[[0, 1], [0, 1]])
    im = ax.imshow(Hm / Hm.sum() * 100, cmap=BLUES, extent=[0, 1, 1, 0], aspect='auto')
    ax.set_title(f'{nm} 박스 중심 ({len(g):,}개)'); ax.set_xticks([0, .5, 1]); ax.set_yticks([0, .5, 1]); ax.set_xlabel('가로 위치 (0 = 왼쪽)')
axes[0].set_ylabel('세로 위치 (0 = 위)')
fig.colorbar(im, ax=axes, shrink=0.85, label='박스 비율 (%)')
plt.show()
for nm in ['화염', '연기']:
    g = B[B.name == nm]
    print(f'{nm} 박스 중심이 화면 아래 절반: {pct((g.y + g.bh / 2) / g.h > .5)}%')
''')
code(r'''
# 사진당 박스 수와 거리 라벨별 화염 크기
nb = main.assign(n_obj=main[CATS].sum(axis=1)).groupby('cls').n_obj.describe()[['mean', '50%', 'max']].reindex(CLS).round(2)
display(nb.rename(columns={'mean': '사진당 박스 평균', '50%': '중앙값', 'max': '최대'}))
fl = Bl[Bl.name == '화염'].copy()
fl['거리'] = pd.cut(fl.dist_m, [0, 19, 49, 99, 199, 1e9], labels=['<20m', '20~49m', '50~99m', '100~199m', '200m+'])
fl.groupby('거리', observed=True)['원본'].agg(사진='size', 중앙값='median', 상위10=lambda s: s.quantile(.9)).round(1)
''')
obs('boxes')

# ============================================================ 9. 방해 요소
md(r'''
## 9. 방해 요소: 구름 · 안개/연무 · 굴뚝연기 (정상 사진 100%)

가설의 핵심 변수다. 음성 사진에는 방해 요소 박스(다각형)가 있어, 종류와 **화면에서 차지하는 면적**을 잴 수 있다.
면적 비율 = 같은 종류 다각형 넓이의 합 ÷ 사진 넓이(겹치면 1로 자름).
''')
code(r'''
DIST = ['구름', '안개/연무', '굴뚝연기']
N = main[main.cls == '정상'].copy()
ar = (ann[ann.cat.isin([4, 5, 6]) & ann.idx.isin(N.idx)].drop_duplicates(['idx', 'cat', 'x', 'y', 'bw', 'bh'])
      .groupby(['idx', 'cat']).poly_area.sum().unstack(fill_value=0))
ar.columns = ['a_' + E.CAT[c] for c in ar.columns]
N = N.merge(ar, left_on='idx', right_index=True, how='left').fillna({f'a_{d}': 0 for d in DIST})
for d in DIST:
    N[f'f_{d}'] = (N[f'a_{d}'] / (N.w * N.h)).clip(0, 1)
N['f_구름안개'] = (N['f_구름'] + N['f_안개/연무']).clip(0, 1)
N['combo'] = N[DIST].gt(0).apply(lambda r: ' + '.join(d for d in DIST if r[d]) or '(없음)', axis=1)
combo = N.combo.value_counts().rename('사진').to_frame()
combo['%'] = (100 * combo.사진 / len(N)).round(1)
display(combo)
pd.crosstab(N.folder_obj, N.combo).rename_axis(index='폴더 대상', columns='실제로 붙은 박스')
''')
code(r'''
fig, axes = plt.subplots(1, 3, figsize=(11, 2.8), sharey=True)
for ax, d in zip(axes, DIST):
    s = N.loc[N[d] > 0, f'f_{d}'] * 100
    ax.hist(s, bins=np.linspace(0, 100, 41), color=CLS_COLOR['정상'])
    ax.set_title(f'{d} ({len(s):,}장)'); ax.set_xlabel('화면 면적 비율 (%)'); ax.grid(axis='y')
axes[0].set_ylabel('사진 수')
plt.show()
N['구간'] = pd.cut(N['f_구름안개'] * 100, [-1, 5, 20, 101], labels=['<5%', '5~20%', '≥20%'])
bins_split = pd.crosstab(N.구간, N.split.map(SPLIT_KO))[['학습', '검증', '시험']]
print('구름+안개/연무 면적 구간별 정상 사진 수')
bins_split.assign(합계=bins_split.sum(axis=1))
''')
code(r'''
# 종류별 장소 수 (분할별) — 장소가 적으면 그 종류의 결과는 몇 곳의 결과다
places = N.groupby(['folder_obj', 'split']).address.nunique().unstack()[SPLITS].rename(columns=SPLIT_KO)
places['전체 장소'] = N.groupby('folder_obj').address.nunique()
places['사진'] = N.groupby('folder_obj').size()
display(places.rename_axis('폴더 대상'))
# 구름이 화면 아래쪽까지 내려와 있는가 (산 능선 근처 구름은 연기처럼 보일 수 있다)
cb = ann[(ann.cat == 4) & ann.idx.isin(N.idx)].merge(N[['idx', 'h']], on='idx')
low = cb.assign(bottom=(cb.y + cb.bh) / cb.h).groupby('idx').bottom.max() > 0.5
print(f'구름 박스가 있는 정상 사진 {len(low):,}장 중 구름이 화면 아래 절반까지 내려온 사진 {pct(low)}%')
''')
code(r'''
# 양성 사진 속 방해 요소: 구름·안개가 낀 산불 사진이 실제로 얼마나 있나
P = main[main.cls != '정상']
pos_d = pd.DataFrame({d: P.groupby(['split', 'cls'])[d].apply(lambda s: int((s > 0).sum())) for d in DIST})
pos_d['양성 사진'] = P.groupby(['split', 'cls']).size()
pos_d.rename(index=SPLIT_KO, level=0).reindex(['학습', '검증', '시험'], level=0)
''')
obs('distractor')

# ============================================================ 10. 픽셀
md(r'''
## 10. 픽셀 통계 (10% 표본)

모든 값은 짧은 변을 256px로 줄여서 잰다. 밝기 = 회색조 평균(0~255), 대비 = 회색조 표준편차, 채도 = HSV 채도 평균, 선명도 = 라플라시안 분산,
어두운 화소 = 회색조 30 미만 비율, 회색 화소 = 채도 40 미만 비율, 불꽃색 화소 = 색상 0~60°·340~360°이면서 채도 >100, 명도 >150인 비율.

### 10.1 표본 뽑기와 대표성 확인

본 실험 범위의 영상마다 프레임을 순서대로 놓고 10장 중 1장을 고른다(시작 위치는 영상마다 무작위, 영상이 10장보다 짧으면 1장).
기본 5개 값은 63만 장 전수 측정 결과가 있으므로, 표본의 중앙값이 전수와 얼마나 다른지 확인한다.
''')
code(r'''
S = use.sort_values(['vid', 'frame']).copy()
rank = S.groupby('vid').cumcount()
size = S.groupby('vid').vid.transform('size')
u = pd.Series(np.random.default_rng(0).random(S.vid.nunique()), index=S.vid.unique())
S = S[(rank % 10) == np.floor(S.vid.map(u) * np.minimum(10, size)).astype(int)]
px = E.aihub_pixels(S[['official', 'jpg']].rename(columns={'official': 'src'}), REBUILD)
S = S.merge(px.rename(columns={'src': 'official', 'w': 'pw', 'h': 'ph'}), on=['official', 'jpg'], how='left')
S['bpp'] = S.nbytes * 8 / (S.pw * S.ph)
S['short'] = S[['pw', 'ph']].min(axis=1)
print(f'표본 {len(S):,}장 = 본 실험 범위의 {100 * len(S) / len(use):.1f}%, 열기 실패 {int(S["error"].notna().sum()) if "error" in S else 0}장')
display(pd.crosstab(S.cls, S.split.map(SPLIT_KO))[['학습', '검증', '시험']])

F5 = ['bright', 'contrast', 'sat', 'sharp', 'dark_px']
uf = use.merge(full[['src', 'jpg'] + F5].rename(columns={'src': 'official', 'jpg': 'jpg_name'}), on=['official', 'jpg_name'])
rep = pd.concat({'전수': uf.groupby('cls')[F5].median(), '10% 표본': S.groupby('cls')[F5].median()}, axis=1)
display(rep.round(3))
print('표본 중앙값 ÷ 전수 중앙값 - 1 (%):')
pd.DataFrame({f: (rep[('10% 표본', f)] / rep[('전수', f)] - 1) * 100 for f in F5 if f != 'dark_px'}).round(2)
''')
md(r'''
### 10.2 클래스별 분포
''')
code(r'''
sm = S[S.cls.isin(CLS)].copy()
FEATS = [('bright', '밝기', np.linspace(0, 255, 52), False), ('contrast', '대비', np.linspace(0, 120, 49), False),
         ('sat', '채도', np.linspace(0, 255, 52), False), ('sharp', '선명도 (로그 축)', np.logspace(0, 4.5, 50), True),
         ('gray_px', '회색 화소 비율', np.linspace(0, 1, 41), False)]
fig, axes = plt.subplots(1, len(FEATS), figsize=(14, 2.8), sharey=False)
for ax, (f, title, bins, logx) in zip(axes, FEATS):
    step_hist(ax, {c: sm.loc[sm.cls == c, f] for c in CLS}, CLS_COLOR, bins=bins)
    if logx: ax.set_xscale('log')
    ax.set_title(title)
axes[0].set_ylabel('사진 비율 (%)'); axes[-1].legend(loc='upper right')
plt.tight_layout(); plt.show()
display(sm.groupby('cls')[[f for f, *_ in FEATS] + ['dark_px', 'fire_px']].median().reindex(CLS).round(3))
fire_tab = pd.DataFrame({f'불꽃색 화소 ≥ {t * 100:g}%': sm.groupby('cls').fire_px.apply(lambda s: pct(s >= t)) for t in (0.001, 0.01, 0.05)}).reindex(CLS)
print('불꽃색 화소가 일정 비율 이상인 사진 (%)'); fire_tab
''')
code(r'''
# 색상 분포: 채도 60 이상 화소의 색상(30도 간격)별 비율의 평균
HUE = [f'hue{i:02d}' for i in range(E.HUE_BINS)]
fig, ax = plt.subplots(figsize=(7.5, 2.9))
for c in CLS:
    ax.plot(np.arange(15, 360, 30), sm.loc[sm.cls == c, HUE].mean() * 100, color=CLS_COLOR[c], marker='o', markersize=4, label=c)
ax.set_xticks(range(0, 361, 60)); ax.set_xlabel('색상 (도, 0 = 빨강 · 60 = 노랑 · 120 = 초록 · 240 = 파랑)')
ax.set_ylabel('화소 비율 (%)'); ax.set_title('색상 분포 (채도 60 이상 화소)'); ax.grid(axis='y'); ax.legend()
plt.show()
''')
md(r'''
### 10.3 방해 요소가 있는 정상 사진과 연기 사진 비교

구름·안개가 낀 정상 사진이 픽셀 값으로 연기 사진과 얼마나 비슷한지 본다. 점 = 중앙값, 선 = 25~75% 범위.
''')
code(r'''
sn = S[S.cls == '정상'].merge(N[['idx', 'f_구름', 'f_안개/연무', 'f_굴뚝연기', 'f_구름안개']], on='idx')
groups = {'연기 (양성)': S[S.cls == '연기'], '불꽃 (양성)': S[S.cls == '불꽃'],
          '정상 · 구름 20% 이상': sn[sn['f_구름'] >= .2], '정상 · 안개/연무 20% 이상': sn[sn['f_안개/연무'] >= .2],
          '정상 · 굴뚝연기 있음': sn[sn['f_굴뚝연기'] > 0], '정상 · 구름+안개 5% 미만': sn[sn['f_구름안개'] < .05]}
gcol = {k: CLS_COLOR['연기'] if k.startswith('연기') else CLS_COLOR['불꽃'] if k.startswith('불꽃') else CLS_COLOR['정상'] for k in groups}
fig, axes = plt.subplots(1, 4, figsize=(13, 2.9), sharey=True)
for ax, (f, title) in zip(axes, [('gray_px', '회색 화소 비율'), ('sat', '채도'), ('contrast', '대비'), ('bright', '밝기')]):
    for yi, (k, g) in enumerate(groups.items()):
        q1, q2, q3 = g[f].quantile([.25, .5, .75])
        ax.plot([q1, q3], [yi, yi], color=gcol[k], linewidth=2.5, solid_capstyle='round')
        ax.plot(q2, yi, 'o', color=gcol[k], markersize=7, markeredgecolor=SURFACE, markeredgewidth=2)
    ax.set_title(title); ax.grid(axis='x')
axes[0].set_yticks(range(len(groups)), [f'{k} ({len(g):,})' for k, g in groups.items()]); axes[0].invert_yaxis()
plt.show()
pd.DataFrame({k: g[['gray_px', 'sat', 'contrast', 'bright']].median() for k, g in groups.items()}).T.round(3)
''')
code(r'''
# 화면 글자: 드론 사진 왼쪽 위의 날짜·GPS 글자 점수(analyze_aihub_images.py — 글자가 있으면 140 이상, 없으면 0 근처)
txt = mf.assign(has_text=mf.text > 100).groupby('cls').has_text.mean().reindex(CLS) * 100
print('왼쪽 위에 날짜·GPS 글자가 있는 사진 비율 (%):', txt.round(1).to_dict())
''')
obs('pixels')

# ============================================================ 11. 지름길
md(r'''
## 11. 지름길 점검: 사진 내용 없이 숫자 몇 개로 클래스를 맞히는 정도

학습 쪽에서 깊이 2 결정 나무(클래스 균형 가중치)를 만들고 검증·시험 쪽에서 **균형 정확도**(클래스별 재현율의 평균)를 잰다. 찍기 수준은 33.3%다.
값이 높을수록 모델이 연기·불꽃 대신 해상도·압축·밝기 같은 지름길로 점수를 낼 수 있다는 뜻이다.
''')
code(r'''
FS = {'해상도·파일 크기': ['short', 'bpp'],
      '밝기·색·선명도': ['bright', 'contrast', 'sat', 'sharp', 'dark_px', 'fire_px', 'gray_px'],
      '전부': ['short', 'bpp', 'bright', 'contrast', 'sat', 'sharp', 'dark_px', 'fire_px', 'gray_px']}
tr = sm[sm.split == 'train']
rows, trees = [], {}
for te_name in ['val', 'test']:
    te = sm[sm.split == te_name]
    row = {}
    for fs, cols in FS.items():
        acc, desc = E.shortcut_tree(tr, te, cols, label='cls')
        row[fs + ' (%)'] = round(100 * acc, 1)
        trees[(te_name, fs)] = desc
    rows.append(pd.Series(row, name=f'학습 → {SPLIT_KO[te_name]}'))
short_tab = pd.DataFrame(rows)
short_tab['찍기 (%)'] = 33.3
display(short_tab)
print('[전부 · 학습 → 시험] 찾은 규칙'); print(trees[('test', '전부')])
''')
obs('shortcut')

# ============================================================ 12. 누수
md(r'''
## 12. 중복·누수

사진마다 32×18 회색 썸네일(밝기 평균·대비를 맞춤)의 평균 절대차를 잰다. **차이 < 6이면 거의 같은 사진**이다(`check_aihub_leak.py`와 같은 척도).
- 공식 분할: 공식 검증 사진(클래스당 1,000장)마다 **같은 카메라**의 공식 학습 사진 전부와 비교
- 장소+촬영일 분할(split_v2): 검증·시험 사진(분할·클래스당 1,000장)마다 **같은 장소**의 학습 사진 전부와 비교
''')
code(r'''
key = pd.Series(np.arange(len(full)), index=pd.MultiIndex.from_arrays([full.src, full.jpg]))
main['k'] = key.reindex(pd.MultiIndex.from_arrays([main.official, main.jpg_name])).to_numpy()
print('썸네일 없는 사진:', int(main.k.isna().sum()))

def leak(queries, cands, by):
    cand = {k: g.k.to_numpy(int) for k, g in cands.groupby(by)}
    out = []
    for kv, g in queries.groupby(by):
        if kv in cand:
            d, j = E.nearest_diff(TH, g.k.to_numpy(int), cand[kv])
        else:
            d, j = np.full(len(g), np.inf), np.full(len(g), -1)
        out.append(g.assign(d=d, near=j))
    return pd.concat(out)

t0 = time.time()
q_off = main[main.official == 'val'].groupby('cls').sample(1000, random_state=0)
r_off = leak(q_off, main[main.official == 'train'], 'place_cam').assign(기준='공식 분할', 분할='검증')
q_sp = main[main.split.isin(['val', 'test'])].groupby(['split', 'cls']).sample(1000, random_state=0)
r_sp = leak(q_sp, main[main.split == 'train'], 'address').assign(기준='장소+촬영일 분할', 분할=lambda d: d.split.map(SPLIT_KO))
R = pd.concat([r_off, r_sp])
leak_tab = R.groupby(['기준', '분할', 'cls']).agg(
    사진=('d', 'size'), 같은_카메라·장소가_학습에_있음=('d', lambda s: pct(np.isfinite(s))),
    거의_같음=('d', lambda s: pct(s < 6)),
    차이_중앙값=('d', lambda s: round(float(np.median(s[np.isfinite(s)])), 1) if np.isfinite(s).any() else np.nan))
print(f'{time.time() - t0:.0f}초 (비율은 %)')
leak_tab
''')
code(r'''
fig, axes = plt.subplots(1, 2, figsize=(11, 2.8), sharey=True)
for ax, (t, g) in zip(axes, [('공식 분할: 검증 → 같은 카메라 학습', r_off), ('장소+촬영일 분할: 검증·시험 → 같은 장소 학습', r_sp)]):
    v = g.d[np.isfinite(g.d)]
    ax.hist(v.clip(upper=40), bins=np.arange(0, 41, 1), color=NEUTRAL)
    ax.axvline(6, color=INK, linewidth=1); ax.text(6.5, ax.get_ylim()[1] * 0.85, '거의 같음 < 6', fontsize=8.5)
    ax.set_title(t); ax.set_xlabel('가장 비슷한 학습 사진과의 차이 (40 이상은 40)'); ax.grid(axis='y')
    ax.text(0.98, 0.9, f'비교할 학습 사진이 없는 {int((~np.isfinite(g.d)).sum()):,}장 제외', transform=ax.transAxes, ha='right', fontsize=8.5, color=INK2)
axes[0].set_ylabel('사진 수')
plt.show()
''')
code(r'''
# 공식 분할에서 '거의 같음'으로 나온 쌍 보기: 왼쪽 검증, 오른쪽 학습
kk = main.dropna(subset=['k'])
inv = kk.set_index(kk.k.astype(int))
tiles = []
for c in CLS:
    for r in r_off[(r_off.cls == c) & (r_off.d < 6)].sample(2, random_state=0).itertuples():
        t = inv.loc[r.near]
        tiles.append(dict(img=E.load_image(r.jpg, r.official), caption=f'검증 {c} · 프레임 {r.frame}'))
        tiles.append(dict(img=E.load_image(t.jpg, t.official), caption=f'학습 · 프레임 {t.frame} · 차이 {r.d:.1f}'))
show_jpeg(E.grid(tiles, cols=4, tw=300, title='공식 분할: 검증 사진(왼쪽)과 가장 비슷한 학습 사진(오른쪽)'))
''')
obs('leak')

# ============================================================ 13. 정리
md(r'''
## 13. 정리: 발견 → 전처리·실험 결정

아래 표의 수치는 이 노트북의 계산 결과에서 바로 가져온다.
''')
code(r'''
off_leak = R[R.기준 == '공식 분할'].d.lt(6).mean() * 100
sp_leak = R[R.기준 == '장소+촬영일 분할'].d.lt(6).mean() * 100
fl_orig, fl224 = Bl[Bl.name == '화염']['원본'], Bl[Bl.name == '화염']['224']
summary = pd.DataFrame([
    ('공식 학습/검증 분할 누수', f'공식 검증의 {off_leak:.1f}%가 같은 카메라 학습 사진과 거의 같음 → 장소+촬영일 분할에서는 {sp_leak:.1f}%',
     '공식 분할을 쓰지 않고 split_v2(장소+촬영일, 60:20:20)를 쓴다'),
    ('밤 정상이 없음', f"'야간' 정상 중 실제로 어두운 사진 {dark.loc['야간', '정상']:.1f}% (불꽃 '야간'은 {dark.loc['야간', '불꽃']:.1f}%), 어두운 사진의 {pct(mf[mf.dark].cls == '불꽃')}%가 불꽃",
     "'어두우면 불꽃' 지름길 → 밤 정상은 만들지 않고, 실제 밝기(dark 열)로 밤 결과를 따로 본다 (10/25 결정)"),
    ('불꽃이 아주 작음', f'가장 큰 화염 박스 짧은 변 중앙값: 원본 {fl_orig.median():.0f}px, 224로 줄이면 {fl224.median():.0f}px ({pct(fl224 < 8)}%가 8px 미만)',
     '줄이지 않고 원본 크기(1920×972) 입력, 거리별 결과를 따로 본다'),
    ('해상도가 클래스·분할마다 다름', f"4K 비율: 시험 정상 {k4.loc['시험', '정상']:.0f}%, 시험 연기 {k4.loc['시험', '연기']:.0f}%, 시험 불꽃 {k4.loc['시험', '불꽃']:.0f}%",
     '3840×2160은 1920×1080으로 줄여 통일 (4K = 정상 지름길 차단)'),
    ('안개/연무 정상의 장소가 적음', f"안개/연무 정상 장소 전체 {places.loc['안개연무', '전체 장소']}곳 (학습 {places.loc['안개연무', '학습']} · 검증 {places.loc['안개연무', '검증']} · 시험 {places.loc['안개연무', '시험']}곳)",
     '안개 결과는 장소별로 보고, 합성 안개(같은 사진 비교)로 보강'),
    ('방해 요소 면적', f"시험 정상의 구름+안개 면적 <5% {bins_split.loc['<5%', '시험']:,}장 · 5~20% {bins_split.loc['5~20%', '시험']:,}장 · ≥20% {bins_split.loc['≥20%', '시험']:,}장",
     '면적 구간별 오경보율로 가설 확인, 학습 정상 구성 바꾸기(<5%만 / 원래 / 큰 것 2배)'),
    ('구름 낀 산불 사진은 시험에 거의 없음', f"시험 양성 중 구름 박스가 있는 사진 {int(pos_d.loc[('test', '연기'), '구름'] + pos_d.loc[('test', '불꽃'), '구름'])}장",
     '방해 요소가 있을 때 불을 놓치는지는 합성 안개로만 확인'),
    ('계절이 클래스와 묶임', f"불꽃·연기는 {summer['불꽃']:.0f}%·{summer['연기']:.0f}%가 8~9월, 정상은 {autumn['정상']:.0f}%가 10~12월 (단풍·마른 풀)",
     '색으로 구별하는 지름길 → 색상 증강(hue jitter)을 검토하고, 정상 중 8~9월 사진의 오경보율을 따로 본다'),
    ('음성 속 연기', f"{int((img.cls == '제외: 음성 속 연기').sum()):,}장", "정상에서 빼고 '판별 불확실'로 따로 보관"),
    ('GPS·고도 값 오류', f'솔향수목원 좌표 한 값이 {img.loc[img.gps == top_gps, "address"].nunique()}개 주소에 들어감 (강릉 밖 사진 {int((outside & (img.gps == top_gps)).sum()):,}장 포함)',
     'GPS·고도는 분석에 쓰지 않는다'),
    ('촬영 시각 형식 섞임', f"정상 형식이 아닌 값 {int((fmt != 'YYYYMMDD_HHMMSS (정상)').sum()):,}장", '날짜만 YYYYMMDD로 통일해 쓰고, 시각 대신 시간대 라벨·실제 밝기를 쓴다'),
    ('지름길 크기', f"숫자 9개 깊이 2 나무의 균형 정확도: 시험 {short_tab.loc['학습 → 시험', '전부 (%)']}% (찍기 33.3%)",
     '모델 결과와 함께 이 기준선을 적는다. 해상도 통일·글자 띠 자르기 뒤 다시 잰 값은 전처리 노트북 7.2'),
], columns=['발견', '수치', '전처리·실험 결정'])
summary.style.hide(axis='index').set_properties(**{'text-align': 'left', 'white-space': 'pre-wrap'})
''')
obs('final')

nb = nbformat.v4.new_notebook(cells=cells, metadata={
    'kernelspec': {'name': 'eda', 'display_name': 'Python (wildfire EDA)', 'language': 'python'},
    'language_info': {'name': 'python'}})
if '--no-exec' not in sys.argv:
    t = time.time()
    NotebookClient(nb, timeout=3600, kernel_name='eda', resources={'metadata': {'path': NB_DIR}}).execute()
    print(f'실행 {time.time() - t:.0f}초')
nbformat.write(nb, OUT)
print('saved', OUT, os.path.getsize(OUT))
