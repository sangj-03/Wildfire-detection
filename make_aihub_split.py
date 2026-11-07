"""AI Hub 71330을 '장소+촬영일' 묶음 단위로 학습/검증/시험(60/20/20)으로 다시 나눈다.

공식 학습/검증 분할은 같은 영상의 이웃 프레임이 양쪽에 들어가 쓰지 않는다(check_aihub_leak.py: 검증의 97%가 학습에 거의 같은 사진).
 - 대상: 실제 사진(합성 양성 제외), 쓰기로 한 양성 장소 4곳(use). 음성 중 백색/회색연기 라벨 사진(neg_with_smoke)도
   묶음을 따라 분할만 받아 두고, 쓸지는 학습 코드에서 정한다.
 - 촬영일: datetime 앞 8자리(20220914_...) 또는 6자리(221027_...)를 YYYYMMDD로 맞춘다.
 - 묶음 = 주소 + 촬영일. 한 카메라(장소_ID)가 이틀에 걸치면 그 묶음들은 합친다(같은 화면이 두 분할로 갈라지지 않게).
   또 aihub_group_links.py가 같은 장소의 날짜 묶음끼리 거의 같은 사진을 찾았으면(한쪽 방향이라도 1% 이상, 또는 똑같은 사진이
   1장이라도) 합친다 — 같은 영상이 다른 날짜로 다시 들어간 경우가 있었다(소하동 11/11↔12/01, 장현동 10/17↔10/18 등).
 - 묶음을 무작위로 많이 배정해 보고, 아래 층마다 사진 수 비율이 60/20/20에 가장 가까운 배정을 고른 뒤 묶음 하나씩 옮겨 다듬는다.
   검증·시험은 층마다 5% 이상, 학습은 45% 이상이 되도록 벌점을 준다(밤 불꽃은 묶음 두 개에 94%가 몰려 있어 10%는 못 맞춘다).
   70/15/15는 양성 묶음이 15개뿐이고 커서(최대 4.8만 장) 검증에 밤 불꽃을 넣을 수 없었다(4.9%) → 60/20/20.
   층: 불꽃·연기 × 낮·밤, 양성 날씨 3종, 정상 × 강릉·그 밖, 정상 날씨(비·흐림), neg_with_smoke
사용: python3 -I make_aihub_split.py  →  report/aihub/split_v1.csv.gz, split_v1_summary.csv, split_v1_blocks.csv
"""
import re

import numpy as np
import pandas as pd

OUT = '/data/home/seame/dl-team-project/team6/report/aihub'
TARGET = np.array([.60, .20, .20]); FLOOR = .05; SPLITS = np.array(['train', 'val', 'test'])
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 200); pd.set_option('display.max_columns', 30)

lab = pd.read_csv(f'{OUT}/labels.csv.gz', usecols=['path', 'split', 'db', 'label', 'use', 'vid'])
meta = pd.read_csv(f'{OUT}/images.csv.gz', low_memory=False,
                   usecols=['path', 'address', 'datetime', 'timezone', 'weather', 'place_cam', 'device'])
df = lab.merge(meta, on='path').rename(columns={'split': 'src'})
df = df[(df.db != 'PositiveSyntheticDB') & df.use].reset_index(drop=True)


def norm_date(s):
    s = str(s)
    if m := re.match(r'(20\d{6})_', s):
        return m.group(1)
    if m := re.match(r'(\d{6})_', s):
        return '20' + m.group(1)
    raise ValueError(f'날짜 형식을 모름: {s}')


df['date'] = df.datetime.map(norm_date)
df['group'] = df.address.str.replace('강원도강릉시', '강릉 ') + '/' + df.date
df['jpg'] = df.path.str.rsplit('/', n=1).str[-1].str.replace('.json', '.jpg')

# 같은 카메라가 걸친 묶음끼리 합치기
gid = {g: i for i, g in enumerate(df.group.unique())}
par = list(range(len(gid)))
def find(a):
    while par[a] != a:
        par[a] = par[par[a]]; a = par[a]
    return a
for _, g in df.groupby('place_cam').group.unique().items():
    for x in g[1:]:
        par[find(gid[x])] = find(gid[g[0]])
links = pd.read_csv(f'{OUT}/group_links.csv')
links = links[(links.near_pct >= 1) | (links.same > 0)]
for x, y in zip(links.group, links.other):
    if x in gid and y in gid:
        par[find(gid[x])] = find(gid[y])
print('거의 같은 사진으로 이은 묶음 쌍:', sorted({tuple(sorted(p)) for p in zip(links.group, links.other)}))
root_name = {}
for g, i in gid.items():
    root_name.setdefault(find(i), []).append(g)
df['block'] = df.group.map(lambda g: ' + '.join(sorted(root_name[find(gid[g])])))
print(f'사진 {len(df)}장, 묶음 {len(gid)}개 → 카메라·거의 같은 사진으로 합친 뒤 {df.block.nunique()}개')

# 층별 사진 수 (묶음 × 층)
night = df.timezone == '야간'
pos = df.db == 'PositiveRealDB'
normal = df.label == 'normal'
gn = df.address.str.startswith('강원도강릉시')
STRATA = {                                     # 이름: (조건, 가중치)
    '불꽃·낮': (df.label.eq('fire') & ~night, 1), '불꽃·밤': (df.label.eq('fire') & night, 1),
    '연기·낮': (df.label.eq('smoke') & ~night, 1), '연기·밤': (df.label.eq('smoke') & night, 1),
    '양성·맑음': (pos & df.weather.eq('맑음'), 1), '양성·비': (pos & df.weather.eq('비'), 1), '양성·흐림': (pos & df.weather.eq('흐림'), 1),
    '정상·강릉': (normal & gn, 1), '정상·그밖': (normal & ~gn, 1),
    '정상·비': (normal & df.weather.eq('비'), 1), '정상·흐림': (normal & df.weather.eq('흐림'), 1),
    'neg_with_smoke': (df.label.eq('neg_with_smoke'), .5),
}
blocks = df.block.unique()
bidx = pd.Series(np.arange(len(blocks)), index=blocks)
B = np.zeros((len(blocks), len(STRATA)))
for k, (name, (m, _)) in enumerate(STRATA.items()):
    B[:, k] = df[m].groupby('block').size().reindex(blocks).fillna(0).values
W = np.array([w for _, w in STRATA.values()])
tot = B.sum(0)


def cost(assign, cols=slice(None)):
    """assign: (n배정, n묶음) 정수 0/1/2 → 층(cols)마다 |비율 - 목표| × 가중치 + 최소 비율 벌점."""
    c = np.zeros(len(assign)); Bc, Wc, tc = B[:, cols], W[cols], tot[cols]
    for s in range(3):
        share = (assign == s).astype(np.float32) @ Bc / tc
        c += (np.abs(share - TARGET[s]) * Wc).sum(1)
        floor = FLOOR if s else TARGET[0] - .15
        c += 30 * (np.clip(floor - share, 0, None) * (Wc >= 1)).sum(1)   # neg_with_smoke는 최소 비율을 보지 않는다
    return c


# 1) 양성이 든 묶음(13개 남짓)은 3^n 배정을 전부 따져 양성 층만으로 고른다
pos_cols = [k for k, n in enumerate(STRATA) if n.startswith(('불꽃', '연기', '양성'))]
P = np.nonzero(B[:, pos_cols].sum(1) > 0)[0]; Nn = np.setdiff1d(np.arange(len(blocks)), P)
best = np.zeros(len(blocks), int); best_c, best_p = np.inf, None
digits = 3 ** np.arange(len(P))
for lo in range(0, 3 ** len(P), 500_000):
    idx = np.arange(lo, min(lo + 500_000, 3 ** len(P)))
    a = np.zeros((len(idx), len(blocks)), np.int8); a[:, P] = (idx[:, None] // digits) % 3
    c = cost(a, pos_cols); i = c.argmin()
    if c[i] < best_c:
        best_c, best_p = c[i], a[i, P].copy()
best[P] = best_p
print(f'양성 묶음 {len(P)}개: 배정 {3 ** len(P):,}가지 중 최선 (양성 층 비용 {best_c:.3f})')

# 2) 나머지(음성) 묶음은 무작위 배정 + 하나씩 옮기기로, 양성 배정은 고정한 채 전체 층 비용으로 고른다
rng = np.random.default_rng(0); best_c = np.inf; fixed = best.copy()
for _ in range(60):                                # 무작위 배정 60 × 5,000개
    a = np.tile(fixed, (5000, 1)); a[:, Nn] = rng.choice(3, size=(5000, len(Nn)), p=TARGET)
    c = cost(a); i = c.argmin()
    if c[i] < best_c:
        best, best_c = a[i].copy(), c[i]
improved = True
while improved:                                    # 음성 묶음 하나씩 다른 분할로 옮겨 보기
    improved = False
    for b in Nn:
        for s in range(3):
            if s == best[b]:
                continue
            t = best.copy(); t[b] = s; c = cost(t[None])[0]
            if c < best_c - 1e-9:
                best, best_c, improved = t, c, True
print(f'비용 {best_c:.3f}')
df['split'] = SPLITS[best[bidx[df.block].values]]

# 저장
df[['path', 'jpg', 'src', 'db', 'label', 'split', 'group', 'block', 'address', 'date', 'timezone', 'weather', 'device', 'vid']].to_csv(
    f'{OUT}/split_v1.csv.gz', index=False)
rows = []
for name, (m, _) in STRATA.items():
    c = df[m].split.value_counts().reindex(SPLITS).fillna(0).astype(int)
    rows.append(dict(층=name, **{s: c[s] for s in SPLITS}, **{f'{s}%': round(100 * c[s] / max(1, c.sum()), 1) for s in SPLITS}))
summ = pd.DataFrame(rows).set_index('층'); summ.to_csv(f'{OUT}/split_v1_summary.csv')
print('\n[층별 사진 수와 비율]'); print(summ)
print('\n[클래스별 사진 수]'); print(pd.crosstab(df.label, df.split, margins=True)[list(SPLITS) + ['All']])
bl = df.groupby('block').agg(분할=('split', 'first'), 사진=('path', 'size'), 불꽃=('label', lambda s: (s == 'fire').sum()),
                             연기=('label', lambda s: (s == 'smoke').sum()), 정상=('label', lambda s: (s == 'normal').sum()),
                             날씨=('weather', lambda s: '/'.join(sorted(s.unique()))), 밤pct=('timezone', lambda s: round(100 * (s == '야간').mean())))
bl.sort_values(['분할', '사진'], ascending=[True, False]).to_csv(f'{OUT}/split_v1_blocks.csv')
print('\n[양성 묶음 배정]'); print(bl[(bl.불꽃 + bl.연기) > 0].sort_values(['분할', '사진'], ascending=[True, False]))
print('\n[정상 묶음 수]', bl[bl.정상 > 0].groupby('분할').size().to_dict())
