"""split_v1을 고쳐 split_v2를 만든다: 음성 묶음을 다시 배정해 기기(드론)·구름·안개 등의 비율을 학습·검증·시험에 맞춘다.

split_v1(make_aihub_split.py)은 층에 기기·방해 요소가 없어, 시험 음성의 63%가 드론(학습 음성 10%)이 됐다(전처리 노트북 3.1).
 - 양성이 든 묶음 13개는 그대로 둔다(양성 쪽 사진·비율은 v1과 같다).
 - 음성만 든 묶음 69개를 다시 배정한다. 비용 = v1과 같은 층별 |비율 - 60/20/20| 벌점(음성 층) + 새 층
   (드론, 안개 면적 ≥5%·≥20%, 구름 면적 ≥5%·≥20%, 굴뚝연기, 8~9월; 드론·안개≥20%·구름≥20%는 가중치 2)
   + 시험의 안개 ≥20% 음성이 v1보다 적거나, 그런 안개가 100장 넘는 장소가 시험에 3곳보다 적으면 벌점.
 - 이렇게 정하기까지: ① v1에서 묶음을 가장 적게 옮기는 방법(옮긴 묶음마다 벌점)은 13개를 옮겨 드론은 맞았지만, 구름 층을 보지 않아
   시험에 든 구름 음성이 24% → 13%로 줄었다. ② 구름·안개를 '박스가 있음'으로 세면, 안개가 아주 조금 걸친 큰 묶음(구정면 8/22, 안개
   면적 중앙값 0.3%)이 시험의 안개 몫을 채워 진짜 안개 사진(평창·강동면모전리·홍제동 등)이 모두 검증으로 빠졌다(시험 '안개/연무만' 116장).
   → 방해 요소 층은 면적으로 세고(가설 검증 묶음과 같은 기준), 묶음 수 제한 없이 음성 묶음 전체를 다시 배정한다.
 - 찾기: v1 배정 + 무작위 배정 240개(시드 0)에서 시작해, 묶음 하나 옮기기·두 묶음 맞바꾸기를 비용이 더 줄지 않을 때까지 반복.
   끝난 배정 중 검증·시험에서 층 비율이 20%에서 가장 멀리 벗어난 정도(음성 속 연기 층 제외)가 가장 작은 것을 고른다.
 - 묶음은 거의 같은 사진이 있는 날짜 묶음끼리 이미 합쳐 둔 단위라, 묶음째 옮기면 분할 사이 누수가 새로 생기지 않는다
   (서로 다른 묶음 사이에 거의 같은 사진이 있는 쌍은 구정면 9/13↔10/28 하나뿐, 0.4%·0.1%). check_aihub_split.py로 다시 잰다.
 - 층의 사진 수는 전처리 목록(prep/manifest.csv.gz, uncertain.csv.gz)에 남은 사진으로 센다.
사용: .venv_eda/bin/python -I make_aihub_split_v2.py  →  report/aihub/split_v2.csv.gz, split_v2_summary.csv, split_v2_moves.csv
"""
import numpy as np
import pandas as pd

ROOT = '/data/home/seame/dl-team-project/team6'
OUT = f'{ROOT}/report/aihub'
TARGET = np.array([.60, .20, .20]); FLOOR = .05; SPLITS = np.array(['train', 'val', 'test'])
N_STARTS, SEED, TIE = 240, 0, 0.001       # 무작위 시작 수, 시드, 같은 비용이면 v1에 가까운 쪽(옮긴 묶음 하나에 0.001)
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 200); pd.set_option('display.max_columns', 30)

v1 = pd.read_csv(f'{OUT}/split_v1.csv.gz')
COLS = ['src', 'member', 'y', 'device', 'month', 'crop_fog', 'crop_chimney', 'crop_cloud', 'crop_cloudfog']
man = pd.read_csv(f'{ROOT}/prep/manifest.csv.gz', usecols=COLS)
unc = pd.read_csv(f'{ROOT}/prep/uncertain.csv.gz', usecols=['src', 'member'])
man['jpg'] = man.member.str.rsplit('/', n=1).str[-1]
unc['jpg'] = unc.member.str.rsplit('/', n=1).str[-1]
man = man.merge(pd.read_csv(f'{ROOT}/prep/manifest.csv.gz', low_memory=False)[['src', 'member', 'place']], on=['src', 'member'])
df = v1.merge(man[['src', 'jpg', 'place'] + COLS[2:]].drop(columns='device'), on=['src', 'jpg'], how='left')
df['in_man'] = df.y.notna()
df['in_unc'] = df.set_index(['src', 'jpg']).index.isin(unc.set_index(['src', 'jpg']).index)

normal = df.in_man & df.y.eq(0)
gn = df.address.str.startswith('강원도강릉시')
STRATA = {                                     # 이름: (조건, 가중치) — 앞 5개는 v1의 음성 층
    '정상·강릉': (normal & gn, 1), '정상·그밖': (normal & ~gn, 1),
    '정상·비': (normal & df.weather.eq('비'), 1), '정상·흐림': (normal & df.weather.eq('흐림'), 1),
    'neg_with_smoke': (df.in_unc, .5),
    '정상·드론': (normal & df.device.eq('drone'), 2),
    '정상·안개≥5%': (normal & (df.crop_fog >= .05), 1), '정상·안개≥20%': (normal & (df.crop_fog >= .2), 2),
    '정상·구름≥5%': (normal & (df.crop_cloud >= .05), 1), '정상·구름≥20%': (normal & (df.crop_cloud >= .2), 2),
    '정상·굴뚝연기': (normal & (df.crop_chimney > 0), 1), '정상·8~9월': (normal & df.month.isin([8, 9]), 1),
}
names = list(STRATA)
blocks = df.block.unique()
bidx = pd.Series(np.arange(len(blocks)), index=blocks)
B = np.zeros((len(blocks), len(STRATA)))
for k, (m, _) in enumerate(STRATA.values()):
    B[:, k] = df[m].groupby('block').size().reindex(blocks).fillna(0).values
Wt = np.array([w for _, w in STRATA.values()]); tot = B.sum(0); Bn = B / tot
has_pos = df.label.isin(['fire', 'smoke']).groupby(df.block).any().reindex(blocks).values
v1_assign = pd.Series(df.groupby('block').split.first()).reindex(blocks).map({s: i for i, s in enumerate(SPLITS)}).values
movable = np.nonzero(~has_pos)[0]
FOG = names.index('정상·안개≥20%')
fog_test_v1 = ((v1_assign == 2).astype(float) @ B)[FOG]
# 안개 ≥20% 사진이 100장 넘는 묶음과 그 장소 (시험에 진짜 안개 장소가 몇 곳 드는지 센다)
_f = df[normal & (df.crop_fog >= .2)]
_fc = _f.groupby('block').size()
FOG_BLOCKS = {bidx[b]: p for b, p in _f.groupby('block').place.agg(lambda s: s.mode().iloc[0]).items() if _fc[b] >= 100}
MAIN = [k for k, n in enumerate(names) if n != 'neg_with_smoke']
FLOORS = np.array([TARGET[0] - .15, FLOOR, FLOOR])[:, None]


def shares(a):
    return np.stack([(a == s) for s in range(3)]).astype(float) @ Bn          # 분할 × 층


def strata_cost(a):
    sh = shares(a)
    return float((np.abs(sh - TARGET[:, None]) * Wt).sum() + 30 * (np.clip(FLOORS - sh, 0, None) * (Wt >= 1)).sum())


def cost(a):
    fog_test = ((a == 2).astype(float) @ B)[FOG]
    fog_places_test = len({p for b, p in FOG_BLOCKS.items() if a[b] == 2})
    return (strata_cost(a) + 5 * max(0.0, fog_test_v1 - fog_test) / fog_test_v1 + 0.5 * max(0, 3 - fog_places_test)
            + TIE * (a != v1_assign).sum())


def max_dev(a):
    sh = shares(a) * 100
    return float(np.abs(sh[1:, MAIN] - 20).max()), float(np.abs(sh[0, MAIN] - 60).max())


def local(a):
    """묶음 하나 옮기기·두 묶음 맞바꾸기 중 비용을 가장 많이 줄이는 것을 더 줄지 않을 때까지."""
    bc = cost(a)
    while True:
        cand = (bc, None)
        for b in movable:
            for s in range(3):
                if s != a[b]:
                    t = a.copy(); t[b] = s; c = cost(t)
                    if c < cand[0] - 1e-9:
                        cand = (c, t)
        for i, b1 in enumerate(movable):
            for b2 in movable[i + 1:]:
                if a[b1] != a[b2]:
                    t = a.copy(); t[b1], t[b2] = a[b2], a[b1]; c = cost(t)
                    if c < cand[0] - 1e-9:
                        cand = (c, t)
        if cand[1] is None:
            return a, bc
        bc, a = cand


rng = np.random.default_rng(SEED)
starts = [v1_assign.copy()]
for _ in range(N_STARTS):
    a = v1_assign.copy(); a[movable] = rng.choice(3, size=len(movable), p=TARGET); starts.append(a)
found = []
for a0 in starts:
    a, c = local(a0)
    found.append((max_dev(a)[0], c, a))
found.sort(key=lambda r: (r[0], r[1]))
best = found[0][2]
print(f'v1: 층 비용 {strata_cost(v1_assign):.3f}, 검증·시험 최대 차이 {max_dev(v1_assign)[0]:.1f}%p, 학습 {max_dev(v1_assign)[1]:.1f}%p')
print(f'v2: 층 비용 {strata_cost(best):.3f}, 검증·시험 최대 차이 {max_dev(best)[0]:.1f}%p, 학습 {max_dev(best)[1]:.1f}%p, '
      f'v1과 다른 분할로 간 음성 묶음 {(best != v1_assign).sum()}개 / {len(movable)}개 (시작 {len(starts)}개 중 최선)')

df['split_v1'] = df.split
df['split'] = SPLITS[best[bidx[df.block].values]]
moves = (df[df.split != df.split_v1].groupby('block')
         .agg(v1=('split_v1', 'first'), v2=('split', 'first'), 사진=('path', 'size'), 정상=('label', lambda s: (s == 'normal').sum()),
              드론pct=('device', lambda s: round(100 * (s == 'drone').mean())), 안개20pct=('crop_fog', lambda s: round(100 * (s >= .2).mean())),
              구름20pct=('crop_cloud', lambda s: round(100 * (s >= .2).mean())), 주소=('address', 'first'))
         .sort_values(['v1', '사진'], ascending=[True, False]))
moves.to_csv(f'{OUT}/split_v2_moves.csv')
print('\n[v1과 다른 분할로 간 묶음]'); print(moves)

df[v1.columns].to_csv(f'{OUT}/split_v2.csv.gz', index=False)
rows = []
for name, (m, _) in STRATA.items():
    for ver, col in (('v1', 'split_v1'), ('v2', 'split')):
        c = df[m][col].value_counts().reindex(SPLITS).fillna(0).astype(int)
        rows.append(dict(층=name, 판=ver, **{s: c[s] for s in SPLITS}, **{f'{s}%': round(100 * c[s] / max(1, c.sum()), 1) for s in SPLITS}))
summ = pd.DataFrame(rows).set_index(['층', '판']); summ.to_csv(f'{OUT}/split_v2_summary.csv')
print('\n[층별 사진 수와 비율: v1 → v2]'); print(summ)
print('\n[클래스별 사진 수 v2]'); print(pd.crosstab(df.label, df.split, margins=True)[list(SPLITS) + ['All']])
