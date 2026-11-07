"""AI Hub 71330 라벨 요약: 우리 3클래스(불꽃·연기·정상)로 바꾼 개수와 출처·조건 쏠림.

analyze_aihub_labels.py가 만든 report/aihub/images.csv.gz를 읽는다.
사용: python3 -I summarize_aihub_labels.py  →  표를 출력하고 report/aihub/summary_*.csv로 저장
"""
import pandas as pd

OUT = '/data/home/seame/dl-team-project/team6/report/aihub'
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
df = pd.read_csv(f'{OUT}/images.csv.gz', low_memory=False)

# 우리 라벨: 화염이 있으면 불꽃, 흑색·백색/회색연기만 있으면 연기, 음성 DB에서 연기·화염 라벨이 없으면 정상
smoke = (df.n_black + df.n_white) > 0
df['label'] = 'normal'
df.loc[smoke, 'label'] = 'smoke'
df.loc[df.n_flame > 0, 'label'] = 'fire'
df.loc[(df.db == 'NegativeDB') & smoke, 'label'] = 'neg_with_smoke'   # 음성인데 연기 라벨이 붙은 사진: 눈으로 확인 전까지 따로 둔다
# 실제 양성은 4곳(솔향수목원·죽헌저수지삼거리·시루봉정상입구·춘갑봉공터)만 쓴다 (2026-10-15 팀 결정)
DROP_POS_PLACES = ['경포대공터', '경포대입구', '모래고개입구', '춘갑봉등산로', '춘갑봉민가', '옥천면비닐하우스', '옥천면교각밑',
                   '춘갑봉샛길', '옥천면계곡', '사천면', '죽헌저수지방둑', '죽헌저수지공터']
df['use'] = ~((df.db == 'PositiveRealDB') & df.address.isin(['강원도강릉시' + p for p in DROP_POS_PLACES]))
df['dist'] = pd.to_numeric(df.distance)
df['dist_bin'] = pd.cut(df.dist, [-1, 49.999, 299.999, 1e9], labels=['근(<50m)', '중(50-299m)', '원(300m+)'])
df['vid'] = df.place_cam + '/' + df.video_fuel
short = df[['w', 'h']].min(axis=1)
df['smoke_box224'] = df[['box_black', 'box_white']].max(axis=1) * 224 / short
df['flame_box224'] = df.box_flame * 224 / short


def show(title, t, name):
    print(f'\n[{title}]'); print(t); t.to_csv(f'{OUT}/summary_{name}.csv')


show('DB × 우리 라벨', pd.crosstab(df.db, df.label, margins=True), 'db_label')
print('\n쓰지 않는 실제 양성 장소 사진: %d장' % (~df.use).sum())
real = df[(df.db != 'PositiveSyntheticDB') & df.use]         # 평가에는 합성 양성을 쓰지 않는다
show('실제 사진: 우리 라벨 × 수집 방법', pd.crosstab(real.label, real.collection), 'label_collection')
show('실제 사진: 우리 라벨 × 거리 (%)', (pd.crosstab(real.label, real.dist_bin, normalize='index') * 100).round(1), 'label_dist')
show('실제 사진: 우리 라벨 × 날씨 (%)', (pd.crosstab(real.label, real.weather, normalize='index') * 100).round(1), 'label_weather')
show('실제 사진: 우리 라벨 × 시간대 (%)', (pd.crosstab(real.label, real.timezone, normalize='index') * 100).round(1), 'label_timezone')
show('실제 사진: 우리 라벨 × 기기 (%)', (pd.crosstab(real.label, real.device, normalize='index') * 100).round(1), 'label_device')
show('실제 사진: 우리 라벨 × 해상도 (%)', (pd.crosstab(real.label, real.w.astype(str) + 'x' + real.h.astype(str),
                                                 normalize='index') * 100).round(1), 'label_resolution')
show('실제 사진: 날씨 × 시간대 사진 수 (연기)', pd.crosstab(real[real.label == 'smoke'].weather,
                                                  real[real.label == 'smoke'].timezone, margins=True), 'smoke_weather_time')
show('실제 사진: 날씨 × 시간대 사진 수 (정상)', pd.crosstab(real[real.label == 'normal'].weather,
                                                  real[real.label == 'normal'].timezone, margins=True), 'normal_weather_time')

print('\n[공식 학습/검증 분할]')
v = df.groupby('vid').split.nunique()
val = df[df.split == 'val']
print('검증 사진 중 같은 영상이 학습에도 있는 비율: %.1f%%' % (100 * val.vid.isin(v[v > 1].index).mean()))
g = df.groupby('db').agg(places=('address', 'nunique'), cameras=('camera_id', 'nunique'), videos=('vid', 'nunique'),
                         images=('path', 'size'))
show('DB별 장소·카메라·영상 수', g, 'db_groups')
for a in ('PositiveRealDB', 'PositiveSyntheticDB'):
    print(f'{a} ∩ NegativeDB 장소:', sorted(set(df[df.db == a].address) & set(df[df.db == 'NegativeDB'].address)))

print('\n[224px로 줄였을 때 가장 큰 물체 박스의 짧은 변(px), 실제 양성]')
r = df[df.db == 'PositiveRealDB']
q = pd.DataFrame({'연기': r[r.smoke_box224 > 0].smoke_box224.quantile([.1, .25, .5, .75, .9]),
                  '화염': r[r.flame_box224 > 0].flame_box224.quantile([.1, .25, .5, .75, .9])}).round(1)
print(q); q.to_csv(f'{OUT}/summary_box224.csv')
print('8px 미만: 연기 %.1f%%, 화염 %.1f%%' % (100 * (r[r.smoke_box224 > 0].smoke_box224 < 8).mean(),
                                         100 * (r[r.flame_box224 > 0].flame_box224 < 8).mean()))
df[['path', 'split', 'db', 'label', 'use', 'dist_bin', 'vid', 'smoke_box224', 'flame_box224']].to_csv(
    f'{OUT}/labels.csv.gz', index=False)
