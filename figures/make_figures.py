"""README figures in English and Korean, for light and dark pages. No dataset photo is used.

- overview: leakage of the official split, flame size by input size, season and night by class, distractor area
  (aggregates in figures/data, written by make_figure_data.py)
- preprocessing: the AI Hub frame preprocessing and the shortcut re-check (prep/tables, from notebooks/preprocess.ipynb)

    python figures/make_figures.py   →  figures/{leak,flame,time,distractors,frame,shortcut}_{en,ko}_{light,dark}.png
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, PathPatch, Patch, Rectangle  # noqa: E402
from matplotlib.path import Path  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
plt.rcParams['font.family'] = 'Noto Sans CJK KR'

THEME = {
    'light': dict(surface='#fcfcfb', ink='#0b0b0b', ink2='#52514e', muted='#898781', grid='#e1e0d9', axis='#c3c2b7',
                  s1='#2a78d6', s2='#eb6834', s3='#1baf7a', frame_fill='#f0efec'),
    'dark': dict(surface='#1a1a19', ink='#f0efec', ink2='#c3c2b7', muted='#898781', grid='#2c2c2a', axis='#383835',
                 s1='#3987e5', s2='#d95926', s3='#199e70', frame_fill='#262624'),
}
DPI_VIEW, DPI_SAVE = 100, 200   # sizes below are in display pixels; files are saved at 2x

TEXT = {
    'en': dict(
        val='train → val', test='train → test', chance='chance 50%', xlabel='balanced accuracy (%)',
        cats=['Before\nresolution + file size', 'Before\nall 9 features', 'After\n7 pixel features'],
        k4='4K drone frames are averaged 2×2 first → 1920×1080',
        band='top 108 px (10%): date · time · GPS text → cut',
        orig='AI Hub frame · 1920×1080', crop='crop', out='model input · 1920×972, not resized',
        note='fire/smoke boxes move up 108 px · 24.2% of them touch the band (median 79.8% of their height kept)\n'
             '45 positive frames whose boxes would all be cut away are removed'),
    'ko': dict(
        val='학습 → 검증', test='학습 → 시험', chance='찍기 50%', xlabel='균형 정확도 (%)',
        cats=['처리 전\n해상도·파일 크기', '처리 전\n전부 9개', '처리 후\n픽셀 7개'],
        k4='4K 드론 사진은 먼저 2×2 화소 평균 → 1920×1080',
        band='위 108px(10%): 날짜·시각·GPS 글자 띠 → 잘라 냄',
        orig='AI Hub 사진 · 1920×1080', crop='자르기', out='모델 입력 · 1920×972 (줄이지 않음)',
        note='불·연기 박스는 108px 위로 옮김 · 박스의 24.2%가 띠에 걸림(높이의 79.8%가 남음, 중앙값)\n'
             '잘라 내면 박스가 모두 사라지는 양성 45장은 뺌'),
}


def rounded_hbar(ax, y_center, length, thick, color, rx, ry):
    """Horizontal bar from x=0: square at the baseline, 4px rounded corners at the data end."""
    y0, y1, x1 = y_center - thick / 2, y_center + thick / 2, length
    k = 0.5523
    verts = [(0, y0), (x1 - rx, y0), (x1 - rx + k * rx, y0), (x1, y0 + ry - k * ry), (x1, y0 + ry),
             (x1, y1 - ry), (x1, y1 - ry + k * ry), (x1 - rx + k * rx, y1), (x1 - rx, y1), (0, y1), (0, y0)]
    codes = [Path.MOVETO, Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.LINETO,
             Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.LINETO, Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(verts, codes), facecolor=color, edgecolor='none', zorder=3))


def shortcut(lang, mode):
    t, c = TEXT[lang], THEME[mode]
    tab = pd.read_csv(f'{ROOT}/prep/tables/shortcut.csv', index_col=0)
    rows = ['처리 전: 해상도·파일 크기', '처리 전: 전부 9개', '처리 후: 픽셀 7개 (전체 사진)']
    val, test = tab.loc[rows, '학습 → 검증'].tolist(), tab.loc[rows, '학습 → 시험'].tolist()

    W, H = 880, 300
    L, R, T, B = 200, 48, 58, 52
    fig = plt.figure(figsize=(W / DPI_VIEW, H / DPI_VIEW), dpi=DPI_VIEW, facecolor=c['surface'])
    ax = fig.add_axes([L / W, B / H, (W - L - R) / W, (H - T - B) / H], facecolor=c['surface'])
    aw, ah = W - L - R, H - T - B
    n = len(rows)
    ax.set_xlim(0, 100); ax.set_ylim(-0.5, n - 0.5)
    px_x, px_y = 100 / aw, n / ah                    # data units per display pixel
    thick, gap = 22 * px_y, 2 * px_y
    for i, (v, s) in enumerate(zip(val, test)):
        yc = n - 1 - i
        rounded_hbar(ax, yc + (thick + gap) / 2, v, thick, c['s1'], 4 * px_x, 4 * px_y)
        rounded_hbar(ax, yc - (thick + gap) / 2, s, thick, c['s2'], 4 * px_x, 4 * px_y)
        ax.text(s + 6 * px_x, yc - (thick + gap) / 2, f'{s:.1f}', va='center', ha='left', fontsize=10.5,
                color=c['ink'], fontweight='bold', zorder=4)
        ax.text(v + 6 * px_x, yc + (thick + gap) / 2, f'{v:.1f}', va='center', ha='left', fontsize=9.5,
                color=c['ink2'], zorder=4)
    for x in range(0, 101, 20):
        ax.axvline(x, color=c['grid'], linewidth=1, zorder=1)
    ax.axvline(0, color=c['axis'], linewidth=1, zorder=2)
    ax.axvline(50, color=c['muted'], linewidth=1.5, zorder=2)
    ax.text(50, n - 0.5 + 8 * px_y, t['chance'], ha='center', va='bottom', fontsize=9.5, color=c['ink2'])
    ax.set_yticks(range(n), t['cats'][::-1], fontsize=10.5, color=c['ink'])
    ax.set_xticks(range(0, 101, 20), [str(x) for x in range(0, 101, 20)], fontsize=9.5, color=c['ink2'])
    ax.tick_params(length=0, pad=8)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xlabel(t['xlabel'], fontsize=10, color=c['ink2'], labelpad=6)
    handles = [Patch(facecolor=c['s1'], label=t['val']), Patch(facecolor=c['s2'], label=t['test'])]
    leg = fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(L / W - 0.005, 1 - 6 / H), ncol=2,
                     frameon=False, fontsize=10.5, handlelength=1.0, handleheight=1.0, columnspacing=1.6)
    for txt in leg.get_texts():
        txt.set_color(c['ink'])
    out = f'{HERE}/shortcut_{lang}_{mode}.png'
    fig.savefig(out, dpi=DPI_SAVE, facecolor=c['surface'])
    plt.close(fig)
    return out


def frame(lang, mode):
    t, c = TEXT[lang], THEME[mode]
    facts = json.load(open(f'{ROOT}/prep/tables/facts.json', encoding='utf-8'))
    assert facts['cut_pct'] == 24.2 and facts['cut_kept_median'] == 79.8 and sum(facts['box_lost'].values()) == 45

    W, H = 880, 350
    fig = plt.figure(figsize=(W / DPI_VIEW, H / DPI_VIEW), dpi=DPI_VIEW, facecolor=c['surface'])
    ax = fig.add_axes([0, 0, 1, 1], facecolor=c['surface'])
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis('off')

    fw = 300
    fh, ch = fw * 1080 / 1920, fw * 972 / 1920       # true proportions: the band is 10% of the height
    bx, by = 60, 112
    ax.add_patch(Rectangle((bx, by), fw, fh, facecolor=c['frame_fill'], edgecolor=c['ink2'], linewidth=1.2))
    band = fh * 108 / 1080
    ax.add_patch(Rectangle((bx, by + fh - band), fw, band, facecolor=c['s2'], alpha=0.85, edgecolor='none'))
    ax.text(bx + fw / 2, by + fh + 10, t['band'], ha='center', va='bottom', fontsize=10.5, color=c['ink'])
    ax.text(bx + fw / 2, by + fh + 40, t['k4'], ha='center', va='bottom', fontsize=10, color=c['ink2'])
    ax.text(bx + fw / 2, by - 12, t['orig'], ha='center', va='top', fontsize=10.5, color=c['ink'])

    cx, cy = 520, by                                   # bottom-aligned: the cut removes only the top band
    ax.add_patch(Rectangle((cx, cy), fw, ch, facecolor=c['frame_fill'], edgecolor=c['ink2'], linewidth=1.2))
    ax.text(cx + fw / 2, cy - 12, t['out'], ha='center', va='top', fontsize=10.5, color=c['ink'])

    ay = by + ch / 2
    ax.add_patch(FancyArrowPatch((bx + fw + 22, ay), (cx - 22, ay), arrowstyle='-|>', mutation_scale=14,
                                 color=c['muted'], linewidth=1.5))
    ax.text((bx + fw + cx) / 2, ay + 8, t['crop'], ha='center', va='bottom', fontsize=10.5, color=c['ink2'])
    ax.text(W / 2, 34, t['note'], ha='center', va='center', fontsize=9.5, color=c['ink2'], linespacing=1.6)
    out = f'{HERE}/frame_{lang}_{mode}.png'
    fig.savefig(out, dpi=DPI_SAVE, facecolor=c['surface'])
    plt.close(fig)
    return out


# ---------------------------------------------------------------- overview figures (figures/data)

OV = {
    'en': dict(
        leak_t=['Official split · validation → same video in training', 'Place + date split · val/test → same place in training'],
        leak_x='difference to the nearest training frame (32×18 thumbnails; 40 or more shown at 40)',
        leak_y='% of compared frames', near='near-identical < 6',
        leak_n=['{near:,} of {n:,} frames ({pct:.1f}%) below 6',
                '{near} of {cmp:,} frames below 6\n{none:,} more have no training frame of their place'],
        fl_series={'model_input': 'model input (1920 wide)', 'short_224': 'shrunk to 224 px on the short side'},
        fl_x='short side of the largest flame box (px, log scale)', fl_y='% of fire frames',
        fl_note={'model_input': 'median {median:.0f} px', 'short_224': 'median {median:.0f} px · {u8:.1f}% under 8 px'},
        cls={'fire': 'fire', 'smoke': 'smoke only', 'negative': 'negative'},
        tm_t=['Shooting month: share of each class', 'Frames that are actually dark, by time-of-day label'],
        months={8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'},
        tz={'morning': 'morning', 'noon': 'noon', 'afternoon': 'afternoon', 'night': 'night'},
        tm_y=['% of the class', '% dark (mean brightness < 50)'],
        ds_t={'cloud': 'cloud', 'fog_haze': 'fog / haze', 'chimney_smoke': 'chimney smoke'},
        ds_n='{n:,} negatives · {pct:.1f}% at ≥ 20%', ds_x='share of the cropped frame (%)', ds_y='frames (thousands)'),
    'ko': dict(
        leak_t=['공식 분할 · 검증 → 같은 영상의 학습 프레임', '장소+촬영일 분할 · 검증·시험 → 같은 장소의 학습 사진'],
        leak_x='가장 비슷한 학습 사진과의 차이 (32×18 썸네일, 40 이상은 40)',
        leak_y='비교한 사진 중 %', near='거의 같음 < 6',
        leak_n=['{n:,}장 중 {near:,}장({pct:.1f}%)이 6 미만',
                '{cmp:,}장 중 {near}장이 6 미만\n나머지 {none:,}장은 같은 장소의 학습 사진이 없음'],
        fl_series={'model_input': '모델 입력 (가로 1920)', 'short_224': '짧은 변 224px로 줄였을 때'},
        fl_x='가장 큰 화염 박스의 짧은 변 (px, 로그 눈금)', fl_y='불꽃 사진 중 %',
        fl_note={'model_input': '중앙값 {median:.0f}px', 'short_224': '중앙값 {median:.0f}px · {u8:.1f}%가 8px 미만'},
        cls={'fire': '불꽃', 'smoke': '연기만', 'negative': '음성'},
        tm_t=['촬영 월: 클래스별 비율', '시간대 라벨별 실제로 어두운 사진'],
        months={8: '8월', 9: '9월', 10: '10월', 11: '11월', 12: '12월'},
        tz={'morning': '오전', 'noon': '정오', 'afternoon': '오후', 'night': '야간'},
        tm_y=['클래스 안의 %', '어두운 사진 % (평균 밝기 < 50)'],
        ds_t={'cloud': '구름', 'fog_haze': '안개/연무', 'chimney_smoke': '굴뚝연기'},
        ds_n='음성 {n:,}장 · {pct:.1f}%가 20% 이상', ds_x='자른 화면에서 차지하는 비율 (%)', ds_y='사진 수 (천 장)'),
}
DATA = os.path.join(HERE, 'data')
CLS_KEY = {'fire': 's2', 'smoke': 's1', 'negative': 's3'}   # fire orange, smoke blue, negative aqua (validated set)


def new_fig(c, w, h):
    return plt.figure(figsize=(w / DPI_VIEW, h / DPI_VIEW), dpi=DPI_VIEW, facecolor=c['surface'])


def style(ax, c, xgrid=False):
    ax.set_facecolor(c['surface'])
    for sp in ('top', 'right', 'left'):
        ax.spines[sp].set_visible(False)
    ax.spines['bottom'].set_color(c['axis'])
    ax.grid(axis='x' if xgrid else 'y', color=c['grid'], linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(colors=c['ink2'], labelsize=9.5, length=0)


def legend(fig, c, items, x, y):
    leg = fig.legend(handles=[Patch(facecolor=c[k], label=t) for t, k in items], loc='upper left',
                     bbox_to_anchor=(x, y), ncol=len(items), frameon=False, fontsize=10.5,
                     handlelength=1.0, handleheight=1.0, columnspacing=1.6)
    for t in leg.get_texts():
        t.set_color(c['ink'])


def save(fig, name, lang, mode, c):
    out = f'{HERE}/{name}_{lang}_{mode}.png'
    fig.savefig(out, dpi=DPI_SAVE, facecolor=c['surface'])
    plt.close(fig)
    return out


def leak(lang, mode):
    t, c = OV[lang], THEME[mode]
    d = pd.read_csv(f'{DATA}/leak.csv')
    W, H = 880, 340
    fig = new_fig(c, W, H)
    for i, name in enumerate(['official', 'place_date']):
        g = d[d.split == name]
        ax = fig.add_axes([(56 + i * 430) / W, 62 / H, 376 / W, 190 / H])
        style(ax, c)
        share = 100 * g['count'] / g.n_compared.iloc[0]
        ax.bar(g['bin'] + 0.5, share, width=0.8, color=c['s1'], zorder=3)
        ax.set_xlim(0, 41)
        ax.axvline(6, color=c['muted'], linewidth=1.5, zorder=4)
        ax.text(6.6, ax.get_ylim()[1] * 0.96, t['near'], color=c['ink2'], fontsize=9, va='top')
        first = g.iloc[0]
        note = t['leak_n'][i].format(near=int(first.n_near), n=int(first.n), cmp=int(first.n_compared),
                                     pct=100 * first.n_near / first.n, none=int(first.n - first.n_compared))
        ax.set_title(t['leak_t'][i], fontsize=10.5, color=c['ink'], loc='left', pad=8 + 16 * 2)  # both titles at the same height (the longer note has two lines)
        ax.text(0, 1.21, note, transform=ax.transAxes, ha='left', va='top', fontsize=9.5, color=c['ink2'],
                linespacing=1.4)
        if i == 0:
            ax.set_ylabel(t['leak_y'], fontsize=9.5, color=c['ink2'])
    fig.text(0.5, 14 / H, t['leak_x'], ha='center', fontsize=9.5, color=c['ink2'])
    return save(fig, 'leak', lang, mode, c)


def flame(lang, mode):
    t, c = OV[lang], THEME[mode]
    d = pd.read_csv(f'{DATA}/flame_size.csv')
    W, H = 880, 300
    fig = new_fig(c, W, H)
    ax = fig.add_axes([60 / W, 54 / H, 790 / W, 190 / H])
    style(ax, c)
    ax.set_xscale('log')
    for name, key in (('model_input', 's1'), ('short_224', 's2')):
        g = d[d['size'] == name]
        share = 100 * g['count'] / g.n.iloc[0]
        ax.stairs(share.to_numpy(), np.r_[g.lo.to_numpy(), g.hi.to_numpy()[-1]], color=c[key], linewidth=2, zorder=3)
        peak = g.iloc[int(np.argmax(share.to_numpy()))]
        note = t['fl_note'][name].format(median=g['median'].iloc[0], u8=g.pct_under_8.iloc[0])
        if name == 'short_224':   # left of the 8 px line, above the curve
            ax.text(7.2, share.max() * 1.08, note, ha='right', va='bottom', fontsize=9.5, color=c['ink'])
        else:
            ax.text(np.sqrt(peak.lo * peak.hi), share.max() * 1.08, note, ha='center', va='bottom', fontsize=9.5,
                    color=c['ink'])
    ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
    ax.axvline(8, color=c['muted'], linewidth=1.5, zorder=2)
    ax.text(8 * 1.07, ax.get_ylim()[1] * 0.97, '8 px', color=c['ink2'], fontsize=9, va='top')
    ticks = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    ax.set_xticks(ticks, [str(x) for x in ticks])
    ax.minorticks_off()
    ax.set_xlim(1.5, 700)
    ax.set_xlabel(t['fl_x'], fontsize=9.5, color=c['ink2'])
    ax.set_ylabel(t['fl_y'], fontsize=9.5, color=c['ink2'])
    legend(fig, c, [(t['fl_series']['model_input'], 's1'), (t['fl_series']['short_224'], 's2')], 60 / W - 0.005, 1 - 8 / H)
    return save(fig, 'flame', lang, mode, c)


def grouped(ax, c, groups, series, values, bar_px, ax_px, labels=None):
    """Grouped columns: rows = groups, one column per series, 2 px surface gap, value labels where asked."""
    n = len(series)
    px = len(groups) / ax_px   # data units per display pixel along x
    width, gap = bar_px * px, 2 * px
    for j, sname in enumerate(series):
        off = (j - (n - 1) / 2) * (width + gap)
        for i, v in enumerate(values[sname]):
            ax.bar(i + off, v, width=width, color=c[CLS_KEY[sname]], zorder=3)
            if labels and (groups[i], sname) in labels:
                ax.text(i + off, v, f'{v:.0f}' if v >= 1 else f'{v:.1f}', ha='center', va='bottom', fontsize=9,
                        color=c['ink'])
    ax.set_xticks(range(len(groups)))
    ax.set_xlim(-0.5, len(groups) - 0.5)


def timecues(lang, mode):
    t, c = OV[lang], THEME[mode]
    se = pd.read_csv(f'{DATA}/season.csv').set_index('month')
    dk = pd.read_csv(f'{DATA}/dark_by_time.csv').pivot(index='timezone', columns='sub', values='dark_pct')
    series = ['fire', 'smoke', 'negative']
    W, H = 880, 330
    fig = new_fig(c, W, H)
    months = [8, 9, 10, 11, 12]
    ax1 = fig.add_axes([58 / W, 46 / H, 380 / W, 210 / H])
    style(ax1, c)
    grouped(ax1, c, months, series, {k: se.loc[months, k].tolist() for k in series}, 18, 380,
            labels={(9, 'fire'), (11, 'negative'), (10, 'smoke')})
    ax1.set_xticklabels([t['months'][m] for m in months], fontsize=10, color=c['ink'])
    ax1.set_ylabel(t['tm_y'][0], fontsize=9.5, color=c['ink2'])
    ax1.set_title(t['tm_t'][0], fontsize=10.5, color=c['ink'], loc='left', pad=8)
    tzs = ['morning', 'noon', 'afternoon', 'night']
    ax2 = fig.add_axes([500 / W, 46 / H, 360 / W, 210 / H])
    style(ax2, c)
    grouped(ax2, c, tzs, series, {k: dk.loc[tzs, k].tolist() for k in series}, 20, 360,
            labels={('night', 'fire'), ('night', 'smoke'), ('night', 'negative')})
    ax2.set_xticklabels([t['tz'][z] for z in tzs], fontsize=10, color=c['ink'])
    ax2.set_ylabel(t['tm_y'][1], fontsize=9.5, color=c['ink2'])
    ax2.set_title(t['tm_t'][1], fontsize=10.5, color=c['ink'], loc='left', pad=8)
    for ax in (ax1, ax2):
        ax.set_ylim(0, ax.get_ylim()[1] * 1.1)
    legend(fig, c, [(t['cls'][k], CLS_KEY[k]) for k in series], 58 / W - 0.005, 1 - 6 / H)
    return save(fig, 'time', lang, mode, c)


def distractors(lang, mode):
    t, c = OV[lang], THEME[mode]
    d = pd.read_csv(f'{DATA}/distractor_area.csv')
    W, H = 880, 270
    fig = new_fig(c, W, H)
    for i, name in enumerate(['cloud', 'fog_haze', 'chimney_smoke']):
        g = d[d.distractor == name]
        ax = fig.add_axes([(58 + i * 280) / W, 58 / H, 240 / W, 160 / H])
        style(ax, c)
        ax.bar(g.lo + 1.25, g['count'] / 1000, width=2.5 * 0.72, color=c['s1'], zorder=3)
        for x in (5, 20):
            ax.axvline(x, color=c['muted'], linewidth=1.2, zorder=2)
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 20, 50, 100], ['0', '20', '50', '100'])
        first = g.iloc[0]
        ax.set_title(f"{t['ds_t'][name]}\n{t['ds_n'].format(n=int(first.n), pct=first.pct_ge_20)}", fontsize=10,
                     color=c['ink'], loc='left', pad=6, linespacing=1.5)
        if i == 0:
            ax.set_ylabel(t['ds_y'], fontsize=9.5, color=c['ink2'])
    fig.text(0.5, 12 / H, t['ds_x'], ha='center', fontsize=9.5, color=c['ink2'])
    return save(fig, 'distractors', lang, mode, c)


if __name__ == '__main__':
    for lang in TEXT:
        for mode in THEME:
            print(leak(lang, mode), flame(lang, mode), timecues(lang, mode), distractors(lang, mode),
                  shortcut(lang, mode), frame(lang, mode))
