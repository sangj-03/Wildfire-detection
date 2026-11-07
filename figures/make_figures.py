"""README figures: the AI Hub frame preprocessing and the shortcut re-check, in English and Korean, for light and dark pages.

Numbers come from prep/tables (written by notebooks/preprocess.ipynb). No dataset photo is used.

    python figures/make_figures.py   →  figures/{frame,shortcut}_{en,ko}_{light,dark}.png
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, PathPatch, Patch, Rectangle  # noqa: E402
from matplotlib.path import Path  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
plt.rcParams['font.family'] = 'Noto Sans CJK KR'

THEME = {
    'light': dict(surface='#fcfcfb', ink='#0b0b0b', ink2='#52514e', muted='#898781', grid='#e1e0d9', axis='#c3c2b7',
                  s1='#2a78d6', s2='#eb6834', frame_fill='#f0efec'),
    'dark': dict(surface='#1a1a19', ink='#f0efec', ink2='#c3c2b7', muted='#898781', grid='#2c2c2a', axis='#383835',
                 s1='#3987e5', s2='#d95926', frame_fill='#262624'),
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


if __name__ == '__main__':
    for lang in TEXT:
        for mode in THEME:
            print(shortcut(lang, mode), frame(lang, mode))
