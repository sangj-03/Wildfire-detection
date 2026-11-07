"""산불 탐지 프로젝트: 다른 출처 시험 데이터(D-Fire, DeepFire) 가공·전처리 공용 코드.

두 데이터는 학습에 쓰지 않고 시험에만 쓴다(출처가 다른 사진에서도 통하는지). 이진 라벨은 AI Hub와 같다:
양성(1) = 불꽃 또는 연기가 있는 사진, 음성(0) = 없는 사진.

사진 처리 (wildfire_data.preprocess와 짝):
  1) RGB로 읽기
  2) D-Fire 감시 카메라(AoF) 사진만: 화면에 찍힌 날짜·시각·카메라 이름 글자 상자를 찾아 주변 화소로 메운다(inpainting).
     회색 상자로 덮지 않는 이유: 평평한 회색 덩어리는 연기·안개처럼 보여 오경보를 만들 수 있다.
  3) 'fit': 비율을 지켜 1920×972 안에 맞추고 남는 곳은 ImageNet 평균색으로 채운다(정규화하면 0).
     'native': 원래 크기 그대로(DeepFire 250×250 같은 작은 사진을 키우지 않고 볼 때).
AI Hub처럼 위 10%를 자르지 않는다 — 그 처리는 AI Hub 글자 띠 때문이었고, 다른 사진에서는 위로 올라가는 연기만 잘린다.
"""
import io

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

OUT_W, OUT_H = 1920, 972                      # wildfire_data.OUT_W, OUT_H와 같다
FILL_RGB = (124, 116, 104)                    # ImageNet 평균 (0.485, 0.456, 0.406) × 255


# ---------------------------------------------------------------- 화면 글자 상자 찾기 (AoF 감시 카메라)
# D-Fire 감시 카메라(AoF) 사진에는 카메라 시스템마다 정해진 글자가 찍혀 있다(가로 1280 화면 기준):
#   top_strip   맨 위 작은 글자 줄 'D619-C0002-CA - 23/06/20 07:46:40' (자리 고정)
#   cam_label   왼쪽 위 'Camera0001' 검은 상자
#   date_br     오른쪽 아래 날짜 '23.Jun 2020  07:46:39' (검은 상자, 또는 흰 글자만). 화면 일부를 확대해 캡처한 사진은 글자가 크고
#               아래·오른쪽 끝에서 잘려 있다
#   band_bottom 아래 검은 띠 'Camera0005 - 2020-11-04 12:12:02' (CEMIG 카메라, 자리 고정)
#   date_big    왼쪽 위 큰 날짜 '12-15-2020 Tue 16:06:47' (CEMIG 정자·적외선 카메라)
#   label_br    오른쪽 아래 'Camera 01' 또는 'Preset 7' (같은 카메라, 테두리 있는 흰 글자)
# 화면 캡처마다 잘린 정도가 달라 글자 크기·자리가 조금씩 다르므로, 정해진 넓은 영역 안에서 '글자 줄'(기준선이 같은 흰 조각이
# 가로로 한 줄)을 찾아 그 줄만 가린다(검은 상자가 있으면 상자까지 넓힌다).
REF_W = 1280
SEARCH = {   # 이름: (기준, 영역 (x0, y0, x1, y1) — 'br'이면 오른쪽·아래 끝에서 잰 거리, 글자 높이 범위, 최소 글자 수, 가로 길이 범위(글자 높이의 배수))
    'cam_label':   ('tl', (10, 10, 300, 140), (8, 30), 6, (4, 14)),
    'date_br':     ('br', (560, 230, 0, 0), (8, 90), 10, (5, 24)),
    'date_big':    ('tl', (0, 15, 680, 125), (14, 44), 8, (8, 32)),
    'label_br':    ('br', (600, 150, 90, 0), (14, 40), 6, (4, 14)),
}


def _area(name, W, H):
    anchor, (a0, b0, a1, b1), *_ = SEARCH[name]
    if anchor == 'br':
        return max(0, W - a0), max(0, H - b0), W - a1, H - b1
    return a0, b0, min(W, a1), min(H, b1)


def _text_line(white, hrange, kmin, span, gray=None):
    """흰 글자 줄 찾기(기준선 맞춤). 높이가 hrange 안이고 너비가 높이의 3배 이하(사진 아래 끝에 닿은 조각은 4배)인 흰 조각 중에서,
    한 조각(기준, 높이 h)을 잡아 아래 끝이 max(3, 0.15h) px 안이고 높이가 0.5h~1.1h인 조각을 모은다 — 대문자·숫자와 소문자가 섞인
    'Camera 01', 'Preset 7'도 한 줄로 잡힌다. 가로로 붙은 무리(조각 사이 간격 ≤ 3h)가 kmin개 이상이고, 가로 길이가 span[0]h ~ span[1]h
    이며, 글자 모양(흰 화소 비율 중앙값 ≤ 0.7, 너비/높이 중앙값 ≤ 1.1 — 흰 자동차처럼 꽉 찬 덩어리는 걸러진다)인 무리 중 가장 큰 것.
    줄이 영역 아래 끝에 닿으면(사진 아래에서 잘린 글자): 최소 글자 수 kmin-4(≥5), 너비/높이 ≤ 1.6, 가로 길이 상한 없음.
    gray를 주면 조각마다 둘레 2px 중 어두운 화소(밝기 ≤ 90) 비율도 잰다(테두리 있는 글자는 높다).
    반환: dict(box=(x0, y0, x1, y1), cut=아래가 잘림, n=글자 조각 수, h=기준 높이, ring=둘레 어두움 중앙값) 또는 None."""
    lbl, _ = ndimage.label(white)
    Hm = white.shape[0]
    comps = []
    for k, sl in enumerate(ndimage.find_objects(lbl), 1):
        if sl is None:
            continue
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        edge = sl[0].stop >= Hm - 1
        if hrange[0] <= h <= hrange[1] and w <= (4 if edge else 3) * h:
            ring = np.nan
            if gray is not None:
                y0, y1 = max(0, sl[0].start - 2), sl[0].stop + 2
                x0, x1 = max(0, sl[1].start - 2), sl[1].stop + 2
                sub = lbl[y0:y1, x0:x1] == k
                rg = ndimage.binary_dilation(sub, iterations=2) & ~sub
                ring = float((gray[y0:y1, x0:x1][rg] <= 90).mean()) if rg.any() else 0.0
            comps.append((sl, h, w, sl[0].stop, float((lbl[sl] == k).mean()), ring))
    kmin_cut = max(5, kmin - 4)
    if len(comps) < min(kmin, kmin_cut):
        return None
    best = None
    for _, ha, _, ba, _, _ in comps:
        t = max(3, 0.15 * ha)
        grp = sorted((c for c in comps if abs(c[3] - ba) <= t and 0.5 * ha <= c[1] <= 1.1 * ha), key=lambda c: c[0][1].start)
        runs, cur = [], [grp[0]]
        for c in grp[1:]:
            if c[0][1].start - max(x[0][1].stop for x in cur) <= 3 * ha:
                cur.append(c)
            else:
                runs.append(cur); cur = [c]
        runs.append(cur)
        for run in runs:
            cut = max(c[3] for c in run) >= Hm - 1
            if len(run) < (kmin_cut if cut else kmin) or (best is not None and len(run) <= best['n']):
                continue
            x0, x1 = min(c[0][1].start for c in run), max(c[0][1].stop for c in run)
            if x1 - x0 < span[0] * ha or (not cut and x1 - x0 > span[1] * ha):
                continue
            if np.median([c[4] for c in run]) > 0.7 or np.median([c[2] / c[1] for c in run]) > (1.6 if cut else 1.1):
                continue
            best = dict(box=(x0, min(c[0][0].start for c in run), x1, max(c[0][0].stop for c in run)), cut=cut, n=len(run), h=ha,
                        ring=float(np.nanmedian([c[5] for c in run])) if gray is not None else np.nan)
    return best


def _grow(mask, box, limit):
    """상자 바깥 줄이 mask(검정)로 60% 넘게 차 있으면 한 줄씩 넓힌다(네 방향, 각 limit px까지)."""
    H, W = mask.shape
    x0, y0, x1, y1 = box
    for _ in range(limit):
        grown = False
        if y0 > 0 and mask[y0 - 1, x0:x1].mean() >= 0.6:
            y0 -= 1; grown = True
        if y1 < H and mask[y1, x0:x1].mean() >= 0.6:
            y1 += 1; grown = True
        if x0 > 0 and mask[y0:y1, x0 - 1].mean() >= 0.6:
            x0 -= 1; grown = True
        if x1 < W and mask[y0:y1, x1].mean() >= 0.6:
            x1 += 1; grown = True
        if not grown:
            break
    return x0, y0, x1, y1


def find_overlays(rgb, mirror=True):
    """AoF 감시 카메라 사진에서 글자가 있는 자리 [(이름, (x0, y0, x1, y1))] (원본 좌표). 가로 1280으로 맞춰 검사한다.
    흰 글자 = 밝기 ≥150·무채색(채도 차 ≤50), 검정 = 밝기 ≤40·무채색(≤30). 밝은 적외선 화면처럼 배경이 밝아 글자가 배경과 붙으면
    더 엄격한 흰색(밝기 ≥215·채도 차 ≤40)으로 한 번 더 찾는다.
      top_strip   맨 위 1~12줄의 40% 이상이 검정이고 작은 흰 조각(높이 2~10)이 20개 이상 → (0, 0, 228, 16)
      band_bottom 맨 아래 띠의 80% 이상이 검정이고 왼쪽 450px에 흰 조각이 15개 이상 → 아래 32px 전체
      그 밖       SEARCH 영역에서 글자 줄을 찾으면 줄 범위 + 여백 6px, 글자 둘레가 검은 바탕이면 검은 상자 끝까지(최대 14px),
                  아래가 잘린 줄은 사진 아래 끝까지, 오른쪽 끝에 닿은 줄은 오른쪽 끝까지 덮는다.
                  cam_label은 검은 바탕(글자 둘레 3px의 글자 아닌 화소 중 50% 이상 검정)이어야 한다.
                  date_big은 'MM-DD-YYYY Ddd HH:MM:SS' 길이가 정해져 있어 x 8~630을 덮는다.
                  date_big·label_br은 밝은 적외선 화면에서 글자가 조각나므로, 줄이 안 잡히면 글자 4개 이상 + 검은 테두리로 한 번 더
                  찾는다(cam_label이 있는 카메라는 이 형식이 아니라 건너뛴다). label_br은 date_br이 있으면 찾지 않는다.
                  date_big은 화면 왼쪽 끝(x ≤ 60)에서 시작해야 하고, label_br은 글자 둘레 2px 중 어두운 화소가 25% 이상이어야 한다
                  (테두리 있는 글자 — 줄무늬 차단봉 제외). cam_label이 잡힌 사진에서는 x 340 넘게 이어지고 글자 조각이 14개 이상인
                  줄만 date_big으로 받고, 그 줄과 높이가 겹치는 cam_label(큰 날짜 앞부분)은 뺀다.
                  label_br이 있는 사진에서 'Camera0001' 상자로 잡힌 줄은 큰 날짜로 바꾼다.
                  label_br이 있는데 큰 날짜가 안 잡히면(같은 카메라라 늘 있다) 큰 날짜 자리(x 8~630, y 15~88)를 덮는다('date_big+').
                  이름 끝 '*' = 테두리 조건으로 짧은 줄까지 받은 것, '+' = 같은 카메라 형식으로 미루어 덮은 것.
    아무것도 못 찾으면 좌우를 뒤집어 한 번 더 찾는다(D-Fire에는 좌우를 뒤집어 늘린 사진이 있어 글자가 거울상으로 오른쪽 위·왼쪽
    아래에 있다). 우연히 잡힌 줄을 거르려고 cam_label과 date_br이 둘 다 잡힐 때만 받는다(이름 끝 '~')."""
    h0, w0 = rgb.shape[:2]
    s = REF_W / w0
    interp = cv2.INTER_NEAREST if s > 1 else cv2.INTER_AREA
    img = cv2.resize(rgb, (REF_W, max(1, round(h0 * s))), interpolation=interp) if s != 1 else rgb
    H, W = img.shape[:2]
    a = img.astype(np.int16)
    g = a.mean(2)
    chroma = a.max(2) - a.min(2)
    black = (g <= 40) & (chroma <= 30)
    white = (g >= 150) & (chroma <= 50)
    white_hi = (g >= 215) & (chroma <= 40)
    found = []
    lbl, _ = ndimage.label(white[0:16, 0:240])
    small = [sl for sl in ndimage.find_objects(lbl) if sl is not None and 2 <= sl[0].stop - sl[0].start <= 10]
    if black[1:12, 2:215].mean() >= 0.4 and len(small) >= 20:
        found.append(('top_strip', (0, 0, 228, 16)))
    lbl, _ = ndimage.label(white[H - 26:H, :450])
    small = [sl for sl in ndimage.find_objects(lbl) if sl is not None and 4 <= sl[0].stop - sl[0].start <= 20]
    if black[H - 24:H - 2, :].mean() >= 0.8 and len(small) >= 15:
        found.append(('band_bottom', (0, H - 32, W, H)))
    names = []
    for name in ('cam_label', 'date_br', 'date_big', 'label_br'):
        if name == 'label_br' and 'date_br' in names:
            continue                               # 'Camera 01'·'Preset' 카메라에는 오른쪽 아래 날짜가 없다(같은 자리를 두 번 세지 않게)
        if name == 'date_big' and any(n == 'band_bottom' for n, _ in found):
            continue                               # 큰 날짜는 정자·적외선 카메라 형식이다(아래 띠 카메라에는 없다)
        _, _, hr, kmin, span = SEARCH[name]
        ax0, ay0, ax1, ay1 = _area(name, W, H)
        outlined = name == 'label_br'              # 테두리(둘레가 어두움)가 있는 글자 — 줄무늬 차단봉 같은 흰 물체를 거른다
        line = None
        for wm in (white, white_hi):
            line = _text_line(wm[ay0:ay1, ax0:ax1], hr, kmin, span, gray=g[ay0:ay1, ax0:ax1] if outlined else None)
            if line is not None and outlined and line['ring'] < 0.25:
                line = None
            if line is not None and name == 'date_big' and line['box'][0] + ax0 > 60:
                line = None                        # 큰 날짜는 화면 왼쪽 끝에서 시작한다
            if line is not None and name == 'date_big' and 'cam_label' in names and (line['box'][2] + ax0 < 340 or line['n'] < 14):
                line = None                        # 'Camera0001'(10글자, x ≤ 300) 상자를 큰 날짜(약 19조각, x 340 넘게)로 다시 읽지 않게
            if line is not None:
                break
        if line is None and name in ('date_big', 'label_br') and not {'cam_label', 'band_bottom'} & set(names + [n for n, _ in found]):
            # 밝은 적외선 화면에서 글자가 조각나 줄이 짧게 잡히는 경우: 글자 4개 이상 + 검은 테두리(둘레 2px의 30% 이상이 어두움)
            for wm in (white_hi, white):
                cand = _text_line(wm[ay0:ay1, ax0:ax1], hr, 4, (3.5, span[1]), gray=g[ay0:ay1, ax0:ax1])
                if cand is not None and cand['ring'] >= 0.3 and (name != 'date_big' or cand['box'][0] + ax0 <= 60):
                    line = dict(cand, relaxed=True)
                    break
        if line is None:
            continue
        lx0, ly0, lx1, ly1 = line['box']
        # 글자 바로 둘레(3px)에서 흰 글자가 아닌 화소 중 검정 비율 — 검은 상자 위 글자면 높다
        bx0, by0 = max(0, lx0 + ax0 - 3), max(0, ly0 + ay0 - 3)
        bx1, by1 = min(W, lx1 + ax0 + 3), min(H, ly1 + ay0 + 3)
        nonwhite = ~wm[by0:by1, bx0:bx1]
        dark_bg = black[by0:by1, bx0:bx1][nonwhite].mean() if nonwhite.any() else 0.0
        if name == 'cam_label' and dark_bg < 0.5:
            continue
        x0, y0, x1, y1 = max(0, lx0 + ax0 - 6), max(0, ly0 + ay0 - 6), min(W, lx1 + ax0 + 6), min(H, ly1 + ay0 + 6)
        if dark_bg >= 0.5:
            x0, y0, x1, y1 = _grow(black, (x0, y0, x1, y1), 14)
        if line['cut']:
            y1 = H                                 # 사진 아래에서 잘린 글자
            if name == 'date_br':
                x0 = max(0, min(x0, x1 - 270))     # 잘린 날짜는 앞 글자가 줄에서 빠지기 쉽다(날짜 줄 너비 약 260px)
        if lx1 + ax0 >= W - 4:
            x1 = W                                 # 오른쪽 끝에서 잘린 글자
        if name == 'date_big':
            x0, x1 = min(x0, 8), min(W, max(x1, 630))
            if 'cam_label' in names:
                # 큰 날짜 줄의 앞부분이 'Camera0001' 상자로 잡혔던 것(x ≤ 300만 덮였다) → 그 상자를 뺀다
                found = [(n, b) for n, b in found if not (n == 'cam_label' and b[1] < y1 and b[3] > y0)]
                names = [n for n in names if n != 'cam_label' or any(k == 'cam_label' for k, _ in found)]
        names.append(name)
        found.append((name + ('*' if line.get('relaxed') else ''), (int(x0), int(y0), int(x1), int(y1))))
    if 'label_br' in names and 'cam_label' in names and 'date_br' not in names:
        # 'Camera 01'·'Preset' 카메라에는 'Camera0001' 상자가 없다 → 왼쪽 위에서 잡힌 줄은 큰 날짜(어두운 지붕 위)다
        found = [('date_big', (min(b[0], 8), b[1], max(b[2], 630), b[3])) if n == 'cam_label' else (n, b) for n, b in found]
        names = ['date_big' if n == 'cam_label' else n for n in names]
    if 'label_br' in names and 'date_big' not in names and any(n == 'top_strip' for n, _ in found):
        # 'Camera 01'·'Preset' 카메라는 늘 왼쪽 위에 큰 날짜가 있다(정자 카메라 y 46~76, 적외선 카메라 y 24~44). 날짜 줄이 배경과 붙어
        # 안 잡혀도 두 자리를 모두 덮는다
        found.append(('date_big+', (8, 15, 630, 88)))
    out = [(n, (int(x0 / s), int(y0 / s), min(w0, int(np.ceil(x1 / s))), min(h0, int(np.ceil(y1 / s)))))
           for n, (x0, y0, x1, y1) in found]
    if not out and mirror:
        fl = find_overlays(np.ascontiguousarray(rgb[:, ::-1]), mirror=False)
        if {'cam_label', 'date_br'} <= {n for n, _ in fl}:
            out = [(n + '~', (w0 - x1, y0, w0 - x0, y1)) for n, (x0, y0, x1, y1) in fl]
    return out


# 자동 검출이 놓친 글자 자리 (원본 좌표). AoF 사진 8,384장의 오른쪽 아래(아무것도 못 찾은 1,712장)와 왼쪽 위(상자·큰 날짜를 못 찾은
# 1,676장 + 아래 띠 카메라 60장)를 눈으로 전부 점검하고, 감시 카메라 글자가 하나도 안 잡힌 221장은 화면 전체를 보고, 오른쪽 위에서
# 글자 줄이 잡히는 사진을 기계로 골라 확인해 찾았다. overlay_boxes가 이름 끝에 '#'을 붙여 자동 검출 결과 뒤에 더한다.
_IR_DATE = ('date_big', (8, 14, 630, 56))            # 적외선·정원 카메라 큰 날짜(y 24~45) — 밝은 잎·바닥과 글자가 붙어 줄이 안 잡힘
_PAVILION_LABEL = ('label_br', (918, 614, 1155, 668))  # 정자 카메라 'Camera 01' — 같은 카메라 AoF03205에서 잡힌 자리
_PRESET = ('label_br', (775, 640, 979, 680))         # 적외선 카메라 'Preset 5'·'Preset 7'
_DATE_TR = ('date_tr', (1017, 55, 1280, 89))          # 난간 카메라 오른쪽 위 '31.May 2017 15:32:51' (1280×720)
MANUAL_OVERLAYS = {
    **{f'AoF{i:05d}': [_DATE_TR] for i in (357, 367, 378, 5969, 5977, 5994, 6002, 6012, 6022, 6028, 6803, 8191)},
    'AoF00229': [('date_tr', (639, 34, 800, 56))],       # 같은 날짜, 크기가 다른 캡처
    'AoF06110': [('date_tr', (838, 39, 1055, 66))],
    'AoF08231': [('date_tr', (835, 36, 1050, 63))],
    'AoF05085': [('cam_label', (32, 42, 182, 78))],      # 작은 사진(500×354)이라 'Camera0001'이 찾는 범위보다 크고 아래에 있다
    **{f'AoF{i:05d}': [_IR_DATE] for i in (3222, 3224, 3225, 3226, 3228, 3230, 3239, 3240, 3242, 3245, 3246, 3248, 3251, 3252,
                                          3254, 3592, 3600, 3604, 3606, 7547, 7549, 7625, 7626)},
    'AoF03241': [_IR_DATE, _PRESET],
    'AoF07548': [_PRESET],
    **{f'AoF{i:05d}': [_PAVILION_LABEL] for i in (3209, 3210, 3211, 3213, 3214, 3215)},
    'AoF05094': [('cam_label', (62, 0, 312, 27))],       # 확대 캡처라 'Camera0001' 글자 높이가 약 40px(찾는 범위 8~30px 밖)
    'AoF05095': [('cam_label', (73, 6, 402, 80))],
    'AoF04563': [('date_br', (1005, 783, 1129, 799))],   # 화면 일부를 확대해 캡처해 아래 끝에 날짜 윗부분만 남은 사진
    'AoF04568': [('date_br', (929, 684, 1292, 698))],
    'AoF04575': [('date_br', (476, 819, 838, 833))],
    'AoF04578': [('date_br', (273, 874, 662, 890))],
    'AoF04581': [('date_br', (557, 734, 921, 746))],
    'AoF07871': [('date_br', (376, 741, 751, 752))],
    'AoF04560': [('date_br', (123, 693, 474, 712))],     # 확대 캡처라 날짜가 커서 자동 상자가 앞 글자('14 Sep')를 덜 덮음
    'AoF04573': [('date_br', (549, 656, 889, 676))],
    'AoF04587': [('date_br', (1128, 806, 1325, 843))],
    'AoF03635': [('watermark', (535, 450, 640, 478))],   # 감시 카메라 글자는 아니지만 사진에 찍힌 출처 표시·방송사 로고
    'AoF03636': [('watermark', (535, 450, 640, 478))],
    'AoF06106': [('watermark', (1092, 568, 1214, 695))],
    'AoF01226': [('watermark', (600, 192, 620, 306))],   # 세로 글자 'FOTO: WESLEY ORTIZ' (아래 로고는 자동으로 잡힘)
    'AoF07295': [('watermark', (912, 8, 1196, 44)), ('watermark', (995, 1138, 1200, 1198))],  # '#FOTOGRAFEOTEMPO', 'CLIMATEMPO' 전체
}


def overlay_boxes(name, rgb):
    """사진 한 장에서 가릴 자리 = find_overlays(자동) + MANUAL_OVERLAYS(눈으로 점검해 적은 자리, 이름 끝 '#')."""
    return find_overlays(rgb) + [(k + '#', b) for k, b in MANUAL_OVERLAYS.get(name, [])]


def mask_overlays(rgb, boxes):
    """글자 상자 자리를 주변 화소로 메운다 (OpenCV Telea inpainting, 반경 5). boxes: [(x0, y0, x1, y1)] 또는 find_overlays 결과.
    상자 둘레에 JPEG 압축 번짐이 남지 않도록 자리를 3px 넓혀서 메운다."""
    if not boxes:
        return rgb
    m = np.zeros(rgb.shape[:2], np.uint8)
    for b in boxes:
        x0, y0, x1, y1 = b[1] if isinstance(b[0], str) else b
        m[y0:y1, x0:x1] = 255
    m = cv2.dilate(m, np.ones((7, 7), np.uint8))
    return cv2.inpaint(rgb, m, 5, cv2.INPAINT_TELEA)


# ---------------------------------------------------------------- 크기 맞추기

def fit_geometry(w, h):
    """'fit': 비율을 지켜 OUT_W×OUT_H 안에 맞출 때의 배율과 왼쪽 위 위치."""
    s = min(OUT_W / w, OUT_H / h)
    nw, nh = round(w * s), round(h * s)
    return s, (OUT_W - nw) // 2, (OUT_H - nh) // 2, nw, nh


def preprocess_external(jpeg_bytes, overlay_boxes=(), mode='fit'):
    """다른 출처 사진 → 모델 입력용 PIL 사진. overlay_boxes가 있으면 먼저 메운다.
    mode='fit': 1920×972 캔버스(ImageNet 평균색) 가운데에 비율을 지켜 넣는다. mode='native': 크기를 바꾸지 않는다."""
    rgb = np.asarray(Image.open(io.BytesIO(jpeg_bytes)).convert('RGB'))
    rgb = mask_overlays(rgb, list(overlay_boxes))
    im = Image.fromarray(rgb)
    if mode == 'native':
        return im
    s, ox, oy, nw, nh = fit_geometry(im.width, im.height)
    canvas = Image.new('RGB', (OUT_W, OUT_H), FILL_RGB)
    canvas.paste(im.resize((nw, nh), Image.BICUBIC if s > 1 else Image.LANCZOS), (ox, oy))
    return canvas


def to_fit_xy(x0, y0, x1, y1, w, h):
    """원본 좌표 박스 → 'fit' 사진 좌표."""
    s, ox, oy, _, _ = fit_geometry(w, h)
    return x0 * s + ox, y0 * s + oy, x1 * s + ox, y1 * s + oy
