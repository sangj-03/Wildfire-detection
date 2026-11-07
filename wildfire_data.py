"""산불 탐지 프로젝트: AI Hub 71330 사진의 전처리 공용 코드 (전처리 노트북과 이후 학습·평가 코드가 같이 쓴다).

이진 분류: 양성(1) = 불꽃·연기 박스가 있는 실제 양성 사진, 음성(0) = 연기·화염 박스가 없는 음성 DB 사진(정상).
목록(manifest)은 notebooks/preprocess.ipynb가 만든다 → prep/manifest.csv.gz (split, y, src, member, vid ...). 분할은 split_v2.

사진 처리 (학습·검증·시험 모두 같음, 원본 JPEG를 다시 저장하지 않고 읽을 때마다 처리):
  1) JPEG → RGB   2) 3840×2160이면 2×2 화소 평균으로 1920×1080   3) 위 108px(10%) 잘라 1920×972
사진은 AI Hub 원천 zip(TS 분할 zip, VS.zip)에서 바로 읽는다(aihub_split_zip.SplitZip).
학습 때의 처리(영상당 N장 추출, 클래스 가중치, 증강, torch Dataset)는 모델링 단계 코드 modeling/wildfire_train.py에 있다.

    import wildfire_data as W
    m = W.load_manifest()
    im = W.load_image(m.iloc[0])          # 1920×972 PIL 사진
"""

import io, os, sys

import numpy as np
import pandas as pd
from PIL import Image

ROOT = '/data/home/seame/dl-team-project/team6'
sys.path.insert(0, ROOT)
from aihub_split_zip import SplitZip  # noqa: E402

AIHUB = ('/data/aihub/71330/265.지역안전재난(산불)_방재의_고도화를_위한_대규모_인공지능_데이터베이스_구축/'
         '01-1.정식개방데이터')
IMAGE_ZIP = {'train': f'{AIHUB}/Training/01.원천데이터/TS.zip', 'val': f'{AIHUB}/Validation/01.원천데이터/VS.zip'}
PREP = f'{ROOT}/prep'
MANIFEST = f'{PREP}/manifest.csv.gz'

BASE_W, BASE_H = 1920, 1080     # 해상도 통일 기준
CROP_TOP = 108                  # 위 10%: 날짜·시각·GPS 글자 띠 (평창 CCTV 1대를 빼면 모든 카메라의 글자가 이 안에 있음)
OUT_W, OUT_H = BASE_W, BASE_H - CROP_TOP   # 1920×972
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------- 사진 한 장 처리

def preprocess(jpeg_bytes):
    """원천 JPEG 바이트 → 1920×972 RGB PIL 사진. 1920×1080·3840×2160만 받는다(나머지 해상도는 목록에서 뺐다)."""
    im = Image.open(io.BytesIO(jpeg_bytes)).convert('RGB')
    if im.size == (2 * BASE_W, 2 * BASE_H):
        im = im.reduce(2)                     # 2×2 화소 평균 (정확히 절반)
    elif im.size != (BASE_W, BASE_H):
        raise ValueError(f'지원하지 않는 해상도 {im.size}')
    return im.crop((0, CROP_TOP, OUT_W, BASE_H))


def to_processed_xy(x, y, w, h, orig_w):
    """원본 픽셀 좌표 박스 [x, y, w, h] → 처리한 사진(1920×972) 좌표 (x0, y0, x1, y1), 화면 밖은 자른다."""
    s = BASE_W / orig_w
    x0, x1 = np.clip(x * s, 0, OUT_W), np.clip((x + w) * s, 0, OUT_W)
    y0, y1 = np.clip(y * s - CROP_TOP, 0, OUT_H), np.clip((y + h) * s - CROP_TOP, 0, OUT_H)
    return x0, y0, x1, y1


class ZipReader:
    """원천 zip 읽기. 파일 목록(중앙 디렉터리)은 한 번만 읽어 fork한 작업 프로세스와 같이 쓰고,
    파일 핸들은 프로세스마다 새로 연다(DataLoader 작업 프로세스끼리 파일 위치를 같이 쓰지 않게)."""
    _zips, _pid = {}, {}

    @classmethod
    def open(cls, src):
        if src not in cls._zips:
            cls._zips[src] = SplitZip(IMAGE_ZIP[src])
            cls._pid[src] = os.getpid()
        elif cls._pid[src] != os.getpid():
            cls._zips[src].close()            # 부모에게서 물려받은 핸들은 닫고 이 프로세스에서 다시 연다
            cls._pid[src] = os.getpid()
        return cls._zips[src]

    @classmethod
    def read(cls, src, member):
        return cls.open(src).read(member)


def load_image(row):
    """목록의 한 줄(src, member) → 처리한 사진."""
    return preprocess(ZipReader.read(row.src, row.member))


def load_manifest(path=MANIFEST):
    return pd.read_csv(path, dtype={'date': str, 'area_bin': str, 'combo': str})
