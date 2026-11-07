"""check_fire_lookalike.py가 고른 사진(음성 전부 + 비교용 불꽃 사진)을 SigLIP 제로샷으로 '무엇처럼 보이는지' 분류한다.

종류마다 문장 여러 개를 두고, 사진과 가장 잘 맞는 문장의 종류를 그 사진의 종류로 삼는다(문장 점수 최댓값).
라벨이 아니라 모델 추정이므로, 종류별 상위·무작위 격자를 사람이 보고 맞는지 확인해야 한다.
사용(GPU 환경):
  PYTHONPATH=team6/pylib_hf HF_HOME=/data/huggingface /data/seotturi/.venv/bin/python tag_lookalike_clip.py
  → report/lookalike/clip_tags.csv.gz,  ... tag_lookalike_clip.py sheets → report/lookalike/clip_<종류>_<데이터>.jpg
"""
import os, sys

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, '/data/home/seame/dl-team-project/team6')
import check_fire_lookalike as L

OUT = L.OUT
MODEL = 'google/siglip-so400m-patch14-384'
CATS = {
    '불꽃놀이': ['fireworks exploding in the night sky', 'colorful fireworks display'],
    '노을·해': ['a sunset sky', 'a sunrise over the horizon', 'the bright sun shining in the sky', 'sun glare and lens flare in a photo'],
    '밤 불빛': ['street lights at night', 'city lights at night', 'car headlights on a road at night', 'a dark road at night with lamps'],
    '실내 조명': ['a room with warm indoor lighting', 'a glowing lamp in a room'],
    '통제된 불': ['a campfire', 'a candle flame', 'a fire in a fireplace', 'a barbecue grill with fire', 'a bonfire'],
    '단풍·마른 풀': ['red and orange autumn leaves on trees', 'a field of dry yellow grass'],
    '번개': ['lightning in the sky'],
    '구름': ['clouds in the sky'],
    '안개': ['fog and mist over mountains', 'a foggy hazy landscape'],
    '굴뚝·공장 연기': ['smoke coming out of a factory chimney', 'white steam rising from a building'],
    '산불': ['a forest fire with flames', 'smoke rising from a wildfire on a mountain', 'a wildfire burning at night'],
    '일반 풍경': ['a mountain landscape', 'a green forest', 'a village with houses', 'a city street in the daytime',
               'a farm field', 'a lake or river', 'a road', 'a parking lot', 'an empty room', 'a building'],
}


class DS(torch.utils.data.Dataset):
    def __init__(self, rows, proc):
        self.rows, self.proc = rows, proc

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        try:
            im = L.load((r['src'], r['key'])).convert('RGB'); ok = True
        except Exception:
            im = Image.new('RGB', (384, 384)); ok = False
        return self.proc(images=im, return_tensors='pt')['pixel_values'][0], i, ok


def run():
    from transformers import AutoModel, AutoProcessor
    df = pd.read_csv(f'{OUT}/measure.csv.gz')
    # AI Hub 음성은 영상 프레임이라 비슷한 사진이 많다 → 폴더별 25% 표본(weight=4로 전체 추정), 다른 데이터 음성은 전부,
    # 불꽃 사진은 비교용으로 데이터마다 1,000장
    a = df[(df.dataset == 'AIHub') & (df.kind == 'neg')].groupby('sub', group_keys=False).apply(lambda g: g.sample(frac=0.25, random_state=0))
    o = df[(df.dataset != 'AIHub') & (df.kind == 'neg')]
    p = df[df.kind == 'pos_fire'].groupby('dataset', group_keys=False).apply(lambda g: g.sample(min(1000, len(g)), random_state=0))
    df = pd.concat([a.assign(weight=4.0), o.assign(weight=1.0), p.assign(weight=1.0)]).reset_index(drop=True)
    print('분류할 사진', len(df), df.groupby(['dataset', 'kind']).size().to_dict(), flush=True)
    for src in ('val', 'train'):                      # 메인에서 연 zip은 fork 전에 닫는다 (train은 목록만 남김)
        L.opener(src)
    L.ZF['train'].close(); del L.ZF['val']
    proc = AutoProcessor.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL, dtype=torch.bfloat16).cuda().eval()
    names, owner = [], []
    for c, ps in CATS.items():
        names += ps; owner += [c] * len(ps)
    with torch.no_grad():
        t = proc(text=[f'a photo of {p}' for p in names], padding='max_length', return_tensors='pt')
        temb = model.get_text_features(**{k: v.cuda() for k, v in t.items()})
        temb = temb if torch.is_tensor(temb) else temb.pooler_output
        temb = temb / temb.norm(dim=-1, keepdim=True)
    rows = df.to_dict('records')
    dl = torch.utils.data.DataLoader(DS(rows, proc), batch_size=64, num_workers=16, shuffle=False)
    owner = np.array(owner); cats = list(CATS)
    best, bestp, score, oks = np.empty(len(rows), object), np.empty(len(rows), object), np.zeros((len(rows), len(cats)), np.float32), np.ones(len(rows), bool)
    with torch.no_grad():
        for k, (x, idx, ok) in enumerate(dl):
            e = model.get_image_features(pixel_values=x.cuda().to(torch.bfloat16))
            e = e if torch.is_tensor(e) else e.pooler_output
            e = e / e.norm(dim=-1, keepdim=True)
            lg = (e.float() @ temb.float().T).cpu().numpy()            # 코사인 유사도 (문장마다)
            for c_i, c in enumerate(cats):
                score[idx.numpy(), c_i] = lg[:, owner == c].max(1)
            j = lg.argmax(1)
            best[idx.numpy()] = owner[j]; bestp[idx.numpy()] = np.array(names)[j]; oks[idx.numpy()] = ok.numpy()
            if k % 200 == 0:
                print(k * 64, flush=True)
    out = df[['dataset', 'kind', 'sub', 'name', 'src', 'key', 'fire_px', 'weight']].copy()
    out['tag'] = best; out['prompt'] = bestp; out['ok'] = oks
    for c_i, c in enumerate(cats):
        out[f's_{c}'] = score[:, c_i]
    out.to_csv(f'{OUT}/clip_tags.csv.gz', index=False)
    print('saved', len(out), '열기 실패', int((~oks).sum()))


def sheets():
    df = pd.read_csv(f'{OUT}/clip_tags_full.csv.gz', low_memory=False)
    neg = df[df.kind == 'neg']
    font = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 12, index=3)
    for c in ['불꽃놀이', '노을·해', '밤 불빛', '실내 조명', '통제된 불', '번개', '산불']:
        for ds, g in neg[neg.tag == c].groupby('dataset'):
            g = g.sample(min(40, len(g)), random_state=0)          # 그 종류로 분류된 사진 중 무작위 (정확도 확인용)
            cell, cols = 200, 8
            S = Image.new('RGB', (cols * cell, ((len(g) + cols - 1) // cols) * (cell + 16)), 'white')
            for k, r in enumerate(g.itertuples()):
                im = L.load((r.src, r.key)).convert('RGB'); im.thumbnail((cell, cell))
                t = Image.new('RGB', (cell, cell + 16), 'white'); t.paste(im, ((cell - im.width) // 2, (cell - im.height) // 2))
                ImageDraw.Draw(t).text((2, cell), f'{k} {r.sub[:12]}', fill='black', font=font)
                S.paste(t, ((k % cols) * cell, (k // cols) * (cell + 16)))
            S.save(f'{OUT}/clip_{c}_{ds}.jpg', quality=85)


if __name__ == '__main__':
    sheets() if len(sys.argv) > 1 and sys.argv[1] == 'sheets' else run()
