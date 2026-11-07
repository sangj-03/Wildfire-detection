"""모델링 단계 초안: wildfire_train의 torch Dataset·Sampler·증강이 학습 환경에서 그대로 도는지 확인한다.
사용: /data/seotturi/.venv/bin/python -I modeling/check_loader.py  →  modeling/draft_outputs/loader_check.json, aug_examples.jpg
"""
import json, sys, time

sys.path[:0] = ['/data/home/seame/dl-team-project/team6', '/data/home/seame/dl-team-project/team6/modeling']
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

import wildfire_data as W
import wildfire_train as WT

OUT = '/data/home/seame/dl-team-project/team6/modeling/draft_outputs'

m = W.load_manifest()
tr = m[m.split == 'train']
sampler = WT.VideoSampler(tr, n_per_video=8)
ds = WT.WildfireDataset(tr, train=True)
loader = DataLoader(ds, batch_size=16, sampler=sampler, num_workers=12, persistent_workers=True)
out = {'train_rows': len(tr), 'epoch_len_n8': len(sampler)}

# 에폭마다 다른 프레임을 뽑는지, 뽑힌 수가 늘 같은지
a = set(WT.video_epoch_indices(tr, 8, 0)); b = set(WT.video_epoch_indices(tr, 8, 1))
out['epoch0_epoch1_overlap'] = round(len(a & b) / len(a), 3)

t0 = time.time(); n = 0; first = None
for x, y in loader:
    first = (x, y) if first is None else first
    n += len(y)
    if n >= 16 * 40:
        break
dt = time.time() - t0
x, y = first
out.update(batch_shape=list(x.shape), dtype=str(x.dtype), y_first_batch=y.tolist(),
           mean_per_channel=[round(float(v), 3) for v in x.mean((0, 2, 3))],
           images_per_sec_12workers=round(n / dt, 1))

# 증강 예: 같은 양성 사진 한 장에 학습용 증강을 4번
torch.manual_seed(0)
r = tr[tr.y == 1].sample(1, random_state=3).iloc[0]
im = W.load_image(r)
tf = WT.make_transform(train=True)
mean, std = torch.tensor(W.IMAGENET_MEAN)[:, None, None], torch.tensor(W.IMAGENET_STD)[:, None, None]
tiles = [im.resize((480, 243))]
for _ in range(4):
    t = (tf(im) * std + mean).clamp(0, 1)
    tiles.append(Image.fromarray((t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)).resize((480, 243)))
S = Image.new('RGB', (480 * 5 + 16, 243), '#ffffff')
for i, t in enumerate(tiles):
    S.paste(t, (i * 484, 0))
ImageDraw.Draw(S).rectangle([0, 0, 479, 242], outline='#1d3461', width=3)
S.save(f'{OUT}/aug_examples.jpg', quality=88)
out['aug_example'] = r.member
json.dump(out, open(f'{OUT}/loader_check.json', 'w'), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
