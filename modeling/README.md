# 모델링 단계 초안 (전처리에서 옮겨 둔 것)

전처리 단계를 가공·전처리까지만으로 정리하면서(2026-10-25), 학습 때 쓰는 처리를 이 폴더로 옮겼다.

- `wildfire_train.py`: 영상당 최대 N장 추출(에폭마다 다시 뽑음), 한 에폭 장수 기준 클래스 가중치, 증강(좌우 뒤집기·작은 확대/이동), torch Dataset·Sampler. 사진 처리는 `../wildfire_data.py`의 `preprocess()`를 그대로 쓴다.
- `check_loader.py`: 학습 환경(`/data/seotturi/.venv`)에서 위 코드가 도는지 확인.
- `draft_outputs/`: 옮기기 전에 낸 결과(클래스 가중치, 로더 점검, 증강 예). **split_v1 기준 수치**라 모델링을 시작할 때 split_v2 목록(`../prep/manifest.csv.gz`)으로 다시 계산한다.

N·가중치/균형 추출은 튜닝 R3에서 고른다.
