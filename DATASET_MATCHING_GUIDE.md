# 데이터셋 매칭 및 학습 가이드

## 개요

노이즈 데이터와 원본 데이터를 폴더명 기반으로 매칭하여 모델 학습을 수행하는 방법을 설명합니다.

## 데이터 구조

### 노이즈 데이터 구조
```
DATASET_FAST_FINAL copy/
├── test/
│   ├── day/
│   └── night/
├── train/
│   ├── day/
│   └── night/
└── val/
    ├── day/
    └── night/
```

### 원본 데이터 구조
```
원본데이터폴더/
├── test/
│   ├── berlin/
│   ├── bielefeld/
│   ├── bonn/
│   ├── leverkusen/
│   ├── mainz/
│   └── munich/
├── train/
│   └── ...
└── val/
    └── ...
```

## 사용 방법

### 1단계: 데이터 확인 및 개수 확인

먼저 두 데이터셋의 구조와 개수를 확인합니다.

```bash
python check_dataset.py --noisy_dir "DATASET_FAST_FINAL copy" --clean_dir "원본데이터폴더"
```

**출력 예시:**
- 노이즈 데이터셋 분석: split별, category별(day/night) 개수
- 원본 데이터셋 분석: split별, 폴더별 개수
- 매칭 가능 여부 확인

### 2단계: 데이터 매칭

노이즈 데이터와 원본 데이터를 폴더명 기반으로 매칭합니다.

```bash
python match_datasets.py --noisy_dir "DATASET_FAST_FINAL copy" --clean_dir "원본데이터폴더" --output matched_pairs.json
```

**매칭 로직:**
1. 각 split(test/train/val)별로 처리
2. 노이즈 데이터는 day/night 구분 없이 모두 수집
3. 원본 데이터는 각 폴더(berlin, bielefeld 등)별로 수집
4. 같은 split 내에서 순차적으로 1:1 매칭
5. 매칭 결과를 JSON 파일로 저장

**출력 파일: `matched_pairs.json`**
```json
{
  "matched_pairs": [
    {
      "split": "test",
      "folder_name": "berlin",
      "noisy_path": "...",
      "clean_path": "...",
      "noisy_relative": "test/day/image1.png",
      "clean_relative": "test/berlin/image1.png"
    },
    ...
  ],
  "stats": {
    "total_pairs": 1000,
    "by_split": {"test": 200, "train": 700, "val": 100},
    "by_folder": {"test/berlin": 50, ...}
  }
}
```

### 3단계: 모델 학습

매칭된 데이터를 사용하여 모델을 학습합니다.

**두 가지 방법이 있습니다:**

#### 방법 1: 매칭 파일 사용 (권장)
매칭 파일을 미리 만들어서 사용하는 방법입니다.

```bash
# CNN 모델
python train_cnn.py \
  --noisy_dir "DATASET_FAST_FINAL copy" \
  --clean_dir "원본데이터폴더" \
  --matched_pairs_file matched_pairs.json \
  --batch_size 2 \
  --epochs 10 \
  --lr 5e-5

# U-Net 모델
python train_unet.py \
  --noisy_dir "DATASET_FAST_FINAL copy" \
  --clean_dir "원본데이터폴더" \
  --matched_pairs_file matched_pairs.json \
  --batch_size 2 \
  --epochs 10 \
  --lr 5e-5
```

#### 방법 2: 자동 매칭 (간편)
매칭 파일 없이 학습 시 자동으로 매칭하는 방법입니다.

```bash
# CNN 모델
python train_cnn.py \
  --noisy_dir "DATASET_FAST_FINAL copy" \
  --clean_dir "원본데이터폴더" \
  --auto_match \
  --batch_size 2 \
  --epochs 10 \
  --lr 5e-5

# U-Net 모델
python train_unet.py \
  --noisy_dir "DATASET_FAST_FINAL copy" \
  --clean_dir "원본데이터폴더" \
  --auto_match \
  --batch_size 2 \
  --epochs 10 \
  --lr 5e-5
```

**자동 매칭 모드 (`--auto_match`):**
- 학습 시작 시 자동으로 폴더명 기반 매칭 수행
- 매칭 파일을 만들 필요 없음
- 매칭 로직은 `match_datasets.py`와 동일
- day/night 구분 없이 모두 함께 매칭

## 코드 설명

### check_dataset.py

**주요 기능:**
- `analyze_noisy_dataset()`: 노이즈 데이터셋 구조 분석
  - split별(test/train/val) 개수
  - category별(day/night) 개수
  - split/category별 상세 개수
  
- `analyze_clean_dataset()`: 원본 데이터셋 구조 분석
  - split별 개수
  - 폴더별 개수
  - split/폴더별 상세 개수
  
- `match_folders()`: 매칭 가능 여부 분석
  - 각 split별로 노이즈/원본 데이터 개수 비교
  - 매칭 가능 여부 및 비율 계산

### match_datasets.py

**주요 기능:**
- `match_by_folder_name()`: 폴더명 기반 매칭
  - 각 split별로 처리
  - 노이즈 데이터: day/night 폴더의 모든 이미지 수집
  - 원본 데이터: 각 폴더별로 이미지 수집
  - 순차적으로 1:1 매칭
  - 매칭 결과를 JSON 파일로 저장

**매칭 알고리즘:**
1. split별로 순회 (test → train → val)
2. 노이즈 데이터 수집: `split/day` + `split/night`의 모든 이미지
3. 원본 데이터 수집: `split/폴더명/`의 모든 이미지
4. 원본 이미지 순서대로 노이즈 이미지와 매칭
5. 노이즈 이미지가 부족하면 순환 배치

### ImageDataset 클래스 (cnn_model.py)

**새로운 기능:**
- `matched_pairs_file` 파라미터 추가
- JSON 파일에서 매칭 정보를 읽어와 이미지 쌍 로드
- 기존 방식(파일명 패턴, 이중 폴더)과 호환

**사용 예시:**
```python
# 매칭 파일 사용
dataset = ImageDataset(
    noisy_dir="DATASET_FAST_FINAL copy",
    clean_dir="원본데이터폴더",
    matched_pairs_file="matched_pairs.json"
)
```

### train_cnn.py / train_unet.py

**새로운 옵션:**
- `--matched_pairs_file`: 매칭 파일 경로 지정
- 이 옵션이 제공되면 ImageDataset이 매칭 파일 모드로 동작

## 특징

1. **day/night 구분 없음**: 노이즈 데이터의 day/night 폴더를 구분하지 않고 모두 함께 학습
2. **폴더명 기반 매칭**: 원본 데이터의 폴더명(berlin, bielefeld 등)을 기준으로 매칭
3. **split 유지**: test/train/val split을 유지하여 학습/검증/테스트 분리
4. **유연한 매칭**: 원본 이미지와 노이즈 이미지 개수가 달라도 최대한 매칭

## 주의사항

1. **경로 설정**: 노이즈 데이터와 원본 데이터의 경로를 정확히 지정해야 합니다.
2. **매칭 파일**: 매칭 파일은 절대 경로 또는 상대 경로로 저장됩니다. 경로가 변경되면 다시 매칭해야 할 수 있습니다.
3. **데이터 개수**: 원본 이미지와 노이즈 이미지 개수가 다를 수 있습니다. 이 경우 일부 이미지는 매칭되지 않을 수 있습니다.

## 문제 해결

### 매칭이 안 되는 경우
1. `check_dataset.py`로 데이터 구조 확인
2. 경로가 올바른지 확인
3. 이미지 파일 확장자 확인 (.jpg, .png 등)

### 학습 중 오류 발생
1. 매칭 파일의 경로가 올바른지 확인
2. 이미지 파일이 실제로 존재하는지 확인
3. 메모리 부족 시 batch_size 줄이기

