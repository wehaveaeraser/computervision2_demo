# 자율주행 영상 노이즈 제거 - CNN 베이스라인 모델

## 개요
본 프로젝트는 자율주행 영상에서 노이즈를 제거하기 위한 CNN 베이스라인 모델을 구현합니다.

## 모델 구조
- **4-layer CNN** (Conv-BN-ReLU 블록)
- 채널 수: 64 → 128 → 256 → 512
- Encoder-Decoder 구조

## 주요 기능
- L1 Loss + SSIM Loss 조합
- Early Stopping
- Validation loss 기반 체크포인트 저장
- PSNR, SSIM 평가 지표
- Inference 시간 및 FPS 측정

## 설치

### 1. 기본 패키지 설치

```bash
pip install -r requirements.txt
```

### 2. PyTorch CUDA 버전 설치 (GPU 사용 시 필수)

**중요**: GPU를 사용하려면 PyTorch CUDA 버전이 필요합니다.

#### 자동 설치 (권장)
```bash
python install_pytorch_cuda.py
```

#### 수동 설치
```bash
# 현재 CPU 버전 제거
pip uninstall torch torchvision torchaudio -y

# CUDA 12.1 버전 설치 (CUDA 12.6과 호환)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

또는 CUDA 12.4 버전:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

**설치 확인:**
```bash
python check_gpu.py
```

CUDA 사용 가능이 `True`로 나오면 성공입니다!

### GPU 설정 확인

GPU 사용 가능 여부를 확인하려면:

```bash
python check_gpu.py
```

이 스크립트는 다음 정보를 출력합니다:
- CUDA 사용 가능 여부
- 사용 가능한 GPU 개수 및 정보
- GPU 메모리 용량
- CUDA 및 cuDNN 버전

## 모델 체크포인트 다운로드

학습된 모델 체크포인트는 Google Drive에서 다운로드할 수 있습니다:

🔗 **[체크포인트 다운로드 (Google Drive)](https://drive.google.com/drive/folders/14_OazjaCb0Yg8gafoHA0x_72LkwP4Wfd?usp=sharing)**

### 다운로드 및 설치 방법

1. 위 링크를 클릭하여 Google Drive 폴더에 접근
2. `cnn` 폴더와 `unet` 폴더를 각각 다운로드
3. 다운로드한 폴더를 프로젝트 루트에 다음과 같이 배치:

```
computervesion/
├── checkpoints/
│   ├── cnn/
│   │   └── best_model.pth
│   └── unet/
│       └── best_model.pth
└── ...
```

또는 `second_checkpoints` 폴더에 배치:

```
computervesion/
├── second_checkpoints/
│   ├── cnn/
│   │   └── best_model.pth
│   └── unet/
│       └── best_model.pth
└── ...
```

**참고:** 모델 파일은 크기가 크므로 GitHub에는 포함되지 않습니다. 위 Google Drive 링크에서 다운로드하세요.

## 데이터 준비

### 방법 1: 정제된 데이터 사용 (권장)

정제된 데이터는 하나의 폴더에 `*_rain.png`와 `*_clean.png` 파일이 함께 있습니다:

```
train_img/
├── 0_rain.png
├── 0_clean.png
├── 1_rain.png
├── 1_clean.png
└── ...
```

이 경우 `--clean_dir` 파라미터를 생략하면 자동으로 파일명 패턴으로 매칭됩니다.

### 방법 2: 두 개의 폴더 사용

데이터셋은 다음과 같은 구조로 준비할 수도 있습니다:

```
data/
├── noisy/          # 노이즈 이미지
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
└── clean/          # 깨끗한 이미지 (Ground Truth)
    ├── img1.jpg
    ├── img2.jpg
    └── ...
```

노이즈 이미지와 깨끗한 이미지는 파일명이 일치해야 합니다.

## 사용 방법

### 1. 모델 학습

#### 현재 프로젝트 데이터 구조 사용 (권장)

현재 프로젝트의 `train_img/` 폴더는 `data/`와 `gt/`로 나뉘어 있습니다:

```bash
python train_cnn.py \
    --noisy_dir ./train_img/data \
    --clean_dir ./train_img/gt \
    --batch_size 16 \
    --lr 1e-4 \
    --epochs 50 \
    --checkpoint_dir checkpoints/cnn
```

간단한 버전:
```bash
python train_cnn.py --noisy_dir ./train_img/data --clean_dir ./train_img/gt
```

#### 정제된 데이터 사용

정제된 데이터는 하나의 폴더에 `*_rain.png`와 `*_clean.png` 파일이 함께 있습니다:

```bash
python train_cnn.py \
    --noisy_dir ./train_img \
    --batch_size 16 \
    --lr 1e-4 \
    --epochs 50 \
    --checkpoint_dir checkpoints/cnn
```

#### 두 개의 폴더 사용

```bash
python train_cnn.py \
    --noisy_dir data/noisy \
    --clean_dir data/clean \
    --batch_size 16 \
    --lr 1e-4 \
    --epochs 50 \
    --checkpoint_dir checkpoints/cnn
```

**주요 파라미터:**
- `--noisy_dir`: 노이즈 이미지 디렉토리 경로 (또는 정제된 데이터가 있는 폴더)
- `--clean_dir`: 깨끗한 이미지 디렉토리 경로 (선택사항, None이면 noisy_dir에서 파일명 패턴으로 매칭)
- `--batch_size`: 배치 크기 (기본값: 16)
- `--lr`: 학습률 (기본값: 1e-4)
- `--epochs`: 에포크 수 (기본값: 50)
- `--checkpoint_dir`: 체크포인트 저장 디렉토리
- `--val_split`: 검증 데이터 비율 (기본값: 0.2)
- `--device`: 사용할 디바이스 (cuda/cpu) (기본값: cuda)
- `--gpu_id`: 사용할 GPU ID (기본값: 0)

**추가 옵션 예시:**
```bash
# 현재 프로젝트 데이터 사용 (GPU 0, 기본)
python train_cnn.py --noisy_dir ./train_img/data --clean_dir ./train_img/gt

# 특정 GPU 사용 (예: GPU 1)
python train_cnn.py --noisy_dir ./train_img/data --clean_dir ./train_img/gt --gpu_id 1

# CPU 사용
python train_cnn.py --noisy_dir ./train_img/data --clean_dir ./train_img/gt --device cpu

# 배치 크기 조정 (GPU 메모리 부족 시)
python train_cnn.py --noisy_dir ./train_img/data --clean_dir ./train_img/gt --batch_size 8
```

### 2. 모델 평가

#### 현재 프로젝트 test/syn 데이터 사용 (권장)

`test/syn/` 폴더는 `rainy_vid/`와 `clean_vid/`로 나뉘어 있으며, 각각 서브디렉토리(0001, 0002, ...)를 포함합니다. 코드는 자동으로 모든 서브디렉토리를 재귀적으로 탐색합니다:

```bash
python inference_cnn.py \
    --model_path checkpoints/cnn/best_model.pth \
    --noisy_dir ./test/syn/rainy_vid \
    --clean_dir ./test/syn/clean_vid \
    --batch_size 16 \
    --visualize
```

간단한 버전:
```bash
python inference_cnn.py \
    --model_path checkpoints/cnn/best_model.pth \
    --noisy_dir ./test/syn/rainy_vid \
    --clean_dir ./test/syn/clean_vid \
    --visualize
```

#### 정제된 테스트 데이터 사용

```bash
python inference_cnn.py \
    --model_path checkpoints/cnn/best_model.pth \
    --noisy_dir ./test \
    --batch_size 16 \
    --visualize
```

#### 두 개의 폴더 사용

```bash
python inference_cnn.py \
    --model_path checkpoints/cnn/best_model.pth \
    --noisy_dir data/test_noisy \
    --clean_dir data/test_clean \
    --batch_size 16 \
    --visualize
```

**주요 파라미터:**
- `--model_path`: 학습된 모델 체크포인트 경로 (필수)
- `--noisy_dir`: 테스트 노이즈 이미지 디렉토리 (또는 정제된 데이터가 있는 폴더)
- `--clean_dir`: 테스트 깨끗한 이미지 디렉토리 (선택사항, None이면 noisy_dir에서 파일명 패턴으로 매칭)
- `--batch_size`: 배치 크기 (기본값: 16)
- `--visualize`: 결과 시각화 저장 여부
- `--save_dir`: 결과 저장 디렉토리 (기본값: results)
- `--device`: 사용할 디바이스 (cuda/cpu) (기본값: cuda)
- `--gpu_id`: 사용할 GPU ID (기본값: 0)

**참고:** 
- 서브디렉토리가 있는 경우 자동으로 재귀적으로 탐색합니다 (예: `test/syn/rainy_vid/0001/`, `test/syn/rainy_vid/0002/` 등)
- `test/real/` 데이터는 clean 이미지가 없어 PSNR/SSIM 평가가 불가능합니다

### 3. 단일 이미지 추론

```bash
python inference_cnn.py \
    --model_path checkpoints/cnn/best_model.pth \
    --image_path path/to/noisy_image.jpg \
    --output_path path/to/output_image.jpg
```

## 평가 지표

모델은 다음 지표들을 측정합니다:

- **PSNR** (Peak Signal-to-Noise Ratio): 화질 지표
- **SSIM** (Structural Similarity Index): 구조적 유사도
- **Inference Time**: 추론 시간 (ms/frame)
- **FPS**: 초당 프레임 수

## 파일 구조

```
team/
├── cnn_model.py          # CNN 모델 정의 및 유틸리티
├── train_cnn.py          # 학습 스크립트
├── inference_cnn.py       # 추론 및 평가 스크립트
├── check_gpu.py          # GPU 정보 확인 스크립트
├── requirements.txt       # 필요한 패키지 목록
└── README.md             # 이 파일
```

## 모델 아키텍처

```
Input (3 channels)
    ↓
Conv(3→64) + BN + ReLU
    ↓
Conv(64→128) + BN + ReLU
    ↓
Conv(128→256) + BN + ReLU
    ↓
Conv(256→512) + BN + ReLU
    ↓
Conv(512→256) + BN + ReLU
    ↓
Conv(256→128) + BN + ReLU
    ↓
Conv(128→64) + BN + ReLU
    ↓
Conv(64→3) + Sigmoid
    ↓
Output (3 channels)
```

## 다음 단계

1. U-Net 모델 구현 및 비교
2. Attention U-Net 모델 구현
3. YOLOv5를 이용한 객체 검출 성능 평가

## GPU 사용

### GPU 요구사항
- NVIDIA GPU (CUDA 지원)
- CUDA Toolkit 설치
- PyTorch CUDA 버전 설치

### GPU 확인
학습 시작 전에 GPU 정보를 확인할 수 있습니다:
```bash
python check_gpu.py
```

### GPU 메모리 관리
- 배치 크기를 조정하여 GPU 메모리 사용량을 제어할 수 있습니다
- 학습 중 각 epoch마다 GPU 메모리 사용량이 출력됩니다
- GPU 메모리가 부족한 경우 `--batch_size`를 줄이세요

### 멀티 GPU 사용
현재는 단일 GPU만 지원합니다. 여러 GPU를 사용하려면 PyTorch의 `DataParallel` 또는 `DistributedDataParallel`을 사용하도록 코드를 수정해야 합니다.

## 참고사항

- GPU 사용을 권장합니다 (CUDA)
- 이미지는 RGB 형식이어야 합니다
- 이미지는 0-255 범위의 uint8 형식이어야 하며, 모델 내부에서 0-1로 정규화됩니다
- 모든 이미지는 512x512 크기로 자동 리사이즈됩니다
- 정제된 데이터 사용 시 파일명 패턴: `*_rain.png` ↔ `*_clean.png`로 자동 매칭
- **서브디렉토리 지원**: 이중 폴더 모드 사용 시 모든 하위 디렉토리를 재귀적으로 탐색합니다
- GPU가 없어도 CPU 모드로 실행 가능합니다 (느릴 수 있음)

## 실제 데이터 구조 예시

### 학습 데이터 (train_img/)
```
train_img/
├── data/          # 노이즈 이미지
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
└── gt/            # 깨끗한 이미지 (Ground Truth)
    ├── img1.jpg
    ├── img2.jpg
    └── ...
```

### 테스트 데이터 (test/)
```
test/
├── syn/
│   ├── rainy_vid/     # 노이즈 이미지 (서브디렉토리 포함)
│   │   ├── 0001/
│   │   │   ├── frame001.png
│   │   │   └── ...
│   │   ├── 0002/
│   │   └── ...
│   └── clean_vid/     # 깨끗한 이미지 (서브디렉토리 포함)
│       ├── 0001/
│       │   ├── frame001.png
│       │   └── ...
│       ├── 0002/
│       └── ...
└── real/              # 같은 사진 여러 장 (clean 이미지 없음)
    └── ...
```

**실행 예시:**
```bash
# 1. 학습
python train_cnn.py --noisy_dir ./train_img/data --clean_dir ./train_img/gt --epochs 50

# 2. 평가 (test/syn 사용)
python inference_cnn.py \
    --model_path checkpoints/cnn/best_model.pth \
    --noisy_dir ./test/syn/rainy_vid \
    --clean_dir ./test/syn/clean_vid \
    --visualize
```

