# 모델 코드 설명

## 1. CNN 모델

### 1.1 모델 구조

CNN 모델은 이미지 노이즈 제거를 위한 4-layer Encoder-Decoder 구조를 사용합니다.

#### 네트워크 아키텍처

```
입력 (3 channels) 
  ↓
Encoder Path:
  - Conv1: 3 → 64 channels (kernel=3, padding=1)
  - BatchNorm + ReLU
  - Conv2: 64 → 128 channels (kernel=3, padding=1)
  - BatchNorm + ReLU
  - Conv3: 128 → 256 channels (kernel=3, padding=1)
  - BatchNorm + ReLU
  - Conv4: 256 → 512 channels (kernel=3, padding=1)
  - BatchNorm + ReLU
  ↓
Decoder Path:
  - Conv5: 512 → 256 channels (kernel=3, padding=1)
  - BatchNorm + ReLU
  - Conv6: 256 → 128 channels (kernel=3, padding=1)
  - BatchNorm + ReLU
  - Conv7: 128 → 64 channels (kernel=3, padding=1)
  - BatchNorm + ReLU
  - Conv8: 64 → 3 channels (kernel=3, padding=1)
  ↓
Residual Learning 적용
  - 출력을 tanh로 정규화하여 노이즈 예측
  - 입력에서 노이즈를 빼서 clean 이미지 생성
  - [0, 1] 범위로 클리핑
```

#### 주요 특징

1. **Residual Learning**: 모델이 전체 이미지를 예측하는 대신 노이즈만 예측하고, 입력 이미지에서 노이즈를 빼서 깨끗한 이미지를 생성합니다. 이는 학습을 더 안정적으로 만들고 성능을 향상시킵니다.

2. **Batch Normalization**: 각 컨볼루션 레이어 후에 Batch Normalization을 적용하여 학습 안정성을 높이고 수렴 속도를 개선합니다.

3. **채널 확장**: Encoder에서 채널 수를 점진적으로 증가시켜 (64 → 128 → 256 → 512) 더 복잡한 특징을 학습할 수 있도록 합니다.

### 1.2 학습 방법

#### 손실 함수 (Loss Function)

**Combined Loss**를 사용하여 세 가지 손실을 조합합니다:

```
Loss = L1_weight × L1_Loss + SSIM_weight × SSIM_Loss + Gradient_weight × Gradient_Loss
```

1. **L1 Loss (Mean Absolute Error)**
   - 픽셀 단위의 절대 오차를 측정
   - 가중치: 1.0 (기본값)

2. **SSIM Loss (Structural Similarity Index Measure)**
   - 이미지의 구조적 유사성을 측정
   - 밝기, 대비, 구조를 종합적으로 고려
   - 가중치: 1.0 (기본값)

3. **Gradient Loss**
   - Sobel 필터를 사용하여 이미지의 gradient(엣지/경계) 차이를 측정
   - 노이즈는 고주파 성분이므로 gradient가 크고, 이를 줄이도록 학습
   - 가중치: 0.5 (기본값)

#### 최적화 파라미터

- **Optimizer**: Adam
- **Learning Rate**: 5e-5 (기본값)
- **Batch Size**: 2
- **Epochs**: 50 (기본값)
- **Early Stopping**: 
  - Patience: 5 epochs
  - Min Delta: 0.001

#### 학습 기법

1. **Mixed Precision Training**
   - GPU에서 자동으로 활성화
   - FP16 연산을 사용하여 메모리 사용량을 줄이고 학습 속도를 향상
   - Loss Scaling을 통해 수치 안정성 유지

2. **Gradient Clipping**
   - 20배치마다 gradient norm을 10.0으로 제한
   - 학습 불안정을 방지

3. **Checkpoint 저장**
   - Validation loss가 가장 낮은 모델을 자동으로 저장
   - Optimizer state와 scaler state도 함께 저장하여 학습 재개 가능

### 1.3 모델 파라미터

- **입력 채널**: 3 (RGB)
- **출력 채널**: 3 (RGB)
- **이미지 크기**: 512 × 512
- **총 파라미터 수**: 약 수백만 개 (모델 구조에 따라 다름)

### 1.4 데이터 전처리

- 이미지를 512 × 512 크기로 리사이즈
- 픽셀 값을 [0, 255]에서 [0, 1]로 정규화
- BGR → RGB 변환

---

## 2. U-Net 모델

### 2.1 모델 구조

U-Net 모델은 Encoder-Decoder 구조에 Skip Connections를 추가한 네트워크입니다.

#### 네트워크 아키텍처

```
입력 (3 channels)
  ↓
Encoder Path (Downsampling):
  - Enc1: 3 → 64 channels (Conv-BN-ReLU × 2)
    ↓ MaxPool (2×2)
  - Enc2: 64 → 128 channels (Conv-BN-ReLU × 2)
    ↓ MaxPool (2×2)
  - Enc3: 128 → 256 channels (Conv-BN-ReLU × 2)
    ↓ MaxPool (2×2)
  - Enc4: 256 → 512 channels (Conv-BN-ReLU × 2)
    ↓ MaxPool (2×2)
  ↓
Bottleneck:
  - 512 → 1024 channels (Conv-BN-ReLU × 2)
  ↓
Decoder Path (Upsampling with Skip Connections):
  - Up4: 1024 → 512 (Transposed Conv, stride=2)
    + Skip Connection (Enc4: 512 channels)
    → Concat → 1024 channels
  - Dec4: 1024 → 512 channels (Conv-BN-ReLU × 2)
    ↓
  - Up3: 512 → 256 (Transposed Conv, stride=2)
    + Skip Connection (Enc3: 256 channels)
    → Concat → 512 channels
  - Dec3: 512 → 256 channels (Conv-BN-ReLU × 2)
    ↓
  - Up2: 256 → 128 (Transposed Conv, stride=2)
    + Skip Connection (Enc2: 128 channels)
    → Concat → 256 channels
  - Dec2: 256 → 128 channels (Conv-BN-ReLU × 2)
    ↓
  - Up1: 128 → 64 (Transposed Conv, stride=2)
    + Skip Connection (Enc1: 64 channels)
    → Concat → 128 channels
  - Dec1: 128 → 64 channels (Conv-BN-ReLU × 2)
    ↓
  - Final Conv: 64 → 3 channels (kernel=1)
  ↓
Residual Learning 적용
  - 출력을 tanh로 정규화하여 노이즈 예측
  - 입력에서 노이즈를 빼서 clean 이미지 생성
  - [0, 1] 범위로 클리핑
```

#### 주요 특징

1. **Skip Connections (U-Net의 핵심)**
   - Encoder의 각 레이어 출력을 Decoder의 대응되는 레이어에 연결
   - 저수준 특징(엣지, 텍스처)을 보존하여 디테일 복원 성능 향상
   - 정보 손실을 최소화

2. **Encoder-Decoder 구조**
   - Encoder: MaxPool을 사용하여 공간 해상도를 점진적으로 감소시키며 고수준 특징 추출
   - Decoder: Transposed Convolution을 사용하여 공간 해상도를 복원

3. **Residual Learning**
   - CNN과 동일하게 노이즈만 예측하여 입력에서 빼는 방식 사용

4. **Double Convolution Block**
   - 각 Encoder/Decoder 단계에서 두 번의 컨볼루션을 수행하여 더 강력한 특징 추출

### 2.2 학습 방법

#### 손실 함수 (Loss Function)

CNN과 동일하게 **Combined Loss**를 사용합니다:

```
Loss = L1_weight × L1_Loss + SSIM_weight × SSIM_Loss + Gradient_weight × Gradient_Loss
```

- **L1 Loss**: 가중치 1.0
- **SSIM Loss**: 가중치 1.0
- **Gradient Loss**: 가중치 0.5

#### 최적화 파라미터

- **Optimizer**: Adam
- **Learning Rate**: 5e-5 (기본값)
- **Batch Size**: 2 (U-Net은 메모리를 많이 사용하므로 작은 배치 크기 사용)
- **Epochs**: 50 (기본값)
- **Early Stopping**: 
  - Patience: 5 epochs
  - Min Delta: 0.001

#### 학습 기법

1. **Mixed Precision Training**
   - GPU에서 자동 활성화
   - FP16 연산으로 메모리 효율성 향상

2. **Gradient Clipping**
   - 20배치마다 gradient norm을 10.0으로 제한

3. **Checkpoint 저장**
   - Best validation loss 모델 자동 저장

### 2.3 모델 파라미터

- **입력 채널**: 3 (RGB)
- **출력 채널**: 3 (RGB)
- **Base Channels**: 64
- **이미지 크기**: 512 × 512
- **총 파라미터 수**: 약 수천만 개 (Skip Connections로 인해 CNN보다 많음)

### 2.4 데이터 전처리

- CNN과 동일한 전처리 과정
- 이미지를 512 × 512 크기로 리사이즈
- 픽셀 값을 [0, 255]에서 [0, 1]로 정규화
- BGR → RGB 변환

---

## 3. 두 모델의 비교

### 3.1 구조적 차이

| 특징 | CNN | U-Net |
|------|-----|-------|
| **Skip Connections** | 없음 | 있음 (핵심 특징) |
| **Downsampling** | 없음 (해상도 유지) | MaxPool 사용 |
| **Upsampling** | 없음 | Transposed Conv 사용 |
| **파라미터 수** | 상대적으로 적음 | 상대적으로 많음 |
| **메모리 사용량** | 적음 | 많음 |

### 3.2 학습 방법의 공통점

두 모델 모두 동일한 학습 방법을 사용합니다:

- **Loss Function**: Combined Loss (L1 + SSIM + Gradient)
- **Optimizer**: Adam
- **Learning Rate**: 5e-5
- **Batch Size**: 2
- **Mixed Precision Training**: 활성화
- **Gradient Clipping**: 적용
- **Early Stopping**: 적용
- **Residual Learning**: 적용

### 3.3 예상 성능 차이

- **CNN**: 
  - 빠른 학습 및 추론 속도
  - 상대적으로 적은 메모리 사용
  - 전역적인 특징 학습에 강점

- **U-Net**: 
  - Skip Connections로 인한 우수한 디테일 복원
  - 더 많은 파라미터로 인한 높은 표현력
  - 공간적 정보 보존에 강점

---

## 4. 학습 프로세스

### 4.1 데이터셋 구성

- **Train/Val/Test 분할**: 7:1.5:1.5 비율
- **데이터 증강**: 필요시 적용 가능
- **이미지 크기**: 512 × 512로 통일

### 4.2 학습 루프

1. **Training Phase**
   - 모델을 train mode로 설정
   - 각 배치에 대해:
     - Forward pass (Mixed Precision)
     - Loss 계산
     - Backward pass (Gradient Clipping)
     - Optimizer step

2. **Validation Phase**
   - 모델을 eval mode로 설정
   - Gradient 계산 없이 Forward pass만 수행
   - PSNR, SSIM 메트릭 계산
   - Best model 저장

3. **Early Stopping**
   - Validation loss가 개선되지 않으면 조기 종료

### 4.3 평가 메트릭

- **PSNR (Peak Signal-to-Noise Ratio)**: 이미지 품질 측정
- **SSIM (Structural Similarity Index)**: 구조적 유사성 측정
- **Validation Loss**: Combined Loss 값

---

## 5. 코드 구조

### 5.1 주요 클래스 및 함수

#### CNN 모델 (`cnn_model.py`)
- `CNNModel`: CNN 모델 클래스
- `SSIMLoss`: SSIM 손실 함수
- `CombinedLoss`: 조합 손실 함수
- `ImageDataset`: 데이터셋 클래스
- `EarlyStopping`: 조기 종료 클래스
- `train_model()`: 학습 함수
- `evaluate_model()`: 평가 함수

#### U-Net 모델 (`unet_model.py`)
- `UNet`: U-Net 모델 클래스
- `train_model()`: 학습 함수
- `evaluate_model()`: 평가 함수
- (SSIMLoss, CombinedLoss 등은 cnn_model에서 import)

### 5.2 학습 스크립트

- `train_cnn_combined.py`: CNN 모델 학습 스크립트
- `train_unet_combined.py`: U-Net 모델 학습 스크립트

두 스크립트 모두 동일한 구조를 가지며, 모델만 다릅니다.

---

## 6. 하이퍼파라미터 요약

### 6.1 공통 하이퍼파라미터

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| Learning Rate | 5e-5 | Adam optimizer 학습률 |
| Batch Size | 2 | 배치 크기 |
| Epochs | 50 | 최대 에포크 수 |
| L1 Weight | 1.0 | L1 Loss 가중치 |
| SSIM Weight | 1.0 | SSIM Loss 가중치 |
| Gradient Weight | 0.5 | Gradient Loss 가중치 |
| Early Stopping Patience | 5 | 조기 종료 patience |
| Gradient Clipping | 10.0 | Gradient norm 제한 |
| Image Size | 512×512 | 입력 이미지 크기 |

### 6.2 모델별 하이퍼파라미터

#### CNN
- Encoder 채널: 64 → 128 → 256 → 512
- Decoder 채널: 512 → 256 → 128 → 64 → 3
- Kernel Size: 3×3 (모든 레이어)
- Padding: 1

#### U-Net
- Base Channels: 64
- Encoder 채널: 64 → 128 → 256 → 512 → 1024
- Decoder 채널: 1024 → 512 → 256 → 128 → 64 → 3
- MaxPool: 2×2, stride=2
- Transposed Conv: kernel=2, stride=2

---

이 문서는 보고서에 포함할 수 있도록 모델의 구조, 학습 방법, 파라미터 등을 상세히 설명합니다.

