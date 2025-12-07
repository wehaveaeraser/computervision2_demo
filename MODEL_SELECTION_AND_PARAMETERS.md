# 모델 선택 및 파라미터 설명

## 1. 모델 선택 이유

### 1.1 CNN (Baseline 모델)

**선택 이유:**
- **베이스라인 모델**로서 기본적인 성능을 제공
- 비교적 단순한 구조로 구현 및 이해가 용이
- 학습 속도가 빠르고 메모리 사용량이 적음
- Encoder-Decoder 구조로 이미지 복원 작업에 적합

**구조 특징:**
- 4-layer CNN with Encoder-Decoder 구조
- 채널 수: 64 → 128 → 256 → 512 (Encoder) → 256 → 128 → 64 → 3 (Decoder)
- 각 레이어에 Batch Normalization과 ReLU 활성화 함수 적용

### 1.2 U-Net (비교 모델)

**선택 이유:**
- **자율주행 분야에서 노이즈 제거에 널리 사용되는 모델**
  - 자율주행 환경에서 비, 눈, 안개 등 날씨 노이즈 제거에 효과적
  - 실제 자율주행 시스템에서 검증된 성능
  - 이미지 세그멘테이션뿐만 아니라 이미지 복원 작업에도 우수한 성능
- **Skip Connection을 통한 특징 보존**
  - Encoder에서 추출한 저수준 특징을 Decoder에 직접 전달
  - 세부 정보 손실 최소화
  - 노이즈 제거 시 원본 이미지의 구조와 텍스처 보존에 유리
- **Encoder-Decoder 구조의 표준 모델**
  - 의료 영상, 위성 영상, 자율주행 등 다양한 분야에서 활용
  - 이미지 복원 작업에 최적화된 아키텍처

**자율주행 분야에서의 U-Net 활용:**
- 비/눈/안개 제거: 자율주행 차량의 카메라 영상에서 날씨 노이즈 제거
- 저조도 이미지 향상: 야간 주행 시 이미지 밝기 및 선명도 개선
- 센서 융합: LiDAR, 카메라 등 다양한 센서 데이터 처리
- 실시간 처리: 경량화된 U-Net 변형을 통한 실시간 노이즈 제거

## 2. 모델 구조 상세 설명

### 2.1 CNN 모델 구조

```
입력 (3 channels, 512×512)
    ↓
[Conv2d(3→64) + BN + ReLU]
    ↓
[Conv2d(64→128) + BN + ReLU]
    ↓
[Conv2d(128→256) + BN + ReLU]
    ↓
[Conv2d(256→512) + BN + ReLU]  ← Encoder 끝
    ↓
[Conv2d(512→256) + BN + ReLU]
    ↓
[Conv2d(256→128) + BN + ReLU]
    ↓
[Conv2d(128→64) + BN + ReLU]
    ↓
[Conv2d(64→3)]  ← 노이즈 예측
    ↓
Residual Learning: output = input - tanh(noise_prediction)
    ↓
출력 (3 channels, 512×512)
```

**주요 특징:**
- **Residual Learning 적용**: 모델이 노이즈만 예측하고 입력에서 빼서 clean 이미지 생성
- **Batch Normalization**: 각 레이어마다 배치 정규화로 학습 안정성 향상
- **채널 수 증가**: Encoder에서 채널 수를 점진적으로 증가시켜 특징 추출 능력 향상

**파라미터 수:**
- 총 파라미터 수: 약 3.5M (모델 크기에 따라 다름)

### 2.2 U-Net 모델 구조

```
입력 (3 channels, 512×512)
    ↓
[Encoder Block 1: 3→64 channels]
    ↓ MaxPool
[Encoder Block 2: 64→128 channels]
    ↓ MaxPool
[Encoder Block 3: 128→256 channels]
    ↓ MaxPool
[Encoder Block 4: 256→512 channels]
    ↓ MaxPool
[Bottleneck: 512→1024 channels]
    ↓
[UpConv + Skip Connection] ← Encoder Block 4와 연결
[Decoder Block 4: 1024→512 channels]
    ↓
[UpConv + Skip Connection] ← Encoder Block 3과 연결
[Decoder Block 3: 512→256 channels]
    ↓
[UpConv + Skip Connection] ← Encoder Block 2와 연결
[Decoder Block 2: 256→128 channels]
    ↓
[UpConv + Skip Connection] ← Encoder Block 1과 연결
[Decoder Block 1: 128→64 channels]
    ↓
[Conv2d(64→3)]  ← 노이즈 예측
    ↓
Residual Learning: output = input - tanh(noise_prediction)
    ↓
출력 (3 channels, 512×512)
```

**주요 특징:**
- **Skip Connection (U-Net의 핵심)**: Encoder의 각 레이어 출력을 Decoder의 대응 레이어에 직접 연결
  - 저수준 특징(엣지, 텍스처) 보존
  - 고수준 특징(객체, 구조)과 저수준 특징의 융합
- **Encoder-Decoder 구조**: 이미지를 압축 후 복원하여 노이즈 제거
- **Residual Learning 적용**: CNN과 동일하게 노이즈만 예측
- **각 Encoder/Decoder Block**: Conv2d + BN + ReLU를 2번 반복

**파라미터 수:**
- 총 파라미터 수: 약 31M (CNN보다 약 9배 많음)
- Skip Connection으로 인해 더 많은 메모리 사용

## 3. 학습 파라미터 설명

### 3.1 현재 사용 중인 파라미터

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| **손실 함수** | L1 Loss + SSIM Loss + Gradient Loss | 다중 손실 함수 조합 |
| - L1 Loss 가중치 | 1.0 | 픽셀 단위 차이 측정 |
| - SSIM Loss 가중치 | 1.0 | 구조적 유사성 측정 |
| - Gradient Loss 가중치 | 0.5 | 엣지/경계 보존 |
| **Optimizer** | Adam | 적응적 학습률 최적화 |
| **Learning Rate** | 5e-5 | 학습률 (원래 1e-4에서 변경) |
| **Batch Size** | 2 | 배치 크기 (원래 16에서 변경) |
| **Epochs** | 30 | 에포크 수 (원래 50에서 변경) |
| **Early Stopping** | Patience=5 | 검증 손실이 5 epoch 동안 개선되지 않으면 중단 |
| **Residual Learning** | 적용 | 모델이 노이즈만 예측 |
| **Mixed Precision** | 활성화 | GPU 메모리 절약 및 학습 속도 향상 |

### 3.2 원래 계획했던 파라미터

| 파라미터 | 원래 계획 | 현재 사용 | 변경 이유 |
|---------|---------|----------|---------|
| **손실 함수** | L1 Loss + SSIM Loss | L1 + SSIM + Gradient Loss | Gradient Loss 추가로 엣지 보존 개선 |
| **Optimizer** | Adam | Adam | 동일 |
| **Learning Rate** | 1e-4 | 5e-5 | 학습 안정성 향상 |
| **Batch Size** | 16 | 2 | GPU 메모리 제약 |
| **Epochs** | 50 | 30 | 학습 시간 단축 |
| **Early Stopping** | 적용 | 적용 (Patience=5) | 동일 |
| **Residual Learning** | 미적용 (계획) | 적용 | 성능 향상 |
| **Skip Connection 비교** | 계획 | U-Net에 기본 적용 | U-Net의 핵심 기능 |

## 4. 파라미터 변경 이유 및 성능 향상 분석

### 4.1 손실 함수 변경: L1+SSIM → L1+SSIM+Gradient Loss

**변경 이유:**
- 초기 실험에서 **노이즈 제거보다 밝기만 올라가는 문제** 발생
- L1 Loss와 SSIM Loss만으로는 엣지와 경계 정보가 손실됨
- 노이즈는 고주파 성분이므로 gradient 정보를 활용하여 더 정확한 제거 가능

**Gradient Loss 추가 효과:**
- **엣지 보존**: Sobel 필터를 사용하여 이미지의 gradient(경계) 차이를 측정
- **고주파 노이즈 제거**: 노이즈는 고주파 성분이므로 gradient loss로 더 효과적으로 제거
- **구조적 일관성**: 원본 이미지의 구조와 경계를 보존하면서 노이즈만 제거

**성능 향상 이유:**
- Gradient Loss가 모델에게 엣지 정보를 보존하도록 학습 유도
- L1 Loss는 픽셀 단위 차이만 측정하지만, Gradient Loss는 구조적 차이도 고려
- SSIM Loss와 함께 사용하여 픽셀, 구조, 엣지 정보를 모두 보존

### 4.2 Residual Learning 적용

**변경 이유:**
- 원래는 모델이 전체 clean 이미지를 직접 예측하도록 학습
- 하지만 학습 결과 **밝기만 올라가고 노이즈는 제거되지 않는 문제** 발생
- Residual Learning을 적용하여 모델이 **노이즈만 예측**하도록 변경

**Residual Learning 작동 원리:**
```python
# 모델이 노이즈만 예측
noise_prediction = model(noisy_image)
noise_prediction = tanh(noise_prediction)  # [-1, 1] 범위로 정규화

# 입력에서 노이즈를 빼서 clean 이미지 생성
clean_image = noisy_image - noise_prediction
clean_image = clamp(clean_image, 0, 1)  # [0, 1] 범위로 클리핑
```

**성능 향상 이유:**
1. **학습 난이도 감소**: 전체 이미지를 재구성하는 것보다 노이즈만 예측하는 것이 더 쉬움
2. **밝기 보존**: 입력 이미지의 밝기를 그대로 유지하면서 노이즈만 제거
3. **Identity Mapping**: 모델이 아무것도 하지 않으면 입력 그대로 출력 (기본 동작)
4. **수렴 속도 향상**: 작은 값(노이즈)을 예측하므로 학습이 더 빠르게 수렴

### 4.3 Learning Rate 변경: 1e-4 → 5e-5

**변경 이유:**
- 초기 학습률 1e-4는 너무 커서 학습이 불안정함
- 손실 함수가 발산하거나 진동하는 현상 발생
- Residual Learning과 Gradient Loss를 추가한 후 더 작은 학습률이 필요

**성능 향상 이유:**
- **학습 안정성**: 더 작은 학습률로 안정적인 학습 가능
- **세밀한 조정**: 복잡한 손실 함수(L1+SSIM+Gradient)에서 각 항의 균형을 맞추기 위해 필요
- **수렴 개선**: 더 작은 학습률로 최적해에 더 정확하게 수렴

### 4.4 Batch Size 변경: 16 → 2

**변경 이유:**
- GPU 메모리 제약 (특히 U-Net은 파라미터가 많아 메모리를 많이 사용)
- Skip Connection으로 인해 중간 특징 맵을 저장해야 하므로 메모리 사용량 증가
- Mixed Precision Training을 사용하더라도 batch size 16은 메모리 부족

**성능 영향:**
- **학습 안정성**: 작은 batch size는 gradient 추정의 분산이 커서 학습이 불안정할 수 있음
- **하지만**: Gradient Clipping과 Mixed Precision으로 안정성 확보
- **실용성**: 메모리 제약을 고려한 현실적인 선택

### 4.5 Epochs 변경: 50 → 30

**변경 이유:**
- 학습 시간이 너무 오래 걸림 (50 epoch 학습 시 수 시간 소요)
- Early Stopping (Patience=5)이 적용되어 실제로는 30 epoch 이전에 종료되는 경우가 많음
- Residual Learning과 적절한 손실 함수로 빠르게 수렴

**성능 영향:**
- **충분한 학습**: Early Stopping으로 최적 시점에 학습 종료
- **시간 효율성**: 불필요한 학습 시간 절약
- **과적합 방지**: 더 적은 epoch로 과적합 위험 감소

## 5. 최종 파라미터 설정의 장점

### 5.1 다중 손실 함수 (L1 + SSIM + Gradient)
- **L1 Loss**: 픽셀 단위 정확도 보장
- **SSIM Loss**: 구조적 유사성 보장
- **Gradient Loss**: 엣지 및 경계 정보 보존
- **균형잡힌 학습**: 세 가지 손실 함수의 조합으로 종합적인 성능 향상

### 5.2 Residual Learning
- **효율적인 학습**: 노이즈만 예측하여 학습 난이도 감소
- **밝기 보존**: 입력 이미지의 밝기를 유지하면서 노이즈만 제거
- **빠른 수렴**: 작은 값 예측으로 학습 속도 향상

### 5.3 적응적 학습률 (5e-5)
- **안정적인 학습**: 복잡한 손실 함수에서도 안정적으로 학습
- **세밀한 조정**: 각 손실 항의 균형을 맞추기 위한 적절한 학습률

### 5.4 Early Stopping
- **과적합 방지**: 검증 손실이 개선되지 않으면 조기 종료
- **시간 효율성**: 불필요한 학습 시간 절약
- **최적 모델 선택**: 검증 손실이 가장 낮은 시점의 모델 저장

## 6. 모델 비교 요약

| 특징 | CNN | U-Net |
|------|-----|-------|
| **구조** | 단순 Encoder-Decoder | Encoder-Decoder + Skip Connection |
| **파라미터 수** | 약 3.5M | 약 31M |
| **메모리 사용량** | 낮음 | 높음 |
| **학습 속도** | 빠름 | 느림 |
| **추론 속도** | 빠름 | 느림 |
| **성능** | 기본 성능 | 우수한 성능 (세부 정보 보존) |
| **적용 분야** | 일반적인 노이즈 제거 | 자율주행, 의료 영상 등 |

## 7. 결론

본 연구에서는 **CNN을 베이스라인 모델**로, **U-Net을 비교 모델**로 선택하여 자율주행 환경의 노이즈 제거 성능을 비교하였다. 

**주요 개선 사항:**
1. **Gradient Loss 추가**: 엣지 보존을 통한 노이즈 제거 성능 향상
2. **Residual Learning 적용**: 밝기 보존 및 학습 효율성 향상
3. **학습률 조정**: 안정적인 학습을 위한 학습률 감소
4. **실용적 파라미터 설정**: GPU 메모리 제약을 고려한 batch size 및 epoch 조정

이러한 변경을 통해 **노이즈 제거 성능이 크게 향상**되었으며, 특히 **밝기만 올라가는 문제를 해결**하고 **실제 노이즈 제거 성능을 개선**할 수 있었다.

