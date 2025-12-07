# 모델 기능 확인 요약

## ✅ 적용 완료 사항

### 1. Residual Learning (잔차 학습)

**CNN 모델 (`cnn_model.py`)**
- ✅ 적용 완료
- `use_residual=True` 파라미터 추가 (기본값: True)
- 모델이 노이즈만 예측하고 입력에서 빼서 clean 이미지 생성
- `out = x - tanh(residual)` 형태로 구현
- 입력의 밝기/구조를 유지하면서 노이즈만 제거

**U-Net 모델 (`unet_model.py`)**
- ✅ 적용 완료
- `use_residual=True` 파라미터 추가 (기본값: True)
- 모델이 노이즈만 예측하고 입력에서 빼서 clean 이미지 생성
- `out = x - tanh(residual)` 형태로 구현
- 입력의 밝기/구조를 유지하면서 노이즈만 제거

**적용 효과:**
- 모델이 전체 이미지를 재구성하지 않고 노이즈만 학습 (작은 값 예측)
- 입력의 밝기/구조를 유지하면서 노이즈만 제거
- 학습 난이도 감소 (identity mapping이 기본)
- 밝기 편향 문제 완화

### 2. Loss 가중치 튜닝

**CNN 모델 (`cnn_model.py`)**
- ✅ 적용 완료
- `train_model()` 함수에 `l1_weight`, `ssim_weight` 파라미터 추가
- `train_cnn.py`에 `--l1_weight`, `--ssim_weight` 옵션 추가
- 기본값: L1=1.0, SSIM=1.0

**U-Net 모델 (`unet_model.py`)**
- ✅ 적용 완료
- `train_model()` 함수에 `l1_weight`, `ssim_weight` 파라미터 추가
- `train_unet.py`에 `--l1_weight`, `--ssim_weight` 옵션 추가
- 기본값: L1=1.0, SSIM=1.0

**사용 예시:**
```bash
# L1 Loss에 더 높은 가중치 부여
python train_cnn.py --l1_weight 2.0 --ssim_weight 1.0 ...

# SSIM Loss에 더 높은 가중치 부여
python train_cnn.py --l1_weight 1.0 --ssim_weight 2.0 ...
```

### 3. 자동 매칭 기능

**ImageDataset 클래스 (`cnn_model.py`)**
- ✅ 적용 완료
- `auto_match` 파라미터 추가
- 학습 시 자동으로 폴더명 기반 매칭 수행
- `unet_model.py`는 `cnn_model.py`의 ImageDataset을 import하므로 자동으로 사용 가능

**학습 스크립트**
- ✅ `train_cnn.py`에 `--auto_match` 옵션 추가
- ✅ `train_unet.py`에 `--auto_match` 옵션 추가

## 📋 확인 체크리스트

### CNN 모델
- [x] Residual Learning 적용
- [x] Loss 가중치 튜닝 가능
- [x] 자동 매칭 기능 지원

### U-Net 모델
- [x] Residual Learning 적용
- [x] Loss 가중치 튜닝 가능
- [x] 자동 매칭 기능 지원 (cnn_model의 ImageDataset 사용)

## 🔍 코드 확인 위치

### Residual Learning 구현
- **CNN**: `cnn_model.py`의 `CNNModel.forward()` 메서드
- **U-Net**: `unet_model.py`의 `UNet.forward()` 메서드

### Loss 가중치 튜닝
- **CNN**: `cnn_model.py`의 `train_model()` 함수
- **U-Net**: `unet_model.py`의 `train_model()` 함수
- **학습 스크립트**: `train_cnn.py`, `train_unet.py`의 argument parser

### 자동 매칭
- **ImageDataset**: `cnn_model.py`의 `ImageDataset.__init__()` 메서드
- **학습 스크립트**: `train_cnn.py`, `train_unet.py`의 argument parser

## 💡 사용 팁

### Residual Learning 비활성화 (기존 방식 사용)
```python
# CNN
model = CNNModel(use_residual=False)

# U-Net
model = UNet(use_residual=False)
```

### Loss 가중치 튜닝 예시
```bash
# L1에 더 집중 (픽셀 단위 정확도)
python train_cnn.py --l1_weight 3.0 --ssim_weight 1.0 ...

# SSIM에 더 집중 (구조적 유사성)
python train_cnn.py --l1_weight 1.0 --ssim_weight 3.0 ...

# 균형잡힌 설정
python train_cnn.py --l1_weight 2.0 --ssim_weight 1.5 ...
```

