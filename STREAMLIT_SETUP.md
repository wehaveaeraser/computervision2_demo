# Streamlit Cloud 배포 시 모델 자동 준비 방법

## 목표
사용자가 처음 방문해도 모델이 이미 준비되어 있어서 바로 사용할 수 있게 하기

## 방법: Hugging Face Hub 사용 (권장)

### 1. Hugging Face 계정 생성 및 모델 업로드

```bash
# Hugging Face CLI 설치
pip install huggingface_hub

# 로그인
huggingface-cli login
```

### 2. 모델 업로드 스크립트 생성

`upload_models_to_hf.py` 파일 생성:

```python
from huggingface_hub import HfApi, login

# 로그인
login()

api = HfApi()

# CNN 모델 업로드
api.upload_folder(
    folder_path="checkpoints/cnn",
    repo_id="your-username/image-denoising-cnn",  # 본인의 Hugging Face 사용자명으로 변경
    repo_type="model",
    create_pr=False
)

# U-Net 모델 업로드
api.upload_folder(
    folder_path="second_checkpoints/unet",
    repo_id="your-username/image-denoising-unet",  # 본인의 Hugging Face 사용자명으로 변경
    repo_type="model",
    create_pr=False
)

print("✅ 모델 업로드 완료!")
```

### 3. 모델 업로드 실행

```bash
python upload_models_to_hf.py
```

### 4. streamlit_app.py에 Hugging Face 저장소 ID 설정

`team/streamlit_app.py` 파일에서:

```python
# 75-76번째 줄 수정
HF_REPO_CNN = "your-username/image-denoising-cnn"  # 실제 저장소 ID로 변경
HF_REPO_UNET = "your-username/image-denoising-unet"  # 실제 저장소 ID로 변경
```

### 5. Streamlit Cloud에 배포

1. GitHub에 코드 푸시
2. Streamlit Cloud에서 저장소 연결
3. Main file path: `team/streamlit_app.py`
4. Deploy!

## 결과

- ✅ 사용자가 처음 방문해도 모델이 자동으로 준비됨
- ✅ 다운로드 과정이 보이지 않음 (백그라운드에서 자동 처리)
- ✅ Hugging Face 캐싱으로 빠른 로드
- ✅ 무료로 사용 가능

## 참고

- Hugging Face Hub는 무료이며, 공개 저장소는 제한 없이 사용 가능
- 모델은 자동으로 캐싱되어 두 번째 방문부터는 더 빠름
- `@st.cache_resource` 데코레이터로 모델이 메모리에 캐싱됨

