# Streamlit Cloud 배포 가이드

## 모델을 사이트에 포함시키는 방법

사용자가 방문하면 모델이 이미 준비되어 있도록 하려면, Streamlit Cloud에 배포할 때 모델을 포함시켜야 합니다.

## 방법 1: Git LFS 사용 (권장)

### 1. Git LFS 설치 및 설정

```bash
# Git LFS 설치 (한 번만)
git lfs install

# .pth 파일을 LFS로 추적
git lfs track "*.pth"
git lfs track "checkpoints/**/*.pth"
git lfs track "second_checkpoints/**/*.pth"

# .gitattributes 파일 커밋
git add .gitattributes
git commit -m "Add Git LFS tracking for model files"
```

### 2. 모델 파일 추가

```bash
# 모델 파일을 Git에 추가
git add checkpoints/
git add second_checkpoints/
git commit -m "Add model checkpoints"
git push
```

### 3. Streamlit Cloud 배포

1. GitHub 저장소에 푸시
2. Streamlit Cloud에서 저장소 연결
3. Main file path: `team/streamlit_app.py`
4. Deploy!

**장점**: 모델이 저장소에 포함되어 자동으로 배포됨
**단점**: Git LFS는 유료 플랜에서만 무제한 사용 가능

## 방법 2: Hugging Face Hub 사용 (추천)

### 1. Hugging Face에 모델 업로드

```python
# upload_models_to_hf.py
from huggingface_hub import HfApi, login

# Hugging Face 로그인
login()

api = HfApi()

# 모델 업로드
api.upload_folder(
    folder_path="checkpoints",
    repo_id="your-username/image-denoising-cnn",
    repo_type="model"
)

api.upload_folder(
    folder_path="second_checkpoints",
    repo_id="your-username/image-denoising-unet",
    repo_type="model"
)
```

### 2. streamlit_app.py 수정

```python
from huggingface_hub import hf_hub_download

# Hugging Face에서 모델 자동 다운로드
@st.cache_resource
def load_model_from_hf(model_type):
    """Hugging Face Hub에서 모델을 자동으로 다운로드하고 로드"""
    if model_type == "CNN":
        model_path = hf_hub_download(
            repo_id="your-username/image-denoising-cnn",
            filename="cnn/best_model.pth",
            cache_dir=cache_dir
        )
    else:
        model_path = hf_hub_download(
            repo_id="your-username/image-denoising-unet",
            filename="unet/best_model.pth",
            cache_dir=cache_dir
        )
    return model_path
```

**장점**: 무료, 버전 관리, 자동 캐싱
**단점**: 초기 설정 필요

## 방법 3: 현재 방식 (자동 다운로드)

현재 구현된 방식은 Google Drive에서 자동으로 다운로드합니다.

**장점**: 설정 간단
**단점**: 매번 다운로드 필요 (캐시되지만 Streamlit Cloud는 임시 파일 시스템 사용)

## 추천: 방법 2 (Hugging Face Hub)

가장 실용적인 방법은 Hugging Face Hub를 사용하는 것입니다:

1. **무료**: 개인/공개 저장소 무료
2. **자동 캐싱**: 한 번 다운로드하면 캐시됨
3. **버전 관리**: 모델 버전 관리 가능
4. **빠른 다운로드**: CDN 사용으로 빠름

### Hugging Face Hub 통합 코드 예시

```python
# requirements.txt에 추가
# huggingface_hub>=0.20.0

# streamlit_app.py에 추가
from huggingface_hub import hf_hub_download

# 모델 경로를 Hugging Face에서 가져오기
HF_REPO_CNN = "your-username/image-denoising-cnn"
HF_REPO_UNET = "your-username/image-denoising-unet"

@st.cache_resource
def get_model_path_from_hf(model_type):
    """Hugging Face에서 모델 경로 가져오기 (자동 다운로드)"""
    try:
        if model_type == "CNN":
            model_path = hf_hub_download(
                repo_id=HF_REPO_CNN,
                filename="cnn/best_model.pth",
                cache_dir=str(cache_dir)
            )
        else:
            model_path = hf_hub_download(
                repo_id=HF_REPO_UNET,
                filename="unet/best_model.pth",
                cache_dir=str(cache_dir)
            )
        return model_path
    except Exception as e:
        st.error(f"모델 다운로드 실패: {e}")
        return None
```

## 빠른 시작 (Hugging Face Hub)

1. Hugging Face 계정 생성: https://huggingface.co
2. 새 모델 저장소 생성
3. 모델 업로드 (웹 UI 또는 Python)
4. `streamlit_app.py`에 Hugging Face Hub 통합 코드 추가
5. 배포!

이렇게 하면 사용자가 방문하면 모델이 자동으로 준비됩니다! 🚀

