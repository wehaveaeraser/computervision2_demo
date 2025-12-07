# Git LFS로 모델 포함하기 (Hugging Face 대안)

## 방법: Git LFS 사용

모델을 GitHub 저장소에 직접 포함시켜서 Streamlit Cloud에 배포할 때 자동으로 포함되게 합니다.

## 1. Git LFS 설치

```bash
# Git LFS 설치 (한 번만)
git lfs install
```

## 2. .gitignore 수정

`.gitignore` 파일에서 모델 파일 제외를 제거하거나 주석 처리:

```gitignore
# 모델 체크포인트 파일 (Git LFS로 관리)
# *.pth  <- 주석 처리
# checkpoints/  <- 주석 처리
# second_checkpoints/  <- 주석 처리
```

또는 특정 모델만 포함:

```gitignore
# 모델 체크포인트 파일 (Git LFS로 관리)
# *.pth는 Git LFS로 추적
!checkpoints/cnn/best_model.pth
!second_checkpoints/unet/best_model.pth
```

## 3. Git LFS 추적 설정

```bash
# .pth 파일을 LFS로 추적
git lfs track "*.pth"
git lfs track "checkpoints/**/*.pth"
git lfs track "second_checkpoints/**/*.pth"

# .gitattributes 파일 커밋
git add .gitattributes
git commit -m "Add Git LFS tracking for model files"
```

## 4. 모델 파일 추가

```bash
# 모델 파일을 Git에 추가
git add checkpoints/cnn/best_model.pth
git add second_checkpoints/unet/best_model.pth
git commit -m "Add model checkpoints with Git LFS"
git push
```

## 5. Streamlit Cloud 배포

1. GitHub 저장소에 푸시 완료
2. Streamlit Cloud에서 저장소 연결
3. Main file path: `team/streamlit_app.py`
4. Deploy!

## 장점

- ✅ 모델이 저장소에 포함되어 자동으로 배포됨
- ✅ 사용자가 다운로드할 필요 없음
- ✅ 처음 방문 시에도 모델이 준비됨
- ✅ 외부 서비스 불필요

## 주의사항

- Git LFS 무료 플랜: 1GB 저장소, 1GB 대역폭/월
- 모델이 1GB 이하면 무료로 사용 가능
- 모델이 크면 유료 플랜 필요할 수 있음

## 대안: 모델 압축

모델이 너무 크면 압축해서 포함:

```bash
# 모델 압축
tar -czf models.tar.gz checkpoints/ second_checkpoints/

# 압축 파일은 Git LFS 없이도 포함 가능 (100MB 이하)
git add models.tar.gz
```

그리고 앱에서 압축 해제:

```python
import tarfile
import os

if not os.path.exists("checkpoints"):
    with tarfile.open("models.tar.gz", "r:gz") as tar:
        tar.extractall()
```

