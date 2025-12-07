"""
Streamlit 앱: 이미지 노이즈 제거 시각화
lastcheckpoints에 저장된 모델을 사용합니다.
"""
import streamlit as st
import torch
import cv2
import numpy as np
from pathlib import Path
import time
from PIL import Image
import io
import sys
import os

# 프로젝트 루트 경로 추가 (second_checkpoints 접근용)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent))

# 모델 import (경로에 따라 자동으로 찾기)
try:
    from team.cnn_model import CNNModel
    from team.unet_model import UNet
except ImportError:
    # team 폴더에서 직접 실행하는 경우
    from cnn_model import CNNModel
    from unet_model import UNet

# 페이지 설정
st.set_page_config(
    page_title="이미지 노이즈 제거",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일링
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        font-size: 1.2rem;
        height: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# 제목
st.markdown('<p class="main-header">✨ 이미지 노이즈 제거</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">딥러닝 모델을 사용한 실시간 노이즈 제거 시연</p>', unsafe_allow_html=True)
st.markdown("---")

# 모델 경로 설정 (lastcheckpoints 사용)
# team/lastcheckpoints 폴더에 있는 모델 사용
team_dir = Path(__file__).parent
lastcheckpoints_dir = team_dir / "lastcheckpoints"
cnn_model_path = lastcheckpoints_dir / "cnn" / "best_model.pth"
unet_model_path = lastcheckpoints_dir / "unet" / "best_model.pth"

cnn_exists = cnn_model_path.exists()
unet_exists = unet_model_path.exists()

# 모델 정보는 표시하지 않음 (사용자에게 보이지 않게)

models_ready = cnn_exists and unet_exists

# 디바이스 선택 (사이드바에 표시하지 않음, 자동으로 설정)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 로드 함수 (캐싱)
@st.cache_resource
def load_model(model_type, model_path, device):
    """모델을 로드하고 캐시합니다"""
    try:
        if model_type == "CNN":
            model = CNNModel(in_channels=3, out_channels=3)
        else:
            model = UNet(in_channels=3, out_channels=3)
        
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        # 모델 정보 추출
        info = {
            'epoch': checkpoint.get('epoch', 'N/A'),
            'val_loss': checkpoint.get('val_loss', 'N/A'),
            'val_psnr': checkpoint.get('val_psnr', 'N/A'),
            'val_ssim': checkpoint.get('val_ssim', 'N/A')
        }
        
        return model, info
    except FileNotFoundError:
        st.error(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
        return None, None
    except Exception as e:
        st.error(f"❌ 모델 로드 실패: {str(e)}")
        return None, None

# 이미지 전처리 함수
def preprocess_image(image, target_size=(512, 512)):
    """이미지를 모델 입력 형식으로 변환"""
    # PIL Image를 numpy array로 변환
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image
    
    # RGB로 변환
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    # 원본 크기 저장 (나중에 복원용)
    original_size = img_array.shape[:2]
    
    # 모델이 기대하는 크기로 리사이즈 (512x512)
    if img_array.shape[:2] != target_size:
        img_array = cv2.resize(img_array, target_size, interpolation=cv2.INTER_LINEAR)
    
    # [0, 1] 범위로 정규화
    img_array = img_array.astype(np.float32) / 255.0
    
    # 텐서로 변환: (H, W, C) -> (1, C, H, W)
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
    
    return img_tensor, original_size

# 노이즈 제거 함수
def denoise_image(model, image_tensor, device, original_size=None):
    """이미지에서 노이즈를 제거합니다"""
    model.eval()
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
        output = output.squeeze(0).cpu().numpy()
        output = output.transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
        output = np.clip(output, 0, 1)
        
        # 원본 크기로 복원 (필요한 경우)
        if original_size is not None and output.shape[:2] != original_size:
            output = cv2.resize(output, (original_size[1], original_size[0]), interpolation=cv2.INTER_LINEAR)
    
    return output

# 메인 컨텐츠
st.subheader("📤 입력 이미지")

# 모델이 없을 때 안내 (최소화 - Hugging Face가 자동으로 처리)
if not models_ready:
    st.info("⏳ 모델을 준비하는 중입니다. 잠시만 기다려주세요...")

# 이미지 업로드
uploaded_file = st.file_uploader(
    "노이즈가 있는 이미지를 업로드하세요",
    type=['png', 'jpg', 'jpeg'],
    help="JPG, PNG 형식의 이미지를 업로드할 수 있습니다",
    disabled=not models_ready  # 모델이 없으면 업로드 비활성화
)

# 이미지 표시
input_image = None
if uploaded_file is not None:
    input_image = Image.open(uploaded_file)
    st.image(input_image, caption="업로드된 이미지", use_container_width=True)
    
    # 이미지 정보
    st.info(f"📏 크기: {input_image.size[0]} × {input_image.size[1]} pixels")
    
    # 노이즈 제거 실행 버튼
    if st.button("🚀 CNN & U-Net 동시 실행", type="primary", use_container_width=True):
        if not models_ready:
            st.error("모델 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
        else:
            # 두 모델 모두 로드
            with st.spinner("모델 로딩 중..."):
                cnn_model, cnn_info = load_model("CNN", cnn_model_path, device)
                unet_model, unet_info = load_model("U-Net", unet_model_path, device)
            
            if cnn_model is None or unet_model is None:
                st.error("모델을 로드할 수 없습니다.")
            else:
                # 이미지 전처리 (한 번만)
                image_tensor, original_size = preprocess_image(input_image)
                
                # CNN 처리
                with st.spinner("CNN 모델 처리 중..."):
                    cnn_start = time.time()
                    cnn_result = denoise_image(cnn_model, image_tensor, device, original_size)
                    cnn_time = time.time() - cnn_start
                
                # U-Net 처리
                with st.spinner("U-Net 모델 처리 중..."):
                    unet_start = time.time()
                    unet_result = denoise_image(unet_model, image_tensor, device, original_size)
                    unet_time = time.time() - unet_start
                
                # 결과를 3열로 표시
                st.markdown("---")
                st.subheader("✨ 노이즈 제거 결과 비교")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.image(input_image, caption="원본 (노이즈 있음)", use_container_width=True)
                
                with col2:
                    st.image(cnn_result, caption="CNN 결과", use_container_width=True)
                    st.metric("처리 시간", f"{cnn_time*1000:.1f} ms")
                    
                    # CNN 다운로드
                    cnn_pil = Image.fromarray((cnn_result * 255).astype(np.uint8))
                    cnn_buf = io.BytesIO()
                    cnn_pil.save(cnn_buf, format='PNG')
                    cnn_buf.seek(0)
                    st.download_button(
                        label="📥 CNN 결과 다운로드",
                        data=cnn_buf,
                        file_name=f"denoised_cnn_{int(time.time())}.png",
                        mime="image/png",
                        key="cnn_download"
                    )
                
                with col3:
                    st.image(unet_result, caption="U-Net 결과", use_container_width=True)
                    st.metric("처리 시간", f"{unet_time*1000:.1f} ms")
                    
                    # U-Net 다운로드
                    unet_pil = Image.fromarray((unet_result * 255).astype(np.uint8))
                    unet_buf = io.BytesIO()
                    unet_pil.save(unet_buf, format='PNG')
                    unet_buf.seek(0)
                    st.download_button(
                        label="📥 U-Net 결과 다운로드",
                        data=unet_buf,
                        file_name=f"denoised_unet_{int(time.time())}.png",
                        mime="image/png",
                        key="unet_download"
                    )
                
                # 모델 정보는 표시하지 않음
                
                # 세션 상태에 결과 저장
                st.session_state['cnn_result'] = cnn_result
                st.session_state['unet_result'] = unet_result
                st.session_state['input_image'] = input_image
else:
    st.info("👆 위에서 이미지를 업로드하세요")

# 추가 비교 뷰 (결과가 있을 때)
if 'cnn_result' in st.session_state and 'unet_result' in st.session_state and 'input_image' in st.session_state:
    st.markdown("---")
    st.subheader("📊 상세 비교")
    
    # 슬라이더로 확대/축소 가능한 비교 뷰
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.image(st.session_state['input_image'], caption="원본", use_container_width=True)
    
    with col2:
        st.image(st.session_state['cnn_result'], caption="CNN 결과", use_container_width=True)
    
    with col3:
        st.image(st.session_state['unet_result'], caption="U-Net 결과", use_container_width=True)

# 푸터
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; padding: 1rem;'>이미지 노이즈 제거 시각화 도구 | 딥러닝 기반</div>",
    unsafe_allow_html=True
)

