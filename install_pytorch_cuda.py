"""
PyTorch CUDA 버전 설치 가이드 스크립트
"""
import subprocess
import sys

def install_pytorch_cuda():
    """
    PyTorch CUDA 버전 설치
    CUDA 12.6은 CUDA 12.1/12.4와 호환됩니다
    """
    print("=" * 60)
    print("PyTorch CUDA 버전 설치")
    print("=" * 60)
    print("\n현재 PyTorch CPU 버전을 제거하고 CUDA 버전을 설치합니다.")
    print("CUDA 12.1 또는 12.4 버전을 설치합니다.\n")
    
    # 현재 PyTorch 제거
    print("1단계: 현재 PyTorch 제거 중...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "torch", "torchvision", "torchaudio", "-y"])
        print("✓ PyTorch 제거 완료\n")
    except Exception as e:
        print(f"⚠️  제거 중 오류 (계속 진행): {e}\n")
    
    # PyTorch CUDA 12.1 설치 (CUDA 12.6과 호환)
    print("2단계: PyTorch CUDA 12.1 버전 설치 중...")
    print("이 작업은 몇 분 걸릴 수 있습니다.\n")
    
    install_command = [
        sys.executable, "-m", "pip", "install", 
        "torch", "torchvision", "torchaudio", 
        "--index-url", "https://download.pytorch.org/whl/cu121"
    ]
    
    try:
        subprocess.check_call(install_command)
        print("\n✓ PyTorch CUDA 버전 설치 완료!\n")
    except Exception as e:
        print(f"\n❌ 설치 실패: {e}")
        print("\n대안: CUDA 12.4 버전을 시도합니다...\n")
        
        # CUDA 12.4로 재시도
        install_command_124 = [
            sys.executable, "-m", "pip", "install",
            "torch", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cu124"
        ]
        
        try:
            subprocess.check_call(install_command_124)
            print("\n✓ PyTorch CUDA 12.4 버전 설치 완료!\n")
        except Exception as e2:
            print(f"\n❌ CUDA 12.4 설치도 실패: {e2}")
            print("\n수동 설치 방법:")
            print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
            return False
    
    # 설치 확인
    print("3단계: 설치 확인 중...")
    try:
        import torch
        print(f"PyTorch 버전: {torch.__version__}")
        print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA 버전: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print("\n✅ GPU 사용 준비 완료!")
        else:
            print("\n⚠️  CUDA를 사용할 수 없습니다. 드라이버를 확인하세요.")
    except Exception as e:
        print(f"확인 중 오류: {e}")
    
    print("=" * 60)
    return True

if __name__ == "__main__":
    install_pytorch_cuda()

