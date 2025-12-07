"""
GPU 사용 가능 여부 및 정보 확인 스크립트
"""
import torch


def check_gpu():
    """GPU 정보 확인"""
    print('=' * 60)
    print('GPU 정보 확인')
    print('=' * 60)
    
    # CUDA 사용 가능 여부
    cuda_available = torch.cuda.is_available()
    print(f'CUDA 사용 가능: {cuda_available}')
    
    if cuda_available:
        # GPU 개수
        gpu_count = torch.cuda.device_count()
        print(f'사용 가능한 GPU 개수: {gpu_count}')
        print()
        
        # 각 GPU 정보
        for i in range(gpu_count):
            print(f'--- GPU {i} ---')
            print(f'이름: {torch.cuda.get_device_name(i)}')
            
            props = torch.cuda.get_device_properties(i)
            print(f'총 메모리: {props.total_memory / 1024**3:.2f} GB')
            print(f'컴퓨팅 능력: {props.major}.{props.minor}')
            print(f'멀티프로세서 수: {props.multi_processor_count}')
            print()
        
        # 현재 GPU
        current_device = torch.cuda.current_device()
        print(f'현재 사용 중인 GPU: {current_device}')
        print(f'현재 GPU 이름: {torch.cuda.get_device_name(current_device)}')
        print()
        
        # CUDA 버전
        print(f'CUDA 버전: {torch.version.cuda}')
        print(f'cuDNN 버전: {torch.backends.cudnn.version()}')
        print(f'cuDNN 활성화: {torch.backends.cudnn.enabled}')
        
        # 메모리 사용량
        print()
        print('현재 GPU 메모리 사용량:')
        for i in range(gpu_count):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            props = torch.cuda.get_device_properties(i)
            total = props.total_memory / 1024**3
            print(f'  GPU {i}: {allocated:.2f} GB / {reserved:.2f} GB / {total:.2f} GB')
    else:
        print('\n⚠️  CUDA를 사용할 수 없습니다.')
        print('   - NVIDIA GPU가 설치되어 있는지 확인하세요.')
        print('   - CUDA가 설치되어 있는지 확인하세요.')
        print('   - PyTorch가 CUDA 버전으로 설치되어 있는지 확인하세요.')
        print('\n   CPU 모드로 실행됩니다.')
    
    print('=' * 60)
    print(f'PyTorch 버전: {torch.__version__}')
    print('=' * 60)


if __name__ == '__main__':
    check_gpu()

