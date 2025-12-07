"""
CNN 모델 추론 및 성능 측정 스크립트
FPS, Inference 시간 측정 포함
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from cnn_model import CNNModel, ImageDataset, calculate_psnr, calculate_ssim
import argparse
import time
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def custom_collate_fn(batch):
    """
    None 값을 처리할 수 있는 커스텀 collate 함수
    clean 이미지가 None인 경우를 처리
    """
    # batch는 [(noisy, clean), ...] 형태
    # clean이 None인 경우가 있을 수 있음
    noisy_batch = [item[0] for item in batch]
    clean_batch = [item[1] for item in batch]
    
    # noisy는 항상 텐서이므로 기본 collate 사용
    noisy_collated = default_collate(noisy_batch)
    
    # clean이 모두 None이 아닌 경우에만 collate
    if all(c is not None for c in clean_batch):
        clean_collated = default_collate(clean_batch)
    else:
        # None이 하나라도 있으면 None으로 설정
        clean_collated = None
    
    return noisy_collated, clean_collated


def measure_inference_time(model, test_loader, device='cuda', num_runs=100):
    """
    추론 시간 및 FPS 측정
    """
    model.eval()
    model.to(device)
    
    total_batches = len(test_loader)
    
    # Warm-up
    print('  Warm-up 중... (5 배치)')
    with torch.no_grad():
        for i, (noisy, _) in enumerate(test_loader):
            if i >= 5:
                break
            noisy = noisy.to(device)
            _ = model(noisy)
    
    # 실제 측정
    print(f'  Inference 시간 측정 중... (총 {total_batches} 배치)')
    times = []
    with torch.no_grad():
        for batch_idx, (noisy, _) in enumerate(test_loader):
            noisy = noisy.to(device)
            
            # GPU 동기화
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            start_time = time.time()
            _ = model(noisy)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # ms로 변환
            
            # 진행사항 출력 (20배치마다 또는 마지막 배치)
            if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == total_batches:
                progress_pct = (batch_idx + 1) / total_batches * 100
                print(f'    [{batch_idx+1}/{total_batches}] ({progress_pct:.1f}%)')
    
    avg_time = np.mean(times)
    fps = 1000.0 / avg_time  # FPS 계산
    
    return avg_time, fps


def inference_single_image(model, image_path, device='cuda', save_path=None):
    """
    단일 이미지 추론
    """
    model.eval()
    model.to(device)
    
    # 이미지 로드
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
    
    # 추론
    with torch.no_grad():
        output = model(img_tensor)
        output = output.squeeze(0).cpu().numpy()
        output = output.transpose(1, 2, 0)
        output = np.clip(output, 0, 1)
    
    # 저장
    if save_path:
        output_uint8 = (output * 255).astype(np.uint8)
        output_bgr = cv2.cvtColor(output_uint8, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(save_path), output_bgr)
    
    return output


def evaluate_with_metrics(model, test_loader, device='cuda'):
    """
    PSNR, SSIM, Inference 시간 측정
    """
    model.eval()
    model.to(device)
    
    # GPU 메모리 정리
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    psnr_values = []
    ssim_values = []
    
    total_batches = len(test_loader)
    start_time = time.time()
    
    print(f'\n평가 시작: 총 {total_batches}개 배치 처리 예정')
    print('-' * 50)
    
    with torch.no_grad():
        for batch_idx, (noisy, clean) in enumerate(test_loader):
            noisy = noisy.to(device)
            output = model(noisy)
            
            # Clean 이미지가 있는 경우에만 PSNR, SSIM 계산
            if clean is not None:
                # 배치 내 None 체크
                if isinstance(clean, torch.Tensor):
                    clean = clean.to(device)
                    psnr_val = calculate_psnr(output, clean)
                    ssim_val = calculate_ssim(output, clean)
                    psnr_values.append(psnr_val)
                    ssim_values.append(ssim_val)
            
            # 메모리 정리 (10배치마다)
            if device.type == 'cuda' and (batch_idx + 1) % 10 == 0:
                torch.cuda.empty_cache()
            
            # 진행사항 출력 (10배치마다 또는 마지막 배치)
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
                elapsed_time = time.time() - start_time
                avg_time_per_batch = elapsed_time / (batch_idx + 1)
                remaining_batches = total_batches - (batch_idx + 1)
                estimated_remaining = avg_time_per_batch * remaining_batches
                
                progress_pct = (batch_idx + 1) / total_batches * 100
                if device.type == 'cuda':
                    allocated = torch.cuda.memory_allocated(device) / 1024**3
                    reserved = torch.cuda.memory_reserved(device) / 1024**3
                    print(f'  Progress: [{batch_idx+1}/{total_batches}] ({progress_pct:.1f}%) | '
                          f'경과: {elapsed_time:.1f}초 | '
                          f'예상 남은 시간: {estimated_remaining/60:.1f}분 | '
                          f'GPU 메모리: {allocated:.2f}GB / {reserved:.2f}GB')
                else:
                    print(f'  Progress: [{batch_idx+1}/{total_batches}] ({progress_pct:.1f}%) | '
                          f'경과: {elapsed_time:.1f}초 | '
                          f'예상 남은 시간: {estimated_remaining/60:.1f}분')
    
    # 결과 반환
    result = {
        'inference_time_ms': 0,
        'fps': 0
    }
    
    if len(psnr_values) > 0:
        result['psnr'] = np.mean(psnr_values)
        result['ssim'] = np.mean(ssim_values)
    else:
        result['psnr'] = None
        result['ssim'] = None
        print('\n⚠️  Clean 이미지가 없어 PSNR/SSIM을 계산할 수 없습니다.')
    
    total_eval_time = time.time() - start_time
    print(f'\n평가 완료! 총 소요 시간: {total_eval_time/60:.2f}분')
    print('Inference 시간 측정 중...')
    
    # Inference 시간 측정
    avg_time, fps = measure_inference_time(model, test_loader, device)
    result['inference_time_ms'] = avg_time
    result['fps'] = fps
    
    return result


def visualize_results(model, test_loader, device='cuda', num_samples=5, save_dir='results'):
    """
    결과 시각화
    각 서브디렉토리에서 균등하게 샘플링하여 다양한 이미지 표시
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    model.eval()
    model.to(device)
    
    print(f'시각화 중... (총 {num_samples}개 샘플)')
    
    # 전체 데이터셋에서 균등하게 샘플링하기 위해 인덱스 계산
    dataset = test_loader.dataset
    total_samples = len(dataset)
    
    # 균등하게 분산된 인덱스 선택
    if total_samples <= num_samples:
        selected_indices = list(range(total_samples))
    else:
        step = total_samples / num_samples
        selected_indices = [int(i * step) for i in range(num_samples)]
    
    print(f'  전체 {total_samples}개 이미지 중 {len(selected_indices)}개 샘플 선택')
    
    with torch.no_grad():
        sample_count = 0
        processed_indices = set()
        
        # 선택된 인덱스가 속한 배치를 찾아서 처리
        for idx, (noisy, clean) in enumerate(test_loader):
            batch_size = noisy.size(0)
            batch_start_idx = idx * test_loader.batch_size
            
            noisy = noisy.to(device)
            output = model(noisy)
            
            for i in range(batch_size):
                current_idx = batch_start_idx + i
                
                # 선택된 인덱스인지 확인
                if current_idx not in selected_indices:
                    continue
                
                if current_idx in processed_indices:
                    continue
                
                if sample_count >= num_samples:
                    break
                
                noisy_np = noisy[i].cpu().numpy().transpose(1, 2, 0)
                output_np = output[i].cpu().numpy().transpose(1, 2, 0)
                
                noisy_np = np.clip(noisy_np, 0, 1)
                output_np = np.clip(output_np, 0, 1)
                
                # Clean 이미지가 있는 경우와 없는 경우 분기
                if clean is not None and isinstance(clean, torch.Tensor):
                    clean = clean.to(device)
                    clean_np = clean[i].cpu().numpy().transpose(1, 2, 0)
                    clean_np = np.clip(clean_np, 0, 1)
                    
                    # 3개 이미지: Noisy, Output, Clean
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    axes[0].imshow(noisy_np)
                    axes[0].set_title(f'Noisy Image #{sample_count+1} (Idx: {current_idx})')
                    axes[0].axis('off')
                    
                    axes[1].imshow(output_np)
                    axes[1].set_title(f'Denoised (CNN) #{sample_count+1}')
                    axes[1].axis('off')
                    
                    axes[2].imshow(clean_np)
                    axes[2].set_title(f'Clean (GT) #{sample_count+1}')
                    axes[2].axis('off')
                else:
                    # 2개 이미지: Noisy, Output (clean 없음)
                    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                    axes[0].imshow(noisy_np)
                    axes[0].set_title(f'Noisy Image #{sample_count+1} (Idx: {current_idx})')
                    axes[0].axis('off')
                    
                    axes[1].imshow(output_np)
                    axes[1].set_title(f'Denoised (CNN) #{sample_count+1}')
                    axes[1].axis('off')
                
                plt.tight_layout()
                plt.savefig(f'{save_dir}/result_{sample_count+1}.png', dpi=150, bbox_inches='tight')
                plt.close()
                
                processed_indices.add(current_idx)
                sample_count += 1
                print(f'  [{sample_count}/{num_samples}] 저장 완료 (인덱스 {current_idx}, 배치 {idx+1}, 이미지 {i+1})')
            
            if sample_count >= num_samples:
                break
    
    print(f'시각화 결과가 {save_dir}에 저장되었습니다.')


def main():
    parser = argparse.ArgumentParser(description='CNN 모델 추론 및 평가')
    parser.add_argument('--model_path', type=str, required=True,
                        help='학습된 모델 체크포인트 경로')
    parser.add_argument('--noisy_dir', type=str, default=None,
                        help='노이즈 이미지 디렉토리 (평가용, test 폴더가 있으면 자동으로 사용)')
    parser.add_argument('--clean_dir', type=str, default=None,
                        help='깨끗한 이미지 디렉토리 (평가용, None이면 noisy_dir에서 파일명 패턴으로 매칭)')
    parser.add_argument('--use_test_split', action='store_true',
                        help='test 폴더를 자동으로 사용 (기본값: True, 폴더 구조가 있으면 자동 감지)')
    parser.add_argument('--image_path', type=str, default=None,
                        help='단일 이미지 추론 경로')
    parser.add_argument('--output_path', type=str, default=None,
                        help='출력 이미지 저장 경로')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='배치 크기 (기본값: 4, 메모리 부족 시 더 줄이세요)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='사용할 디바이스 (cuda/cpu)')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='사용할 GPU ID (default: 0)')
    parser.add_argument('--visualize', action='store_true',
                        help='결과 시각화 저장')
    parser.add_argument('--save_dir', type=str, default='results',
                        help='결과 저장 디렉토리')
    
    args = parser.parse_args()
    
    # GPU 설정 및 확인
    if args.device == 'cuda':
        if torch.cuda.is_available():
            device = torch.device(f'cuda:{args.gpu_id}')
            torch.cuda.set_device(args.gpu_id)
            print(f'=' * 50)
            print(f'GPU 사용 가능!')
            print(f'GPU 장치: {torch.cuda.get_device_name(args.gpu_id)}')
            print(f'GPU 메모리: {torch.cuda.get_device_properties(args.gpu_id).total_memory / 1024**3:.2f} GB')
            print(f'=' * 50)
        else:
            print('⚠️  경고: CUDA를 사용할 수 없습니다. CPU로 전환합니다.')
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')
        print(f'CPU 모드로 실행합니다.')
    
    print(f'사용 디바이스: {device}')
    
    # 모델 로드
    print('모델 로딩 중...')
    model = CNNModel(in_channels=3, out_channels=3)
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # GPU 메모리 정리
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        print(f'GPU 메모리 정리 완료')
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        print(f'  모델 로드 후 GPU 메모리: {allocated:.2f}GB / {reserved:.2f}GB')
    
    print('모델 로드 완료!')
    
    # 단일 이미지 추론
    if args.image_path:
        print(f'단일 이미지 추론: {args.image_path}')
        output = inference_single_image(
            model, 
            args.image_path, 
            device=device, 
            save_path=args.output_path
        )
        if args.output_path:
            print(f'결과 저장: {args.output_path}')
        return
    
    # 데이터셋 평가
    if args.noisy_dir:
        print('데이터셋 평가 중...')
        
        # 경로 확인
        noisy_path = Path(args.noisy_dir)
        print(f'Noisy 디렉토리: {noisy_path.absolute()}')
        print(f'  존재 여부: {noisy_path.exists()}')
        if noisy_path.exists():
            image_files = list(noisy_path.glob('**/*.jpg')) + list(noisy_path.glob('**/*.png'))
            print(f'  이미지 파일 개수: {len(image_files)}')
        
        if args.clean_dir:
            clean_path = Path(args.clean_dir)
            print(f'Clean 디렉토리: {clean_path.absolute()}')
            print(f'  존재 여부: {clean_path.exists()}')
            if clean_path.exists():
                image_files = list(clean_path.glob('**/*.jpg')) + list(clean_path.glob('**/*.png'))
                print(f'  이미지 파일 개수: {len(image_files)}')
        
        # test 폴더 자동 감지
        test_noisy_path = noisy_path / 'test'
        test_clean_path = Path(args.clean_dir) / 'test' if args.clean_dir else None
        
        # clean 폴더 구조 확인
        clean_has_split_structure = False
        if args.clean_dir:
            clean_path = Path(args.clean_dir)
            # clean 폴더에 test/train/val 폴더가 있는지 확인
            if (clean_path / 'test').exists() or (clean_path / 'train').exists() or (clean_path / 'val').exists():
                clean_has_split_structure = True
        
        # test 폴더가 있으면 자동으로 사용
        if (args.use_test_split or test_noisy_path.exists()) and test_noisy_path.exists():
            print(f'✅ test 폴더를 자동으로 감지하여 사용합니다.')
            actual_noisy_dir = str(test_noisy_path)
            # clean 폴더는 항상 전체 사용 (test/train/val 모두 포함)
            actual_clean_dir = args.clean_dir
            if clean_has_split_structure:
                print(f'  clean 폴더의 모든 split (test/train/val)을 사용합니다.')
        else:
            print(f'⚠️  test 폴더를 찾을 수 없습니다. 전체 폴더를 사용합니다.')
            actual_noisy_dir = args.noisy_dir
            actual_clean_dir = args.clean_dir
        
        if actual_clean_dir is None:
            print(f'단일 폴더 모드: {actual_noisy_dir}에서 파일명 패턴으로 매칭합니다.')
            # Clean 이미지가 없을 가능성이 있으므로 inference_only 모드 사용
            inference_only = True
            test_dataset = ImageDataset(actual_noisy_dir, actual_clean_dir, inference_only=inference_only)
        else:
            print(f'이중 폴더 모드: noisy_dir={actual_noisy_dir}, clean_dir={actual_clean_dir}')
            inference_only = False
            # 폴더명 기반 자동 매칭 사용
            # clean 폴더의 test/train/val 모두 사용하기 위해 split=None 설정
            # _auto_match_by_folder_name 함수가 모든 split을 순회하며 매칭 시도
            # (noisy에 없는 split은 자동으로 건너뜀)
            split_value = None
            print(f'  clean 폴더의 모든 split (test/train/val)을 사용하여 매칭합니다.')
            test_dataset = ImageDataset(actual_noisy_dir, actual_clean_dir, inference_only=inference_only, auto_match=True, split=split_value)
        test_loader = DataLoader(
            test_dataset, 
            batch_size=args.batch_size, 
            shuffle=False,
            num_workers=0,  # Windows 호환성
            pin_memory=True if device.type == 'cuda' else False,
            collate_fn=custom_collate_fn  # None 처리용 커스텀 collate
        )
        
        total_samples = len(test_dataset)
        total_batches = len(test_loader)
        print(f'테스트 데이터셋: {total_samples}개 이미지, {total_batches}개 배치 (batch_size={args.batch_size})')
        
        # 예상 소요 시간 계산 (학습 시 배치당 약 1.1초 기준)
        estimated_time_per_batch = 1.1  # 초 (학습 로그 기준)
        estimated_total_time = estimated_time_per_batch * total_batches
        print(f'예상 소요 시간: 약 {estimated_total_time/60:.1f}분 ({estimated_total_time:.0f}초)')
        print()
        
        # 성능 측정
        metrics = evaluate_with_metrics(model, test_loader, device=device)
        
        print('\n=== 평가 결과 ===')
        if metrics['psnr'] is not None:
            print(f'PSNR: {metrics["psnr"]:.4f} dB')
            print(f'SSIM: {metrics["ssim"]:.4f}')
        else:
            print('PSNR/SSIM: 계산 불가 (clean 이미지 없음)')
        print(f'Inference Time: {metrics["inference_time_ms"]:.2f} ms/frame')
        print(f'FPS: {metrics["fps"]:.2f}')
        
        # 시각화
        if args.visualize:
            print('\n결과 시각화 중...')
            visualize_results(model, test_loader, device=device, save_dir=args.save_dir)
    else:
        print('평가를 위해 --noisy_dir를 제공해주세요. (--clean_dir는 선택사항입니다)')


if __name__ == '__main__':
    main()

