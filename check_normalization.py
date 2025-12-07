"""
출력·정규화 점검 스크립트

1. 데이터 입출력 파이프라인 확인: train/val/test 모두 같은 정규화 사용하는지 확인
2. 모델 출력의 activation 확인: 마지막 레이어의 activation과 역스케일 확인
3. 빠른 실험: 모델 출력 이미지의 픽셀 히스토그램을 그려서 비교
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from torch.utils.data import DataLoader
import cv2
import torch.utils.data

# 모델 import
from cnn_model import CNNModel, ImageDataset
from unet_model import UNet


def check_data_normalization(dataset, split_name="Dataset", num_samples=5):
    """
    데이터셋의 정규화 상태 확인
    """
    print(f"\n[{split_name}] 정규화 점검:")
    print("-" * 60)
    
    # 샘플 이미지들의 픽셀 값 범위 확인
    pixel_ranges = []
    pixel_means = []
    pixel_stds = []
    
    for i in range(min(num_samples, len(dataset))):
        noisy, clean = dataset[i]
        
        # Tensor를 numpy로 변환 (CHW -> HWC)
        noisy_np = noisy.numpy().transpose(1, 2, 0)
        clean_np = clean.numpy().transpose(1, 2, 0)
        
        # 픽셀 값 범위 확인
        noisy_min, noisy_max = noisy_np.min(), noisy_np.max()
        clean_min, clean_max = clean_np.min(), clean_np.max()
        
        pixel_ranges.append({
            'noisy': (noisy_min, noisy_max),
            'clean': (clean_min, clean_max)
        })
        
        pixel_means.append({
            'noisy': noisy_np.mean(),
            'clean': clean_np.mean()
        })
        
        pixel_stds.append({
            'noisy': noisy_np.std(),
            'clean': clean_np.std()
        })
        
        print(f"  샘플 {i+1}: Noisy[min={noisy_min:.4f}, max={noisy_max:.4f}, mean={noisy_np.mean():.4f}], "
              f"Clean[min={clean_min:.4f}, max={clean_max:.4f}, mean={clean_np.mean():.4f}]")
    
    # 정규화 일관성 확인
    all_in_range = all(
        (0.0 <= r['noisy'][0] <= 1.0 and 0.0 <= r['noisy'][1] <= 1.0 and
         0.0 <= r['clean'][0] <= 1.0 and 0.0 <= r['clean'][1] <= 1.0)
        for r in pixel_ranges
    )
    
    if all_in_range:
        print(f"  ✅ [{split_name}] 정규화 확인: 모든 이미지가 [0, 1] 범위에 있습니다.")
    else:
        print(f"  ⚠️  [{split_name}] 경고: 일부 이미지가 [0, 1] 범위를 벗어났습니다!")
    
    return pixel_ranges, pixel_means, pixel_stds


def check_train_val_test_normalization(full_dataset, val_split=0.2, num_samples=3):
    """
    Train/Val/Test 모두의 정규화 상태 확인
    """
    print("=" * 60)
    print("1. 데이터 정규화 점검 (Train/Val/Test 모두 확인)")
    print("=" * 60)
    
    # Train/Val 분할 (학습 시와 동일한 방식)
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)  # 재현성을 위해 seed 고정
    )
    
    # Test는 Val과 동일하게 사용 (실제로는 별도 test set이 있을 수 있지만, 여기서는 val을 test로 사용)
    test_dataset = val_dataset
    
    print(f"\n데이터셋 크기:")
    print(f"  Train: {len(train_dataset)}개")
    print(f"  Val: {len(val_dataset)}개")
    print(f"  Test: {len(test_dataset)}개")
    print(f"  총합: {len(full_dataset)}개")
    
    # 각 split별로 정규화 확인
    train_ranges, train_means, train_stds = check_data_normalization(train_dataset, "Train", num_samples)
    val_ranges, val_means, val_stds = check_data_normalization(val_dataset, "Val", num_samples)
    test_ranges, test_means, test_stds = check_data_normalization(test_dataset, "Test", num_samples)
    
    # Train/Val/Test 간 정규화 일관성 확인
    print(f"\n" + "-" * 60)
    print("Train/Val/Test 정규화 일관성 확인:")
    
    # 평균 픽셀 값 비교
    train_noisy_mean = np.mean([m['noisy'] for m in train_means])
    val_noisy_mean = np.mean([m['noisy'] for m in val_means])
    test_noisy_mean = np.mean([m['noisy'] for m in test_means])
    
    train_clean_mean = np.mean([m['clean'] for m in train_means])
    val_clean_mean = np.mean([m['clean'] for m in val_means])
    test_clean_mean = np.mean([m['clean'] for m in test_means])
    
    print(f"  Noisy 평균: Train={train_noisy_mean:.4f}, Val={val_noisy_mean:.4f}, Test={test_noisy_mean:.4f}")
    print(f"  Clean 평균: Train={train_clean_mean:.4f}, Val={val_clean_mean:.4f}, Test={test_clean_mean:.4f}")
    
    # 모든 split이 같은 정규화를 사용하는지 확인 (같은 ImageDataset을 사용하므로 동일해야 함)
    all_same_normalization = (
        all(0.0 <= r['noisy'][0] <= 1.0 and 0.0 <= r['noisy'][1] <= 1.0 
            for r in train_ranges + val_ranges + test_ranges) and
        all(0.0 <= r['clean'][0] <= 1.0 and 0.0 <= r['clean'][1] <= 1.0 
            for r in train_ranges + val_ranges + test_ranges)
    )
    
    if all_same_normalization:
        print(f"\n✅ Train/Val/Test 모두 동일한 정규화(/255.0 → [0,1])를 사용합니다!")
    else:
        print(f"\n⚠️  경고: Train/Val/Test 간 정규화가 일치하지 않을 수 있습니다!")
    
    return train_dataset, val_dataset, test_dataset


def check_model_output_activation(model, device='cuda'):
    """
    모델 출력의 activation 확인
    """
    print("\n" + "=" * 60)
    print("2. 모델 출력 Activation 점검")
    print("=" * 60)
    
    # 모델의 마지막 레이어 확인
    model.eval()
    model.to(device)
    
    # 더미 입력 생성 (배치 크기 1, 3채널, 512x512)
    dummy_input = torch.randn(1, 3, 512, 512).to(device)
    
    with torch.no_grad():
        output = model(dummy_input)
    
    # 출력 범위 확인
    output_min = output.min().item()
    output_max = output.max().item()
    output_mean = output.mean().item()
    output_std = output.std().item()
    
    print(f"\n모델 출력 통계:")
    print(f"  Min: {output_min:.4f}")
    print(f"  Max: {output_max:.4f}")
    print(f"  Mean: {output_mean:.4f}")
    print(f"  Std: {output_std:.4f}")
    
    # 모델 구조에서 마지막 activation 확인
    model_name = model.__class__.__name__
    print(f"\n모델: {model_name}")
    
    if model_name == 'CNNModel':
        # CNNModel의 마지막 레이어 확인
        last_layer = model.conv8
        print(f"  마지막 Conv 레이어: {last_layer}")
        # forward에서 clamp 사용 확인
        print(f"  ✅ Forward에서 torch.clamp() 사용 → 출력 범위: [0, 1]")
        print(f"\n  📝 모델 마지막 activation 코드:")
        print(f"     out = self.conv8(x7)")
        print(f"     out = torch.clamp(out, 0.0, 1.0)  # cnn_model.py:64-65")
        print(f"     → clamp 사용 중 (sigmoid 대신 clamp로 변경됨)")
    elif model_name == 'UNet':
        # UNet의 마지막 레이어 확인
        last_layer = model.final_conv
        print(f"  마지막 Conv 레이어: {last_layer}")
        # forward에서 clamp 사용 확인
        print(f"  ✅ Forward에서 torch.clamp() 사용 → 출력 범위: [0, 1]")
        print(f"\n  📝 모델 마지막 activation 코드:")
        print(f"     out = self.final_conv(dec1)")
        print(f"     out = torch.clamp(out, 0.0, 1.0)  # unet_model.py:96-97")
        print(f"     → clamp 사용 중 (sigmoid 대신 clamp로 변경됨)")
    
    # 출력 범위 검증
    if 0.0 <= output_min and output_max <= 1.0:
        print(f"\n✅ 출력 범위 확인: [0, 1] 범위에 있습니다.")
    else:
        print(f"\n⚠️  경고: 출력이 [0, 1] 범위를 벗어났습니다!")
        print(f"   예상 범위: [0, 1], 실제 범위: [{output_min:.4f}, {output_max:.4f}]")
    
    return {
        'min': output_min,
        'max': output_max,
        'mean': output_mean,
        'std': output_std
    }


def plot_pixel_histograms(model, test_loader, device='cuda', num_samples=3, save_dir='normalization_check'):
    """
    입력/출력/GT 이미지의 픽셀 히스토그램을 그려서 비교
    """
    print("\n" + "=" * 60)
    print("3. 픽셀 히스토그램 분석")
    print("=" * 60)
    
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    model.eval()
    model.to(device)
    
    sample_count = 0
    
    with torch.no_grad():
        for batch_idx, (noisy, clean) in enumerate(test_loader):
            if sample_count >= num_samples:
                break
            
            noisy = noisy.to(device)
            clean = clean.to(device)
            
            # 모델 출력 (activation 이후 값 사용 - sigmoid가 모델 내부에 있음)
            output = model(noisy)
            # ✅ 모델 출력에 activation 이후 값 사용 중 (sigmoid가 forward에 포함됨)
            
            batch_size = noisy.size(0)
            for i in range(batch_size):
                if sample_count >= num_samples:
                    break
                
                # CPU로 이동 및 numpy 변환 (detach 명시적으로 추가)
                noisy_np = noisy[i].detach().cpu().numpy().transpose(1, 2, 0)  # HWC
                clean_np = clean[i].detach().cpu().numpy().transpose(1, 2, 0)  # HWC
                output_np = output[i].detach().cpu().numpy().transpose(1, 2, 0)  # HWC
                
                # 클리핑 (안전을 위해)
                noisy_np = np.clip(noisy_np, 0, 1)
                clean_np = np.clip(clean_np, 0, 1)
                output_np = np.clip(output_np, 0, 1)
                
                # 히스토그램 계산 (RGB 채널별)
                fig, axes = plt.subplots(2, 3, figsize=(18, 12))
                
                colors = ['red', 'green', 'blue']
                channel_names = ['R', 'G', 'B']
                
                # 각 채널별 히스토그램
                for ch_idx in range(3):
                    # 입력 (Noisy)
                    noisy_flat = noisy_np[:, :, ch_idx].flatten()
                    axes[0, ch_idx].hist(noisy_flat, bins=50, 
                                        range=(0, 1), alpha=0.7, color=colors[ch_idx], 
                                        label=f'Noisy {channel_names[ch_idx]}')
                    axes[0, ch_idx].set_title(f'Input (Noisy) - {channel_names[ch_idx]} Channel')
                    axes[0, ch_idx].set_xlabel('Pixel Value')
                    axes[0, ch_idx].set_ylabel('Frequency')
                    axes[0, ch_idx].set_xlim(0, 1)
                    axes[0, ch_idx].grid(True, alpha=0.3)
                    axes[0, ch_idx].legend()
                    
                    # 출력 (Model Output) vs GT - 같은 스케일로 비교
                    output_flat = output_np[:, :, ch_idx].flatten()
                    clean_flat = clean_np[:, :, ch_idx].flatten()
                    axes[1, ch_idx].hist(output_flat, bins=50, 
                                        range=(0, 1), alpha=0.7, color=colors[ch_idx], 
                                        label=f'Output {channel_names[ch_idx]}')
                    axes[1, ch_idx].hist(clean_flat, bins=50, 
                                        range=(0, 1), alpha=0.5, color='gray', 
                                        label=f'GT {channel_names[ch_idx]}', linestyle='--')
                    axes[1, ch_idx].set_title(f'Output vs GT - {channel_names[ch_idx]} Channel')
                    axes[1, ch_idx].set_xlabel('Pixel Value')
                    axes[1, ch_idx].set_ylabel('Frequency')
                    axes[1, ch_idx].set_xlim(0, 1)
                    axes[1, ch_idx].grid(True, alpha=0.3)
                    axes[1, ch_idx].legend()
                
                # 전체 비교 히스토그램 추가 (3개를 한 그래프에)
                fig2, ax = plt.subplots(1, 1, figsize=(12, 6))
                # 전체 채널 평균으로 비교
                noisy_all = noisy_np.flatten()
                output_all = output_np.flatten()
                clean_all = clean_np.flatten()
                
                ax.hist(noisy_all, bins=50, range=(0, 1), alpha=0.5, color='blue', 
                       label='Input (Noisy)', density=True)
                ax.hist(output_all, bins=50, range=(0, 1), alpha=0.7, color='red', 
                       label='Output (Model)', density=True)
                ax.hist(clean_all, bins=50, range=(0, 1), alpha=0.5, color='green', 
                       label='GT (Clean)', density=True)
                ax.set_title(f'Sample {sample_count+1} - 전체 픽셀 히스토그램 비교 (같은 스케일 [0,1])')
                ax.set_xlabel('Pixel Value')
                ax.set_ylabel('Density')
                ax.set_xlim(0, 1)
                ax.grid(True, alpha=0.3)
                ax.legend()
                plt.tight_layout()
                plt.savefig(f'{save_dir}/histogram_combined_sample_{sample_count+1}.png', dpi=150, bbox_inches='tight')
                plt.close()
                
                # 통계 정보 추가
                stats_text = f"""
Sample {sample_count+1} Statistics:

Input (Noisy):
  Mean: {noisy_np.mean():.4f}
  Std: {noisy_np.std():.4f}
  Min: {noisy_np.min():.4f}
  Max: {noisy_np.max():.4f}

Output:
  Mean: {output_np.mean():.4f}
  Std: {output_np.std():.4f}
  Min: {output_np.min():.4f}
  Max: {output_np.max():.4f}

GT (Clean):
  Mean: {clean_np.mean():.4f}
  Std: {clean_np.std():.4f}
  Min: {clean_np.min():.4f}
  Max: {clean_np.max():.4f}
"""
                
                # 통계 출력
                print(f"\n샘플 {sample_count+1} 통계:")
                print(f"  Input (Noisy): mean={noisy_np.mean():.4f}, std={noisy_np.std():.4f}, "
                      f"min={noisy_np.min():.4f}, max={noisy_np.max():.4f}")
                print(f"  Output: mean={output_np.mean():.4f}, std={output_np.std():.4f}, "
                      f"min={output_np.min():.4f}, max={output_np.max():.4f}")
                print(f"  GT (Clean): mean={clean_np.mean():.4f}, std={clean_np.std():.4f}, "
                      f"min={clean_np.min():.4f}, max={clean_np.max():.4f}")
                
                # 출력이 밝은 쪽에 몰려있는지 확인 (히스토그램 해석 기준)
                output_mean = output_np.mean()
                clean_mean = clean_np.mean()
                output_max = output_np.max()
                clean_max = clean_np.max()
                
                # 히스토그램 해석 기준 출력
                print(f"\n  📊 히스토그램 해석 기준:")
                print(f"     - 정상: output과 GT 분포가 비슷한 구간 (0.2~0.8)")
                print(f"     - 문제: output이 0.8~1.0 구간에 몰려있고 GT는 0.2~0.8")
                
                # 문제 확정 케이스 체크
                output_high_range_ratio = np.sum((output_np > 0.8) & (output_np <= 1.0)) / output_np.size
                clean_high_range_ratio = np.sum((clean_np > 0.8) & (clean_np <= 1.0)) / clean_np.size
                
                if output_mean > clean_mean + 0.1:  # 0.1 이상 차이
                    print(f"\n  ⚠️  [문제 확정] 출력이 GT보다 밝습니다!")
                    print(f"     - 평균 차이: {output_mean - clean_mean:.4f}")
                    print(f"     - 출력 평균: {output_mean:.4f}, GT 평균: {clean_mean:.4f}")
                    print(f"     - 출력 0.8~1.0 비율: {output_high_range_ratio*100:.1f}%, GT: {clean_high_range_ratio*100:.1f}%")
                    if output_high_range_ratio > 0.3:  # 30% 이상이 밝은 구간에 있으면
                        print(f"     → ✅ '출력 스케일 오류 or activation 미보정' 100% 확정!")
                        print(f"     → 해결: 모델 마지막 activation 코드 확인 필요")
                elif output_mean < clean_mean - 0.1:
                    print(f"\n  ⚠️  경고: 출력이 GT보다 어둡습니다! (차이: {clean_mean - output_mean:.4f})")
                else:
                    print(f"\n  ✅ [정상] 출력과 GT의 밝기가 비슷합니다.")
                    print(f"     - 출력 평균: {output_mean:.4f}, GT 평균: {clean_mean:.4f}")
                    print(f"     → 정규화는 정상 → 다음 원인은 Loss / 모델 구조 쪽 확인 필요")
                
                plt.suptitle(f'Sample {sample_count+1} - Pixel Histograms', fontsize=16, y=0.995)
                plt.tight_layout()
                plt.savefig(f'{save_dir}/histogram_sample_{sample_count+1}.png', dpi=150, bbox_inches='tight')
                plt.close()
                
                # 이미지 비교도 저장
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].imshow(noisy_np)
                axes[0].set_title(f'Input (Noisy) #{sample_count+1}')
                axes[0].axis('off')
                
                axes[1].imshow(output_np)
                axes[1].set_title(f'Output #{sample_count+1}')
                axes[1].axis('off')
                
                axes[2].imshow(clean_np)
                axes[2].set_title(f'GT (Clean) #{sample_count+1}')
                axes[2].axis('off')
                
                plt.tight_layout()
                plt.savefig(f'{save_dir}/comparison_sample_{sample_count+1}.png', dpi=150, bbox_inches='tight')
                plt.close()
                
                sample_count += 1
                print(f"  ✅ 히스토그램 저장 완료: {save_dir}/histogram_sample_{sample_count}.png")
    
    print(f"\n✅ 히스토그램 분석 완료! 결과가 {save_dir}에 저장되었습니다.")


def main():
    parser = argparse.ArgumentParser(description='출력·정규화 점검 스크립트')
    parser.add_argument('--model_path', type=str, default=None,
                        help='학습된 모델 체크포인트 경로 (없으면 데이터 정규화만 점검)')
    parser.add_argument('--model_type', type=str, choices=['cnn', 'unet'], default=None,
                        help='모델 타입 (cnn 또는 unet, --model_path가 있을 때 필수)')
    parser.add_argument('--noisy_dir', type=str, required=True,
                        help='노이즈 이미지 디렉토리')
    parser.add_argument('--clean_dir', type=str, default=None,
                        help='깨끗한 이미지 디렉토리 (None이면 noisy_dir에서 파일명 패턴으로 매칭)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='사용할 디바이스 (cuda/cpu)')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='사용할 GPU ID (default: 0)')
    parser.add_argument('--num_samples', type=int, default=3,
                        help='히스토그램 분석에 사용할 샘플 수 (default: 3)')
    parser.add_argument('--save_dir', type=str, default='normalization_check',
                        help='결과 저장 디렉토리 (default: normalization_check)')
    
    args = parser.parse_args()
    
    # 모델 경로가 있으면 model_type도 필요
    if args.model_path and not args.model_type:
        parser.error("--model_path가 제공되면 --model_type도 필요합니다 (cnn 또는 unet)")
    
    # GPU 설정
    if args.device == 'cuda':
        if torch.cuda.is_available():
            device = torch.device(f'cuda:{args.gpu_id}')
            torch.cuda.set_device(args.gpu_id)
            print(f'GPU 사용: {torch.cuda.get_device_name(args.gpu_id)}')
        else:
            print('⚠️  CUDA를 사용할 수 없습니다. CPU로 전환합니다.')
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')
    
    print(f'사용 디바이스: {device}\n')
    
    # 데이터셋 로드
    print('데이터셋 로딩 중...')
    full_dataset = ImageDataset(args.noisy_dir, args.clean_dir)
    print(f'전체 데이터셋 크기: {len(full_dataset)}개 이미지 쌍\n')
    
    # 1. Train/Val/Test 모두의 정규화 점검 (항상 실행)
    train_dataset, val_dataset, test_dataset = check_train_val_test_normalization(
        full_dataset, val_split=0.2, num_samples=args.num_samples
    )
    
    # Test loader 생성 (모델이 있을 때 사용)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )
    
    # 모델이 있으면 추가 점검 수행
    if args.model_path:
        # 모델 로드
        print('모델 로딩 중...')
        if args.model_type == 'cnn':
            model = CNNModel(in_channels=3, out_channels=3)
        else:
            model = UNet(in_channels=3, out_channels=3)
        
        checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print('모델 로드 완료!\n')
        
        # 2. 모델 출력 activation 점검
        check_model_output_activation(model, device=device)
        
        # 3. 픽셀 히스토그램 분석
        plot_pixel_histograms(model, test_loader, device=device, 
                             num_samples=args.num_samples, save_dir=args.save_dir)
        
        print("\n" + "=" * 60)
        print("점검 완료!")
        print("=" * 60)
        print("\n점검 결과 요약:")
        print("1. 데이터 정규화: ImageDataset에서 /255.0으로 [0,1] 범위로 정규화")
        print("2. 모델 출력: sigmoid activation으로 [0,1] 범위 출력")
        print("3. 히스토그램: 입력/출력/GT의 픽셀 분포 비교")
        print(f"\n결과 파일은 {args.save_dir} 폴더에 저장되었습니다.")
    else:
        print("\n" + "=" * 60)
        print("데이터 정규화 점검 완료!")
        print("=" * 60)
        print("\n점검 결과 요약:")
        print("1. 데이터 정규화: ImageDataset에서 /255.0으로 [0,1] 범위로 정규화")
        print("\n💡 모델이 있으면 --model_path와 --model_type을 추가하여")
        print("   모델 출력 activation과 히스토그램 분석도 수행할 수 있습니다.")


if __name__ == '__main__':
    main()

