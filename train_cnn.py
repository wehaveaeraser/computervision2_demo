"""
CNN 모델 학습 스크립트
"""
import torch
from torch.utils.data import DataLoader
from cnn_model import CNNModel, ImageDataset, train_model, evaluate_model
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='CNN 모델 학습')
    parser.add_argument('--noisy_dir', type=str, required=True,
                        help='노이즈 이미지 디렉토리 경로 (또는 정제된 데이터가 있는 폴더)')
    parser.add_argument('--clean_dir', type=str, default=None,
                        help='깨끗한 이미지 디렉토리 경로 (None이면 noisy_dir에서 파일명 패턴으로 매칭)')
    parser.add_argument('--matched_pairs_file', type=str, default=None,
                        help='매칭된 이미지 쌍 정보가 담긴 JSON 파일 경로 (폴더명 기반 매칭 결과)')
    parser.add_argument('--auto_match', action='store_true',
                        help='학습 시 자동으로 폴더명 기반 매칭 수행 (matched_pairs_file 없이 사용 가능)')
    parser.add_argument('--l1_weight', type=float, default=1.0,
                        help='L1 Loss 가중치 (default: 1.0)')
    parser.add_argument('--ssim_weight', type=float, default=1.0,
                        help='SSIM Loss 가중치 (default: 1.0)')
    parser.add_argument('--gradient_weight', type=float, default=0.5,
                        help='Gradient Loss 가중치 (default: 0.5)')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='배치 크기 (default: 2, Mixed Precision 없을 때 메모리 절약)')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='학습률 (default: 5e-5)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='에포크 수 (default: 50)')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/cnn',
                        help='체크포인트 저장 디렉토리 (default: checkpoints/cnn)')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='학습 데이터 비율 (default: 0.7)')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='검증 데이터 비율 (default: 0.15)')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                        help='테스트 데이터 비율 (default: 0.15)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='사용할 디바이스 (cuda/cpu) (default: cuda)')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='사용할 GPU ID (default: 0)')
    parser.add_argument('--resume', type=str, default=None,
                        help='체크포인트 경로 (학습 재개용, 예: checkpoints/cnn/best_model.pth)')
    
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
            print(f'CUDA 버전: {torch.version.cuda}')
            print(f'PyTorch 버전: {torch.__version__}')
            print(f'=' * 50)
        else:
            print(' 경고: CUDA를 사용할 수 없습니다. CPU로 전환합니다.')
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')
        print(f'CPU 모드로 실행합니다.')
    
    print(f'사용 디바이스: {device}')
    
    # 데이터셋 생성
    print('데이터셋 로딩 중...')
    if args.matched_pairs_file is not None:
        print(f'매칭 파일 모드: {args.matched_pairs_file}에서 매칭 정보를 읽습니다.')
        print(f'  noisy_dir={args.noisy_dir}, clean_dir={args.clean_dir}')
    elif args.clean_dir is None:
        print(f'단일 폴더 모드: {args.noisy_dir}에서 파일명 패턴으로 매칭합니다.')
    else:
        print(f'이중 폴더 모드: noisy_dir={args.noisy_dir}, clean_dir={args.clean_dir}')
    
    # train/val/test 폴더 구조 자동 감지
    def has_split_structure(base_dir):
        """폴더에 train/val/test 구조가 있는지 확인"""
        base_path = Path(base_dir)
        train_exists = (base_path / 'train').exists()
        val_exists = (base_path / 'val').exists()
        test_exists = (base_path / 'test').exists()
        return train_exists and val_exists and test_exists
    
    # 노이즈 데이터의 train/val/test 구조를 존중하는 모드
    # auto_match 모드이거나 폴더 구조가 있으면 split 구조 사용
    use_split_structure = (args.auto_match and args.clean_dir is not None) or \
                          (has_split_structure(args.noisy_dir) and (args.clean_dir is None or has_split_structure(args.clean_dir)))
    
    if use_split_structure:
        # 노이즈 데이터의 train/val/test 구조를 그대로 사용
        if args.auto_match and args.clean_dir is not None:
            print("노이즈 데이터의 train/val/test 구조를 그대로 사용합니다.")
            print("Train과 Validation 모두 노이즈 데이터 기준으로 매칭합니다.")
            
            # Train 데이터셋: 노이즈 데이터 기준으로 매칭
            train_dataset = ImageDataset(
                args.noisy_dir, args.clean_dir,
                matched_pairs_file=args.matched_pairs_file,
                auto_match=True,
                split='train',
                reverse_match=False  # 노이즈 데이터 기준
            )
            
            # Validation 데이터셋: 노이즈 데이터 기준으로 매칭
            val_dataset = ImageDataset(
                args.noisy_dir, args.clean_dir,
                matched_pairs_file=args.matched_pairs_file,
                auto_match=True,
                split='val',
                reverse_match=False  # 노이즈 데이터 기준
            )
            
            # Train과 Val 데이터를 합쳐서 비율에 맞게 재분할
            # 목표: train:val:test = 7:1.5:1.5
            total_samples = len(train_dataset) + len(val_dataset)
            target_train_ratio = 7.0 / 10.0  # 7 / (7 + 1.5 + 1.5)
            target_val_ratio = 1.5 / 10.0
            target_test_ratio = 1.5 / 10.0
            
            target_train_size = int(total_samples * target_train_ratio)
            target_val_size = int(total_samples * target_val_ratio)
            target_test_size = total_samples - target_train_size - target_val_size
            
            current_train_size = len(train_dataset)
            current_val_size = len(val_dataset)
            
            print(f"\n📊 데이터 분할 조정:")
            print(f"   현재: Train={current_train_size}, Val={current_val_size}, Total={total_samples}")
            print(f"   목표 비율: Train:Val:Test = 7:1.5:1.5")
            print(f"   목표: Train={target_train_size}, Val={target_val_size}, Test={target_test_size}")
            
            # 전체 데이터셋을 합쳐서 재분할
            from torch.utils.data import ConcatDataset
            full_dataset = ConcatDataset([train_dataset, val_dataset])
            
            # 비율에 맞게 분할
            train_dataset, val_dataset, _ = torch.utils.data.random_split(
                full_dataset, 
                [target_train_size, target_val_size, target_test_size],
                generator=torch.Generator().manual_seed(42)  # 재현성을 위해 시드 고정
            )
            
            print(f"   조정 후: Train={len(train_dataset)}, Val={len(val_dataset)}")
            # test 데이터는 inference 코드에서만 사용
        else:
            # 폴더 구조 자동 감지 모드
            print("train/val/test 폴더 구조를 자동으로 감지하여 사용합니다.")
            # 각 split별로 데이터셋 생성
            train_dataset = ImageDataset(
                str(Path(args.noisy_dir) / 'train'), 
                str(Path(args.clean_dir) / 'train') if args.clean_dir else None,
                matched_pairs_file=args.matched_pairs_file,
                auto_match=False
            )
            val_dataset = ImageDataset(
                str(Path(args.noisy_dir) / 'val'),
                str(Path(args.clean_dir) / 'val') if args.clean_dir else None,
                matched_pairs_file=args.matched_pairs_file,
                auto_match=False
            )
            # test 데이터는 inference 코드에서만 사용
        print(f'Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}')
        print('ℹ️  Test 데이터는 inference 코드에서 평가하세요.')
    else:
        # 기존 방식: 전체 데이터셋을 로드 후 랜덤 분할
        full_dataset = ImageDataset(args.noisy_dir, args.clean_dir, 
                                    matched_pairs_file=args.matched_pairs_file,
                                    auto_match=args.auto_match)
        
        # Train/Validation 분할 (test는 inference에서 사용)
        total_size = len(full_dataset)
        train_size = int(total_size * args.train_ratio)
        val_size = int(total_size * args.val_ratio)
        # test_size는 계산하되 사용하지 않음 (inference에서 사용)
        test_size = total_size - train_size - val_size
        
        train_dataset, val_dataset, _ = torch.utils.data.random_split(
            full_dataset, [train_size, val_size, test_size]
        )
        print(f'Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}')
        print(f'분할 비율: Train={len(train_dataset)/total_size:.2%}, Val={len(val_dataset)/total_size:.2%}, Test={test_size/total_size:.2%} (inference에서 사용)')
        print('ℹ️  Test 데이터는 inference 코드에서 평가하세요.')
    
    # DataLoader 생성
    # Windows 호환성을 위해 num_workers=0 사용
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=0,  # Windows 호환성
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=0,  # Windows 호환성
        pin_memory=True if device.type == 'cuda' else False
    )
    
    
    # 모델 생성
    model = CNNModel(in_channels=3, out_channels=3)
    print(f'모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}')
    
    # 체크포인트에서 재개
    start_epoch = 0
    if args.resume:
        print(f'\n체크포인트에서 학습 재개: {args.resume}')
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f'✅ 체크포인트 로드 완료!')
        print(f'   - Epoch: {checkpoint.get("epoch", 0)}')
        print(f'   - Val Loss: {checkpoint.get("val_loss", "N/A"):.4f}')
        print(f'   - Val PSNR: {checkpoint.get("val_psnr", "N/A"):.4f}')
        print(f'   - Val SSIM: {checkpoint.get("val_ssim", "N/A"):.4f}')
        print(f'   - 다음 Epoch부터 재개: {start_epoch}')
    
    # 학습
    print('\n학습 시작...')
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        device=device,
        lr=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        start_epoch=start_epoch,
        resume_checkpoint=args.resume,
        l1_weight=args.l1_weight,
        ssim_weight=args.ssim_weight,
        gradient_weight=args.gradient_weight
    )
    
    # 최종 평가
    print('\n최종 평가 중...')
    best_model = CNNModel(in_channels=3, out_channels=3)
    checkpoint = torch.load(f'{args.checkpoint_dir}/best_model.pth', map_location=device)
    best_model.load_state_dict(checkpoint['model_state_dict'])
    best_model.to(device)
    
    print('\n=== Validation Set 평가 ===')
    evaluate_model(best_model, val_loader, device=device)
    
    print('\n✅ 학습 완료!')
    print('ℹ️  Test 데이터 평가는 inference 코드를 사용하세요:')
    print(f'   python inference_cnn.py --model_path {args.checkpoint_dir}/best_model.pth --noisy_dir {args.noisy_dir} --clean_dir {args.clean_dir if args.clean_dir else ""} --visualize')


if __name__ == '__main__':
    main()

