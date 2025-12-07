import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os
from pathlib import Path
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt

# 기존 코드의 유틸리티 함수들을 재사용
from cnn_model import SSIMLoss, CombinedLoss, ImageDataset, EarlyStopping, calculate_psnr, calculate_ssim


class UNet(nn.Module):
    """
    U-Net 모델 for 이미지 노이즈 제거
    Encoder-Decoder 구조 with Skip Connections
    Residual Learning 적용: 모델이 노이즈만 예측하고 입력에서 빼서 clean 이미지 생성
    """
    def __init__(self, in_channels=3, out_channels=3, base_channels=64, use_residual=True):
        super(UNet, self).__init__()
        self.use_residual = use_residual
        
        # Encoder (Downsampling path)
        self.enc1 = self._conv_block(in_channels, base_channels)
        self.enc2 = self._conv_block(base_channels, base_channels * 2)
        self.enc3 = self._conv_block(base_channels * 2, base_channels * 4)
        self.enc4 = self._conv_block(base_channels * 4, base_channels * 8)
        
        # Bottleneck
        self.bottleneck = self._conv_block(base_channels * 8, base_channels * 16)
        
        # Decoder (Upsampling path) with skip connections
        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.dec4 = self._conv_block(base_channels * 16, base_channels * 8)  # 8*2 = 16 (skip connection)
        
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = self._conv_block(base_channels * 8, base_channels * 4)  # 4*2 = 8 (skip connection)
        
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(base_channels * 4, base_channels * 2)  # 2*2 = 4 (skip connection)
        
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(base_channels * 2, base_channels)  # 1*2 = 2 (skip connection)
        
        # Final output layer
        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        
        # MaxPool for downsampling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def _conv_block(self, in_channels, out_channels):
        """
        Conv-BN-ReLU 블록
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder path (with skip connections 저장)
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))
        
        # Decoder path (with skip connections)
        dec4 = self.up4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)  # Skip connection
        dec4 = self.dec4(dec4)
        
        dec3 = self.up3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)  # Skip connection
        dec3 = self.dec3(dec3)
        
        dec2 = self.up2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)  # Skip connection
        dec2 = self.dec2(dec2)
        
        dec1 = self.up1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)  # Skip connection
        dec1 = self.dec1(dec1)
        
        # Final output: 노이즈만 예측
        residual = self.final_conv(dec1)
        
        # Residual Learning: 모델이 노이즈만 예측하고 입력에서 빼서 clean 이미지 생성
        if self.use_residual:
            # residual을 [-1, 1] 범위로 정규화 (tanh 사용)
            residual = torch.tanh(residual)
            # 입력에서 노이즈를 빼서 clean 이미지 생성
            out = x - residual
            # [0, 1] 범위로 클리핑
            out = torch.clamp(out, 0.0, 1.0)
        else:
            # 기존 방식: 전체 이미지 예측
            out = torch.sigmoid(residual)
        
        return out


def train_model(model, train_loader, val_loader, num_epochs=10, 
                device='cuda', lr=1e-4, checkpoint_dir='checkpoints',
                start_epoch=0, resume_checkpoint=None, l1_weight=1.0, ssim_weight=1.0, gradient_weight=0.5):
    """
    모델 학습 함수 (CNN과 동일한 구조)
    
    Args:
        l1_weight: L1 Loss 가중치 (default: 1.0)
        ssim_weight: SSIM Loss 가중치 (default: 1.0)
        gradient_weight: Gradient Loss 가중치 (default: 0.5)
    """
    # 디렉토리 생성
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 손실 함수, 옵티마이저 설정 (가중치 튜닝 가능)
    criterion = CombinedLoss(l1_weight=l1_weight, ssim_weight=ssim_weight, gradient_weight=gradient_weight)
    print(f"Loss 가중치: L1={l1_weight:.2f}, SSIM={ssim_weight:.2f}, Gradient={gradient_weight:.2f}")
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Mixed Precision Training
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', init_scale=2.**10, growth_factor=2.0, backoff_factor=0.5) if use_amp else None
    
    # 체크포인트에서 재개
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        checkpoint = torch.load(resume_checkpoint, map_location=device, weights_only=False)
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if device.type == 'cuda':
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(device)
            print(f'✅ Optimizer 상태 복원 완료')
        if 'scaler_state_dict' in checkpoint and scaler is not None:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            print(f'✅ Scaler 상태 복원 완료')
        best_val_loss = checkpoint.get('val_loss', float('inf'))
        print(f'✅ Best Val Loss 복원: {best_val_loss:.4f}')
    else:
        best_val_loss = float('inf')
    
    if use_amp:
        print(f'✅ Mixed Precision Training 활성화')
    else:
        print(f'⚠️  Mixed Precision Training 비활성화 (CPU 모드)')
    
    # Early Stopping
    # Early Stopping (patience 줄여서 더 빨리 중단)
    early_stopping = EarlyStopping(patience=5, min_delta=0.001)
    
    # 학습 기록
    train_losses = []
    val_losses = []
    val_psnrs = []
    val_ssims = []
    
    model.to(device)
    
    # GPU 메모리 정리
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print(f'\nGPU 메모리 정리 완료!')
        print(f'GPU 메모리 사용량 (정리 후):')
        print(f'  할당됨: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB')
        print(f'  예약됨: {torch.cuda.memory_reserved(device) / 1024**3:.2f} GB')
        print()
    
    for epoch in range(start_epoch, num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        print(f'\n=== Epoch [{epoch+1}/{num_epochs}] ===')
        print('Training 시작...')
        
        import time
        epoch_start_time = time.time()
        
        for batch_idx, (noisy, clean) in enumerate(train_loader):
            batch_start_time = time.time()
            
            noisy = noisy.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            # Forward pass
            optimizer.zero_grad()
            
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    output = model(noisy)
                    loss = criterion(output, clean)
            else:
                output = model(noisy)
                loss = criterion(output, clean)
            
            # 동기화는 최소화 (시간 측정을 위해 첫 배치에서만)
            if device.type == 'cuda' and batch_idx == 0:
                torch.cuda.synchronize()
            
            # Backward pass
            if scaler is not None:
                scaler.scale(loss).backward()
                # Gradient clipping은 20배치마다만 수행 (속도 개선)
                if (batch_idx + 1) % 20 == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                # Gradient Clipping은 20배치마다만 수행 (속도 개선)
                if (batch_idx + 1) % 20 == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()
            
            # 동기화는 완전히 제거 (시간 측정용이므로 학습에는 영향 없음)
            # if device.type == 'cuda' and (batch_idx + 1) % 10 == 0:
            #     torch.cuda.synchronize()
            
            train_loss += loss.item()
            
            # 진행 상황 출력 (2배치마다)
            if (batch_idx + 1) % 2 == 0:
                avg_loss_so_far = train_loss / (batch_idx + 1)
                elapsed_since_epoch = time.time() - epoch_start_time
                avg_time_per_batch = elapsed_since_epoch / (batch_idx + 1)
                estimated_remaining = avg_time_per_batch * (len(train_loader) - (batch_idx + 1))
                
                print(f'  Progress: [{batch_idx+1}/{len(train_loader)}] batches, Avg Loss: {avg_loss_so_far:.4f}')
                print(f'  ⏱️  배치당 평균: {avg_time_per_batch:.2f}초, 예상 남은 시간: {estimated_remaining/60:.1f}분')
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        print(f'Training 완료! Train Loss: {train_loss:.4f}')
        print('Validation 시작...')
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_psnr_sum = 0.0
        val_ssim_sum = 0.0
        
        with torch.no_grad():
            for val_batch_idx, (noisy, clean) in enumerate(val_loader):
                noisy = noisy.to(device, non_blocking=True)
                clean = clean.to(device, non_blocking=True)
                
                if scaler is not None:
                    with torch.amp.autocast('cuda'):
                        output = model(noisy)
                        loss = criterion(output, clean)
                else:
                    output = model(noisy)
                    loss = criterion(output, clean)
                
                val_loss += loss.item()
                
                # PSNR, SSIM 계산
                val_psnr_sum += calculate_psnr(output, clean)
                val_ssim_sum += calculate_ssim(output, clean)
                
                # Validation 진행 상황 출력 (5배치마다)
                if (val_batch_idx + 1) % 5 == 0:
                    print(f'  Validation Progress: [{val_batch_idx+1}/{len(val_loader)}] batches')
        
        val_loss /= len(val_loader)
        val_psnr = val_psnr_sum / len(val_loader)
        val_ssim = val_ssim_sum / len(val_loader)
        
        val_losses.append(val_loss)
        val_psnrs.append(val_psnr)
        val_ssims.append(val_ssim)
        
        print(f'Validation 완료!')
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        print(f'Val PSNR: {val_psnr:.4f}, Val SSIM: {val_ssim:.4f}')
        
        # GPU 메모리 사용량 출력
        if device.type == 'cuda':
            allocated = torch.cuda.memory_allocated(device) / 1024**3
            reserved = torch.cuda.memory_reserved(device) / 1024**3
            print(f'GPU Memory: {allocated:.2f} GB / {reserved:.2f} GB')
        
        # Checkpoint 저장 (best validation loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_psnr': val_psnr,
                'val_ssim': val_ssim,
            }
            if scaler is not None:
                checkpoint_data['scaler_state_dict'] = scaler.state_dict()
            # 경로 정규화 및 디렉토리 확인
            checkpoint_path = Path(checkpoint_dir) / 'best_model.pth'
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint_data, str(checkpoint_path))
            print(f'Checkpoint saved! (Val Loss: {val_loss:.4f})')
        
        # Early Stopping 체크
        if early_stopping(val_loss, model):
            print(f'Early stopping at epoch {epoch+1}')
            break
        
        print('-' * 50)
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_psnrs': val_psnrs,
        'val_ssims': val_ssims
    }


def evaluate_model(model, test_loader, device='cuda'):
    """
    모델 평가 함수
    """
    model.eval()
    test_psnr_sum = 0.0
    test_ssim_sum = 0.0
    
    with torch.no_grad():
        for noisy, clean in test_loader:
            noisy = noisy.to(device)
            clean = clean.to(device)
            
            output = model(noisy)
            
            test_psnr_sum += calculate_psnr(output, clean)
            test_ssim_sum += calculate_ssim(output, clean)
    
    avg_psnr = test_psnr_sum / len(test_loader)
    avg_ssim = test_ssim_sum / len(test_loader)
    
    print(f'Test PSNR: {avg_psnr:.4f}')
    print(f'Test SSIM: {avg_ssim:.4f}')
    
    return avg_psnr, avg_ssim


if __name__ == '__main__':
    # 예제 사용법
    print("U-Net 모델 정의 완료!")
    print("사용 방법:")
    print("1. 데이터셋 준비: noisy_dir, clean_dir 설정")
    print("2. DataLoader 생성")
    print("3. train_model() 함수로 학습")
    print("4. evaluate_model() 함수로 평가")
    
    # 모델 생성 예제
    model = UNet(in_channels=3, out_channels=3)
    print(f"\n모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")

