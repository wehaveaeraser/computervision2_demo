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


class CNNModel(nn.Module):
    """
    4-layer CNN 모델 for 이미지 노이즈 제거
    Conv-BN-ReLU 블록 구조
    채널 수: 64 -> 128 -> 256 -> 512
    Residual Learning 적용: 모델이 노이즈만 예측하고 입력에서 빼서 clean 이미지 생성
    """
    def __init__(self, in_channels=3, out_channels=3, use_residual=True):
        super(CNNModel, self).__init__()
        self.use_residual = use_residual
        
        # Layer 1: 64 channels
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Layer 2: 128 channels
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        # Layer 3: 256 channels
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        # Layer 4: 512 channels
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        
        # Decoder: 512 -> 256 -> 128 -> 64 -> 3
        self.conv5 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        
        self.conv6 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        
        self.conv7 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(64)
        
        self.conv8 = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        # Encoder path
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = F.relu(self.bn2(self.conv2(x1)))
        x3 = F.relu(self.bn3(self.conv3(x2)))
        x4 = F.relu(self.bn4(self.conv4(x3)))
        
        # Decoder path
        x5 = F.relu(self.bn5(self.conv5(x4)))
        x6 = F.relu(self.bn6(self.conv6(x5)))
        x7 = F.relu(self.bn7(self.conv7(x6)))
        residual = self.conv8(x7)
        
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


class SSIMLoss(nn.Module):
    """
    SSIM Loss 구현
    """
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2

        sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel

        return 1 - self._ssim(img1, img2, window, self.window_size, channel, self.size_average)


class CombinedLoss(nn.Module):
    """
    L1 Loss + SSIM Loss + Gradient Loss 조합
    Gradient Loss는 노이즈 제거에 집중 (노이즈는 고주파 성분)
    """
    def __init__(self, l1_weight=1.0, ssim_weight=1.0, gradient_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.ssim_loss = SSIMLoss()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.gradient_weight = gradient_weight
        
    def gradient_loss(self, pred, target):
        """
        Gradient Loss: 이미지의 gradient(엣지/경계) 차이를 측정
        노이즈는 고주파 성분이므로 gradient가 크고, 이를 줄이도록 학습
        """
        # Sobel 필터 정의
        sobel_x = torch.tensor([[-1, 0, 1], 
                                [-2, 0, 2], 
                                [-1, 0, 1]], dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], 
                                [0, 0, 0], 
                                [1, 2, 1]], dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
        
        # 각 채널에 대해 gradient 계산
        # RGB 채널이므로 3번 반복
        sobel_x = sobel_x.repeat(pred.shape[1], 1, 1, 1)
        sobel_y = sobel_y.repeat(pred.shape[1], 1, 1, 1)
        
        pred_grad_x = F.conv2d(pred, sobel_x, padding=1, groups=pred.shape[1])
        pred_grad_y = F.conv2d(pred, sobel_y, padding=1, groups=pred.shape[1])
        target_grad_x = F.conv2d(target, sobel_x, padding=1, groups=target.shape[1])
        target_grad_y = F.conv2d(target, sobel_y, padding=1, groups=target.shape[1])
        
        # Gradient magnitude (크기)
        pred_grad = torch.sqrt(pred_grad_x**2 + pred_grad_y**2 + 1e-6)
        target_grad = torch.sqrt(target_grad_x**2 + target_grad_y**2 + 1e-6)
        
        # L1 Loss로 gradient 차이 측정
        return F.l1_loss(pred_grad, target_grad)
        
    def forward(self, pred, target):
        l1 = self.l1_loss(pred, target)
        ssim = self.ssim_loss(pred, target)
        grad = self.gradient_loss(pred, target)
        return self.l1_weight * l1 + self.ssim_weight * ssim + self.gradient_weight * grad


class ImageDataset(Dataset):
    """
    노이즈 이미지와 깨끗한 이미지 쌍을 로드하는 Dataset
    
    사용 방법:
    1. 두 개의 폴더 사용: noisy_dir와 clean_dir가 다른 경우
       - 각 폴더에서 같은 파일명의 이미지를 매칭
    2. 하나의 폴더 사용 (정제된 데이터): noisy_dir와 clean_dir가 같은 경우
       - 파일명 패턴으로 매칭 (*_rain.png <-> *_clean.png)
    3. Inference 전용 모드: inference_only=True
       - clean 이미지 없이 noisy 이미지만 반환 (시각적 평가용)
    4. 매칭 파일 사용: matched_pairs_file이 제공된 경우
       - JSON 파일에서 매칭 정보를 읽어옴 (폴더명 기반 매칭 결과)
    """
    def __init__(self, noisy_dir, clean_dir=None, transform=None, 
                 noisy_pattern='*_rain.png', clean_pattern='*_clean.png',
                 inference_only=False, matched_pairs_file=None, auto_match=False, split=None,
                 reverse_match=False):
        # split 파라미터 추가: None(모든 split), 'train', 'val', 'test'
        # reverse_match: True면 원본 데이터 기준으로 매칭, False면 노이즈 데이터 기준으로 매칭
        self.noisy_dir = Path(noisy_dir)
        self.transform = transform
        self.inference_only = inference_only
        self.split = split  # split 정보 저장
        
        # 자동 매칭 모드: 학습 시 자동으로 폴더명 기반 매칭
        if auto_match and clean_dir is not None:
            print("자동 매칭 모드: 폴더명 기반으로 자동 매칭을 수행합니다...")
            if split:
                print(f"  Split 필터: {split}만 로드합니다.")
            if reverse_match:
                print(f"  매칭 방식: 원본 데이터 기준으로 노이즈 데이터 찾기")
            else:
                print(f"  매칭 방식: 노이즈 데이터 기준으로 원본 데이터 찾기")
            self.pairs = self._auto_match_by_folder_name(noisy_dir, clean_dir, split=split, reverse_match=reverse_match)
            print(f"자동 매칭 완료: {len(self.pairs)}개의 이미지 쌍을 찾았습니다.")
            return
        
        # 매칭 파일 모드: JSON 파일에서 매칭 정보 읽기
        if matched_pairs_file is not None:
            import json
            matched_pairs_path = Path(matched_pairs_file)
            if not matched_pairs_path.exists():
                raise ValueError(f"매칭 파일을 찾을 수 없습니다: {matched_pairs_file}")
            
            with open(matched_pairs_path, 'r', encoding='utf-8') as f:
                matched_data = json.load(f)
            
            self.pairs = []
            for pair in matched_data.get('matched_pairs', []):
                noisy_path = Path(pair['noisy_path'])
                clean_path = Path(pair['clean_path'])
                
                # 절대 경로가 아니면 상대 경로로 처리
                if not noisy_path.is_absolute():
                    noisy_path = self.noisy_dir / pair.get('noisy_relative', pair['noisy_path'])
                if not clean_path.is_absolute():
                    if clean_dir:
                        clean_path = Path(clean_dir) / pair.get('clean_relative', pair['clean_path'])
                    else:
                        # clean_dir가 없으면 noisy_dir 기준으로 처리
                        clean_path = self.noisy_dir / pair.get('clean_relative', pair['clean_path'])
                
                if noisy_path.exists() and clean_path.exists():
                    self.pairs.append((noisy_path, clean_path))
            
            if len(self.pairs) == 0:
                raise ValueError(f"매칭 파일에서 유효한 이미지 쌍을 찾을 수 없습니다: {matched_pairs_file}")
            
            print(f"매칭 파일 모드: {len(self.pairs)}개의 이미지 쌍을 로드했습니다.")
            print(f"  매칭 파일: {matched_pairs_file}")
            return
        
        # Inference 전용 모드: clean 이미지 없이 noisy만 반환
        if inference_only:
            self.single_folder_mode = False
            self.noisy_files = sorted(list(self.noisy_dir.glob('**/*.jpg')) + 
                                      list(self.noisy_dir.glob('**/*.png')))
            if len(self.noisy_files) == 0:
                raise ValueError(f"이미지를 찾을 수 없습니다: {self.noisy_dir}")
            self.pairs = [(f, None) for f in self.noisy_files]  # clean은 None
            print(f"Inference 전용 모드: {len(self.pairs)}개의 이미지를 찾았습니다.")
            return
        
        # clean_dir가 None이거나 noisy_dir와 같으면 하나의 폴더에서 파일명 패턴으로 매칭
        if clean_dir is None or str(self.noisy_dir) == str(Path(clean_dir)):
            self.single_folder_mode = True
            self.data_dir = self.noisy_dir
            
            # 파일명 패턴으로 매칭
            noisy_files = sorted(list(self.data_dir.glob(noisy_pattern)))
            # 원본 파일명을 키로 하는 딕셔너리 생성
            clean_files_dict = {}
            for f in self.data_dir.glob(clean_pattern):
                clean_key = f.stem  # 원본 파일명 그대로 사용
                clean_files_dict[clean_key] = f
            
            # 매칭되는 쌍만 저장
            self.pairs = []
            for noisy_file in noisy_files:
                # 노이즈 파일명: aachen_000004_000019_leftImg8bit_rain_alpha_0.02_...
                # _rain 이후 모든 부분 제거
                noisy_stem = noisy_file.stem
                if '_rain' in noisy_stem:
                    base_name = noisy_stem.split('_rain')[0]  # _rain 이전 부분만 추출
                else:
                    base_name = noisy_stem
                
                clean_file = clean_files_dict.get(base_name)
                if clean_file and clean_file.exists():
                    self.pairs.append((noisy_file, clean_file))
            
            if len(self.pairs) == 0:
                raise ValueError(f"매칭되는 이미지 쌍을 찾을 수 없습니다. "
                               f"폴더: {self.data_dir}, "
                               f"패턴: {noisy_pattern} <-> {clean_pattern}")
            
            print(f"단일 폴더 모드: {len(self.pairs)}개의 이미지 쌍을 찾았습니다.")
        else:
            # 기존 방식: 두 개의 폴더 사용
            self.single_folder_mode = False
            self.clean_dir = Path(clean_dir)
            
            # 이미지 파일 목록 가져오기 (재귀적으로 서브디렉토리 탐색)
            self.noisy_files = sorted(list(self.noisy_dir.glob('**/*.jpg')) + 
                                      list(self.noisy_dir.glob('**/*.png')))
            self.clean_files = sorted(list(self.clean_dir.glob('**/*.jpg')) + 
                                      list(self.clean_dir.glob('**/*.png')))
            
            # 파일명 기반 매칭 (확장자 제외)
            # 서브디렉토리 구조를 고려하여 상대 경로로 매칭
            # _rain, _clean 등의 접미사 제거하여 매칭
            noisy_dict = {}
            for f in self.noisy_files:
                # 상대 경로를 키로 사용하여 같은 파일명이 여러 폴더에 있어도 구분
                rel_path = f.relative_to(self.noisy_dir)
                stem = rel_path.stem
                # _rain, _noisy 등의 접미사 제거
                # 노이즈 파일명: aachen_000004_000019_leftImg8bit_rain_alpha_0.02_...
                if '_rain' in stem:
                    base_name = stem.split('_rain')[0]  # '_rain' 이전 부분만 추출
                elif stem.endswith('_noisy'):
                    base_name = stem[:-6]  # '_noisy' 제거
                else:
                    base_name = stem
                
                key = str(rel_path.parent / base_name) if rel_path.parent != Path('.') else base_name
                noisy_dict[key] = f
            
            clean_dict = {}
            for f in self.clean_files:
                rel_path = f.relative_to(self.clean_dir)
                stem = rel_path.stem
                # _clean, _gt 등의 접미사 제거
                if stem.endswith('_clean'):
                    base_name = stem[:-6]  # '_clean' 제거
                elif stem.endswith('_gt'):
                    base_name = stem[:-3]  # '_gt' 제거
                else:
                    base_name = stem
                
                key = str(rel_path.parent / base_name) if rel_path.parent != Path('.') else base_name
                clean_dict[key] = f
            
            # 매칭되는 쌍만 저장
            self.pairs = []
            for stem in noisy_dict.keys():
                if stem in clean_dict:
                    self.pairs.append((noisy_dict[stem], clean_dict[stem]))
            
            if len(self.pairs) == 0:
                # 디버깅 정보 출력
                print(f"디버깅 정보:")
                print(f"  Noisy 파일 수: {len(noisy_dict)}")
                print(f"  Clean 파일 수: {len(clean_dict)}")
                print(f"  Noisy 키 샘플 (처음 5개): {list(noisy_dict.keys())[:5]}")
                print(f"  Clean 키 샘플 (처음 5개): {list(clean_dict.keys())[:5]}")
                raise ValueError("매칭되는 이미지 쌍을 찾을 수 없습니다. "
                               f"noisy_dir: {self.noisy_dir}, clean_dir: {self.clean_dir}")
            
            # 매칭된 서브디렉토리 정보 출력
            matched_subdirs = set()
            for noisy_path, clean_path in self.pairs[:10]:  # 처음 10개만 확인
                noisy_rel = noisy_path.relative_to(self.noisy_dir)
                clean_rel = clean_path.relative_to(self.clean_dir)
                if len(noisy_rel.parts) > 1:
                    matched_subdirs.add(noisy_rel.parts[0])
            
            print(f"이중 폴더 모드: {len(self.pairs)}개의 이미지 쌍을 찾았습니다.")
            if matched_subdirs:
                print(f"  매칭된 서브디렉토리 샘플: {sorted(list(matched_subdirs))[:5]}")
    
    def _auto_match_by_folder_name(self, noisy_dir, clean_dir, split=None, reverse_match=False):
        """
        폴더명 기반 자동 매칭 (학습 시 자동으로 호출)
        reverse_match=False: 노이즈 데이터 기준으로 원본 데이터 찾기 (기본)
        reverse_match=True: 원본 데이터 기준으로 노이즈 데이터 찾기
        
        노이즈 파일명에서 도시명을 추출하여 원본 데이터의 도시 폴더와 매칭
        예: aachen_000004_000019_leftImg8bit_rain_alpha_0.02_... -> aachen 폴더 내 파일들과 매칭
        노이즈 데이터의 day/night 구분 없이 모두 함께 매칭
        split: None이면 모든 split, 'train', 'val', 'test' 중 하나면 해당 split만
        """
        from collections import defaultdict
        import re
        
        noisy_path = Path(noisy_dir)
        clean_path = Path(clean_dir)
        
        def get_image_files(folder_path):
            """폴더 내 모든 이미지 파일 경로 반환"""
            folder = Path(folder_path)
            if not folder.exists():
                return []
            image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
            image_files = []
            for ext in image_extensions:
                image_files.extend(folder.glob(f'**/*{ext}'))
            return sorted([f for f in image_files])
        
        pairs = []
        # split이 지정되면 해당 split만, 아니면 모든 split
        splits = [split] if split else ['test', 'train', 'val']
        
        # noisy가 이미 특정 split 폴더를 가리키는 경우 (예: test 폴더만)
        # clean의 모든 split에서 매칭하도록 처리
        noisy_is_split_folder = noisy_path.name in ['test', 'train', 'val']
        if noisy_is_split_folder and split is None:
            # noisy는 이미 test/train/val 중 하나를 가리키고 있음
            # clean의 모든 split에서 매칭 시도
            actual_noisy_split = noisy_path.name
            print(f"  노이즈 데이터는 {actual_noisy_split} split만 사용합니다.")
            
            # noisy 파일은 한 번만 수집
            noisy_files = []
            day_path = noisy_path / 'day'
            night_path = noisy_path / 'night'
            if day_path.exists():
                noisy_files.extend(get_image_files(day_path))
            if night_path.exists():
                noisy_files.extend(get_image_files(night_path))
            
            if len(noisy_files) == 0:
                print(f"  ⚠️  노이즈 파일을 찾을 수 없습니다: {noisy_path}")
                return []
            
            print(f"  노이즈 파일 {len(noisy_files)}개 수집 완료")
            
            # clean 폴더 구조 확인: split 구조인지, 직접 도시 폴더인지
            clean_has_split = (clean_path / 'test').exists() or (clean_path / 'train').exists() or (clean_path / 'val').exists()
            
            if clean_has_split:
                # clean 폴더 안에 test/train/val이 있는 경우
                print(f"  clean 데이터는 모든 split (test/train/val)에서 매칭을 시도합니다.")
                clean_splits = ['test', 'train', 'val']
            else:
                # clean 폴더 안에 도시명 폴더가 직접 있는 경우
                print(f"  clean 폴더 안의 모든 도시 폴더에서 매칭을 시도합니다.")
                clean_splits = [None]  # None은 clean_path 자체를 의미
            
            # clean의 모든 split/폴더에서 매칭 시도
            for split_name in clean_splits:
                if split_name is None:
                    # clean 폴더 직접 사용
                    clean_split_path = clean_path
                else:
                    clean_split_path = clean_path / split_name
                
                if not clean_split_path.exists():
                    continue
                
                # 원본 데이터 수집 (각 폴더별로)
                clean_folders = {}
                subfolders = [d for d in clean_split_path.iterdir() if d.is_dir()]
                
                if subfolders:
                    for subfolder in subfolders:
                        clean_files = get_image_files(subfolder)
                        if len(clean_files) > 0:
                            clean_folders[subfolder.name] = clean_files
                else:
                    clean_files = get_image_files(clean_split_path)
                    if len(clean_files) > 0:
                        clean_folders['root'] = clean_files
                
                if len(clean_folders) == 0:
                    continue
                
                # 매칭 로직
                split_pairs_count = len(pairs)
                if not reverse_match:
                    # 노이즈 데이터 기준으로 원본 데이터 찾기
                    clean_dict = {}
                    for folder_name, clean_files in clean_folders.items():
                        for clean_file in clean_files:
                            clean_stem = clean_file.stem
                            if clean_stem not in clean_dict:
                                clean_dict[clean_stem] = []
                            clean_dict[clean_stem].append(clean_file)
                    
                    # 노이즈 파일과 매칭
                    for noisy_file in noisy_files:
                        noisy_stem = noisy_file.stem
                        if '_rain' in noisy_stem:
                            noisy_key = noisy_stem.split('_rain')[0]
                        else:
                            noisy_key = noisy_stem
                        
                        if noisy_key in clean_dict:
                            # 매칭된 clean 파일 중 첫 번째 사용
                            pairs.append((noisy_file, clean_dict[noisy_key][0]))
                else:
                    # 원본 데이터 기준으로 노이즈 데이터 찾기
                    noisy_dict = {}
                    for noisy_file in noisy_files:
                        noisy_stem = noisy_file.stem
                        if '_rain' in noisy_stem:
                            noisy_key = noisy_stem.split('_rain')[0]
                        else:
                            noisy_key = noisy_stem
                        if noisy_key not in noisy_dict:
                            noisy_dict[noisy_key] = noisy_file
                    
                    for folder_name, clean_files in clean_folders.items():
                        for clean_file in clean_files:
                            clean_stem = clean_file.stem
                            if clean_stem in noisy_dict:
                                pairs.append((noisy_dict[clean_stem], clean_file))
                
                new_pairs = len(pairs) - split_pairs_count
                print(f"  {split_name}: {len(clean_folders)}개 폴더, {new_pairs}개 매칭")
            
            if len(pairs) > 0:
                print(f"  총 {len(pairs)}개의 이미지 쌍을 찾았습니다.")
            return pairs
        
        # 기존 로직 (각 split별로 매칭)
        for split_name in splits:
            noisy_split_path = noisy_path / split_name
            clean_split_path = clean_path / split_name
            
            if not noisy_split_path.exists() or not clean_split_path.exists():
                if split:  # split이 지정되었는데 폴더가 없으면 경고
                    print(f"⚠️  경고: {split_name} 폴더를 찾을 수 없습니다.")
                continue
            
            # 노이즈 데이터 수집 (day/night 구분 없이 모두)
            noisy_files = []
            day_path = noisy_split_path / 'day'
            night_path = noisy_split_path / 'night'
            
            if day_path.exists():
                noisy_files.extend(get_image_files(day_path))
            if night_path.exists():
                noisy_files.extend(get_image_files(night_path))
            
            if len(noisy_files) == 0:
                continue
            
            # 원본 데이터 수집 (각 폴더별로)
            clean_folders = {}
            subfolders = [d for d in clean_split_path.iterdir() if d.is_dir()]
            
            if subfolders:
                for subfolder in subfolders:
                    clean_files = get_image_files(subfolder)
                    if len(clean_files) > 0:
                        clean_folders[subfolder.name] = clean_files
            else:
                clean_files = get_image_files(clean_split_path)
            if len(clean_files) > 0:
                clean_folders['root'] = clean_files
            
            if reverse_match:
                # 원본 데이터 기준으로 노이즈 데이터 찾기
                noisy_dict = {}  # 노이즈 데이터를 딕셔너리로 변환
                noisy_dict_multiple = {}  # 같은 키에 여러 파일이 있을 수 있음
                
                import re  # 정규표현식 사용
                
                for noisy_file in noisy_files:
                    noisy_stem = noisy_file.stem
                    # 노이즈 파일명: aachen_000004_000019_leftImg8bit_rain_alpha_0.02_...
                    # _rain 이후 모든 부분 제거
                    if '_rain' in noisy_stem:
                        noisy_key = noisy_stem.split('_rain')[0]  # aachen_000004_000019_leftImg8bit
                    else:
                        noisy_key = noisy_stem
                    
                    # 같은 키가 여러 개 있을 수 있으므로 리스트로 저장
                    if noisy_key not in noisy_dict_multiple:
                        noisy_dict_multiple[noisy_key] = []
                    noisy_dict_multiple[noisy_key].append(noisy_file)
                    # 첫 번째 것만 사용 (기존 동작 유지)
                    if noisy_key not in noisy_dict:
                        noisy_dict[noisy_key] = noisy_file
                
                print(f"  🔍 노이즈 데이터 키 개수: {len(noisy_dict)}개 (총 파일: {len(noisy_files)}개)")
                if len(noisy_dict) < len(noisy_files):
                    print(f"     ⚠️  중복 키가 있습니다. (중복: {len(noisy_files) - len(noisy_dict)}개)")
                
                # 원본 데이터를 순회하면서 노이즈 데이터 찾기
                matched_count = 0
                unmatched_samples = []
                matched_samples = []
                
                for folder_name, clean_files in clean_folders.items():
                    for clean_file in clean_files:
                        # 원본 파일명: aachen_000000_000019_leftImg8bit.png
                        clean_stem = clean_file.stem  # aachen_000000_000019_leftImg8bit
                        
                        # 노이즈 데이터에서 매칭
                        if clean_stem in noisy_dict:
                            pairs.append((noisy_dict[clean_stem], clean_file))
                            matched_count += 1
                            # 매칭 성공 샘플 수집 (처음 3개만)
                            if len(matched_samples) < 3:
                                matched_samples.append((clean_file.name, clean_stem))
                        else:
                            # 매칭 실패 샘플 수집 (처음 10개)
                            if len(unmatched_samples) < 10:
                                unmatched_samples.append((clean_file.name, clean_stem))
                
                # 매칭 성공 샘플 출력
                if matched_samples:
                    print(f"  ✅ 매칭 성공 샘플 (처음 {len(matched_samples)}개):")
                    for clean_name, clean_stem in matched_samples:
                        print(f"     - {clean_name[:70]}... -> {clean_stem}")
                
                # 매칭 실패 샘플 출력
                if unmatched_samples:
                    print(f"  ⚠️  매칭 실패 샘플 (처음 {len(unmatched_samples)}개):")
                    for clean_name, clean_stem in unmatched_samples[:10]:
                        # 노이즈 데이터의 키 샘플도 출력
                        noisy_keys_sample = list(noisy_dict.keys())[:5]
                        print(f"     - 원본: {clean_name[:50]}... (stem: {clean_stem})")
                        print(f"       노이즈 키 샘플: {noisy_keys_sample}")
                
                if matched_count > 0:
                    total_clean = sum(len(files) for files in clean_folders.values())
                    unmatched_clean_count = total_clean - matched_count
                    print(f'  ✅ {split_name} (원본 기준): {matched_count}개 매칭 성공')
                    print(f'     - 노이즈 파일: {len(noisy_files)}개')
                    print(f'     - 원본 파일: {total_clean}개')
                    print(f'     - 매칭된 쌍: {matched_count}개')
                    if unmatched_clean_count > 0:
                        print(f'     - ⚠️  매칭되지 않은 원본 파일: {unmatched_clean_count}개 (노이즈 데이터에 없음)')
            else:
                # 노이즈 데이터 기준으로 원본 데이터 찾기 (기존 방식)
                # 1단계 매칭: 정확한 파일명만 매칭
                # 원본 데이터를 딕셔너리로 변환
                clean_dict = {}  # 정확한 파일명으로 매칭
                
                for folder_name, clean_files in clean_folders.items():
                    for clean_file in clean_files:
                        # 원본 파일명: aachen_000000_000019_leftImg8bit.png
                        clean_stem = clean_file.stem  # aachen_000000_000019_leftImg8bit
                        
                        # 정확한 키 (전체 stem)
                        clean_dict[clean_stem] = clean_file
                
                # 노이즈 데이터를 정확한 파일명으로만 매칭
                matched_count = 0
                unmatched_samples = []
                
                for noisy_file in noisy_files:
                    noisy_stem = noisy_file.stem
                    # 노이즈 파일명: aachen_000004_000019_leftImg8bit_rain_alpha_0.02_...
                    # _rain 이후 모든 부분 제거
                    if '_rain' in noisy_stem:
                        noisy_key = noisy_stem.split('_rain')[0]  # aachen_000004_000019_leftImg8bit
                    else:
                        noisy_key = noisy_stem
                    
                    # 정확한 매칭 시도
                    if noisy_key in clean_dict:
                        pairs.append((noisy_file, clean_dict[noisy_key]))
                        matched_count += 1
                    else:
                        # 매칭 실패 샘플 수집 (처음 3개만)
                        if len(unmatched_samples) < 3:
                            unmatched_samples.append((noisy_file.name, noisy_key))
                
                # 매칭 실패 샘플 출력
                if unmatched_samples:
                    print(f"  ⚠️  매칭 실패 샘플 (처음 {len(unmatched_samples)}개):")
                    # 원본 데이터의 키 샘플도 출력
                    clean_keys_sample = list(clean_dict.keys())[:5]
                    print(f"     원본 키 샘플: {clean_keys_sample}")
                    for noisy_name, noisy_key in unmatched_samples[:5]:
                        print(f"     - 노이즈: {noisy_name[:60]}...")
                        print(f"       추출된 키: {noisy_key}")
                        # 유사한 키 찾기
                        similar_keys = [k for k in clean_keys_sample if noisy_key.split('_')[0] in k or k.split('_')[0] in noisy_key]
                        if similar_keys:
                            print(f"       유사한 원본 키: {similar_keys[:3]}")
                
                if matched_count > 0:
                    total_clean = sum(len(files) for files in clean_folders.values())
                    unmatched_noisy_count = len(noisy_files) - matched_count
                    print(f'  ✅ {split_name} (노이즈 기준): {matched_count}개 매칭 성공')
                    print(f'     - 노이즈 파일: {len(noisy_files)}개')
                    print(f'     - 원본 파일: {total_clean}개')
                    print(f'     - 매칭된 쌍: {matched_count}개')
                    if unmatched_noisy_count > 0:
                        print(f'     - ⚠️  매칭되지 않은 노이즈 파일: {unmatched_noisy_count}개 (원본 데이터에 없음)')
                else:
                    # 매칭이 하나도 안 되면 더 자세한 정보 출력
                    total_clean = sum(len(files) for files in clean_folders.values())
                    print(f'  ❌ {split_name} (노이즈 기준): 매칭 실패')
                    print(f'     - 노이즈 파일: {len(noisy_files)}개')
                    print(f'     - 원본 파일: {total_clean}개')
                    if len(noisy_files) > 0 and total_clean > 0:
                        # 노이즈 키 샘플
                        noisy_keys_sample = []
                        for nf in noisy_files[:5]:
                            ns = nf.stem
                            if '_rain' in ns:
                                nk = ns.split('_rain')[0]
                            else:
                                nk = ns
                            noisy_keys_sample.append(nk)
                        print(f'     - 노이즈 키 샘플: {noisy_keys_sample}')
                        print(f'     - 원본 키 샘플: {list(clean_dict.keys())[:5]}')
        
        return pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        noisy_path, clean_path = self.pairs[idx]
        
        # Noisy 이미지 로드
        noisy_img = cv2.imread(str(noisy_path))
        if noisy_img is None:
            raise ValueError(f"이미지를 로드할 수 없습니다: {noisy_path}")
        
        # Inference 전용 모드면 clean 이미지 없이 반환
        if self.inference_only or clean_path is None:
            # 이미지 전처리
            if noisy_img.shape[:2] != (512, 512):
                noisy_img = cv2.resize(noisy_img, (512, 512), interpolation=cv2.INTER_LINEAR)
            noisy_img = cv2.cvtColor(noisy_img, cv2.COLOR_BGR2RGB)
            noisy_img = noisy_img.astype(np.float32) / 255.0
            noisy_img = torch.from_numpy(noisy_img).permute(2, 0, 1)
            
            if self.transform:
                noisy_img = self.transform(noisy_img)
            
            return noisy_img, None  # clean은 None 반환
        
        # 기존 코드 (clean 이미지도 로드)
        clean_img = cv2.imread(str(clean_path))
        if clean_img is None:
            raise ValueError(f"이미지를 로드할 수 없습니다: {clean_path}")
        
        # 이미지 크기 확인 및 리사이즈 (512x512로 통일)
        if noisy_img.shape[:2] != (512, 512):
            noisy_img = cv2.resize(noisy_img, (512, 512), interpolation=cv2.INTER_LINEAR)
        if clean_img.shape[:2] != (512, 512):
            clean_img = cv2.resize(clean_img, (512, 512), interpolation=cv2.INTER_LINEAR)
        
        # BGR to RGB
        noisy_img = cv2.cvtColor(noisy_img, cv2.COLOR_BGR2RGB)
        clean_img = cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB)
        
        # 0-255 -> 0-1로 정규화
        noisy_img = noisy_img.astype(np.float32) / 255.0
        clean_img = clean_img.astype(np.float32) / 255.0
        
        # HWC -> CHW 변환
        noisy_img = torch.from_numpy(noisy_img).permute(2, 0, 1)
        clean_img = torch.from_numpy(clean_img).permute(2, 0, 1)
        
        if self.transform:
            noisy_img = self.transform(noisy_img)
            clean_img = self.transform(clean_img)
        
        return noisy_img, clean_img


class EarlyStopping:
    """
    Early Stopping 구현
    """
    def __init__(self, patience=10, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()


def calculate_psnr(img1, img2):
    """PSNR 계산"""
    img1 = img1.cpu().numpy()
    img2 = img2.cpu().numpy()
    return psnr(img1, img2, data_range=1.0)


def calculate_ssim(img1, img2):
    """SSIM 계산 (최적화 버전)"""
    img1 = img1.cpu().numpy()
    img2 = img2.cpu().numpy()
    
    # win_size를 한 번만 계산 (512x512 이미지이므로 11 사용)
    win_size = 11  # 512x512 이미지에 적합
    
    def compute_ssim_single(img1_single, img2_single):
        """단일 이미지 쌍에 대한 SSIM 계산"""
        # 이미지 크기 확인 (H, W, C)
        h, w = img1_single.shape[:2]
        
        # win_size를 이미지 크기에 맞게 조정
        actual_win_size = min(win_size, min(h, w))
        if actual_win_size % 2 == 0:
            actual_win_size -= 1
        actual_win_size = max(3, actual_win_size)
        
        # 최신 API 사용 (channel_axis)
        try:
            # scikit-image 0.19+ 버전
            if len(img1_single.shape) == 3 and img1_single.shape[2] == 3:
                return ssim(img1_single, img2_single, 
                           win_size=actual_win_size,
                           channel_axis=2,
                           data_range=1.0)
            else:
                return ssim(img1_single, img2_single,
                           win_size=actual_win_size,
                           data_range=1.0)
        except TypeError:
            # 구버전 호환 (multichannel 사용)
            if len(img1_single.shape) == 3 and img1_single.shape[2] == 3:
                return ssim(img1_single, img2_single,
                           win_size=actual_win_size,
                           multichannel=True,
                           data_range=1.0)
            else:
                return ssim(img1_single, img2_single,
                           win_size=actual_win_size,
                           data_range=1.0)
    
    if len(img1.shape) == 4:  # batch dimension
        ssim_values = []
        for i in range(img1.shape[0]):
            img1_transposed = img1[i].transpose(1, 2, 0)
            img2_transposed = img2[i].transpose(1, 2, 0)
            ssim_val = compute_ssim_single(img1_transposed, img2_transposed)
            ssim_values.append(ssim_val)
        return np.mean(ssim_values)
    else:
        img1_transposed = img1.transpose(1, 2, 0)
        img2_transposed = img2.transpose(1, 2, 0)
        return compute_ssim_single(img1_transposed, img2_transposed)


def train_model(model, train_loader, val_loader, num_epochs=10, 
                device='cuda', lr=1e-4, checkpoint_dir='checkpoints',
                start_epoch=0, resume_checkpoint=None, l1_weight=1.0, ssim_weight=1.0, gradient_weight=0.5):
    """
    모델 학습 함수
    
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
    
    # Mixed Precision Training (안정적인 설정으로 활성화)
    use_amp = device.type == 'cuda'  # GPU에서만 사용
    # 더 보수적인 loss scaling으로 학습 안정성 향상
    scaler = torch.cuda.amp.GradScaler(init_scale=2.**10, growth_factor=2.0, backoff_factor=0.5) if use_amp else None
    
    # 체크포인트에서 재개
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # Optimizer state를 GPU로 이동
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
        print(f'✅ Mixed Precision Training 활성화 (안정적인 설정)')
        print(f'   - Loss scaling: init_scale=2^10 (보수적 설정)')
    else:
        print(f'⚠️  Mixed Precision Training 비활성화 (CPU 모드)')
    
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
        torch.cuda.empty_cache()  # 캐시 정리
        torch.cuda.synchronize()  # 동기화
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
        
        # Epoch 시작 시간 기록
        import time
        epoch_start_time = time.time()
        
        # 첫 epoch에서 device 확인
        if epoch == 0:
            print(f'🔍 Device 확인:')
            print(f'   - device 변수: {device}')
            print(f'   - device.type: {device.type}')
            print(f'   - torch.cuda.is_available(): {torch.cuda.is_available()}')
            print(f'   - 모델이 GPU에 있는지: {next(model.parameters()).is_cuda}')
            print(f'   - 모델 device: {next(model.parameters()).device}')
            if device.type == 'cuda':
                print(f'   - 현재 CUDA device: {torch.cuda.current_device()}')
                print(f'   - GPU 이름: {torch.cuda.get_device_name(device)}')
            print()
        
        for batch_idx, (noisy, clean) in enumerate(train_loader):
            # 배치 시작 시간 기록
            batch_start_time = time.time()
            
            # 첫 배치 로딩 완료 알림
            if batch_idx == 0:
                data_load_time = time.time() - batch_start_time
                print(f'첫 배치 로딩 완료! Shape: {noisy.shape}')
                print(f'⏱️  데이터 로딩 시간: {data_load_time:.3f}초')
                print(f'   - 로딩 후: noisy.is_cuda={noisy.is_cuda}, device={noisy.device}')
            
            # 데이터 전송 시간 측정
            data_transfer_start = time.time()
            noisy = noisy.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            data_transfer_time = time.time() - data_transfer_start
            
            # 첫 배치에서 GPU 전송 확인
            if batch_idx == 0:
                print(f'   - GPU 전송 후: noisy.is_cuda={noisy.is_cuda}, device={noisy.device}')
                print(f'   - 모델 device: {next(model.parameters()).device}')
                if not noisy.is_cuda and device.type == 'cuda':
                    print(f'   ⚠️  경고: GPU로 전송 실패! CPU로 실행 중입니다.')
                    print(f'   ⚠️  device={device}인데 데이터가 CPU에 있습니다.')
                    print(f'   ⚠️  CUDA 사용 가능 여부를 확인하세요.')
                elif noisy.is_cuda:
                    print(f'   ✅ GPU 사용 확인됨!')
                print()
            
            # Forward pass 시간 측정 (Mixed Precision 사용)
            forward_start = time.time()
            optimizer.zero_grad()
            
            if scaler is not None:
                # Mixed Precision forward pass
                with torch.amp.autocast('cuda'):
                    output = model(noisy)
                    loss = criterion(output, clean)
            else:
                output = model(noisy)
                loss = criterion(output, clean)
            
            # 동기화는 최소화 (시간 측정을 위해 첫 배치에서만)
            if device.type == 'cuda' and batch_idx == 0:
                torch.cuda.synchronize()
            forward_time = time.time() - forward_start
            
            # Backward pass 시간 측정 (Mixed Precision 사용)
            backward_start = time.time()
            
            if scaler is not None:
                # Mixed Precision backward pass
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
            backward_time = time.time() - backward_start
            
            train_loss += loss.item()
            
            # 총 배치 처리 시간
            total_batch_time = time.time() - batch_start_time
            
            # 2배치마다 진행 상황 및 시간 분석 출력
            if (batch_idx + 1) % 2 == 0:
                avg_loss_so_far = train_loss / (batch_idx + 1)
                elapsed_since_epoch = time.time() - epoch_start_time
                avg_time_per_batch = elapsed_since_epoch / (batch_idx + 1)
                estimated_remaining = avg_time_per_batch * (len(train_loader) - (batch_idx + 1))
                
                print(f'  Progress: [{batch_idx+1}/{len(train_loader)}] batches, Avg Loss: {avg_loss_so_far:.4f}')
                print(f'  ⏱️  시간 분석:')
                print(f'     - 배치당 평균: {avg_time_per_batch:.2f}초')
                print(f'     - 예상 남은 시간: {estimated_remaining/60:.1f}분')
                if batch_idx == 1:  # 첫 2배치에서만 상세 정보
                    print(f'     - 데이터 전송: {data_transfer_time:.3f}초')
                    print(f'     - Forward pass: {forward_time:.3f}초')
                    print(f'     - Backward pass: {backward_time:.3f}초')
                    print(f'     - 총 배치 시간: {total_batch_time:.3f}초')
        
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
                    # Mixed Precision forward pass (validation)
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
        
        # GPU 메모리 사용량 출력 (매 epoch마다)
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
    print("CNN 모델 정의 완료!")
    print("사용 방법:")
    print("1. 데이터셋 준비: noisy_dir, clean_dir 설정")
    print("2. DataLoader 생성")
    print("3. train_model() 함수로 학습")
    print("4. evaluate_model() 함수로 평가")
    
    # 모델 생성 예제
    model = CNNModel(in_channels=3, out_channels=3)
    print(f"\n모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")

