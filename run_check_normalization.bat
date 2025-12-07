@echo off
REM CNN 모델 정규화 점검
python check_normalization.py --model_path team/checkpoints/cnn/best_model.pth --model_type cnn --noisy_dir team/train_img/data --clean_dir team/train_img/gt --num_samples 3

REM U-Net 모델 정규화 점검
python check_normalization.py --model_path team/checkpoints/unet/best_model.pth --model_type unet --noisy_dir team/train_img/data --clean_dir team/train_img/gt --num_samples 3

