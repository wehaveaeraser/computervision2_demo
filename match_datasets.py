"""
데이터셋 매칭 스크립트

노이즈 데이터와 원본 데이터를 폴더명 기반으로 매칭하는 스크립트
- 노이즈: DATASET_FAST_FINAL copy/test/day, test/night 등 (day/night 구분 무시)
- 원본: test/berlin, test/bielefeld 등 (폴더명으로 매칭)
"""
import os
from pathlib import Path
from collections import defaultdict
import argparse
import json


def get_image_files(folder_path):
    """폴더 내 모든 이미지 파일 경로 반환"""
    folder = Path(folder_path)
    if not folder.exists():
        return []
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(folder.glob(f'**/*{ext}'))
    
    return sorted([str(f) for f in image_files])


def match_by_folder_name(noisy_dir, clean_dir, output_file='matched_pairs.json'):
    """
    폴더명 기반으로 노이즈 데이터와 원본 데이터 매칭
    
    매칭 로직:
    1. 원본 데이터의 폴더명을 키로 사용 (예: berlin, bielefeld 등)
    2. 노이즈 데이터는 split(test/train/val)과 상관없이 모든 이미지를 수집
    3. 원본 데이터의 각 폴더에 대해, 같은 split 내의 노이즈 이미지와 매칭
    4. day/night 구분 없이 모두 함께 매칭
    """
    print("=" * 60)
    print("🔗 데이터셋 매칭 시작")
    print("=" * 60)
    
    noisy_path = Path(noisy_dir)
    clean_path = Path(clean_dir)
    
    if not noisy_path.exists():
        raise ValueError(f"노이즈 데이터 경로를 찾을 수 없습니다: {noisy_dir}")
    if not clean_path.exists():
        raise ValueError(f"원본 데이터 경로를 찾을 수 없습니다: {clean_dir}")
    
    matched_pairs = []
    stats = {
        'total_pairs': 0,
        'by_split': defaultdict(int),
        'by_folder': defaultdict(int),
        'unmatched_noisy': [],
        'unmatched_clean': []
    }
    
    splits = ['test', 'train', 'val']
    
    # 각 split에 대해 매칭 수행
    for split in splits:
        print(f"\n📂 {split} split 처리 중...")
        
        noisy_split_path = noisy_path / split
        clean_split_path = clean_path / split
        
        if not noisy_split_path.exists():
            print(f"  ⚠️  노이즈 {split} 폴더가 없습니다: {noisy_split_path}")
            continue
        if not clean_split_path.exists():
            print(f"  ⚠️  원본 {split} 폴더가 없습니다: {clean_split_path}")
            continue
        
        # 노이즈 데이터 수집 (day/night 구분 없이 모두)
        noisy_files = []
        day_path = noisy_split_path / 'day'
        night_path = noisy_split_path / 'night'
        
        if day_path.exists():
            day_files = get_image_files(day_path)
            noisy_files.extend(day_files)
            print(f"    노이즈 day: {len(day_files)}개")
        
        if night_path.exists():
            night_files = get_image_files(night_path)
            noisy_files.extend(night_files)
            print(f"    노이즈 night: {len(night_files)}개")
        
        print(f"    노이즈 총합: {len(noisy_files)}개")
        
        if len(noisy_files) == 0:
            print(f"    ⚠️  노이즈 이미지가 없습니다. 건너뜁니다.")
            continue
        
        # 원본 데이터 수집 (각 폴더별로)
        clean_folders = {}
        if clean_split_path.exists():
            # 하위 폴더 확인
            subfolders = [d for d in clean_split_path.iterdir() if d.is_dir()]
            
            if subfolders:
                # 하위 폴더가 있으면 각 폴더별로 처리
                for subfolder in subfolders:
                    folder_name = subfolder.name
                    clean_files = get_image_files(subfolder)
                    if len(clean_files) > 0:
                        clean_folders[folder_name] = clean_files
            else:
                # 하위 폴더가 없으면 직접 이미지 파일 확인
                clean_files = get_image_files(clean_split_path)
                if len(clean_files) > 0:
                    clean_folders['root'] = clean_files
        
        print(f"    원본 폴더 수: {len(clean_folders)}개")
        for folder_name, files in clean_folders.items():
            print(f"      - {folder_name}: {len(files)}개")
        
        # 매칭 수행
        # 원본 데이터의 각 폴더에 대해, 노이즈 이미지와 1:1 매칭
        # 원본 이미지가 더 많으면 여러 노이즈 이미지와 매칭 가능
        # 노이즈 이미지가 더 많으면 여러 원본 이미지와 매칭 가능
        
        # 간단한 매칭: 원본 이미지 개수와 노이즈 이미지 개수 중 작은 값만큼 매칭
        total_clean = sum(len(files) for files in clean_folders.values())
        
        if total_clean == 0:
            print(f"    ⚠️  원본 이미지가 없습니다. 건너뜁니다.")
            continue
        
        # 각 원본 폴더의 이미지와 노이즈 이미지를 순차적으로 매칭
        noisy_idx = 0
        for folder_name, clean_files in clean_folders.items():
            for clean_file in clean_files:
                if noisy_idx < len(noisy_files):
                    noisy_file = noisy_files[noisy_idx]
                    matched_pairs.append({
                        'split': split,
                        'folder_name': folder_name,
                        'noisy_path': noisy_file,
                        'clean_path': clean_file,
                        'noisy_relative': str(Path(noisy_file).relative_to(noisy_path)),
                        'clean_relative': str(Path(clean_file).relative_to(clean_path))
                    })
                    stats['by_split'][split] += 1
                    stats['by_folder'][f"{split}/{folder_name}"] += 1
                    noisy_idx += 1
                else:
                    # 노이즈 이미지가 부족하면 원본만 기록
                    stats['unmatched_clean'].append(clean_file)
        
        # 남은 노이즈 이미지 기록
        if noisy_idx < len(noisy_files):
            stats['unmatched_noisy'].extend(noisy_files[noisy_idx:])
    
    stats['total_pairs'] = len(matched_pairs)
    
    # 결과 출력
    print("\n" + "=" * 60)
    print("📊 매칭 결과")
    print("=" * 60)
    print(f"총 매칭된 쌍: {stats['total_pairs']:,}개")
    
    print(f"\nSplit별 매칭 수:")
    for split in splits:
        count = stats['by_split'].get(split, 0)
        if count > 0:
            print(f"  {split}: {count:,}개")
    
    print(f"\n폴더별 매칭 수 (상위 10개):")
    sorted_folders = sorted(stats['by_folder'].items(), key=lambda x: x[1], reverse=True)
    for folder_key, count in sorted_folders[:10]:
        print(f"  {folder_key}: {count:,}개")
    
    if len(stats['unmatched_noisy']) > 0:
        print(f"\n⚠️  매칭되지 않은 노이즈 이미지: {len(stats['unmatched_noisy'])}개")
    if len(stats['unmatched_clean']) > 0:
        print(f"⚠️  매칭되지 않은 원본 이미지: {len(stats['unmatched_clean'])}개")
    
    # JSON 파일로 저장
    output_data = {
        'matched_pairs': matched_pairs,
        'stats': {
            'total_pairs': stats['total_pairs'],
            'by_split': dict(stats['by_split']),
            'by_folder': dict(stats['by_folder']),
            'unmatched_noisy_count': len(stats['unmatched_noisy']),
            'unmatched_clean_count': len(stats['unmatched_clean'])
        },
        'metadata': {
            'noisy_dir': str(noisy_path),
            'clean_dir': str(clean_path),
            'matching_method': 'folder_name_based'
        }
    }
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 매칭 결과 저장: {output_path}")
    print(f"   총 {len(matched_pairs):,}개의 매칭 쌍이 저장되었습니다.")
    
    return matched_pairs, stats


def main():
    parser = argparse.ArgumentParser(description='데이터셋 매칭')
    parser.add_argument('--noisy_dir', type=str, required=True,
                        help='노이즈 데이터 디렉토리 경로 (예: "DATASET_FAST_FINAL copy")')
    parser.add_argument('--clean_dir', type=str, required=True,
                        help='원본 데이터 디렉토리 경로')
    parser.add_argument('--output', type=str, default='matched_pairs.json',
                        help='매칭 결과 저장 파일 (default: matched_pairs.json)')
    
    args = parser.parse_args()
    
    matched_pairs, stats = match_by_folder_name(
        args.noisy_dir,
        args.clean_dir,
        args.output
    )
    
    print("\n✅ 매칭 완료!")
    
    return matched_pairs, stats


if __name__ == '__main__':
    main()

