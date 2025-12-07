"""
데이터셋 확인 및 개수 확인 스크립트

노이즈 데이터와 원본 데이터의 구조를 확인하고 개수를 세는 스크립트
"""
import os
from pathlib import Path
from collections import defaultdict
import argparse


def count_images_in_folder(folder_path):
    """폴더 내 이미지 파일 개수 세기"""
    folder = Path(folder_path)
    if not folder.exists():
        return 0
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    count = 0
    for ext in image_extensions:
        count += len(list(folder.glob(f'**/*{ext}')))
    return count


def analyze_noisy_dataset(noisy_dir):
    """
    노이즈 데이터셋 구조 분석
    구조: DATASET_FAST_FINAL copy/test/day, test/night, train/day, train/night, val/day, val/night
    """
    print("=" * 60)
    print("📊 노이즈 데이터셋 분석")
    print("=" * 60)
    
    noisy_path = Path(noisy_dir)
    if not noisy_path.exists():
        print(f"❌ 경로를 찾을 수 없습니다: {noisy_dir}")
        return None
    
    stats = {
        'total': 0,
        'by_split': defaultdict(int),
        'by_category': defaultdict(int),
        'by_split_category': defaultdict(int),
        'folder_structure': {}
    }
    
    # test, train, val 폴더 확인
    splits = ['test', 'train', 'val']
    categories = ['day', 'night']
    
    for split in splits:
        split_path = noisy_path / split
        if not split_path.exists():
            print(f"⚠️  {split} 폴더가 없습니다: {split_path}")
            continue
        
        stats['folder_structure'][split] = {}
        
        for category in categories:
            category_path = split_path / category
            if not category_path.exists():
                print(f"⚠️  {split}/{category} 폴더가 없습니다: {category_path}")
                continue
            
            count = count_images_in_folder(category_path)
            stats['by_split'][split] += count
            stats['by_category'][category] += count
            stats['by_split_category'][f"{split}/{category}"] = count
            stats['total'] += count
            
            stats['folder_structure'][split][category] = {
                'path': str(category_path),
                'count': count
            }
    
    # 결과 출력
    print(f"\n📁 전체 구조:")
    print(f"  루트 경로: {noisy_path}")
    print(f"  총 이미지 수: {stats['total']:,}개")
    
    print(f"\n📂 Split별 분포:")
    for split in splits:
        count = stats['by_split'][split]
        if count > 0:
            print(f"  {split}: {count:,}개")
    
    print(f"\n🌓 Category별 분포 (day/night 합계):")
    for category in categories:
        count = stats['by_category'][category]
        if count > 0:
            print(f"  {category}: {count:,}개")
    
    print(f"\n📋 Split/Category별 상세:")
    for split in splits:
        for category in categories:
            key = f"{split}/{category}"
            count = stats['by_split_category'].get(key, 0)
            if count > 0:
                print(f"  {key}: {count:,}개")
    
    return stats


def analyze_clean_dataset(clean_dir):
    """
    원본 데이터셋 구조 분석
    구조: test/berlin, test/bielefeld, test/bonn, test/leverkusen, test/mainz, test/munich, train/..., val/...
    """
    print("\n" + "=" * 60)
    print("📊 원본 데이터셋 분석")
    print("=" * 60)
    
    clean_path = Path(clean_dir)
    if not clean_path.exists():
        print(f"❌ 경로를 찾을 수 없습니다: {clean_dir}")
        return None
    
    stats = {
        'total': 0,
        'by_split': defaultdict(int),
        'by_folder': defaultdict(int),
        'by_split_folder': defaultdict(int),
        'folder_structure': {}
    }
    
    # test, train, val 폴더 확인
    splits = ['test', 'train', 'val']
    
    for split in splits:
        split_path = clean_path / split
        if not split_path.exists():
            print(f"⚠️  {split} 폴더가 없습니다: {split_path}")
            continue
        
        stats['folder_structure'][split] = {}
        
        # split 폴더 내의 모든 하위 폴더 확인
        subfolders = [d for d in split_path.iterdir() if d.is_dir()]
        
        if not subfolders:
            # 하위 폴더가 없으면 직접 이미지 파일이 있는지 확인
            count = count_images_in_folder(split_path)
            if count > 0:
                stats['by_split'][split] += count
                stats['total'] += count
                stats['folder_structure'][split]['root'] = {
                    'path': str(split_path),
                    'count': count
                }
        else:
            # 각 하위 폴더 확인
            for subfolder in subfolders:
                folder_name = subfolder.name
                count = count_images_in_folder(subfolder)
                
                if count > 0:
                    stats['by_split'][split] += count
                    stats['by_folder'][folder_name] += count
                    stats['by_split_folder'][f"{split}/{folder_name}"] = count
                    stats['total'] += count
                    
                    stats['folder_structure'][split][folder_name] = {
                        'path': str(subfolder),
                        'count': count
                    }
    
    # 결과 출력
    print(f"\n📁 전체 구조:")
    print(f"  루트 경로: {clean_path}")
    print(f"  총 이미지 수: {stats['total']:,}개")
    
    print(f"\n📂 Split별 분포:")
    for split in splits:
        count = stats['by_split'][split]
        if count > 0:
            print(f"  {split}: {count:,}개")
    
    print(f"\n📁 폴더별 분포 (모든 split 합계):")
    for folder_name, count in sorted(stats['by_folder'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {folder_name}: {count:,}개")
    
    print(f"\n📋 Split/폴더별 상세 (상위 20개):")
    sorted_items = sorted(stats['by_split_folder'].items(), key=lambda x: x[1], reverse=True)
    for key, count in sorted_items[:20]:
        print(f"  {key}: {count:,}개")
    
    if len(sorted_items) > 20:
        print(f"  ... 외 {len(sorted_items) - 20}개 폴더")
    
    return stats


def match_folders(noisy_stats, clean_stats):
    """
    노이즈 데이터와 원본 데이터의 폴더명 매칭 분석
    노이즈: test/day, test/night 등
    원본: test/berlin, test/bielefeld 등
    """
    print("\n" + "=" * 60)
    print("🔗 데이터 매칭 분석")
    print("=" * 60)
    
    if noisy_stats is None or clean_stats is None:
        print("❌ 통계 데이터가 없어 매칭 분석을 수행할 수 없습니다.")
        return None
    
    matches = defaultdict(lambda: {'noisy': 0, 'clean': 0, 'matched': False})
    
    # 노이즈 데이터의 split별 카테고리별 개수 (day/night 합산)
    noisy_by_split = {}
    for split in ['test', 'train', 'val']:
        noisy_by_split[split] = noisy_stats['by_split'].get(split, 0)
    
    # 원본 데이터의 split별 폴더별 개수
    clean_by_split = {}
    for split in ['test', 'train', 'val']:
        clean_by_split[split] = clean_stats['by_split'].get(split, 0)
    
    print(f"\n📊 Split별 매칭 가능 여부:")
    for split in ['test', 'train', 'val']:
        noisy_count = noisy_by_split.get(split, 0)
        clean_count = clean_by_split.get(split, 0)
        
        print(f"\n  {split}:")
        print(f"    노이즈 데이터: {noisy_count:,}개 (day+night 합계)")
        print(f"    원본 데이터: {clean_count:,}개")
        
        if noisy_count > 0 and clean_count > 0:
            ratio = min(noisy_count, clean_count) / max(noisy_count, clean_count) * 100
            print(f"    매칭 가능: ✅ (비율: {ratio:.1f}%)")
            matches[split] = {
                'noisy': noisy_count,
                'clean': clean_count,
                'matched': True,
                'ratio': ratio
            }
        else:
            print(f"    매칭 가능: ❌ (데이터 없음)")
            matches[split] = {
                'noisy': noisy_count,
                'clean': clean_count,
                'matched': False
            }
    
    # 원본 데이터의 폴더명 목록 (매칭에 사용될 폴더명)
    print(f"\n📁 원본 데이터 폴더명 목록 (매칭 키로 사용):")
    all_folders = set()
    for split in ['test', 'train', 'val']:
        if split in clean_stats['folder_structure']:
            for folder_name in clean_stats['folder_structure'][split].keys():
                if folder_name != 'root':  # root는 제외
                    all_folders.add(folder_name)
    
    print(f"  총 {len(all_folders)}개의 고유 폴더명:")
    for folder_name in sorted(all_folders):
        total_count = clean_stats['by_folder'].get(folder_name, 0)
        print(f"    - {folder_name}: {total_count:,}개")
    
    return matches


def main():
    parser = argparse.ArgumentParser(description='데이터셋 확인 및 개수 확인')
    parser.add_argument('--noisy_dir', type=str, required=True,
                        help='노이즈 데이터 디렉토리 경로 (예: "DATASET_FAST_FINAL copy")')
    parser.add_argument('--clean_dir', type=str, required=True,
                        help='원본 데이터 디렉토리 경로')
    
    args = parser.parse_args()
    
    # 노이즈 데이터 분석
    noisy_stats = analyze_noisy_dataset(args.noisy_dir)
    
    # 원본 데이터 분석
    clean_stats = analyze_clean_dataset(args.clean_dir)
    
    # 매칭 분석
    matches = match_folders(noisy_stats, clean_stats)
    
    print("\n" + "=" * 60)
    print("✅ 분석 완료!")
    print("=" * 60)
    
    return noisy_stats, clean_stats, matches


if __name__ == '__main__':
    main()

