import zipfile
from pathlib import Path
from collections import defaultdict

# Zip 파일 경로
zip_path = Path(r"C:\Users\dbswl\OneDrive\바탕 화면\3-2\computervesion\team\leftImg8bit_trainvaltest (1).zip")

print(f"Zip 파일 분석: {zip_path.name}")
print("=" * 60)

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    # 모든 파일 목록 가져오기
    all_files = zip_ref.namelist()
    
    # 이미지 파일만 필터링
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp']
    image_files = [f for f in all_files if any(f.lower().endswith(ext) for ext in image_extensions)]
    
    # 디렉토리 구조 분석
    directories = set()
    for file in all_files:
        # Windows와 Unix 경로 모두 처리
        if '/' in file:
            dir_path = '/'.join(file.split('/')[:-1])
        elif '\\' in file:
            dir_path = '\\'.join(file.split('\\')[:-1])
        else:
            dir_path = ''
        if dir_path:
            directories.add(dir_path)
    
    print(f"📦 총 파일 수: {len(all_files)}")
    print(f"🖼️  이미지 파일 수: {len(image_files)}")
    print(f"📁 디렉토리 수: {len(directories)}")
    print()
    
    # 디렉토리별 파일 수
    dir_counts = defaultdict(int)
    dir_image_counts = defaultdict(int)
    
    for file in all_files:
        if '/' in file:
            dir_path = '/'.join(file.split('/')[:-1]) if '/' in file else 'root'
        elif '\\' in file:
            dir_path = '\\'.join(file.split('\\')[:-1]) if '\\' in file else 'root'
        else:
            dir_path = 'root'
        dir_counts[dir_path] += 1
        
        if any(file.lower().endswith(ext) for ext in image_extensions):
            dir_image_counts[dir_path] += 1
    
    print("📊 디렉토리별 파일 수:")
    for dir_path in sorted(dir_counts.keys()):
        total = dir_counts[dir_path]
        images = dir_image_counts.get(dir_path, 0)
        print(f"   {dir_path}: 총 {total}개 (이미지: {images}개)")
    
    print()
    print("📋 상위 30개 파일 샘플:")
    for i, file in enumerate(all_files[:30], 1):
        file_type = "🖼️ 이미지" if any(file.lower().endswith(ext) for ext in image_extensions) else "📄 기타"
        print(f"   {i:2d}. [{file_type}] {file}")
    
    if len(all_files) > 30:
        print(f"   ... 외 {len(all_files) - 30}개 파일")
    
    print()
    print("📋 상위 30개 이미지 파일 샘플:")
    for i, file in enumerate(image_files[:30], 1):
        print(f"   {i:2d}. {file}")
    
    if len(image_files) > 30:
        print(f"   ... 외 {len(image_files) - 30}개 이미지 파일")

