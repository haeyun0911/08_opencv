import os
import glob
import shutil

# 이미지를 찾을 최상위 폴더 경로를 지정합니다.
# 예: './img/101_ObjectCategories'
root_folder = 'C:/Users/405/Downloads/New_Sample/원천데이터'

# 이미지를 모을 새로운 폴더 경로를 지정합니다.
output_folder = '../img/car'

# 이미지를 모을 폴더가 없으면 새로 만듭니다.
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"'{output_folder}' 폴더가 생성되었습니다.")

# root_folder 아래의 모든 하위 폴더와 파일들을 순회합니다.
for folder, subfolders, files in os.walk(root_folder):
    for file in files:
        # 파일 확장자가 '.jpg'인 경우를 찾습니다.
        if file.endswith('.jpg'):
            # 원본 파일 경로와 이동할 파일 경로를 만듭니다.
            source_path = os.path.join(folder, file)
            destination_path = os.path.join(output_folder, file)

            # 파일 이름 충돌을 방지하기 위한 로직 (선택 사항)
            # 만약 같은 이름의 파일이 이미 output_folder에 있다면,
            # 파일 이름에 고유한 숫자를 추가합니다.
            base_name, extension = os.path.splitext(file)
            count = 1
            while os.path.exists(destination_path):
                new_file_name = f"{base_name}_{count}{extension}"
                destination_path = os.path.join(output_folder, new_file_name)
                count += 1
            
            # 파일을 이동시킵니다. (복사를 원하면 shutil.copy2 사용)
            shutil.move(source_path, destination_path)
            print(f"'{source_path}' -> '{destination_path}'로 이동했습니다.")

print("\n모든 이미지 파일 이동이 완료되었습니다.")