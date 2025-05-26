#pdf, img 형태의 파일을 입력받아 이미지로 변환하고, 이미지 파일 경로 리스트를 반환

import os
import shutil
from pdf2image import convert_from_path
# from PIL import Image # pdf2image가 PIL.Image 객체를 반환하지만, 이 함수 내에서 직접 Image를 사용하진 않음

def get_image_paths_from_input(input_file_path, temp_output_folder="pdf_converted_images_temp"):
    """
    입력 파일 경로를 받아 PDF인 경우 이미지로 변환하고, 이미지 파일 경로 리스트를 반환
    PDF가 아닌 이미지 파일이면 해당 파일 경로를 리스트에 담아 반환

    Args:
        input_file_path (str): 처리할 파일의 경로 (PDF 또는 이미지)
        temp_output_folder (str): PDF에서 변환된 이미지를 저장할 임시 폴더 이름

    Returns:
        list: 처리할 이미지 파일 경로들의 리스트. 오류 발생 시 빈 리스트 반환
    """
    file_name, file_ext = os.path.splitext(input_file_path)
    file_ext = file_ext.lower()

    image_paths = []

    if file_ext == '.pdf':
        if os.path.exists(temp_output_folder): # 기존 임시 폴더가 있다면 삭제 후 생성
            try:
                shutil.rmtree(temp_output_folder)
                print(f"기존 임시 폴더 '{temp_output_folder}' 삭제 완료.")
            except Exception as e:
                print(f"기존 임시 폴더 '{temp_output_folder}' 삭제 중 오류: {e}")

        try:
            os.makedirs(temp_output_folder)
            print(f"임시 폴더 '{temp_output_folder}' 생성 완료.")
        except OSError as e:
            print(f"임시 폴더 '{temp_output_folder}' 생성 실패: {e}. 이미 폴더가 존재할 수 있습니다.")
            if not os.path.isdir(temp_output_folder):
                 return []


        try:
            print(f"PDF 파일 변환 중: '{input_file_path}' (시간이 다소 소요될 수 있습니다)...")
            # dpi는 이미지 품질에 영향 (200-300 정도가 일반적)
            pil_images = convert_from_path(input_file_path, dpi=200, poppler_path=None) # poppler_path=None이면 시스템 PATH에서 찾음

            base_pdf_name = os.path.basename(file_name)
            for i, image in enumerate(pil_images):
                image_filename = os.path.join(temp_output_folder, f"{base_pdf_name}_page_{i + 1}.png")
                image.save(image_filename, "PNG") # PIL Image 객체의 save 메소드 사용
                image_paths.append(image_filename)
            print(f"PDF 파일이 성공적으로 {len(image_paths)}개의 이미지로 변환되어 '{temp_output_folder}'에 저장되었습니다.")
        except Exception as e: # pdf2image.exceptions.PDFInfoNotInstalledError 등 포함
            print(f"PDF 파일 변환 중 오류 발생 ('{input_file_path}'): {e}")
            print("Poppler가 시스템에 올바르게 설치되어 있고 PATH에 등록되어 있는지 확인해주세요.")
            print("- Windows: Poppler 바이너리 다운로드 후 bin 폴더를 PATH에 추가")
            print("- macOS: 'brew install poppler'")
            print("- Linux: 'sudo apt-get install poppler-utils'")
            print("- Conda: 'conda install -c conda-forge poppler'")
            return [] # 오류 시 빈 리스트 반환

    elif file_ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp']:
        if not os.path.exists(input_file_path):
            print(f"이미지 파일 경로 오류: '{input_file_path}'를 찾을 수 없습니다.")
            return []
        image_paths.append(input_file_path)
        print(f"입력 파일은 이미지입니다: '{input_file_path}'")
    else:
        print(f"지원하지 않는 파일 형식입니다: '{file_ext}'. PDF 또는 이미지 파일을 제공해주세요.")
        return []

    return image_paths

if __name__ == '__main__':
    # 이 파일 자체를 테스트하기 위한 간단한 코드 (선택 사항)
    # 예를 들어, 특정 PDF 파일로 테스트
    # test_pdf_path = "여기에_테스트할_PDF_경로.pdf"
    # test_output_folder = "pdf_test_output"
    # if os.path.exists(test_pdf_path):
    #     paths = get_image_paths_from_input(test_pdf_path, test_output_folder)
    #     if paths:
    #         print(f"\n테스트 변환된 이미지 경로:")
    #         for p in paths:
    #             print(p)
    #         # 테스트 후 임시 폴더 삭제 (선택)
    #         # if os.path.exists(test_output_folder):
    #         #     shutil.rmtree(test_output_folder)
    #     else:
    #         print("\n테스트 변환 실패.")
    # else:
    #     print(f"\n테스트 PDF 파일({test_pdf_path})을 찾을 수 없습니다.")
    pass
