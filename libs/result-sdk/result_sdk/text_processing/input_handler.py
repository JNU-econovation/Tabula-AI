"""
입력 파일(PDF 또는 이미지)을 처리, 이미지 파일 경로 리스트를 반환하는 모듈.
PDF 파일은 이미지로 변환 후 임시 폴더에 저장함.
"""

import os
import shutil
from pdf2image import convert_from_path

def get_image_paths_from_input(input_file_path: str, temp_output_folder: str = None) -> list[str]:
    """
    입력 파일 경로를 받아, 처리할 이미지 파일 경로의 리스트를 반환.

    - PDF 파일인 경우:
      1. `temp_output_folder`에 각 페이지를 이미지 파일로 변환 및 저장.
      2. 저장된 이미지 파일들의 경로 리스트를 반환함.
    - 이미지 파일인 경우:
      1. 해당 파일 경로를 리스트에 담아 반환.

    Args:
        input_file_path (str): 처리할 PDF 또는 이미지 파일 경로.
        temp_output_folder (str, optional): PDF 변환 이미지를 저장할 임시 폴더. PDF 처리 시 필수.

    Returns:
        list[str]: 이미지 파일 경로 리스트. 오류 시 빈 리스트 반환.
    """
    file_name, file_ext = os.path.splitext(input_file_path)
    file_ext = file_ext.lower()

    image_paths = []

    if file_ext == '.pdf':
        if not temp_output_folder:
            print("[Error] 'temp_output_folder' argument is required for PDF processing.")
            return []

        # 기존 임시 폴더가 존재하면 삭제.
        if os.path.exists(temp_output_folder):
            try:
                shutil.rmtree(temp_output_folder)
                print(f"Removed existing temporary folder: '{temp_output_folder}'")
            except Exception as e:
                print(f"Error removing existing temporary folder '{temp_output_folder}': {e}")

        try:
            os.makedirs(temp_output_folder)
            print(f"Created temporary folder: '{temp_output_folder}'")
        except OSError as e:
            print(f"Failed to create temporary folder '{temp_output_folder}': {e}")
            if not os.path.isdir(temp_output_folder):
                 return []

        try:
            print(f"Converting PDF: '{input_file_path}' (this may take a moment)...")
            # DPI는 이미지 해상도. 200-300이 일반적.
            # poppler_path=None이면 시스템 PATH에서 Poppler를 찾음.
            pil_images = convert_from_path(input_file_path, dpi=200, poppler_path=None)

            base_pdf_name = os.path.basename(file_name)
            for i, image in enumerate(pil_images):
                image_filename = os.path.join(temp_output_folder, f"{base_pdf_name}_page_{i + 1}.png")
                image.save(image_filename, "PNG")
                image_paths.append(image_filename)
            print(f"Successfully converted PDF to {len(image_paths)} images in '{temp_output_folder}'.")
        except Exception as e:
            print(f"Error during PDF conversion ('{input_file_path}'): {e}")
            print("Please ensure Poppler is installed and in your system's PATH.")
            print("  - Windows: Download Poppler binaries and add the bin folder to PATH.")
            print("  - macOS (Homebrew): 'brew install poppler'")
            print("  - Linux (apt): 'sudo apt-get install poppler-utils'")
            print("  - Conda: 'conda install -c conda-forge poppler'")
            return []

    elif file_ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp']:
        if not os.path.exists(input_file_path):
            print(f"[Error] Image file not found: '{input_file_path}'")
            return []
        image_paths.append(input_file_path)
        print(f"Input file is an image: '{input_file_path}'")
    else:
        print(f"[Error] Unsupported file format: '{file_ext}'. Please provide a PDF or image file.")
        return []

    return image_paths

if __name__ == '__main__':
    # 스크립트 직접 실행 시 테스트용 코드.
    # 'test_pdf_path'를 실제 PDF 파일 경로로 변경하여 사용.
    #
    # test_pdf_path = "path/to/your/test.pdf"
    # test_output_folder = "pdf_test_output"
    #
    # if os.path.exists(test_pdf_path):
    #     paths = get_image_paths_from_input(test_pdf_path, test_output_folder)
    #     if paths:
    #         print("\nTest conversion successful. Image paths created:")
    #         for p in paths:
    #             print(p)
    #
    #         # 테스트 후 임시 폴더 삭제 시 주석 해제.
    #         # if os.path.exists(test_output_folder):
    #         #     shutil.rmtree(test_output_folder)
    #     else:
    #         print("\nTest conversion failed.")
    # else:
    #     print(f"\nTest PDF file not found: {test_pdf_path}")
    pass
