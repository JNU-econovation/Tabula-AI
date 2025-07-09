# libs/result-sdk/tests/test_output_visualization.py

import os
import shutil
import argparse
import google.generativeai as genai

# Adjust imports for the new structure
from result_sdk.config import settings
from result_sdk.output_visualization import draw_underlines_for_incorrect_answers_enhanced
from result_sdk.text_processing.core import process_document
from common_sdk.prompt_loader import PromptLoader # 올바른 프롬프트 로더를 import

def run_integration_visualization_test(input_file_path: str):
    """
    Runs an integrated test: processes a real document, then visualizes incorrect answers.
    """
    print(f"Integrated Output Visualization Test Started for: '{input_file_path}'")
    print("=" * 70)

    # YAML 파일에서 최신 프롬프트를 로드합니다.
    prompt_loader = PromptLoader()
    try:
        ocr_prompt_data = prompt_loader.load_prompt('ocr-prompt')
        PROMPT_TEMPLATE = ocr_prompt_data['template']
        print("Successfully loaded prompt template from 'OCR-PROMPT.yaml'.")
    except Exception as e:
        print(f"Error loading prompt from YAML file: {e}")
        raise

    if not settings.GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY가 config.py 또는 환경 변수를 통해 설정되지 않았습니다.")
    try:
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        print("Google Generative AI SDK 설정 완료.")
    except Exception as e:
        print(f"오류: Google Generative AI SDK 설정 실패 - {e}")

    # Service account file path resolution
    service_file_from_settings = settings.SERVICE_ACCOUNT_FILE
    if not service_file_from_settings:
        raise ValueError("SERVICE_ACCOUNT_FILE_PATH is not set in the environment/config.")

    if os.path.isabs(service_file_from_settings):
        actual_service_account_file = service_file_from_settings
    else:
        # Assuming service_file_from_settings is relative to the project root.
        # Project root is three levels up from this test file's directory (libs/result-sdk/tests/).
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        actual_service_account_file = os.path.join(project_root, service_file_from_settings)
        # Check if the path needs to be adjusted if common-sdk is a sibling of result-sdk
        # Example: if service_file_from_settings starts with "libs/common-sdk"
        # This logic assumes SERVICE_ACCOUNT_FILE_PATH in .env.dev is like "libs/common-sdk/common_sdk/file.json"
        # and the test script is in "libs/result-sdk/tests/"
        # So, project_root (Tabula-AI) + "libs/common-sdk/common_sdk/file.json" should be correct.

    if not os.path.exists(actual_service_account_file):
        # Attempt to construct path assuming it's relative to common_sdk's root if previous failed
        # This part might be redundant if the above logic is correct for the .env.dev structure
        common_sdk_path_segment = "libs/common-sdk/common_sdk/"
        if service_file_from_settings.startswith(common_sdk_path_segment):
             # This case should have been handled by the project_root logic if .env.dev is correct
             pass # Path already formed relative to project root
        
        # Final check or raise error
        if not os.path.exists(actual_service_account_file): # Re-check after any alternative construction
            raise ValueError(
                f"Resolved SERVICE_ACCOUNT_FILE ('{actual_service_account_file}') does not exist. "
                f"Original path from settings: '{service_file_from_settings}'. "
                f"Current PWD: '{os.getcwd()}'"
            )
    
    print(f"Using service account file: {actual_service_account_file}")

    print(f"\n[단계 1] '{input_file_path}' 문서 처리 중...")
    
    # process_document는 이제 5개의 값을 반환합니다.
    all_visualization_data, all_rag_ready_data, all_original_ocr_data, image_paths, created_temp_folder_path = process_document(
        input_file_path=input_file_path,
        service_account_file=actual_service_account_file,
        temp_base_dir=settings.BASE_TEMP_DIR,
        prompt_template=PROMPT_TEMPLATE, # YAML에서 로드한 프롬프트 사용
        generation_config=settings.GENERATION_CONFIG,
        safety_settings=settings.SAFETY_SETTINGS
    )
    print(f"문서 처리 완료. Visualization: {len(all_visualization_data)}, RAG_ready: {len(all_rag_ready_data)}, Original_OCR: {len(all_original_ocr_data)}, Images: {len(image_paths)}")

    # 시각화 결과 저장 폴더 설정 (result_sdk의 settings.BASE_OUTPUT_DIR 사용)
    # output_base_dir은 result_sdk.settings.BASE_OUTPUT_DIR을 사용
    output_visualization_folder = os.path.join(settings.BASE_OUTPUT_DIR, "integration_test_visualization_sdk")
    if os.path.exists(output_visualization_folder):
        shutil.rmtree(output_visualization_folder)
    os.makedirs(output_visualization_folder, exist_ok=True)
    
    # 시각화는 visualization_data와 original_ocr_data를 사용합니다.
    if image_paths and all_visualization_data and all_original_ocr_data:
        # 테스트용 오답 목데이터 (기존 테스트 스크립트에서 가져옴)
        mock_incorrect_rag_ids_to_test = [
            (1, 0, 3, 1), 
            (1, 0, 5, 2),
            (1, 0, 7, 1),
            (1, 1, 3, 1),
            (1, 1, 4, 1),
            (2, 0, 6, 1)
        ]
        
        valid_mock_ids = []
        if all_rag_ready_data:
            rag_ids_set = {tuple(entry[0]) for entry in all_rag_ready_data if isinstance(entry, list) and len(entry) > 0 and isinstance(entry[0], list) and len(entry[0]) == 4}
            for mock_id in mock_incorrect_rag_ids_to_test:
                if mock_id in rag_ids_set:
                    valid_mock_ids.append(mock_id)
                else:
                    print(f"경고: 목업 오답 ID {mock_id}가 생성된 RAG 데이터에 없습니다. 시각화에서 제외됩니다.")
        else:
             print("경고: RAG 데이터가 생성되지 않아 목업 오답 ID를 검증할 수 없습니다.")

        if not valid_mock_ids:
            print("\n[단계 2] 시각화할 유효한 오답 ID가 없습니다. 시각화 단계를 건너<0xEB><0x9C><0x85>니다.")
        else:
            print(f"\n[단계 2] 오답 시각화 진행 (유효 오답 ID: {valid_mock_ids})...")
            # 수정한 visualizer 함수 시그니처에 맞게 인자 전달
            draw_underlines_for_incorrect_answers_enhanced(
                incorrect_rag_ids=valid_mock_ids,
                all_visualization_data=all_visualization_data, # 시각화용 데이터 전달
                all_original_ocr_data=all_original_ocr_data, # 원본 OCR 데이터 전달
                page_image_paths=image_paths,
                output_folder=output_visualization_folder,
                underline_color=(0, 0, 255),
                underline_thickness=2
            )
            print(f"시각화 결과는 '{output_visualization_folder}' 폴더를 확인하세요.")
    else:
        print("\n문서 처리 결과(이미지 경로, 시각화 데이터, 원본 OCR 데이터)가 없어 시각화를 진행할 수 없습니다.")

    if created_temp_folder_path and os.path.exists(created_temp_folder_path):
        try:
            shutil.rmtree(created_temp_folder_path)
            print(f"\n최종: 임시 PDF 변환 폴더 '{created_temp_folder_path}' 삭제 완료.")
        except Exception as e:
            print(f"\n최종: 임시 PDF 변환 폴더 '{created_temp_folder_path}' 삭제 중 오류: {e}")
    
    print("=" * 70)
    print(f"Integrated Output Visualization Test Finished for: '{input_file_path}'")

if __name__ == '__main__':
    default_input_file = '/Users/ki/Desktop/Google Drive/Dev/Ecode/OCR_Test/예시_한국사.pdf'
    
    parser = argparse.ArgumentParser(description="통합 문서 처리 및 오답 시각화 테스트 스크립트")
    parser.add_argument(
        "--input_file",  # 옵션 인자로 변경 (--input_file)
        type=str,
        default=default_input_file,
        help=f"처리할 PDF 또는 이미지 파일의 전체 경로 (기본값: {default_input_file})"
    )
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"오류: 입력 파일 '{args.input_file}'을 찾을 수 없습니다.")
        exit()
        
    run_integration_visualization_test(args.input_file)
