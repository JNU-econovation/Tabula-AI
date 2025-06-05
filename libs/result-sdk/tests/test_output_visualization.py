# libs/result-sdk/tests/test_output_visualization.py

import os
import shutil
import argparse
import google.generativeai as genai

# Adjust imports for the new structure
from result_sdk.config import settings # result_sdk의 설정을 가져옴
from result_sdk.output_visualization import draw_underlines_for_incorrect_answers_enhanced
from result_sdk.text_processing.core import process_document # 실제 문서 처리 함수
# from common_sdk.prompt_loader import PromptLoader # common_sdk의 프롬프트 로더 대신 직접 임포트
from result_sdk.result_processor.Prompt import gemini_prompt as PROMPT_TEMPLATE # Prompt.py에서 직접 가져옴

def run_integration_visualization_test(input_file_path: str):
    """
    Runs an integrated test: processes a real document, then visualizes incorrect answers.
    """
    print(f"Integrated Output Visualization Test Started for: '{input_file_path}'")
    print("=" * 70)

    if not settings.GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY가 config.py 또는 환경 변수를 통해 설정되지 않았습니다.")
    try:
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        print("Google Generative AI SDK 설정 완료.")
    except Exception as e:
        print(f"오류: Google Generative AI SDK 설정 실패 - {e}")
        # SDK 설정 실패 시, LLM을 사용하는 process_document가 실패할 수 있으므로 여기서 중단하는 것이 좋을 수 있음
        # 또는 LLM 없이 OCR만 진행하도록 process_document를 수정하거나, 테스트를 제한적으로 실행할 수 있음
        # 여기서는 일단 에러를 출력하고 계속 진행 (process_document 내부에서 LLM 호출 시 에러 처리될 것으로 기대)

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
    print(f"Using prompt template from result_sdk.result_processor.Prompt")

    print(f"\n[단계 1] '{input_file_path}' 문서 처리 중...")
    
    all_consolidated_data, all_rag_ready_data, image_paths, created_temp_folder_path = process_document(
        input_file_path=input_file_path,
        service_account_file=actual_service_account_file, # Resolved absolute path
        temp_base_dir=settings.BASE_TEMP_DIR,
        prompt_template=PROMPT_TEMPLATE, # Directly imported
        generation_config=settings.GENERATION_CONFIG,
        safety_settings=settings.SAFETY_SETTINGS
    )
    print(f"문서 처리 완료. Consolidated: {len(all_consolidated_data)}, RAG_ready: {len(all_rag_ready_data)}, Images: {len(image_paths)}")

    # 시각화 결과 저장 폴더 설정 (result_sdk의 settings.BASE_OUTPUT_DIR 사용)
    # output_base_dir은 result_sdk.settings.BASE_OUTPUT_DIR을 사용
    output_visualization_folder = os.path.join(settings.BASE_OUTPUT_DIR, "integration_test_visualization_sdk")
    if os.path.exists(output_visualization_folder):
        shutil.rmtree(output_visualization_folder)
    os.makedirs(output_visualization_folder, exist_ok=True)
    
    if image_paths and all_consolidated_data:
        # 테스트용 오답 목데이터 (기존 테스트 스크립트에서 가져옴)
        mock_incorrect_rag_ids_to_test = [
            (1, 0, 3, 1), 
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
            draw_underlines_for_incorrect_answers_enhanced(
                incorrect_rag_ids=valid_mock_ids,
                all_consolidated_data=all_consolidated_data, # 실제 처리된 데이터 사용
                all_rag_ready_data=all_rag_ready_data,       # 실제 처리된 데이터 사용
                page_image_paths=image_paths,                # 실제 처리된 이미지 경로 사용
                output_folder=output_visualization_folder,
                underline_color=(0, 0, 255), # BGR: 빨간색으로 변경
                underline_thickness=2
            )
            print(f"시각화 결과는 '{output_visualization_folder}' 폴더를 확인하세요.")
    else:
        print("\n문서 처리 결과(이미지 경로 또는 consolidated 데이터)가 없어 시각화를 진행할 수 없습니다.")

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
