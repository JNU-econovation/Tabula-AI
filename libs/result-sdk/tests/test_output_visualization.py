"""
출력 시각화 통합 테스트 스크립트
실제 문서를 처리하고, 가상의 오답 데이터에 대해 밑줄을 그어 시각화 결과를 생성
"""

import os
import shutil
import argparse
import google.generativeai as genai

from result_sdk.config import settings
from result_sdk.output_visualization import draw_underlines_for_incorrect_answers_enhanced
from result_sdk.text_processing.core import process_document
from common_sdk.prompt_loader import PromptLoader

def run_integration_visualization_test(input_file_path: str):
    """
    실제 문서를 처리하고 오답을 시각화하는 통합 테스트 실행
    """
    print(f"Integrated Output Visualization Test Started for: '{input_file_path}'")
    print("=" * 70)

    # --- 설정 및 초기화 ---
    # YAML 파일에서 프롬프트 로드
    prompt_loader = PromptLoader()
    try:
        ocr_prompt_data = prompt_loader.load_prompt('ocr-prompt')
        PROMPT_TEMPLATE = ocr_prompt_data['template']
        print("Successfully loaded prompt template from 'OCR-PROMPT.yaml'.")
    except Exception as e:
        print(f"Error loading prompt from YAML file: {e}")
        raise

    # Google AI SDK 설정
    if not settings.GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY is not set in the config.")
    genai.configure(api_key=settings.GOOGLE_API_KEY)
    print("Google Generative AI SDK configured.")

    # 서비스 계정 파일 경로 확인
    if not settings.SERVICE_ACCOUNT_FILE or not os.path.exists(settings.SERVICE_ACCOUNT_FILE):
        raise ValueError(f"Service account file not found at: {settings.SERVICE_ACCOUNT_FILE}")
    print(f"Using service account file: {settings.SERVICE_ACCOUNT_FILE}")

    # --- 단계 1: 문서 처리 ---
    print(f"\n[Step 1] Processing document: '{input_file_path}'...")
    
    all_visualization_data, all_rag_ready_data, all_original_ocr_data, image_paths, temp_folder = process_document(
        input_file_path=input_file_path,
        service_account_file=settings.SERVICE_ACCOUNT_FILE,
        temp_base_dir=settings.BASE_TEMP_DIR,
        prompt_template=PROMPT_TEMPLATE,
        generation_config=settings.GENERATION_CONFIG,
        safety_settings=settings.SAFETY_SETTINGS
    )
    print(f"Document processing complete. Visualization data: {len(all_visualization_data)}, RAG data: {len(all_rag_ready_data)}, Original OCR: {len(all_original_ocr_data)}")

    # --- 단계 2: 시각화 ---
    output_folder = os.path.join(settings.BASE_OUTPUT_DIR, "integration_test_visualization_sdk")
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)
    
    if not all( (image_paths, all_visualization_data, all_original_ocr_data) ):
        print("\nSkipping visualization due to missing document processing results.")
    else:
        # 테스트용 가상 오답 ID
        mock_incorrect_ids = [
            (1, 0, 3, 1), (1, 0, 5, 2), (1, 0, 7, 1),
            (1, 1, 3, 1), (1, 1, 4, 1), (2, 0, 6, 1)
        ]
        
        # 생성된 RAG 데이터에 실제 존재하는 ID만 필터링
        rag_ids_set = {tuple(entry[0]) for entry in all_rag_ready_data if isinstance(entry, list) and len(entry) > 0}
        valid_mock_ids = [mid for mid in mock_incorrect_ids if mid in rag_ids_set]
        
        if not valid_mock_ids:
            print("\n[Step 2] No valid incorrect IDs to visualize. Skipping visualization.")
        else:
            print(f"\n[Step 2] Visualizing incorrect answers (Valid IDs: {len(valid_mock_ids)})...")
            draw_underlines_for_incorrect_answers_enhanced(
                incorrect_rag_ids=valid_mock_ids,
                all_visualization_data=all_visualization_data,
                all_original_ocr_data=all_original_ocr_data,
                page_image_paths=image_paths,
                output_folder=output_folder
            )
            print(f"Visualization results saved in '{output_folder}'.")

    # --- 정리 ---
    if temp_folder and os.path.exists(temp_folder):
        try:
            shutil.rmtree(temp_folder)
            print(f"\nCleaned up temporary folder: '{temp_folder}'")
        except Exception as e:
            print(f"\nError cleaning up temporary folder '{temp_folder}': {e}")
    
    print("=" * 70)
    print(f"Integrated Output Visualization Test Finished for: '{input_file_path}'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Integration test for document processing and visualization.")
    parser.add_argument(
        "--input_file",
        type=str,
        default='/Users/ki/Desktop/Google Drive/Dev/Ecode/OCR_Test/예시_한국사.pdf',
        help="Path to the PDF or image file to process."
    )
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: '{args.input_file}'")
        exit(1)
        
    run_integration_visualization_test(args.input_file)
