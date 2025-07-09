"""
텍스트 처리 통합 테스트 스크립트
실제 문서를 입력받아 전체 텍스트 처리 파이프라인을 실행하고 결과를 출력
"""

import os
import shutil
import argparse
import google.generativeai as genai

from result_sdk.config import settings
from result_sdk.text_processing import process_document
from common_sdk.prompt_loader import PromptLoader

def run_text_processing_test(input_file_path: str):
    """
    텍스트 처리 파이프라인에 대한 통합 테스트 실행
    """
    print(f"Text Processing Test Started: Processing '{input_file_path}'")
    print("=" * 70)

    # --- 설정 및 초기화 ---
    prompt_loader = PromptLoader()
    try:
        ocr_prompt_data = prompt_loader.load_prompt('ocr-prompt')
        PROMPT_TEMPLATE = ocr_prompt_data['template']
        print("Successfully loaded prompt template from 'OCR-PROMPT.yaml'.")
    except Exception as e:
        print(f"Error loading prompt from YAML file: {e}")
        raise

    if not settings.GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY is not set in result_sdk.config.settings")
    genai.configure(api_key=settings.GOOGLE_API_KEY)

    if not settings.SERVICE_ACCOUNT_FILE or not os.path.exists(settings.SERVICE_ACCOUNT_FILE):
        raise ValueError(f"Service account file not found: {settings.SERVICE_ACCOUNT_FILE}")
    print(f"Using service account file: {settings.SERVICE_ACCOUNT_FILE}")

    # --- 문서 처리 실행 ---
    # process_document는 시각화 데이터, RAG 데이터, 원본 OCR 데이터, 이미지 경로, 임시 폴더 경로를 반환
    vis_data, rag_data, ocr_data, img_paths, temp_folder = process_document(
        input_file_path=input_file_path,
        service_account_file=settings.SERVICE_ACCOUNT_FILE,
        temp_base_dir=settings.BASE_TEMP_DIR,
        prompt_template=PROMPT_TEMPLATE,
        generation_config=settings.GENERATION_CONFIG,
        safety_settings=settings.SAFETY_SETTINGS
    )

    # --- 결과 요약 출력 ---
    print("\n--- Document Processing Complete ---")
    print(f"  - Visualization data items: {len(vis_data)}")
    print(f"  - RAG ready data items: {len(rag_data)}")
    print(f"  - Original OCR data items: {len(ocr_data)}")
    print(f"  - Processed image pages: {len(img_paths)}")
    print("-" * 30)

    # --- 샘플 데이터 출력 ---
    print("\n[Visualization Data Sample (Top 3)]")
    if vis_data:
        for i, item in enumerate(vis_data[:3]):
            print(f"  {i+1}: {str(item)[:100]}...")
    else:
        print("  No data produced.")

    print("\n[RAG Ready Data Sample (Top 3)]")
    if rag_data:
        for i, item in enumerate(rag_data[:3]):
            print(f"  {i+1}: {str(item)[:100]}...")
    else:
        print("  No data produced.")

    # --- 정리 ---
    if temp_folder and os.path.exists(temp_folder):
        try:
            shutil.rmtree(temp_folder)
            print(f"\nCleaned up temporary folder: '{temp_folder}'")
        except Exception as e:
            print(f"\nError cleaning up temporary folder '{temp_folder}': {e}")

    print("=" * 70)
    print("Text Processing Test Finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Text Processing Test Script for result-sdk")
    parser.add_argument(
        "--input_file",
        default=os.getenv("TEST_INPUT_FILE", '/Users/ki/Desktop/Google Drive/Dev/Ecode/OCR_Test/예시_한국사.pdf'),
        help="Path to the PDF or image file to process for testing."
    )
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: Test input file not found: '{args.input_file}'")
        exit(1)
        
    run_text_processing_test(args.input_file)
