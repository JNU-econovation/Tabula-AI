"""
OCR-LLM 파이프라인 정확도 테스트 스크립트

이 스크립트는 문서 처리 파이프라인을 실행하고,
중간 단계인 순수 OCR 결과와 최종 LLM 처리 결과를 별도의 텍스트 파일로 저장합니다.
이를 통해 각 단계의 성능을 독립적으로 평가할 수 있습니다.

실행 방법:
python -m tests.test_accuracy --input_file [테스트할 파일 경로] --output_dir [결과 저장 폴더]
"""

import os
import shutil
import argparse
import google.generativeai as genai
from result_sdk.config import settings
from result_sdk.text_processing import process_document
from common_sdk.prompt_loader import PromptLoader

def save_results_to_file(data, output_path, data_type='rag'):
    """
    처리된 데이터를 텍스트 파일로 저장합니다.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        if not data:
            f.write("No data produced.")
            return

        if data_type == 'rag':
            # RAG 데이터는 [id_list, [text, ...]] 형식
            sorted_data = sorted(data, key=lambda x: (x[0][0], x[0][1], x[0][2])) # page, block, id 순으로 정렬
            for item in sorted_data:
                if len(item) > 1 and item[1]:
                    text = item[1][0]
                    f.write(text + '\n')
        elif data_type == 'ocr':
            # OCR 데이터를 ID 기반으로 정렬 후, 줄 단위로 재구성하여 저장
            sorted_data = sorted(data, key=lambda x: (x['page_num'], x['block_id'], x['y_idx'], x['x_idx']))
            
            current_page = -1
            current_block = -1
            current_line_y = -1
            current_line_text = []

            for item in sorted_data:
                page = item['page_num']
                block = item['block_id']
                y_idx = item['y_idx']
                text = item['text']

                if page != current_page or block != current_block:
                    if current_line_text:
                        f.write(" ".join(current_line_text) + '\n')
                    f.write(f"\n--- Page {page}, Block {block} ---\n")
                    current_page, current_block, current_line_y = page, block, -1
                    current_line_text = []

                if y_idx != current_line_y:
                    if current_line_text:
                        f.write(" ".join(current_line_text) + '\n')
                    current_line_y = y_idx
                    current_line_text = [text]
                else:
                    current_line_text.append(text)
            
            if current_line_text:
                f.write(" ".join(current_line_text) + '\n')

def run_accuracy_test(input_file: str, output_dir: str):
    """
    파이프라인을 실행하고 OCR, LLM 결과를 파일로 저장하여 정확도 분석을 돕습니다.
    """
    print(f"Accuracy Test Started: Processing '{input_file}'")
    print("=" * 70)

    # --- 출력 디렉토리 준비 ---
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # --- 설정 및 초기화 ---
    prompt_loader = PromptLoader()
    try:
        ocr_prompt_data = prompt_loader.load_prompt('ocr-prompt')
        PROMPT_TEMPLATE = ocr_prompt_data['template']
    except Exception as e:
        print(f"Error loading prompt: {e}")
        raise

    genai.configure(api_key=settings.GOOGLE_API_KEY)

    # --- 문서 처리 실행 ---
    vis_data, rag_data, ocr_data, img_paths, temp_folder = process_document(
        input_file_path=input_file,
        service_account_file=settings.SERVICE_ACCOUNT_FILE,
        temp_base_dir=settings.BASE_TEMP_DIR,
        prompt_template=PROMPT_TEMPLATE,
        generation_config=settings.GENERATION_CONFIG,
        safety_settings=settings.SAFETY_SETTINGS
    )

    print("\n--- Document Processing Complete ---")
    print(f"  - Original OCR data items: {len(ocr_data)}")
    print(f"  - RAG ready data (LLM processed) items: {len(rag_data)}")

    # --- 결과 파일로 저장 ---
    ocr_output_path = os.path.join(output_dir, "ocr_output.txt")
    llm_output_path = os.path.join(output_dir, "llm_output.txt")

    save_results_to_file(ocr_data, ocr_output_path, data_type='ocr')
    print(f"\nSaved raw OCR results to: {ocr_output_path}")

    save_results_to_file(rag_data, llm_output_path, data_type='rag')
    print(f"Saved LLM processed results to: {llm_output_path}")

    # --- 정리 ---
    if temp_folder and os.path.exists(temp_folder):
        shutil.rmtree(temp_folder)
        print(f"\nCleaned up temporary folder: '{temp_folder}'")

    print("\n--- Next Steps ---")
    print("1. Create a 'ground_truth.txt' file in the output directory with the correct text.")
    print("2. Compare 'ground_truth.txt' with 'ocr_output.txt' to evaluate OCR performance.")
    print("3. Compare 'ocr_output.txt' with 'llm_output.txt' to evaluate LLM's correction performance.")
    print("=" * 70)
    print("Accuracy Test Finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Accuracy Test Script for OCR-LLM Pipeline")
    parser.add_argument(
        "--input_file",
        default='/Users/ki/Desktop/Google Drive/Dev/Ecode/OCR_Test/예시_한국사.pdf',
        #default='/Users/ki/Desktop/test1.pdf',
        help="Path to the PDF or image file to process for testing."
    )
    parser.add_argument(
        "--output_dir",
        default="./accuracy_test_results",
        help="Directory to save the output text files."
    )
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: Test input file not found: '{args.input_file}'")
        exit(1)
        
    run_accuracy_test(args.input_file, args.output_dir)
