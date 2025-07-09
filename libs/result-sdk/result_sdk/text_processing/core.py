"""
문서(PDF 또는 이미지)를 처리하여 구조화된 텍스트 데이터를 추출하는 핵심 모듈입니다.
OCR, LLM을 이용한 텍스트 정제, 데이터 병렬 처리 등의 기능을 포함합니다.
"""

import os
import shutil
import uuid
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import google.generativeai as genai

# result_sdk의 최상위 설정을 가져옵니다.
from .. import config
# 공통 SDK에서 프롬프트 로더를 가져옵니다.
from common_sdk.prompt_loader import PromptLoader
# 현재 패키지(text_processing) 내의 모듈들을 가져옵니다.
from .input_handler import get_image_paths_from_input
from .OCR_Processor import (
    ocr_image, parse_raw_words, find_vertical_split_point, assign_ids_after_split
)
from .LLM_interaction import (
    format_ocr_results_for_prompt, build_full_prompt, get_llm_response, process_llm_and_integrate
)

# Google API 키 설정
# config.py 또는 환경 변수를 통해 GOOGLE_API_KEY가 설정되어 있어야 합니다.
if not config.settings.GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY가 설정되지 않았습니다.")
genai.configure(api_key=config.settings.GOOGLE_API_KEY)


def process_document(
    input_file_path: str,
    service_account_file: str,
    temp_base_dir: str,
    prompt_template: str,
    generation_config: dict,
    safety_settings: list,
    pre_converted_image_paths: list = None
) -> tuple[list, list, list, list, str | None]:
    """
    입력된 문서(PDF 또는 이미지)를 페이지별로 처리하여 RAG용 데이터와 시각화용 데이터를 생성합니다.

    이 함수는 다음 단계를 수행합니다:
    1. 입력이 PDF인 경우, 이미지가 포함된 임시 폴더를 생성합니다.
    2. 각 이미지 페이지에 대해 OCR을 수행하여 텍스트와 위치 정보를 추출합니다.
    3. OCR 결과를 좌우 블록으로 분할하고, 각 텍스트 청크에 고유 ID를 할당합니다.
    4. 각 블록을 병렬로 처리하여 LLM에 전달하고, 정제된 텍스트를 받습니다.
    5. LLM 응답과 원본 OCR 데이터를 통합하여 최종 결과물을 생성합니다.

    Args:
        input_file_path (str): 처리할 PDF 또는 이미지 파일의 경로.
        service_account_file (str): Google Cloud 서비스 계정 파일 경로.
        temp_base_dir (str): PDF를 이미지로 변환할 때 사용할 임시 디렉토리의 기본 경로.
        prompt_template (str): LLM에 전달할 프롬프트 템플릿.
        generation_config (dict): LLM 생성 관련 설정.
        safety_settings (list): LLM 안전 관련 설정.
        pre_converted_image_paths (list, optional): 사전에 이미지로 변환된 파일들의 경로 리스트.
                                                     이 값이 제공되면 PDF 변환을 건너뜁니다. Defaults to None.

    Returns:
        tuple[list, list, list, list, str | None]:
            - all_visualization_data_doc (list): 시각화용 데이터 리스트.
            - all_rag_ready_data_doc (list): RAG에 사용될 데이터 리스트.
            - all_original_ocr_data_doc (list): 원본 OCR 데이터 리스트.
            - image_files_to_process (list): 처리된 이미지 파일 경로 리스트.
            - created_temp_folder_for_this_run (str | None): 생성된 임시 폴더의 경로.
                                                             폴더가 생성되지 않았거나, 사전 변환 이미지를 사용한 경우 None.
    """
    print(f"Processing document: {input_file_path}")
    
    created_temp_folder_for_this_run = None
    image_files_to_process = []

    if pre_converted_image_paths:
        print(f"Using {len(pre_converted_image_paths)} pre-converted images.")
        image_files_to_process = pre_converted_image_paths
        # 사전 변환된 이미지를 사용하므로, 이 실행에서 임시 폴더를 직접 생성하지 않습니다.
    else:
        unique_subfolder_name = f"{config.settings.DEFAULT_PDF_CONVERT_TEMP_SUBDIR_NAME}_{os.path.splitext(os.path.basename(input_file_path))[0]}_{uuid.uuid4().hex[:8]}"
        current_temp_pdf_folder_for_run = os.path.join(temp_base_dir, unique_subfolder_name)

        if input_file_path.lower().endswith('.pdf'):
            image_files_to_process = get_image_paths_from_input(input_file_path, current_temp_pdf_folder_for_run)
            if image_files_to_process:
                 created_temp_folder_for_this_run = current_temp_pdf_folder_for_run
        else:
            # 이미지 파일은 별도의 임시 폴더가 필요 없습니다.
            image_files_to_process = get_image_paths_from_input(input_file_path, None)

    if not image_files_to_process:
        print(f"No images to process found in '{input_file_path}'.")
        if created_temp_folder_for_this_run and os.path.exists(created_temp_folder_for_this_run):
            try:
                shutil.rmtree(created_temp_folder_for_this_run)
                print(f"  (Due to error) Deleted temporary folder '{created_temp_folder_for_this_run}'.")
            except Exception as e_del:
                print(f"  (Due to error) Error deleting temporary folder '{created_temp_folder_for_this_run}': {e_del}")
        return [], [], [], [], None

    all_visualization_data_doc = []
    all_rag_ready_data_doc = []
    all_original_ocr_data_doc = []

    def task_process_block(block_id_for_task, ocr_chunks_this_block, current_page_num, current_image_file_path,
                           current_prompt_template, current_gen_config, current_safety_settings):
        """페이지 내의 한 블록(좌/우)을 처리하는 태스크"""
        if not ocr_chunks_this_block:
            return [], []

        print(f"    [Page {current_page_num}, Block {block_id_for_task}] Preparing for LLM call... ({len(ocr_chunks_this_block)} chunks)")
        ocr_chunk_list_str_block = format_ocr_results_for_prompt(ocr_chunks_this_block)
        full_prompt_for_llm_block = build_full_prompt(ocr_chunk_list_str_block, current_prompt_template)

        print(f"    [Page {current_page_num}, Block {block_id_for_task}] Calling LLM (Model: {config.settings.LLM_MODEL_NAME})...")
        llm_response_text_block = get_llm_response(
            full_prompt_for_llm_block,
            current_image_file_path,
            current_gen_config,
            current_safety_settings
        )
        
        print(f"    [Page {current_page_num}, Block {block_id_for_task}] Processing and integrating LLM results...")
        visualization_data_b, rag_ready_data_b = process_llm_and_integrate(
            llm_response_text_block,
            ocr_chunks_this_block
        )
        
        if llm_response_text_block.startswith("LLM_RESPONSE_ERROR:"):
             print(f"    [Warning] Page {current_page_num}, Block {block_id_for_task}: RAG data not generated due to LLM error.")
        
        print(f"    [Page {current_page_num}, Block {block_id_for_task}] Processing complete. Visualization data: {len(visualization_data_b)}, RAG data: {len(rag_ready_data_b)}")
        return visualization_data_b, rag_ready_data_b

    for page_idx, image_file_path in enumerate(image_files_to_process):
        page_num = page_idx + 1
        print(f"\n\n[Processing Page {page_num}/{len(image_files_to_process)}] File: {os.path.basename(image_file_path)}")
        
        print(f"  [Page {page_num}] Step 1: Running OCR")
        try:
            response_json = ocr_image(image_file_path, service_account_file)
            raw_words = parse_raw_words(response_json)
            if not raw_words:
                print(f"    [Warning] Page {page_num}: No words found in OCR result. Skipping to next page.")
                continue
            
            image_for_size_check = cv2.imread(image_file_path)
            if image_for_size_check is None:
                print(f"    [Error] Could not load image for page {page_num}: {image_file_path}")
                continue
            
            image_height, image_width, _ = image_for_size_check.shape
            split_x = find_vertical_split_point(raw_words, image_width)
            original_ocr_chunks_for_page = assign_ids_after_split(raw_words, split_x, page_num)
            
            # 원본 OCR 데이터 수집
            all_original_ocr_data_doc.extend(original_ocr_chunks_for_page)
            
            if not original_ocr_chunks_for_page:
                print(f"    [Warning] Page {page_num}: No OCR chunks with assigned IDs. Skipping to next page.")
                continue
            print(f"    Page {page_num}: OCR and ID assignment complete (Total {len(original_ocr_chunks_for_page)} chunks).")
        except Exception as e:
            print(f"    [Error] Unexpected error during OCR processing for page {page_num}, skipping: {e}")
            continue
            
        page_visualization_data_accumulator = []
        page_rag_ready_data_accumulator = []
        
        # OCR 결과를 좌/우 블록으로 분리
        ocr_blocks_for_page = {
            0: [item for item in original_ocr_chunks_for_page if item['block_id'] == 0],
            1: [item for item in original_ocr_chunks_for_page if item['block_id'] == 1]
        }

        # 각 블록을 병렬로 처리
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_to_block_map = {}
            if ocr_blocks_for_page[0]:
                future = executor.submit(task_process_block, 0, ocr_blocks_for_page[0], page_num, image_file_path, 
                                         prompt_template, generation_config, safety_settings)
                future_to_block_map[future] = 0
            if ocr_blocks_for_page[1]:
                future = executor.submit(task_process_block, 1, ocr_blocks_for_page[1], page_num, image_file_path,
                                         prompt_template, generation_config, safety_settings)
                future_to_block_map[future] = 1
            
            block_results_temp = {}
            for future_completed in as_completed(future_to_block_map):
                block_id_completed = future_to_block_map[future_completed]
                try:
                    visualization_block_data, rag_block_data = future_completed.result()
                    block_results_temp[block_id_completed] = (visualization_block_data, rag_block_data)
                except Exception as exc:
                    print(f"    [Error] Exception during parallel processing for page {page_num}, block {block_id_completed}: {exc}")
                    block_results_temp[block_id_completed] = ([], [])
            
            # 병렬 처리 결과 취합
            if 0 in block_results_temp:
                page_visualization_data_accumulator.extend(block_results_temp[0][0])
                page_rag_ready_data_accumulator.extend(block_results_temp[0][1])
            if 1 in block_results_temp:
                page_visualization_data_accumulator.extend(block_results_temp[1][0])
                page_rag_ready_data_accumulator.extend(block_results_temp[1][1])

        all_visualization_data_doc.extend(page_visualization_data_accumulator)
        all_rag_ready_data_doc.extend(page_rag_ready_data_accumulator)
        print(f"\n  Parallel processing for all blocks on page {page_num} is complete.")
            
    return all_visualization_data_doc, all_rag_ready_data_doc, all_original_ocr_data_doc, image_files_to_process, created_temp_folder_for_this_run


if __name__ == '__main__':
    """
    커맨드 라인에서 직접 이 스크립트를 실행할 때의 진입점입니다.
    PDF 또는 이미지 파일을 입력받아 처리하고, 결과를 요약하여 출력합니다.
    """
    parser = argparse.ArgumentParser(description="문서에서 텍스트를 추출하고 LLM으로 정제하여 RAG 데이터를 생성합니다.")
    parser.add_argument("input_file", help="처리할 PDF 또는 이미지 파일의 전체 경로")
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found.")
        exit(1)
    
    if not config.settings.SERVICE_ACCOUNT_FILE or not os.path.exists(config.settings.SERVICE_ACCOUNT_FILE):
        print(f"Error: Service account file '{config.settings.SERVICE_ACCOUNT_FILE}' not found or not configured correctly.")
        exit(1)

    print(f"Processing document '{args.input_file}' (Environment: {config.settings.APP_ENV})")
    
    # common_sdk를 통해 YAML 프롬프트 로드
    try:
        prompt_loader = PromptLoader()
        # 'OCR-PROMPT.yaml' 파일을 'ocr-prompt'라는 이름으로 로드
        ocr_prompt_data = prompt_loader.load_prompt('ocr-prompt')
        PROMPT_TEMPLATE_FROM_YAML = ocr_prompt_data['template']
        print("Successfully loaded OCR prompt from common_sdk.")
    except Exception as e:
        print(f"Error: Failed to load OCR prompt from common_sdk: {e}")
        # 오류 발생 시, 필요에 따라 기본값을 사용하거나 여기서 프로그램을 종료할 수 있습니다.
        exit(1)

    final_visualization_data, final_rag_data, final_original_ocr, processed_image_paths, temp_folder_used = process_document(
        input_file_path=args.input_file,
        service_account_file=config.settings.SERVICE_ACCOUNT_FILE,
        temp_base_dir=config.settings.BASE_TEMP_DIR,
        prompt_template=PROMPT_TEMPLATE_FROM_YAML,
        generation_config=config.settings.GENERATION_CONFIG,
        safety_settings=config.settings.SAFETY_SETTINGS
    )

    print("\n\n\n--- All pages processed ---")
    print(f"Total visualization data items: {len(final_visualization_data)}")
    print(f"Total RAG ready data items: {len(final_rag_data)}")
    print(f"Total original OCR data items: {len(final_original_ocr)}")

    if final_rag_data:
        print("\n--- RAG Ready Data Sample ---")
        sample_count = 0
        for i, item_list in enumerate(final_rag_data):
            # 처음 5개와 마지막 2개 샘플을 출력
            if i < 5 or (len(final_rag_data) - i) <= 2:
                if item_list and len(item_list) > 1 and item_list[0] and item_list[1]:
                    id_list_str = f"ID({','.join(map(str, item_list[0]))})"
                    text_str = item_list[1][0][:100] if item_list[1] else "" 
                    print(f"{id_list_str}, Text: '{text_str}...'")
                else:
                    print(f"Warning: Malformed RAG data item at index {i}: {item_list}")
                sample_count += 1
            elif sample_count == 5 and i < len(final_rag_data) - 2:
                print("...")
                sample_count += 1
    
    # PDF 변환 시 생성된 임시 폴더 삭제
    if temp_folder_used and os.path.exists(temp_folder_used):
        try:
            shutil.rmtree(temp_folder_used)
            print(f"\nTemporary PDF conversion folder '{temp_folder_used}' deleted.")
        except Exception as e:
            print(f"\nError deleting temporary PDF conversion folder '{temp_folder_used}': {e}")

    print("\n[core.py execution complete]")
