# 파일명: core.py

import os
import shutil
import uuid
import argparse # if __name__ == "__main__" 에서 사용
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- 모듈 임포트 ---
import config
from Prompt import gemini_prompt as PROMPT_TEMPLATE # 이제 process_document의 인자로 받음
from input_handler import get_image_paths_from_input
from OCR_Processor import (
    ocr_image, parse_raw_words, find_vertical_split_point, assign_ids_after_split
)
from LLM_interaction import (
    format_ocr_results_for_prompt, build_full_prompt, get_llm_response, process_llm_and_integrate
)
import google.generativeai as genai
import cv2


# --- API 키 설정 ---
if not config.GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY가 config.py를 통해 설정되지 않았습니다.")
genai.configure(api_key=config.GOOGLE_API_KEY)


# --- 핵심 처리 함수 ---
def process_document(input_file_path: str,
                     service_account_file: str,   # <--- 호출부와 일치하도록 수정 (p_ 접두사 제거)
                     temp_base_dir: str,          # <--- 호출부와 일치하도록 수정 (p_ 접두사 제거)
                     prompt_template: str,        # <--- 호출부와 일치하도록 수정 (p_ 접두사 제거)
                     generation_config: dict,     # <--- 호출부와 일치하도록 수정 (이름 일관성)
                     safety_settings: list        # <--- 호출부와 일치하도록 수정 (이름 일관성)
                    ) -> tuple[list, list, list, str | None]:
    print(f"문서 처리 시작: {input_file_path}")

    unique_subfolder_name = f"{config.DEFAULT_PDF_CONVERT_TEMP_SUBDIR_NAME}_{os.path.splitext(os.path.basename(input_file_path))[0]}_{uuid.uuid4().hex[:8]}"
    current_temp_pdf_folder_for_run = os.path.join(temp_base_dir, unique_subfolder_name)
    created_temp_folder_for_this_run = None

    if input_file_path.lower().endswith('.pdf'):
        image_files_to_process = get_image_paths_from_input(input_file_path, current_temp_pdf_folder_for_run)
        if image_files_to_process:
             created_temp_folder_for_this_run = current_temp_pdf_folder_for_run
    else:
        image_files_to_process = get_image_paths_from_input(input_file_path, None)

    if not image_files_to_process:
        print(f"'{input_file_path}'에서 처리할 이미지를 찾지 못했습니다.")
        if created_temp_folder_for_this_run and os.path.exists(created_temp_folder_for_this_run):
            try:
                shutil.rmtree(created_temp_folder_for_this_run)
                print(f"  (오류로 인한) 임시 폴더 '{created_temp_folder_for_this_run}' 삭제 완료.")
            except Exception as e_del:
                print(f"  (오류로 인한) 임시 폴더 '{created_temp_folder_for_this_run}' 삭제 중 오류: {e_del}")
        return [], [], [], None

    all_consolidated_data_doc = []
    all_rag_ready_data_doc = []

    def task_process_block(block_id_for_task, ocr_chunks_this_block, current_page_num, current_image_file_path,
                           # 아래 인자들은 process_document로부터 전달받은 값을 사용
                           current_prompt_template, current_gen_config, current_safety_settings):
        if not ocr_chunks_this_block:
            return [], []
        
        print(f"    [페이지 {current_page_num}, 블록 {block_id_for_task} - LLM 호출 준비] (청크 {len(ocr_chunks_this_block)}개)")
        ocr_chunk_list_str_block = format_ocr_results_for_prompt(ocr_chunks_this_block)
        full_prompt_for_llm_block = build_full_prompt(ocr_chunk_list_str_block, current_prompt_template) # 전달받은 프롬프트 사용

        print(f"    페이지 {current_page_num}, 블록 {block_id_for_task}: LLM 호출 중 (모델: {config.LLM_MODEL_NAME})...")
        llm_response_text_block = get_llm_response(
            full_prompt_for_llm_block,
            current_image_file_path,
            current_gen_config, # 전달받은 설정 사용
            current_safety_settings    # 전달받은 설정 사용
        )
        
        print(f"    [페이지 {current_page_num}, 블록 {block_id_for_task} - LLM 결과 처리 및 통합]...")
        consolidated_data_b, rag_ready_data_b = process_llm_and_integrate(
            llm_response_text_block,
            ocr_chunks_this_block
        )
        
        if llm_response_text_block.startswith("LLM_RESPONSE_ERROR:"):
             print(f"    페이지 {current_page_num}, 블록 {block_id_for_task}: LLM 관련 오류로 RAG 데이터 생성 안됨.")
        
        print(f"    페이지 {current_page_num}, 블록 {block_id_for_task} 처리 완료. Consolidated: {len(consolidated_data_b)}, RAG: {len(rag_ready_data_b)}")
        return consolidated_data_b, rag_ready_data_b

    for page_idx, image_file_path in enumerate(image_files_to_process):
        page_num = page_idx + 1
        print(f"\n\n[페이지 {page_num}/{len(image_files_to_process)} 처리 시작] ({os.path.basename(image_file_path)})")
        
        print(f"  [페이지 {page_num} - 단계 1] OCR 실행 중...")
        try:
            response_json = ocr_image(image_file_path, service_account_file) # 전달받은 service_account_file 사용
            raw_words = parse_raw_words(response_json)
            if not raw_words:
                print(f"    페이지 {page_num}: OCR 결과에서 단어를 찾을 수 없습니다. 다음 단계로 넘어갑니다.")
                continue
            
            image_for_size_check = cv2.imread(image_file_path)
            if image_for_size_check is None:
                print(f"    오류: 페이지 {page_num} 이미지를 로드할 수 없습니다 - {image_file_path}")
                continue
            image_height, image_width, _ = image_for_size_check.shape
            split_x = find_vertical_split_point(raw_words, image_width)
            original_ocr_chunks_for_page = assign_ids_after_split(raw_words, split_x, page_num)
            if not original_ocr_chunks_for_page:
                print(f"    페이지 {page_num}: ID가 할당된 OCR 결과가 없습니다. 다음 단계로 넘어갑니다.")
                continue
            print(f"    페이지 {page_num}: 전체 OCR 및 ID 할당 완료 (총 {len(original_ocr_chunks_for_page)} 청크).")
        except Exception as e:
            print(f"    페이지 {page_num} OCR 처리 중 예기치 않은 오류 발생하여 건너뜁니다: {e}") # 상세 오류 출력
            continue
            
        page_consolidated_data_accumulator = []
        page_rag_ready_data_accumulator = []
        ocr_blocks_for_page = {
            0: [item for item in original_ocr_chunks_for_page if item['block_id'] == 0],
            1: [item for item in original_ocr_chunks_for_page if item['block_id'] == 1]
        }

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_to_block_map = {}
            if ocr_blocks_for_page[0]:
                future = executor.submit(task_process_block, 0, ocr_blocks_for_page[0], page_num, image_file_path, 
                                         prompt_template, generation_config, safety_settings) # 수정된 인자 전달
                future_to_block_map[future] = 0
            if ocr_blocks_for_page[1]:
                future = executor.submit(task_process_block, 1, ocr_blocks_for_page[1], page_num, image_file_path,
                                         prompt_template, generation_config, safety_settings) # 수정된 인자 전달
                future_to_block_map[future] = 1
            
            block_results_temp = {}
            for future_completed in as_completed(future_to_block_map):
                block_id_completed = future_to_block_map[future_completed]
                try:
                    consolidated_block_data, rag_block_data = future_completed.result()
                    block_results_temp[block_id_completed] = (consolidated_block_data, rag_block_data)
                except Exception as exc:
                    print(f"    페이지 {page_num}, 블록 {block_id_completed} 병렬 처리 중 예외 발생: {exc}")
                    block_results_temp[block_id_completed] = ([], [])
            
            if 0 in block_results_temp:
                page_consolidated_data_accumulator.extend(block_results_temp[0][0])
                page_rag_ready_data_accumulator.extend(block_results_temp[0][1])
            if 1 in block_results_temp:
                page_consolidated_data_accumulator.extend(block_results_temp[1][0])
                page_rag_ready_data_accumulator.extend(block_results_temp[1][1])

        all_consolidated_data_doc.extend(page_consolidated_data_accumulator)
        all_rag_ready_data_doc.extend(page_rag_ready_data_accumulator)
        print(f"\n  페이지 {page_num} 모든 블록 병렬 처리 완료.")
            
    return all_consolidated_data_doc, all_rag_ready_data_doc, image_files_to_process, created_temp_folder_for_this_run


# `if __name__ == "__main__":` 블록은 이전 답변과 동일하게 유지하되,
# `process_document` 호출 시 인자 이름이 위 함수 정의와 일치하는지 확인합니다.
# (이전 답변의 __main__ 블록은 이미 'service_account_file', 'generation_config', 'safety_settings' 등을 사용하고 있으므로 호환됩니다.)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="문서에서 텍스트를 추출하고 LLM으로 정제하여 RAG 데이터를 생성합니다.")
    parser.add_argument("input_file", help="처리할 PDF 또는 이미지 파일의 전체 경로")
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"오류: 입력 파일 '{args.input_file}'을 찾을 수 없습니다.")
        exit()
    
    if not config.SERVICE_ACCOUNT_FILE or not os.path.exists(config.SERVICE_ACCOUNT_FILE):
        print(f"오류: 서비스 계정 파일 '{config.SERVICE_ACCOUNT_FILE}'을 찾을 수 없거나 config.py 또는 환경 변수에 올바르게 설정되지 않았습니다.")
        exit()

    print(f"'{args.input_file}' 문서에 대한 처리 시작 (환경: {config.APP_ENV})")
    
    final_consolidated_data, final_rag_data, processed_image_paths, temp_folder_used = process_document(
        input_file_path=args.input_file,
        service_account_file=config.SERVICE_ACCOUNT_FILE, # config에서 가져온 값
        temp_base_dir=config.BASE_TEMP_DIR,
        prompt_template=PROMPT_TEMPLATE, # Prompt.py 에서 가져온 값
        generation_config=config.GENERATION_CONFIG, # config에서 가져온 값
        safety_settings=config.SAFETY_SETTINGS     # config에서 가져온 값
    )

    print("\n\n\n--- 모든 페이지 처리 완료 (core.py 실행) ---")
    print(f"총 통합된 데이터 항목 수: {len(final_consolidated_data)}")
    print(f"총 RAG 준비 데이터 항목 수: {len(final_rag_data)}")

    if final_rag_data:
        print("\n--- 전체 RAG 준비 데이터 (일부 샘플) ---")
        sample_count = 0
        for i, item_list in enumerate(final_rag_data):
            if i < 5 or (len(final_rag_data) - i) <= 2 :
                if item_list and len(item_list) > 1 and item_list[0] and item_list[1]:
                    id_list_str = f"ID({','.join(map(str, item_list[0]))})"
                    text_str = item_list[1][0][:100] if item_list[1] else "" 
                    print(f"{id_list_str}, Text: '{text_str}...'")
                else:
                    print(f"Warning: Malformed RAG data item at index {i}: {item_list}")
                sample_count +=1
            elif sample_count == 5 and i < len(final_rag_data) - 2 :
                print("...")
                sample_count +=1 
    
    if temp_folder_used and os.path.exists(temp_folder_used):
        try:
            shutil.rmtree(temp_folder_used)
            print(f"\ncore.py 실행 완료 후 임시 PDF 변환 폴더 '{temp_folder_used}' 삭제 완료.")
        except Exception as e:
            print(f"\ncore.py 실행 완료 후 임시 PDF 변환 폴더 '{temp_folder_used}' 삭제 중 오류: {e}")

    print("\n[core.py 전체 작업 완료]")