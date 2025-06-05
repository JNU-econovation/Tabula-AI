# 파일명: llm_interaction.py

import re
import google.generativeai as genai
from PIL import Image
# PROMPT_TEMPLATE will be passed as an argument

# 이 함수는 OCR 결과를 LLM 프롬프트의 일부로 변환합니다.
def format_ocr_results_for_prompt(ocr_block_results: list) -> str:
    """ID가 할당된 OCR 블록 결과를 LLM 프롬프트용 문자열로 포맷합니다."""
    return "\n".join([f"ID({item.get('page_num')},{item.get('block_id')},{item.get('y_idx')},{item.get('x_idx')}): {item.get('text')}" for item in ocr_block_results])

# 이 함수는 포맷된 OCR 결과 문자열과 프롬프트 템플릿을 결합합니다.
def build_full_prompt(ocr_chunk_list_str: str, prompt_template: str) -> str:
    """OCR 청크 문자열과 프롬프트 템플릿을 결합하여 전체 프롬프트를 생성합니다."""
    return prompt_template.replace("{ocr_chunk_list_placeholder}", ocr_chunk_list_str)

# LLM 호출 함수 (이전 답변에서 개선된 버전)
def get_llm_response(full_prompt: str, image_path: str,
                     generation_config_dict: dict, safety_settings_list: list = None) -> str:
    ERROR_PREFIX = "LLM_RESPONSE_ERROR: "
    try:
        img = Image.open(image_path)
    except FileNotFoundError:
        return f"{ERROR_PREFIX}Image file not found."
    except Exception as e:
        return f"{ERROR_PREFIX}Error opening image: {str(e)}"
    
    # 모델 이름은 필요에 따라 config 등에서 관리하거나 직접 지정할 수 있습니다.
    # 현재 사용 중인 모델로 유지 (또는 'gemini-1.5-flash-latest')
    model = genai.GenerativeModel('gemini-1.5-flash-latest') # 안정적인 버전으로 변경 권장
    contents = [full_prompt, img]
    
    try:
        response = model.generate_content(
            contents,
            generation_config=genai.types.GenerationConfig(**generation_config_dict),
            safety_settings=safety_settings_list
        )

        if not response.candidates:
            feedback_info = "Unknown reason"
            if response.prompt_feedback:
                block_reason_str = str(response.prompt_feedback.block_reason) if hasattr(response.prompt_feedback, 'block_reason') and response.prompt_feedback.block_reason else "N/A"
                safety_ratings_str = str(response.prompt_feedback.safety_ratings) if hasattr(response.prompt_feedback, 'safety_ratings') and response.prompt_feedback.safety_ratings else "N/A"
                feedback_info = f"Block Reason: {block_reason_str}, Safety Ratings: {safety_ratings_str}"
            # print(f"  LLM Error: No candidates returned. Feedback: {feedback_info}") # 상세 로그는 여기서 관리
            return f"{ERROR_PREFIX}Request was blocked or no candidates. Feedback: {feedback_info}"

        candidate = response.candidates[0]
        
        finish_reason_value = 0 
        finish_reason_name_str = "UNSPECIFIED" # 기본값

        if hasattr(candidate, 'finish_reason'):
            if hasattr(candidate.finish_reason, 'value'): # Enum 객체일 경우
                finish_reason_value = candidate.finish_reason.value
            else: # 숫자 값일 경우
                finish_reason_value = candidate.finish_reason
        
        can_use_finish_reason_enum = hasattr(genai.types, 'FinishReason')
        if can_use_finish_reason_enum:
            try:
                finish_reason_name_str = genai.types.FinishReason(finish_reason_value).name
            except ValueError: # enum에 없는 값일 경우 대비
                 finish_reason_name_str = f"UNKNOWN_REASON_VALUE_{finish_reason_value}"
            except Exception as e_enum: # 기타 예외
                print(f"  Warning: Could not get FinishReason enum name for value {finish_reason_value} (Error: {e_enum}). Using numeric value.")
                finish_reason_name_str = str(finish_reason_value)
        else:
            print(f"  Warning: genai.types.FinishReason enum not found in genai.types. Using numeric finish_reason value ({finish_reason_value}).")
            finish_reason_name_str = str(finish_reason_value) # 숫자 값으로 사용
            
        print(f"  LLM Candidate Finish Reason: {finish_reason_name_str} (Value: {finish_reason_value})")

        # FinishReason 값에 따른 처리 (숫자 값으로 비교)
        # (참고: genai.types.FinishReason.SAFETY.value 는 3, .MAX_TOKENS.value는 2, .STOP.value는 1 - 라이브러리 버전에 따라 확인 필요)
        # 사용자 로그에서는 2가 SAFETY로 나왔으므로, 해당 값을 기준으로 함
        OBSERVED_FINISH_REASON_SAFETY_INT = 2 
        OBSERVED_FINISH_REASON_STOP_INT = 1 
        OBSERVED_FINISH_REASON_MAX_TOKENS_INT = -1 # 이전 로그에서 MAX_TOKENS는 관찰되지 않음, 임의의 값. 실제로는 2

        if finish_reason_value == OBSERVED_FINISH_REASON_SAFETY_INT:
            feedback_info = "Not available"
            if response.prompt_feedback:
                block_reason_str = str(response.prompt_feedback.block_reason) if hasattr(response.prompt_feedback, 'block_reason') and response.prompt_feedback.block_reason else "N/A"
                safety_ratings_str = str(response.prompt_feedback.safety_ratings) if hasattr(response.prompt_feedback, 'safety_ratings') and response.prompt_feedback.safety_ratings else "N/A"
                feedback_info = f"Block Reason: {block_reason_str}, Safety Ratings: {safety_ratings_str}"
            # print(f"  LLM Error: Content blocked due to safety settings (Finish Reason Value: {finish_reason_value}). Feedback: {feedback_info}")
            return f"{ERROR_PREFIX}Content blocked by safety settings. Finish Reason: {finish_reason_name_str} ({finish_reason_value}). Feedback: {feedback_info}"
        
        # 응답에 유효한 텍스트 Part가 있는지 확인
        if not candidate.content or not candidate.content.parts:
            return f"{ERROR_PREFIX}No valid content part in LLM response. Finish reason: {finish_reason_name_str} ({finish_reason_value})"
        
        try:
            return response.text # 성공적인 경우
        except Exception as e_text: # response.text 접근 시 발생할 수 있는 추가 오류 (예: 라이브러리 내부 오류)
            # print(f"  LLM Error: Failed to access response.text. Finish Reason: {finish_reason_name_str}, Error: {e_text}")
            return f"{ERROR_PREFIX}Failed to extract text from LLM response. Finish Reason: {finish_reason_name_str} ({finish_reason_value}). Error: {str(e_text)}"

    except AttributeError as e_attr: 
        print(f"  LLM Response Processing AttributeError: {e_attr} (Hint: Check 'google-generativeai' library version or installation. Try: pip install --upgrade google-generativeai)")
        return f"{ERROR_PREFIX}Error processing LLM response attribute: {str(e_attr)}"
    except Exception as e_main:
        # print(f"  LLM API Call Exception: {e_main}")
        return f"{ERROR_PREFIX}Error during LLM API call: {str(e_main)}"

# LLM 결과 처리 및 통합 함수 (변경 없음)
def process_llm_and_integrate(llm_output_string: str, original_ocr_results_for_block: list) -> tuple[list, list]:
    llm_processed_info = {}
    line_pattern = re.compile(r"ID\((\d+),(\d+),(\d+),(\d+)\):\s*(.*)")
    merged_pattern = re.compile(r"__MERGED_TO_ID\((\d+),(\d+),(\d+),(\d+)\)__")

    if llm_output_string.startswith("LLM_RESPONSE_ERROR:"):
        print(f"  LLM Error detected by custom prefix in process_llm_and_integrate: {llm_output_string}")
        consolidated_data_block = []
        for ocr_item in original_ocr_results_for_block:
            item_copy = ocr_item.copy()
            item_copy['llm_processed_text'] = ocr_item['text']
            item_copy['llm_status'] = 'LLM_ERROR_FALLBACK'
            consolidated_data_block.append(item_copy)
        return consolidated_data_block, []

    for line in llm_output_string.strip().split('\n'):
        line_match = line_pattern.match(line)
        if not line_match:
            # print(f"  Warning: Could not parse LLM output line in process_llm_and_integrate: '{line}'")
            continue
        page_num, block_id_from_llm, y_idx, x_idx = map(int, line_match.group(1, 2, 3, 4)) # block_id는 LLM 응답에서 온 것
        content = line_match.group(5).strip()
        # current_id_tuple은 LLM 응답의 ID를 사용
        current_id_tuple = (page_num, block_id_from_llm, y_idx, x_idx) 
        
        merged_match = merged_pattern.match(content)
        if merged_match:
            target_p, target_b, target_y, target_x = map(int, merged_match.group(1, 2, 3, 4))
            llm_processed_info[current_id_tuple] = {
                "status": "MERGED", "text": content,
                "target_id": (target_p, target_b, target_y, target_x)
            }
        elif content == "__REMOVED__":
            llm_processed_info[current_id_tuple] = {"status": "REMOVED", "text": ""}
        else:
            llm_processed_info[current_id_tuple] = {"status": "PROCESSED", "text": content}

    consolidated_data_block = []
    for ocr_item in original_ocr_results_for_block: # 이 ocr_item의 block_id는 현재 처리중인 블록의 ID
        # LLM 응답에서 파싱된 ID와 비교해야 함.
        # LLM은 프롬프트에 제공된 ID(page, block, y, x) 그대로 응답해야 함.
        item_id_tuple = (ocr_item['page_num'], ocr_item['block_id'], ocr_item['y_idx'], ocr_item['x_idx'])
        integrated_item = ocr_item.copy()
        if item_id_tuple in llm_processed_info:
            llm_info = llm_processed_info[item_id_tuple]
            integrated_item['llm_processed_text'] = llm_info['text']
            integrated_item['llm_status'] = llm_info['status']
            if llm_info['status'] == 'MERGED':
                integrated_item['llm_merged_target_id'] = llm_info['target_id']
        else:
            integrated_item['llm_processed_text'] = ocr_item['text']
            integrated_item['llm_status'] = 'UNPROCESSED_BY_LLM'
        consolidated_data_block.append(integrated_item)

    rag_ready_data_block = []
    for item in consolidated_data_block:
        if item.get('llm_status') == 'PROCESSED' and item.get('llm_processed_text','').strip():
            id_list_for_rag = [item['page_num'], item['block_id'], item['y_idx'], item['x_idx']]
            text_for_rag = item['llm_processed_text']
            rag_ready_data_block.append([id_list_for_rag, [text_for_rag]])
    return consolidated_data_block, rag_ready_data_block

# LLM_interaction.py TestCode
if __name__ == '__main__':
    print("--- LLM_interaction.py 테스트 시작 ---")

    # 테스트를 위한 기본 설정 (실제 환경에서는 config.py 등에서 관리)
    # API 키 설정 (환경 변수에서 로드)
    from dotenv import load_dotenv
    import os

    load_dotenv()
    GOOGLE_API_KEY_TEST = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY_TEST:
        print("오류: 테스트를 위해 GOOGLE_API_KEY 환경 변수를 설정해주세요.")
        exit()
    try:
        genai.configure(api_key=GOOGLE_API_KEY_TEST)
        print("Google Generative AI SDK 설정 완료 (테스트용).")
    except Exception as e:
        print(f"오류: Google Generative AI SDK 설정 실패 - {e}")
        exit()

    # --- 1. format_ocr_results_for_prompt 함수 테스트 ---
    print("\n[1. format_ocr_results_for_prompt 함수 테스트]")
    mock_ocr_block_for_format = [
        {'page_num': 1, 'block_id': 0, 'y_idx': 1, 'x_idx': 1, 'text': '첫번째'},
        {'page_num': 1, 'block_id': 0, 'y_idx': 1, 'x_idx': 2, 'text': '텍스트'},
        {'page_num': 1, 'block_id': 0, 'y_idx': 2, 'x_idx': 1, 'text': '다음줄'},
    ]
    formatted_ocr_string = format_ocr_results_for_prompt(mock_ocr_block_for_format)
    print("포맷된 OCR 문자열:")
    print(formatted_ocr_string)
    assert formatted_ocr_string == "ID(1,0,1,1): 첫번째\nID(1,0,1,2): 텍스트\nID(1,0,2,1): 다음줄"
    print("format_ocr_results_for_prompt 함수 테스트 통과!")

    # --- 2. build_full_prompt 함수 테스트 ---
    print("\n[2. build_full_prompt 함수 테스트]")
    try:
        # text_processing 내의 Prompt.py에서 가져오도록 수정
        from .Prompt import gemini_prompt as TEST_PROMPT_TEMPLATE
        print("Prompt.py에서 gemini_prompt를 성공적으로 가져왔습니다.")
    except ImportError:
        print("Warning: Prompt.py 또는 gemini_prompt를 찾을 수 없습니다. 임시 프롬프트 템플릿을 사용합니다.")
        TEST_PROMPT_TEMPLATE = "OCR Data:\n{ocr_chunk_list_placeholder}\n\nImage Context:\n(이미지 있음)"
    
    full_prompt_built = build_full_prompt(formatted_ocr_string, TEST_PROMPT_TEMPLATE)
    print("생성된 전체 프롬프트 (일부):")
    print(full_prompt_built[:200] + "...") # 너무 길면 일부만 출력
    assert "{ocr_chunk_list_placeholder}" not in full_prompt_built
    assert formatted_ocr_string in full_prompt_built
    print("build_full_prompt 함수 테스트 통과!")

    # --- 3. get_llm_response 함수 테스트 (실제 API 호출 발생!) ---
    print("\n[3. get_llm_response 함수 테스트 (실제 API 호출)]")
    print("주의: 이 테스트는 실제 LLM API를 호출하므로 비용이 발생할 수 있고, 시간이 소요될 수 있습니다.")
    
    sample_image_path = "" 

    if os.path.exists(sample_image_path):
        test_generation_config = {"temperature": 0.1, "max_output_tokens": 50} 
        test_safety_settings = [ 
            {"category": genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT, "threshold": genai.types.HarmBlockThreshold.BLOCK_NONE},
            {"category": genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH, "threshold": genai.types.HarmBlockThreshold.BLOCK_NONE},
            {"category": genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, "threshold": genai.types.HarmBlockThreshold.BLOCK_NONE},
            {"category": genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, "threshold": genai.types.HarmBlockThreshold.BLOCK_NONE},
        ]
        
        simple_ocr_for_llm_test = "ID(1,0,1,1): 안녕하세요"
        simple_full_prompt = TEST_PROMPT_TEMPLATE.replace("{ocr_chunk_list_placeholder}", simple_ocr_for_llm_test)
        
        print(f"LLM에 전달할 간단한 프롬프트: {simple_full_prompt[:100]}...")
        llm_response = get_llm_response(simple_full_prompt, sample_image_path, test_generation_config, test_safety_settings)
        print("LLM 응답:")
        print(llm_response)
        if llm_response.startswith("LLM_RESPONSE_ERROR:"):
            print("get_llm_response 함수 테스트 중 오류 발생 (예상된 오류일 수 있음).")
        else:
            print("get_llm_response 함수 테스트 (일단) 통과!")
    else:
        print(f"경고: 테스트 이미지 경로 '{sample_image_path}'를 찾을 수 없어 get_llm_response 테스트를 건너<0xEB><0x9C><0x85>니다.")


    # --- 4. process_llm_and_integrate 함수 테스트 ---
    print("\n[4. process_llm_and_integrate 함수 테스트]")
    mock_llm_output = """ID(1,0,1,1): 수정된 첫번째 텍스트
ID(1,0,1,2): __MERGED_TO_ID(1,0,1,1)__
ID(1,0,2,1): __REMOVED__
ID(1,0,3,1): 이것은 처리됨"""

    mock_original_for_integrate = [
        {'page_num': 1, 'block_id': 0, 'y_idx': 1, 'x_idx': 1, 'text': '원본 첫번째', 'x1':10, 'y1':10, 'x2':20, 'y2':20, 'bounding_box':[]},
        {'page_num': 1, 'block_id': 0, 'y_idx': 1, 'x_idx': 2, 'text': '원본 텍스트', 'x1':20, 'y1':10, 'x2':30, 'y2':20, 'bounding_box':[]},
        {'page_num': 1, 'block_id': 0, 'y_idx': 2, 'x_idx': 1, 'text': '원본 다음줄', 'x1':10, 'y1':20, 'x2':20, 'y2':30, 'bounding_box':[]},
        {'page_num': 1, 'block_id': 0, 'y_idx': 3, 'x_idx': 1, 'text': '이것은', 'x1':10, 'y1':30, 'x2':20, 'y2':40, 'bounding_box':[]},
        {'page_num': 1, 'block_id': 0, 'y_idx': 3, 'x_idx': 2, 'text': '처리안됨', 'x1':20, 'y1':30, 'x2':30, 'y2':40, 'bounding_box':[]}, 
    ]

    consolidated_data, rag_data = process_llm_and_integrate(mock_llm_output, mock_original_for_integrate)
    
    print("통합된 데이터 (Consolidated Data):")
    for item in consolidated_data:
        print(f"  ID({item['page_num']},{item['block_id']},{item['y_idx']},{item['x_idx']}) "
              f"Orig: '{item['text']}', LLM: '{item['llm_processed_text']}', Status: {item['llm_status']}"
              + (f", MergedTo: {item['llm_merged_target_id']}" if 'llm_merged_target_id' in item else ""))

    print("\nRAG 준비 데이터 (RAG Ready Data):")
    for item in rag_data:
        print(f"  {item}")

    assert len(consolidated_data) == 5
    assert consolidated_data[0]['llm_status'] == 'PROCESSED'
    assert consolidated_data[0]['llm_processed_text'] == '수정된 첫번째 텍스트'
    assert consolidated_data[1]['llm_status'] == 'MERGED'
    assert consolidated_data[1]['llm_merged_target_id'] == (1,0,1,1)
    assert consolidated_data[2]['llm_status'] == 'REMOVED'
    assert consolidated_data[3]['llm_status'] == 'PROCESSED'
    assert consolidated_data[4]['llm_status'] == 'UNPROCESSED_BY_LLM' 
    
    assert len(rag_data) == 2 
    assert rag_data[0][0] == [1,0,1,1] 
    assert rag_data[1][0] == [1,0,3,1] 

    print("process_llm_and_integrate 함수 테스트 통과!")

    print("\n--- LLM_interaction.py 테스트 종료 ---")
