# 파일명: llm_interaction.py

import re
import google.generativeai as genai
from PIL import Image
from result_sdk.config import settings # 설정 가져오기
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
    
    # 모델 이름은 config.py의 설정을 사용합니다.
    model_name_to_use = settings.LLM_MODEL_NAME
    print(f"  Using LLM Model: {model_name_to_use}") # 사용될 모델 이름 로깅
    model = genai.GenerativeModel(model_name_to_use)
    contents = [full_prompt, img]

    current_safety_settings = safety_settings_list
    if current_safety_settings is None:
        print("  Warning: safety_settings_list is None. Applying default BLOCK_NONE settings.")
        if hasattr(genai.types, 'HarmCategory') and hasattr(genai.types, 'HarmBlockThreshold'):
            current_safety_settings = [
                {"category": genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT, "threshold": genai.types.HarmBlockThreshold.BLOCK_NONE},
                {"category": genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH, "threshold": genai.types.HarmBlockThreshold.BLOCK_NONE},
                {"category": genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, "threshold": genai.types.HarmBlockThreshold.BLOCK_NONE},
                {"category": genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, "threshold": genai.types.HarmBlockThreshold.BLOCK_NONE},
            ]
        else:
            print("  Error: Cannot apply default safety settings because HarmCategory or HarmBlockThreshold is not available. LLM might use its own defaults.")
            # 이 경우, safety_settings를 None으로 두어 LLM 기본 설정을 따르거나,
            # 혹은 더 강력한 오류 처리를 할 수 있습니다. 여기서는 None으로 유지합니다.
            current_safety_settings = None # 명시적으로 None으로 설정

    try:
        response = model.generate_content(
            contents,
            generation_config=genai.types.GenerationConfig(**generation_config_dict),
            safety_settings=current_safety_settings
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

        # Determine actual enum values for comparison
        actual_fr_safety, actual_fr_max_tokens, actual_fr_stop = None, None, None
        if can_use_finish_reason_enum:
            try:
                actual_fr_safety = genai.types.FinishReason.SAFETY.value
                actual_fr_max_tokens = genai.types.FinishReason.MAX_TOKENS.value
                actual_fr_stop = genai.types.FinishReason.STOP.value
            except AttributeError:
                print("  Warning: Could not access specific FinishReason enum values (SAFETY, MAX_TOKENS, STOP). Falling back to integer comparison if values are known.")
                # Fallback to known integer values if direct enum access fails after all
                actual_fr_stop = 1
                actual_fr_max_tokens = 2
                actual_fr_safety = 3 
        else: # Fallback if enum itself is not available
            print("  Info: Using fallback integer values for FinishReason comparison (STOP=1, MAX_TOKENS=2, SAFETY=3).")
            actual_fr_stop = 1
            actual_fr_max_tokens = 2
            actual_fr_safety = 3

        if actual_fr_safety is not None and finish_reason_value == actual_fr_safety:
            detailed_feedback = _format_prompt_feedback(response.prompt_feedback if hasattr(response, 'prompt_feedback') else None)
            return f"{ERROR_PREFIX}Content blocked by safety settings. Finish Reason: {finish_reason_name_str} ({finish_reason_value}). Detailed Feedback: {detailed_feedback}"
        
        elif actual_fr_max_tokens is not None and finish_reason_value == actual_fr_max_tokens:
            return f"{ERROR_PREFIX}Response stopped due to maximum token limit. Finish Reason: {finish_reason_name_str} ({finish_reason_value}). Check 'max_output_tokens' in generation config and input/output length."

        # If finish reason is STOP or any other reason that implies content should be present
        if (actual_fr_stop is not None and finish_reason_value == actual_fr_stop) or \
           (finish_reason_value not in [actual_fr_safety, actual_fr_max_tokens]): # Other reasons might still have content

            if not candidate.content or not candidate.content.parts:
                # This can happen even with STOP if the content is empty for some reason
                return f"{ERROR_PREFIX}No valid content part in LLM response despite Finish Reason '{finish_reason_name_str}' ({finish_reason_value})."
            
            try:
                return response.text # Successful case
            except Exception as e_text:
                return f"{ERROR_PREFIX}Failed to extract text from LLM response. Finish Reason: {finish_reason_name_str} ({finish_reason_value}). Error: {str(e_text)}"
        
        # Fallback for unhandled finish reasons if any (should be rare if above logic is complete)
        return f"{ERROR_PREFIX}Unhandled LLM finish reason: {finish_reason_name_str} ({finish_reason_value})."

    except AttributeError as e_attr:
        # It's good to suggest checking the library version here.
        print(f"  LLM Response Processing AttributeError: {e_attr} (Hint: Check 'google-generativeai' library version or installation. Try: pip install --upgrade google-generativeai)")
        return f"{ERROR_PREFIX}Error processing LLM response attribute: {str(e_attr)}"
    except Exception as e_main:
        # print(f"  LLM API Call Exception: {e_main}")
        return f"{ERROR_PREFIX}Error during LLM API call: {str(e_main)}"

def _format_prompt_feedback(prompt_feedback) -> str:
    """Helper function to format prompt feedback details."""
    if not prompt_feedback:
        return "PromptFeedback not available"

    details = []
    if hasattr(prompt_feedback, 'block_reason') and prompt_feedback.block_reason:
        # Ensure block_reason is converted to its name if it's an enum, or string directly
        block_reason_val = prompt_feedback.block_reason
        if hasattr(block_reason_val, 'name'): # It's an enum
            details.append(f"Block Reason: {block_reason_val.name} ({block_reason_val.value})")
        else: # It's likely already a string or simple type
            details.append(f"Block Reason: {str(block_reason_val)}")
    else:
        details.append("Block Reason: N/A")

    if hasattr(prompt_feedback, 'safety_ratings') and prompt_feedback.safety_ratings:
        ratings_details = []
        for rating in prompt_feedback.safety_ratings:
            category_str = "UnknownCategory"
            if hasattr(rating, 'category'):
                if hasattr(rating.category, 'name'): # Enum
                    category_str = f"{rating.category.name} ({rating.category.value})"
                else: # String or other
                    category_str = str(rating.category)
            
            probability_str = "UnknownProbability"
            if hasattr(rating, 'probability'):
                if hasattr(rating.probability, 'name'): # Enum
                    probability_str = rating.probability.name
                else: # String or other
                    probability_str = str(rating.probability)

            blocked_str = f", BlockedByThis: {rating.blocked}" if hasattr(rating, 'blocked') else "" # Some SDKs might have 'blocked'
            
            # Gemini specific: harm_severity and harm_probability_score might be present
            severity_str = ""
            if hasattr(rating, 'harm_severity') and rating.harm_severity:
                 severity_str = f", Severity: {rating.harm_severity.name if hasattr(rating.harm_severity, 'name') else str(rating.harm_severity)}"
            
            score_str = ""
            if hasattr(rating, 'harm_probability_score'): # This might be more specific for Gemini
                score_str = f", ProbabilityScore: {rating.harm_probability_score:.2f}" if isinstance(rating.harm_probability_score, float) else f", ProbabilityScore: {rating.harm_probability_score}"


            ratings_details.append(f"  - Category: {category_str}, Probability: {probability_str}{severity_str}{score_str}{blocked_str}")
        
        if ratings_details:
            details.append("Safety Ratings:\n" + "\n".join(ratings_details))
        else:
            details.append("Safety Ratings: N/A or empty")
    else:
        details.append("Safety Ratings: N/A")
    
    return "; ".join(details)

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
