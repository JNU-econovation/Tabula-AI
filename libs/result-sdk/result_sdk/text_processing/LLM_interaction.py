"""
LLM(Large Language Model) 상호작용 모듈
OCR 결과를 LLM 프롬프트로 변환, API 호출, 응답 처리 기능 포함
"""

import re
import json
import google.generativeai as genai
from PIL import Image
from result_sdk.config import settings

def format_ocr_results_for_prompt(ocr_block_results: list) -> str:
    """ID 할당된 OCR 블록 결과를 LLM 프롬프트용 문자열로 포맷"""
    return "\n".join([f"ID({item.get('page_num')},{item.get('block_id')},{item.get('y_idx')},{item.get('x_idx')}): {item.get('text')}" for item in ocr_block_results])

def build_full_prompt(ocr_chunk_list_str: str, prompt_template: str) -> str:
    """OCR 청크 문자열과 프롬프트 템플릿을 결합해 전체 프롬프트 생성"""
    return prompt_template.replace("{ocr_chunk_list_placeholder}", ocr_chunk_list_str)

def get_llm_response(full_prompt: str, image_path: str,
                     generation_config_dict: dict, safety_settings_list: list = None) -> str:
    """
    LLM API 호출 후 응답 반환

    Args:
        full_prompt (str): LLM에 전달할 전체 프롬프트
        image_path (str): 함께 전달할 이미지 파일 경로
        generation_config_dict (dict): LLM 생성 관련 설정
        safety_settings_list (list, optional): LLM 안전 설정, 없으면 기본값 사용

    Returns:
        str: LLM 응답 텍스트. 오류 시 "LLM_RESPONSE_ERROR:" 접두사 붙은 오류 메시지 반환
    """
    ERROR_PREFIX = "LLM_RESPONSE_ERROR: "
    try:
        img = Image.open(image_path)
    except FileNotFoundError:
        return f"{ERROR_PREFIX}이미지 파일 없음"
    except Exception as e:
        return f"{ERROR_PREFIX}이미지 열기 오류: {str(e)}"
    
    model_name_to_use = settings.LLM_MODEL_NAME
    print(f"  Using LLM Model: {model_name_to_use}")
    model = genai.GenerativeModel(model_name_to_use)
    contents = [full_prompt, img]

    current_safety_settings = safety_settings_list
    if current_safety_settings is None:
        print("  Warning: safety_settings_list is None. Applying default BLOCK_NONE settings.")
        if hasattr(genai.types, 'HarmCategory') and hasattr(genai.types, 'HarmBlockThreshold'):
            current_safety_settings = [
                {"category": cat, "threshold": genai.types.HarmBlockThreshold.BLOCK_NONE}
                for cat in [
                    genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                    genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                    genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                ]
            ]
        else:
            print("  Error: Cannot apply default safety settings because HarmCategory or HarmBlockThreshold is not available.")
            current_safety_settings = None

    try:
        response = model.generate_content(
            contents,
            generation_config=genai.types.GenerationConfig(**generation_config_dict),
            safety_settings=current_safety_settings
        )

        if not response.candidates:
            feedback_info = _format_prompt_feedback(getattr(response, 'prompt_feedback', None))
            return f"{ERROR_PREFIX}요청 차단 또는 후보 응답 없음. 피드백: {feedback_info}"

        candidate = response.candidates[0]
        finish_reason_val = getattr(getattr(candidate, 'finish_reason', None), 'value', getattr(candidate, 'finish_reason', 0))
        
        if finish_reason_val == 3: # SAFETY
            detailed_feedback = _format_prompt_feedback(getattr(response, 'prompt_feedback', None))
            return f"{ERROR_PREFIX}안전 설정에 의해 콘텐츠 차단됨. 상세 피드백: {detailed_feedback}"
        elif finish_reason_val == 2: # MAX_TOKENS
            return f"{ERROR_PREFIX}최대 출력 토큰 제한으로 응답 중단"

        if not getattr(candidate, 'content', None) or not getattr(candidate.content, 'parts', None):
            return f"{ERROR_PREFIX}응답에 유효한 콘텐츠 파트 없음 (종료 사유: {finish_reason_val})"
        
        return response.text

    except Exception as e:
        return f"{ERROR_PREFIX}LLM API 호출 중 오류 발생: {str(e)}"

def _format_prompt_feedback(prompt_feedback) -> str:
    """프롬프트 피드백 객체를 읽기 좋은 문자열로 포맷"""
    if not prompt_feedback:
        return "피드백 정보 없음"

    reason = getattr(prompt_feedback, 'block_reason', 'N/A')
    ratings = "\n".join([
        f"  - 카테고리: {getattr(r.category, 'name', r.category)}, 확률: {getattr(r.probability, 'name', r.probability)}"
        for r in getattr(prompt_feedback, 'safety_ratings', [])
    ])
    return f"차단 사유: {reason}, 안전 등급: \n{ratings if ratings else 'N/A'}"

def process_llm_and_integrate(llm_output_string: str, original_ocr_results_for_block: list) -> tuple[list, list]:
    """
    LLM의 JSON 응답을 파싱해 RAG 및 시각화 데이터 생성

    - 반환값 1: 시각화용 데이터 (LLM 출력 형식과 거의 동일)
    - 반환값 2: RAG용 데이터
    """
    def _fallback_to_error_state(original_data):
        # 오류 시 원본 데이터를 RAG 형식으로 변환해 최소 데이터 보존
        rag_fallback = [
            [[item['page_num'], item['block_id'], item['y_idx'], item['x_idx']], [item['text']]]
            for item in original_data
        ]
        return [], rag_fallback

    if llm_output_string.startswith("LLM_RESPONSE_ERROR:"):
        print(f"  LLM Error detected by custom prefix: {llm_output_string}")
        return _fallback_to_error_state(original_ocr_results_for_block)

    try:
        # LLM 응답에서 JSON 코드 블록 마커 제거
        clean_json_string = re.sub(r'^\s*```json\s*|\s*```\s*$', '', llm_output_string.strip(), flags=re.DOTALL)
        visualization_data = json.loads(clean_json_string)
        if not isinstance(visualization_data, list):
            raise json.JSONDecodeError("LLM 출력이 리스트 형식이 아님", clean_json_string, 0)
    except (json.JSONDecodeError, TypeError) as e:
        print(f"  Error parsing LLM JSON response: {e}\n  Response was: '{llm_output_string[:200]}...'")
        return _fallback_to_error_state(original_ocr_results_for_block)

    id_pattern = re.compile(r"ID\((\d+),(\d+),(\d+),(\d+)\)")
    rag_ready_data_block = []

    for id_list, sentence in visualization_data:
        if not id_list:
            continue
        
        # 첫 ID를 대표 ID로 사용해 RAG 데이터 생성
        match = id_pattern.match(id_list[0])
        if match:
            rag_id_tuple = tuple(map(int, match.groups()))
            rag_ready_data_block.append([list(rag_id_tuple), [sentence]])

    return visualization_data, rag_ready_data_block

if __name__ == '__main__':
    print("--- LLM_interaction.py 테스트 시작 ---")

    from dotenv import load_dotenv
    import os

    load_dotenv()
    GOOGLE_API_KEY_TEST = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY_TEST:
        print("오류: 테스트를 위해 GOOGLE_API_KEY 환경 변수 설정 필요")
        exit()
    genai.configure(api_key=GOOGLE_API_KEY_TEST)

    print("\n[1. format_ocr_results_for_prompt 테스트]")
    mock_ocr_data = [
        {'page_num': 1, 'block_id': 0, 'y_idx': 1, 'x_idx': 1, 'text': '첫번째 텍스트'},
        {'page_num': 1, 'block_id': 0, 'y_idx': 2, 'x_idx': 1, 'text': '다음 줄'},
    ]
    formatted_string = format_ocr_results_for_prompt(mock_ocr_data)
    print(f"포맷된 문자열:\n{formatted_string}")
    assert formatted_string == "ID(1,0,1,1): 첫번째 텍스트\nID(1,0,2,1): 다음 줄"
    print("-> 통과")

    print("\n[2. build_full_prompt 테스트]")
    test_template = "데이터:\n{ocr_chunk_list_placeholder}\n---"
    full_prompt = build_full_prompt(formatted_string, test_template)
    print(f"생성된 프롬프트(일부):\n{full_prompt[:100]}...")
    assert "{ocr_chunk_list_placeholder}" not in full_prompt
    print("-> 통과")

    print("\n[3. get_llm_response 테스트 (API 호출)]")
    sample_image_path = "sample_image.png" # 테스트용 이미지 경로
    if os.path.exists(sample_image_path):
        test_config = {"temperature": 0.1, "max_output_tokens": 50}
        response = get_llm_response("ID(1,0,1,1): 안녕하세요", sample_image_path, test_config)
        print(f"LLM 응답:\n{response}")
        if response.startswith("LLM_RESPONSE_ERROR:"):
            print("-> 오류 발생 (예상 가능)")
        else:
            print("-> 통과")
    else:
        print(f"경고: 테스트 이미지 '{sample_image_path}'가 없어 get_llm_response 테스트 건너뜀")

    print("\n[4. process_llm_and_integrate 테스트]")
    mock_llm_json = """
    ```json
    [
      [["ID(1,0,1,1)"], "수정된 텍스트"],
      [["ID(1,0,2,1)", "ID(1,0,2,2)"], "병합된 텍스트"]
    ]
    ```
    """
    viz_data, rag_data = process_llm_and_integrate(mock_llm_json, [])
    print("시각화 데이터:", json.dumps(viz_data, indent=2, ensure_ascii=False))
    print("RAG 데이터:", rag_data)
    assert len(rag_data) == 2
    assert rag_data[0] == [[1, 0, 1, 1], ['수정된 텍스트']]
    assert rag_data[1] == [[1, 0, 2, 1], ['병합된 텍스트']]
    print("-> 통과")

    print("\n--- LLM_interaction.py 테스트 종료 ---")
