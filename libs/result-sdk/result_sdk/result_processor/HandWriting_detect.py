import os
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image
import json
import requests
import base64
import cv2
import re
from google.oauth2 import service_account
import google.auth.transport.requests
from pdf2image import convert_from_path # PDF 처리를 위해 추가
import shutil # 임시 폴더 삭제를 위해 추가 (선택 사항)
from Prompt import gemini_prompt

# .env파일 로드 (API관련 키 저장)
load_dotenv()

# API키 가져오기
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY 환경 변수를 설정해주세요.")
genai.configure(api_key=GOOGLE_API_KEY)

# --- PDF 및 이미지 입력 처리 함수 ---
def get_image_paths_from_input(input_file_path, temp_output_folder="pdf_converted_images_temp"):
    """
    입력 파일 경로를 받아 PDF인 경우 이미지로 변환하고, 이미지 파일 경로 리스트를 반환
    PDF가 아닌 이미지 파일이면 해당 파일 경로를 리스트에 담아 반환

    Args:
        input_file_path (str): 처리할 파일의 경로 (PDF 또는 이미지)
        temp_output_folder (str): PDF에서 변환된 이미지를 저장할 임시 폴더 이름

    Returns:
        list: 처리할 이미지 파일 경로들의 리스트. 오류 발생 시 빈 리스트 반환
    """
    file_name, file_ext = os.path.splitext(input_file_path)
    file_ext = file_ext.lower()

    image_paths = []

    if file_ext == '.pdf':
        if os.path.exists(temp_output_folder): # 기존 임시 폴더가 있다면 삭제 후 생성
            try:
                shutil.rmtree(temp_output_folder)
                print(f"기존 임시 폴더 '{temp_output_folder}' 삭제 완료.")
            except Exception as e:
                print(f"기존 임시 폴더 '{temp_output_folder}' 삭제 중 오류: {e}")

        try:
            os.makedirs(temp_output_folder)
            print(f"임시 폴더 '{temp_output_folder}' 생성 완료.")
        except OSError as e:
            print(f"임시 폴더 '{temp_output_folder}' 생성 실패: {e}. 이미 폴더가 존재할 수 있습니다.")
            if not os.path.isdir(temp_output_folder):
                 return []


        try:
            print(f"PDF 파일 변환 중: '{input_file_path}' (시간이 다소 소요될 수 있습니다)...")
            # dpi는 이미지 품질에 영향 (200-300 정도가 일반적)
            pil_images = convert_from_path(input_file_path, dpi=200, poppler_path=None) # poppler_path=None이면 시스템 PATH에서 찾음

            base_pdf_name = os.path.basename(file_name)
            for i, image in enumerate(pil_images):
                image_filename = os.path.join(temp_output_folder, f"{base_pdf_name}_page_{i + 1}.png")
                image.save(image_filename, "PNG")
                image_paths.append(image_filename)
            print(f"PDF 파일이 성공적으로 {len(image_paths)}개의 이미지로 변환되어 '{temp_output_folder}'에 저장되었습니다.")
        except Exception as e: # pdf2image.exceptions.PDFInfoNotInstalledError 등 포함
            print(f"PDF 파일 변환 중 오류 발생 ('{input_file_path}'): {e}")
            print("Poppler가 시스템에 올바르게 설치되어 있고 PATH에 등록되어 있는지 확인해주세요.")
            print("- Windows: Poppler 바이너리 다운로드 후 bin 폴더를 PATH에 추가")
            print("- macOS: 'brew install poppler'")
            print("- Linux: 'sudo apt-get install poppler-utils'")
            print("- Conda: 'conda install -c conda-forge poppler'")
            return [] # 오류 시 빈 리스트 반환

    elif file_ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp']:
        if not os.path.exists(input_file_path):
            print(f"이미지 파일 경로 오류: '{input_file_path}'를 찾을 수 없습니다.")
            return []
        image_paths.append(input_file_path)
        print(f"입력 파일은 이미지입니다: '{input_file_path}'")
    else:
        print(f"지원하지 않는 파일 형식입니다: '{file_ext}'. PDF 또는 이미지 파일을 제공해주세요.")
        return []

    return image_paths

# --- OCR 관련 함수들 ---
def ocr_image(file_path, service_account_file):
    credentials = service_account.Credentials.from_service_account_file(
        service_account_file,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    auth_req = google.auth.transport.requests.Request()
    credentials.refresh(auth_req)
    access_token = credentials.token
    VISION_API_URL = "https://vision.googleapis.com/v1/images:annotate"
    headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}
    with open(file_path, "rb") as image_file:
        content = base64.b64encode(image_file.read()).decode("utf-8")
    request_payload = {"requests": [{"image": {"content": content}, "features": [{"type": "TEXT_DETECTION"}]}]}
    response = requests.post(VISION_API_URL, headers=headers, data=json.dumps(request_payload))
    response.raise_for_status()
    return response.json()

def parse_raw_words(response_json):
    if not response_json['responses'] or not response_json['responses'][0]:
        print("Warning: Empty or invalid response from OCR API.")
        return []
    texts = response_json['responses'][0].get('textAnnotations')
    if not texts: return []
    raw_words = []
    for word_info in texts[1:]:
        text = word_info['description'].strip()
        if not text: continue
        vertices = word_info['boundingPoly']['vertices']
        x_list = [v.get('x', 0) for v in vertices]
        y_list = [v.get('y', 0) for v in vertices]
        raw_words.append({"text": text, "x1": min(x_list), "y1": min(y_list), "x2": max(x_list), "y2": max(y_list), "bounding_box": vertices})
    return raw_words

def find_vertical_split_point(words, image_width):
    if not words: return image_width // 2
    histogram = [0] * image_width
    for item in words:
        start_x, end_x = max(0, item["x1"]), min(image_width, item["x2"])
        for x in range(start_x, end_x): histogram[x] += 1
    if not any(histogram): return image_width // 2
    center, search_half_range = image_width // 2, image_width // 8
    min_val, split_x = float('inf'), center
    start_search, end_search = max(0, center - search_half_range), min(image_width, center + search_half_range)
    if start_search >= end_search: return center
    for x_candidate in range(start_search, end_search):
        if histogram[x_candidate] < min_val:
            min_val, split_x = histogram[x_candidate], x_candidate
    return split_x

# page_num 인자 추가
def assign_ids_after_split(raw_words, split_x, page_num):
    def assign_block_ids(group, block_id, current_page_num): # page_num 전달
        if not group: return []
        group.sort(key=lambda w: (w['y1'] // 10, w['x1']))
        lines, current_line, line_threshold = [], [], 20
        if group:
            first_word_height = group[0]['y2'] - group[0]['y1']
            line_threshold = max(10, first_word_height * 0.7)
        last_word_y_center = -1
        for word in group:
            word_y_center = (word['y1'] + word['y2']) / 2
            if not current_line or abs(word_y_center - last_word_y_center) < line_threshold:
                current_line.append(word)
            else:
                if current_line:
                    current_line.sort(key=lambda w: w['x1'])
                    lines.append(current_line)
                current_line = [word]
            last_word_y_center = word_y_center
        if current_line:
            current_line.sort(key=lambda w: w['x1'])
            lines.append(current_line)
        result_with_ids = []
        for y_idx, line_words in enumerate(lines, start=1):
            for x_idx, word_data in enumerate(line_words, start=1):
                word_data_copy = word_data.copy()
                word_data_copy['page_num'] = current_page_num # 페이지 번호 추가
                word_data_copy['block_id'] = block_id
                word_data_copy['y_idx'] = y_idx
                word_data_copy['x_idx'] = x_idx
                result_with_ids.append(word_data_copy)
        return result_with_ids

    left_col = [w for w in raw_words if (w['x1'] + w['x2']) / 2 < split_x]
    right_col = [w for w in raw_words if (w['x1'] + w['x2']) / 2 >= split_x]

    # page_num을 assign_block_ids로 전달
    processed_results = assign_block_ids(left_col, 0, page_num)
    processed_results.extend(assign_block_ids(right_col, 1, page_num))
    processed_results.sort(key=lambda w: (w['page_num'], w['block_id'], w['y_idx'], w['x_idx']))
    return processed_results

# ID 형식 변경
def display_ocr_results(ocr_results_list):
    for item in ocr_results_list:
        print(f"ID({item['page_num']},{item['block_id']},{item['y_idx']},{item['x_idx']}): {item['text']}")

# --- LLM 호출 함수 ---
def get_llm_response(full_prompt: str, image_path: str,
                     generation_config_dict: dict, safety_settings_list: list = None) -> str:
    try: img = Image.open(image_path)
    except FileNotFoundError: return "Error: Image file not found."
    except Exception as e: return f"Error opening image: {e}"
    model = genai.GenerativeModel('gemini-1.5-flash-latest') #gemini-2.5-flash-preview-04-17을 사용하려고 했으나, 간헐적으로 오류가 발생하여 일단 1.5 flash사용.
    contents = [full_prompt, img]
    try:
        response = model.generate_content(contents,
            generation_config=genai.types.GenerationConfig(**generation_config_dict),
            safety_settings=safety_settings_list)
        if not response.candidates:
            return f"Error: Request was blocked. Feedback: {response.prompt_feedback if response.prompt_feedback else 'Unknown'}"
        return response.text
    except Exception as e: return f"Error during LLM API call: {str(e)}"

# --- LLM 결과 처리 함수  ---
def process_llm_and_integrate(llm_output_string, original_ocr_results_for_page):
    llm_processed_info = {}
    # ID 파싱 정규식
    line_pattern = re.compile(r"ID\((\d+),(\d+),(\d+),(\d+)\):\s*(.*)")
    merged_pattern = re.compile(r"__MERGED_TO_ID\((\d+),(\d+),(\d+),(\d+)\)__")

    if llm_output_string.startswith("Error:"):
        print(f"LLM Error detected: {llm_output_string}")
        consolidated_data = []
        for ocr_item in original_ocr_results_for_page:
            item_copy = ocr_item.copy()
            item_copy['llm_processed_text'] = ocr_item['text']
            item_copy['llm_status'] = 'LLM_ERROR_FALLBACK'
            consolidated_data.append(item_copy)
        return consolidated_data, []

    for line in llm_output_string.strip().split('\n'):
        line_match = line_pattern.match(line)
        if not line_match:
            print(f"Warning: Could not parse LLM output line: {line}")
            continue
        # ID 파싱
        page_num, block_id, y_idx, x_idx = map(int, line_match.group(1, 2, 3, 4))
        content = line_match.group(5).strip()
        current_id_tuple = (page_num, block_id, y_idx, x_idx) # 4-element tuple
        merged_match = merged_pattern.match(content)
        if merged_match:
            # target ID 파싱
            target_p, target_b, target_y, target_x = map(int, merged_match.group(1, 2, 3, 4))
            llm_processed_info[current_id_tuple] = {
                "status": "MERGED", "text": content,
                "target_id": (target_p, target_b, target_y, target_x) # 4-element tuple
            }
        elif content == "__REMOVED__":
            llm_processed_info[current_id_tuple] = {"status": "REMOVED", "text": ""}
        else:
            llm_processed_info[current_id_tuple] = {"status": "PROCESSED", "text": content}

    consolidated_data = []
    for ocr_item in original_ocr_results_for_page:
        # item_id_tuple 생성 시 page_num 포함
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
        consolidated_data.append(integrated_item)

    rag_ready_data = []
    for item in consolidated_data:
        if item.get('llm_status') == 'PROCESSED' and item.get('llm_processed_text','').strip():
            # RAG 데이터 형식 변경: [[id_list], [text_string_in_list]]
            id_list_for_rag = [item['page_num'], item['block_id'], item['y_idx'], item['x_idx']]
            text_for_rag = item['llm_processed_text']
            rag_ready_data.append([id_list_for_rag, [text_for_rag]]) # 텍스트를 리스트로 감쌈
            # RAG에 바운딩 박스 정보도 필요하다면 아래와 같이 추가 가능
            # rag_ready_data.append([id_list_for_rag, [text_for_rag], item['bounding_box']])
    return consolidated_data, rag_ready_data

#ID형식 지정
def format_ocr_results_for_prompt(ocr_page_results):
    return "\n".join([f"ID({item['page_num']},{item['block_id']},{item['y_idx']},{item['x_idx']}): {item['text']}" for item in ocr_page_results])

# --- 프롬프트 (ID 설명 부분 수정) ---
PROMPT_TEMPLATE = gemini_prompt

def assign_ids_after_split(raw_words, split_x, page_num):
    def assign_block_ids(group, block_id, current_page_num): # page_num 전달
        if not group: return []
        group.sort(key=lambda w: (w['y1'] // 10, w['x1']))
        lines, current_line, line_threshold = [], [], 20
        if group:
            first_word_height = group[0]['y2'] - group[0]['y1']
            line_threshold = max(10, first_word_height * 0.7)
        last_word_y_center = -1
        for word in group:
            word_y_center = (word['y1'] + word['y2']) / 2
            if not current_line or abs(word_y_center - last_word_y_center) < line_threshold:
                current_line.append(word)
            else:
                if current_line:
                    current_line.sort(key=lambda w: w['x1'])
                    lines.append(current_line)
                current_line = [word]
            last_word_y_center = word_y_center
        if current_line:
            current_line.sort(key=lambda w: w['x1'])
            lines.append(current_line)
        result_with_ids = []
        for y_idx, line_words in enumerate(lines, start=1):
            for x_idx, word_data in enumerate(line_words, start=1):
                word_data_copy = word_data.copy()
                word_data_copy['page_num'] = current_page_num # 페이지 번호 추가
                word_data_copy['block_id'] = block_id
                word_data_copy['y_idx'] = y_idx
                word_data_copy['x_idx'] = x_idx
                result_with_ids.append(word_data_copy)
        return result_with_ids

    left_col = [w for w in raw_words if (w['x1'] + w['x2']) / 2 < split_x]
    right_col = [w for w in raw_words if (w['x1'] + w['x2']) / 2 >= split_x]

    processed_results = assign_block_ids(left_col, 0, page_num)
    processed_results.extend(assign_block_ids(right_col, 1, page_num))
    processed_results.sort(key=lambda w: (w['page_num'], w['block_id'], w['y_idx'], w['x_idx']))
    return processed_results


if __name__ == "__main__":
    SERVICE_ACCOUNT_FILE = '/Users/ki/Desktop/Google Drive/Dev/Ecode/ecode-458109-73d063ae5f2a.json'

    # 처리할 단일 입력 파일 (PDF 또는 이미지)
    # INPUT_FILE = "/path/to/your/document.pdf"
    # INPUT_FILE = "/path/to/your/single_image.png"
    INPUT_FILE = '/Users/ki/Desktop/Google Drive/Dev/Ecode/OCR_Test/예시_한국사.pdf'# << 실제 파일 경로로 입력하기!

    TEMP_PDF_IMAGE_FOLDER = "temp_pdf_converted_images" # PDF 변환 시 이미지가 저장될 임시 폴더

    # 입력 파일 경로 유효성 검사 (하드코딩된 플레이스홀더 방지)
    if "YOUR_INPUT_FILE_PATH" in INPUT_FILE or not INPUT_FILE:
        print("오류: INPUT_FILE 변수에 실제 처리할 파일의 전체 경로를 입력해주세요.")
        print("예: INPUT_FILE = '/Users/username/Documents/my_document.pdf'")
        exit()
    if not os.path.exists(INPUT_FILE):
        print(f"오류: 입력 파일 '{INPUT_FILE}'을 찾을 수 없습니다. 경로를 확인해주세요.")
        exit()
    if "YOUR_SERVICE_ACCOUNT_FILE_PATH" in SERVICE_ACCOUNT_FILE or not SERVICE_ACCOUNT_FILE:
        print("오류: SERVICE_ACCOUNT_FILE 변수에 실제 서비스 계정 파일의 전체 경로를 입력해주세요.")
        exit()


    # 입력 파일로부터 처리할 이미지 경로 리스트 가져오기
    image_files_to_process = get_image_paths_from_input(INPUT_FILE, TEMP_PDF_IMAGE_FOLDER)

    if not image_files_to_process:
        print("처리할 이미지를 준비하지 못했습니다. 프로그램을 종료합니다.")
        exit()

    all_consolidated_data = []
    all_rag_ready_data = []

    generation_config = {"temperature": 0.2, "max_output_tokens": 8000} # Flash 모델 최대 8192, 여유있게 설정
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    safety_settings = [
        {"category": HarmCategory.HARM_CATEGORY_HARASSMENT, "threshold": HarmBlockThreshold.BLOCK_ONLY_HIGH},
        {"category": HarmCategory.HARM_CATEGORY_HATE_SPEECH, "threshold": HarmBlockThreshold.BLOCK_ONLY_HIGH},
        {"category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, "threshold": HarmBlockThreshold.BLOCK_ONLY_HIGH},
        {"category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, "threshold": HarmBlockThreshold.BLOCK_ONLY_HIGH},
    ]


    for page_idx, image_file_path in enumerate(image_files_to_process):
        page_num = page_idx + 1
        print(f"\n\n[페이지 {page_num}/{len(image_files_to_process)} 처리 시작] ({os.path.basename(image_file_path)})")

        print(f"[페이지 {page_num} - 단계 1] OCR 실행 중...")
        try:
            response_json = ocr_image(image_file_path, SERVICE_ACCOUNT_FILE)
            raw_words = parse_raw_words(response_json)
            if not raw_words:
                print(f"페이지 {page_num}: OCR 결과에서 단어를 찾을 수 없습니다. 다음 페이지로 넘어갑니다.")
                continue
            image_for_size_check = cv2.imread(image_file_path)
            if image_for_size_check is None:
                print(f"오류: 페이지 {page_num} 이미지를 로드할 수 없습니다 - {image_file_path}")
                continue
            image_height, image_width, _ = image_for_size_check.shape
            split_x = find_vertical_split_point(raw_words, image_width)
            original_ocr_results_page = assign_ids_after_split(raw_words, split_x, page_num)
            print(f"페이지 {page_num}: OCR 및 ID 할당 완료 (총 {len(original_ocr_results_page)} 청크).")
        except FileNotFoundError as e:
            print(f"오류: 파일 경로를 확인해주세요 ({image_file_path}) - {e}")
            continue
        except requests.exceptions.RequestException as e:
            print(f"오류: 페이지 {page_num} Vision API 요청 중 문제가 발생했습니다 - {e}")
            continue
        except Exception as e:
            print(f"페이지 {page_num} OCR 처리 중 예기치 않은 오류 발생: {e}")
            continue

        if not original_ocr_results_page:
            print(f"페이지 {page_num}: ID가 할당된 OCR 결과가 없습니다. 다음 페이지로 넘어갑니다.")
            continue

        print(f"\n[페이지 {page_num} - 단계 2] LLM 호출 준비 중...")
        ocr_chunk_list_str = format_ocr_results_for_prompt(original_ocr_results_page)
        full_prompt_for_llm = PROMPT_TEMPLATE.replace("{ocr_chunk_list_placeholder}", ocr_chunk_list_str)

        print(f"페이지 {page_num}: LLM 호출 중...")
        llm_response_text = get_llm_response(full_prompt_for_llm, image_file_path, generation_config, safety_settings)

        if llm_response_text.startswith("Error:"):
            print(f"페이지 {page_num} LLM 호출 실패: {llm_response_text}")
        else:
            print(f"페이지 {page_num}: LLM 응답 수신 완료.")

        print(f"\n[페이지 {page_num} - 단계 3] LLM 결과 처리 및 통합 중...")
        consolidated_data_page, rag_ready_data_page = process_llm_and_integrate(llm_response_text, original_ocr_results_page)

        all_consolidated_data.extend(consolidated_data_page)
        all_rag_ready_data.extend(rag_ready_data_page)
        print(f"페이지 {page_num} 처리 완료. RAG 데이터 {len(rag_ready_data_page)}개 추가됨.")

    print("\n\n\n--- 모든 페이지 처리 완료 ---")
    # ... (이후 출력 로직은 이전 답변과 동일) ...
    print(f"총 통합된 데이터 항목 수: {len(all_consolidated_data)}")
    print(f"총 RAG 준비 데이터 항목 수: {len(all_rag_ready_data)}")

    print("\n--- 전체 RAG 준비 데이터 (일부 샘플) ---")
    sample_count = 0
    for i, item_list in enumerate(all_rag_ready_data):
        if i < 5 or (len(all_rag_ready_data) - i) <= 2 :
            id_list_str = f"ID({','.join(map(str, item_list[0]))})"
            text_str = item_list[1][0][:100]
            print(f"{id_list_str}, Text: '{text_str}...'")
            sample_count +=1
        elif i == 5 and len(all_rag_ready_data) > 7 and sample_count == 5 :
            print("...")
            sample_count +=1

    # 임시 폴더 자동 삭제 (선택 사항)
    # if os.path.exists(TEMP_PDF_IMAGE_FOLDER) and any(fname.lower().endswith('.pdf') for fname in [INPUT_FILE]): # PDF 처리 시에만
    #     try:
    #         shutil.rmtree(TEMP_PDF_IMAGE_FOLDER)
    #         print(f"\n임시 폴더 '{TEMP_PDF_IMAGE_FOLDER}' 삭제 완료.")
    #     except Exception as e:
    #         print(f"\n임시 폴더 '{TEMP_PDF_IMAGE_FOLDER}' 삭제 중 오류: {e}")

    print("\n[전체 작업 완료]")
    print(all_rag_ready_data)

#all_rag_ready_data에서 RAG(다음과정)으로 넘길 리스트파일 확인가능