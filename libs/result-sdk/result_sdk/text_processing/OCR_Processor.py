#OCR API활용 및 중앙분할 알고리즘

import json
import requests
import base64
from google.oauth2 import service_account
import google.auth.transport.requests
# 참고: 이 모듈 내 함수들이 직접 cv2나 os를 사용하지는 않지만,
#       main 스크립트에서 이 함수들의 인자를 준비할 때 cv2 (이미지 너비) 등이 사용됩니다.

def ocr_image(file_path: str, service_account_file: str) -> dict:
    """Google Cloud Vision API를 호출하여 이미지에서 텍스트를 추출합니다."""
    try:
        credentials = service_account.Credentials.from_service_account_file(
            service_account_file,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        auth_req = google.auth.transport.requests.Request()
        credentials.refresh(auth_req)
        access_token = credentials.token
    except Exception as e:
        # print(f"오류: 서비스 계정 파일 또는 인증 처리 중 문제 발생 - {e}") # 상세 오류는 호출부에서 처리
        raise ValueError(f"서비스 계정 파일 또는 인증 오류: {e}")

    VISION_API_URL = "https://vision.googleapis.com/v1/images:annotate"
    headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}
    
    try:
        with open(file_path, "rb") as image_file:
            content = base64.b64encode(image_file.read()).decode("utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(f"OCR 이미지 파일을 찾을 수 없습니다: {file_path}")
    except Exception as e:
        raise IOError(f"OCR 이미지 파일 로딩 중 오류 발생: {e}")

    request_payload = {"requests": [{"image": {"content": content}, "features": [{"type": "DOCUMENT_TEXT_DETECTION"}]}]}
    
    try:
        response = requests.post(VISION_API_URL, headers=headers, data=json.dumps(request_payload))
        response.raise_for_status() # HTTP 오류 발생 시 예외 발생
        return response.json()
    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"Vision API 요청 중 네트워크 오류 발생: {e}")
    except Exception as e:
        raise RuntimeError(f"Vision API 응답 처리 또는 기타 오류 발생: {e}")

def parse_raw_words(response_json: dict) -> list:
    """OCR API 응답을 파싱하여 단어 목록과 좌표를 추출합니다."""
    if not response_json.get('responses') or not response_json['responses'][0]:
        print("Warning: OCR API로부터 비어있거나 유효하지 않은 응답을 받았습니다 (parse_raw_words).")
        return []
    
    texts = response_json['responses'][0].get('textAnnotations')
    if not texts:
        # print("Warning: No textAnnotations found in OCR response (parse_raw_words).")
        return [] # 텍스트 없는 이미지일 수 있음
        
    raw_words = []
    for word_info in texts[1:]: # texts[0]은 전체 텍스트 블록, [1:]부터 개별 단어
        text = word_info.get('description', '').strip()
        if not text: 
            continue
        
        vertices = word_info.get('boundingPoly', {}).get('vertices', [])
        if not vertices or len(vertices) < 4:
            # print(f"Warning: Invalid boundingPoly for text '{text}' in parse_raw_words. Skipping.")
            continue

        x_list = [v.get('x', 0) for v in vertices] # get으로 안전하게 접근, 없으면 0
        y_list = [v.get('y', 0) for v in vertices] # get으로 안전하게 접근, 없으면 0
        
        raw_words.append({
            "text": text, 
            "x1": min(x_list) if x_list else 0, 
            "y1": min(y_list) if y_list else 0, 
            "x2": max(x_list) if x_list else 0, 
            "y2": max(y_list) if y_list else 0, 
            "bounding_box": vertices
        })
    return raw_words

def find_vertical_split_point(words: list, image_width: int) -> int:
    """텍스트 분포를 기반으로 이미지의 수직 분할 지점을 찾습니다."""
    if not words or image_width == 0: 
        return image_width // 2
        
    histogram = [0] * image_width
    for item in words:
        start_x_val = item.get("x1")
        end_x_val = item.get("x2")

        if start_x_val is None or end_x_val is None:
            continue 

        start_x = max(0, int(start_x_val))
        end_x = min(image_width, int(end_x_val))
        
        for x in range(start_x, end_x): 
            if 0 <= x < image_width:
                histogram[x] += 1
            
    if not any(histogram): 
        return image_width // 2
        
    center = image_width // 2
    search_half_range = image_width // 8 
    
    min_val = float('inf')
    split_x = center
    
    start_search = max(0, center - search_half_range)
    end_search = min(image_width, center + search_half_range)
    
    if start_search >= end_search: 
        return center
        
    for x_candidate in range(start_search, end_search):
        if histogram[x_candidate] < min_val:
            min_val = histogram[x_candidate]
            split_x = x_candidate
            
    return split_x

def assign_ids_after_split(raw_words: list, split_x: int, page_num: int) -> list:
    """단어들을 좌우로 나누고 각 청크에 ID를 할당합니다."""
    
    def assign_block_ids(group: list, block_id: int, current_page_num: int) -> list:
        if not group: 
            return []
            
        group.sort(key=lambda w: (w.get('y1', 0) // 10, w.get('x1', 0)))
        
        lines = []
        current_line = []
        line_threshold = 20 
        if group and group[0].get('y1') is not None and group[0].get('y2') is not None: # 키 존재 확인
            first_word_height = group[0]['y2'] - group[0]['y1']
            if first_word_height > 0 : # 높이가 0보다 클 때만 의미 있음
                line_threshold = max(10, first_word_height * 0.7)
            
        last_word_y_center = -1

        for word in group:
            y1 = word.get('y1')
            y2 = word.get('y2')
            if y1 is None or y2 is None:
                continue

            word_y_center = (y1 + y2) / 2
            
            if not current_line or abs(word_y_center - last_word_y_center) < line_threshold:
                current_line.append(word)
            else:
                if current_line:
                    current_line.sort(key=lambda w: w.get('x1', 0))
                    lines.append(current_line)
                current_line = [word]
            last_word_y_center = word_y_center
            
        if current_line:
            current_line.sort(key=lambda w: w.get('x1', 0))
            lines.append(current_line)
            
        result_with_ids = []
        for y_idx, line_words in enumerate(lines, start=1):
            for x_idx, word_data in enumerate(line_words, start=1):
                word_data_copy = word_data.copy()
                word_data_copy['page_num'] = current_page_num
                word_data_copy['block_id'] = block_id
                word_data_copy['y_idx'] = y_idx
                word_data_copy['x_idx'] = x_idx
                result_with_ids.append(word_data_copy)
        return result_with_ids

    left_col = [w for w in raw_words if w.get("x1") is not None and w.get("x2") is not None and (w['x1'] + w['x2']) / 2 < split_x]
    right_col = [w for w in raw_words if w.get("x1") is not None and w.get("x2") is not None and (w['x1'] + w['x2']) / 2 >= split_x]

    processed_results = assign_block_ids(left_col, 0, page_num)
    processed_results.extend(assign_block_ids(right_col, 1, page_num))
    
    processed_results.sort(key=lambda w: (
        w.get('page_num', 0), 
        w.get('block_id', 0), 
        w.get('y_idx', 0), 
        w.get('x_idx', 0)
    ))
    return processed_results

def display_ocr_results(ocr_results_list: list):
    """ID가 할당된 OCR 결과를 출력합니다 (디버깅용)."""
    for item in ocr_results_list:
        print(f"ID({item.get('page_num')},{item.get('block_id')},{item.get('y_idx')},{item.get('x_idx')}): {item.get('text')}")

if __name__ == '__main__':
    # 이 모듈을 직접 실행했을 때 테스트할 수 있는 코드 (선택 사항)
    # 예시:
    # dummy_ocr_response = { ... } # 실제 Vision API 응답과 유사한 더미 데이터
    # raw_words = parse_raw_words(dummy_ocr_response)
    # if raw_words:
    #     print("Parsed raw words sample:", raw_words[:2])
    #     image_width_sample = 1000 # 예시 이미지 너비
    #     split_x_sample = find_vertical_split_point(raw_words, image_width_sample)
    #     print(f"Sample split_x: {split_x_sample}")
    #     ids_assigned = assign_ids_after_split(raw_words, split_x_sample, 1)
    #     if ids_assigned:
    #         print("Assigned IDs sample:")
    #         display_ocr_results(ids_assigned[:2])
    pass
