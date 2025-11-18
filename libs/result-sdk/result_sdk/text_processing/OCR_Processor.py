"""
OCR API 활용 및 중앙 분할 알고리즘 모듈
Google Cloud Vision API를 사용해 이미지에서 텍스트를 추출하고,
텍스트 분포를 기반으로 좌우 블록을 나눈 뒤 ID를 할당함
"""

import json
import requests
import base64
from google.oauth2 import service_account
import google.auth.transport.requests

def ocr_image(file_path: str, service_account_file: str) -> dict:
    """Google Cloud Vision API를 호출하여 이미지에서 텍스트 추출"""
    try:
        credentials = service_account.Credentials.from_service_account_file(
            service_account_file,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        auth_req = google.auth.transport.requests.Request()
        credentials.refresh(auth_req)
        access_token = credentials.token
    except Exception as e:
        raise ValueError(f"서비스 계정 파일 또는 인증 오류: {e}")

    VISION_API_URL = "https://vision.googleapis.com/v1/images:annotate"
    headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}
    
    try:
        with open(file_path, "rb") as image_file:
            content = base64.b64encode(image_file.read()).decode("utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(f"OCR 이미지 파일을 찾을 수 없음: {file_path}")
    except Exception as e:
        raise IOError(f"OCR 이미지 파일 로딩 중 오류: {e}")

    request_payload = {"requests": [{"image": {"content": content}, "features": [{"type": "DOCUMENT_TEXT_DETECTION"}]}]}
    
    try:
        response = requests.post(VISION_API_URL, headers=headers, data=json.dumps(request_payload))
        response.raise_for_status() # HTTP 오류 시 예외 발생
        return response.json()
    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"Vision API 요청 중 네트워크 오류: {e}")
    except Exception as e:
        raise RuntimeError(f"Vision API 응답 처리 중 기타 오류: {e}")

def parse_raw_words(response_json: dict) -> list:
    """OCR API 응답을 파싱하여 단어 목록과 좌표 추출"""
    if not response_json.get('responses') or not response_json['responses'][0]:
        print("Warning: Received empty or invalid response from OCR API")
        return []
    
    texts = response_json['responses'][0].get('textAnnotations')
    if not texts:
        return [] # 텍스트 없는 이미지일 수 있음
        
    raw_words = []
    for word_info in texts[1:]: # texts[0]은 전체 텍스트, [1:]부터 개별 단어
        text = word_info.get('description', '').strip()
        if not text: 
            continue
        
        vertices = word_info.get('boundingPoly', {}).get('vertices', [])
        if not vertices or len(vertices) < 4:
            continue

        x_list = [v.get('x', 0) for v in vertices]
        y_list = [v.get('y', 0) for v in vertices]
        
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
    """텍스트 분포 기반으로 이미지의 수직 분할 지점 탐색"""
    if not words or image_width == 0: 
        return image_width // 2
        
    histogram = [0] * image_width
    for item in words:
        start_x = item.get("x1", 0)
        end_x = item.get("x2", 0)
        
        for x in range(int(start_x), int(end_x)): 
            if 0 <= x < image_width:
                histogram[x] += 1
            
    if not any(histogram): 
        return image_width // 2
        
    center = image_width // 2
    search_range = image_width // 8
    start_search = max(0, center - search_range)
    end_search = min(image_width, center + search_range)
    
    if start_search >= end_search: 
        return center
        
    # 텍스트가 가장 적은(가장 하얀) 부분을 분할점으로 선택
    split_x = min(range(start_search, end_search), key=lambda x: histogram[x])
            
    return split_x

def assign_ids_after_split(raw_words: list, split_x: int, page_num: int) -> list:
    """단어들을 좌우로 나누고 각 청크에 ID 할당"""
    
    def assign_block_ids(group: list, block_id: int, current_page_num: int) -> list:
        if not group: 
            return []
            
        group.sort(key=lambda w: (w.get('y1', 0) // 10, w.get('x1', 0)))
        
        lines = []
        current_line = []
        # 단어 높이의 70%를 같은 줄로 판단하는 임계값으로 사용
        first_word_height = group[0]['y2'] - group[0]['y1'] if group and group[0].get('y2') is not None and group[0].get('y1') is not None else 0
        line_threshold = max(10, first_word_height * 0.7) if first_word_height > 0 else 20
            
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
                word_data_copy.update({
                    'page_num': current_page_num,
                    'block_id': block_id,
                    'y_idx': y_idx,
                    'x_idx': x_idx
                })
                result_with_ids.append(word_data_copy)
        return result_with_ids

    left_col = [w for w in raw_words if w.get("x1") is not None and w.get("x2") is not None and (w['x1'] + w['x2']) / 2 < split_x]
    right_col = [w for w in raw_words if w.get("x1") is not None and w.get("x2") is not None and (w['x1'] + w['x2']) / 2 >= split_x]

    processed_results = assign_block_ids(left_col, 0, page_num)
    processed_results.extend(assign_block_ids(right_col, 1, page_num))
    
    processed_results.sort(key=lambda w: (w.get('page_num', 0), w.get('block_id', 0), w.get('y_idx', 0), w.get('x_idx', 0)))
    return processed_results

def display_ocr_results(ocr_results_list: list):
    """ID 할당된 OCR 결과를 디버깅용으로 출력"""
    for item in ocr_results_list:
        print(f"ID({item.get('page_num')},{item.get('block_id')},{item.get('y_idx')},{item.get('x_idx')}): {item.get('text')}")

if __name__ == '__main__':
    # 모듈 직접 실행 시 테스트용 코드
    # 예:
    # dummy_response = { ... } # Vision API 응답과 유사한 더미 데이터
    # raw_words = parse_raw_words(dummy_response)
    # ... (이하 테스트 로직)
    pass
