"""
Google Document AI를 활용한 OCR 처리 모듈
"""

import os
from google.cloud import documentai
from google.api_core.client_options import ClientOptions

def process_document_ocr(
    file_path: str,
    project_id: str,
    location: str,
    processor_id: str,
    service_account_file: str,
) -> dict:
    """Google Document AI를 호출하여 이미지에서 텍스트를 추출합니다."""
    
    # Document AI 클라이언트 옵션 설정 (서비스 계정 파일 사용)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_file
    opts = ClientOptions(api_endpoint=f"{location}-documentai.googleapis.com")
    client = documentai.DocumentProcessorServiceClient(client_options=opts)

    name = client.processor_path(project_id, location, processor_id)

    try:
        with open(file_path, "rb") as image:
            image_content = image.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"OCR 이미지 파일을 찾을 수 없음: {file_path}")
    except Exception as e:
        raise IOError(f"OCR 이미지 파일 로딩 중 오류: {e}")

    # Document AI에 처리할 문서 요청
    raw_document = documentai.RawDocument(
        content=image_content, mime_type="application/pdf" if file_path.lower().endswith(".pdf") else "image/jpeg"
    )

    request = documentai.ProcessRequest(name=name, raw_document=raw_document)

    try:
        result = client.process_document(request=request)
        document = result.document
        # OCR 결과를 JSON 직렬화 가능한 형태로 변환하여 반환 (필요시 구조 수정)
        return documentai.Document.to_dict(document)
    except Exception as e:
        raise ConnectionError(f"Document AI API 요청 중 오류 발생: {e}")

def parse_docai_raw_words(docai_response: dict) -> list:
    """Document AI 응답을 파싱하여 단어 목록과 좌표를 추출합니다."""
    raw_words = []
    
    # Document AI 응답 구조에 따라 텍스트와 바운딩 박스 정보 추출
    # 페이지, 블록, 문단, 줄, 토큰(단어) 계층 구조를 가짐
    for page in docai_response.get("pages", []):
        # 페이지 크기 정보
        page_width = page.get("dimension", {}).get("width", 1)
        page_height = page.get("dimension", {}).get("height", 1)

        for token in page.get("tokens", []):
            # Document AI 응답에서 텍스트 추출
            text_segments = token.get("layout", {}).get("text_anchor", {}).get("text_segments", [])
            if not text_segments:
                continue
            
            # text_segments를 사용하여 전체 텍스트에서 해당 토큰의 텍스트를 가져옴
            start_index = int(text_segments[0].get("start_index", 0))
            end_index = int(text_segments[0].get("end_index", 0))
            text = docai_response.get("text", "")[start_index:end_index].strip()

            if not text:
                continue

            vertices = token.get("layout", {}).get("bounding_poly", {}).get("vertices", [])
            if not vertices or len(vertices) < 4:
                continue
            
            # Document AI는 정규화된 좌표(0~1)를 반환하지 않고 바로 픽셀 좌표를 제공합니다.
            # 따라서 별도의 변환이 필요 없습니다.
            x_list = [v.get("x", 0) for v in vertices]
            y_list = [v.get("y", 0) for v in vertices]

            raw_words.append({
                "text": text,
                "x1": min(x_list),
                "y1": min(y_list),
                "x2": max(x_list),
                "y2": max(y_list),
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

if __name__ == '__main__':
    # 모듈 직접 실행 시 테스트용 코드
    # 예:
    # project_id = "your-gcp-project-id"
    # location = "us"  # e.g., "us" or "eu"
    # processor_id = "your-processor-id"
    # file_path = "path/to/your/image.jpg"
    # service_account_file = "path/to/your/service-account.json"
    #
    # docai_result = process_document_ocr(file_path, project_id, location, processor_id, service_account_file)
    # words = parse_docai_raw_words(docai_result)
    # print(words)
    pass
