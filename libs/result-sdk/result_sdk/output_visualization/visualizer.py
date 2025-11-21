"""
결과 시각화 모듈
오답으로 판정된 부분에 밑줄을 그어 이미지로 저장하는 기능을 담당
"""

import os
import cv2
import re
from collections import defaultdict

def draw_underlines_for_incorrect_answers_enhanced(
    incorrect_rag_ids: list,
    all_visualization_data: list,
    all_original_ocr_data: list,
    page_image_paths: list,
    output_folder: str = "underlined_visualization_output",
    underline_color: tuple = (0, 0, 255), # BGR: 빨간색
    underline_thickness: int = 2,
    underline_offset: int = 3
):
    """
    오답 RAG ID를 기반으로, 문장 전체에 해당하는 원본 OCR 청크들에 밑줄을 그림

    Args:
        incorrect_rag_ids (list): 오답 판정된 RAG 앵커 ID 튜플 리스트. 예: [(p,b,y,x), ...]
        all_visualization_data (list): LLM이 출력한 시각화용 데이터. 예: [[["ID(..)",..], "문장"], ..]
        all_original_ocr_data (list): ID가 할당된 전체 원본 OCR 데이터
        page_image_paths (list): 페이지 이미지 경로 리스트
        output_folder (str): 결과물 저장 폴더
        underline_color (tuple): 밑줄 색상 (BGR)
        underline_thickness (int): 밑줄 두께
        underline_offset (int): 텍스트와 밑줄 사이 간격
    """
    print(f"\n--- Starting sentence-level underlining task ---")
    if not incorrect_rag_ids:
        print("  No incorrect IDs to process.")
        return

    if not all_visualization_data or not all_original_ocr_data:
        print("  Error: Cannot draw underlines because 'all_visualization_data' or 'all_original_ocr_data' is empty.")
        return

    # 빠른 조회를 위해 원본 OCR 데이터를 딕셔너리로 변환
    ocr_data_lookup = {
        (item['page_num'], item['block_id'], item['y_idx'], item['x_idx']): item
        for item in all_original_ocr_data
    }

    # 오답 ID를 set으로 변환해 조회 성능 향상
    incorrect_anchor_ids_set = {tuple(item) for item in incorrect_rag_ids}
    
    id_pattern = re.compile(r"ID\((\d+),(\d+),(\d+),(\d+)\)")
    lines_to_draw_on_page = defaultdict(list)

    # 시각화 데이터를 순회하며 오답 문장 찾기
    for id_list_str, sentence in all_visualization_data:
        if not id_list_str:
            continue
        
        match = id_pattern.match(id_list_str[0])
        if not match:
            continue
        
        representative_id = tuple(map(int, match.groups()))

        # 이 문장이 오답 문장인지 확인
        if representative_id in incorrect_anchor_ids_set:
            chunks_for_this_sentence = []
            page_num_of_sentence = representative_id[0]

            # 문장을 구성하는 모든 ID에 대해 원본 OCR 데이터 조회
            for id_str in id_list_str:
                id_match = id_pattern.match(id_str)
                if id_match:
                    id_tuple = tuple(map(int, id_match.groups()))
                    if id_tuple in ocr_data_lookup:
                        chunks_for_this_sentence.append(ocr_data_lookup[id_tuple])
            
            if chunks_for_this_sentence:
                # 여러 줄에 걸친 문장 및 다단(Multi-column) 처리를 위해 (block_id, y_idx) 기준으로 그룹화
                # block_id가 다르면 같은 y_idx라도 분리된 밑줄을 그어야 함
                chunks_by_line_block = defaultdict(list)
                for chunk in chunks_for_this_sentence:
                    # 키: (블록ID, 줄번호)
                    key = (chunk['block_id'], chunk['y_idx'])
                    chunks_by_line_block[key].append(chunk)
                
                # 각 그룹(블록+줄)별로 밑줄 계산
                # OCR이 좌우 단을 같은 블록으로 인식했거나(잘못된 병합), 
                # 좌측 단어가 우측 블록에 포함된 경우를 대비해 물리적 간격(Gap)을 체크하여 분리
                for _, line_chunks in chunks_by_line_block.items():
                    if not line_chunks:
                        continue
                    
                    # x1 좌표 기준으로 정렬
                    sorted_chunks = sorted(line_chunks, key=lambda c: c['x1'])
                    
                    segments = []
                    if sorted_chunks:
                        current_segment = [sorted_chunks[0]]
                        
                        for i in range(1, len(sorted_chunks)):
                            prev = sorted_chunks[i-1]
                            curr = sorted_chunks[i]
                            
                            gap = curr['x1'] - prev['x2']
                            # 임계값: 글자 높이의 3배 이상 떨어져 있으면 단 분리 등으로 간주하여 선을 끊음
                            avg_height = ((prev['y2'] - prev['y1']) + (curr['y2'] - curr['y1'])) / 2
                            threshold = avg_height * 5.0
                            
                            if gap > threshold:
                                segments.append(current_segment)
                                current_segment = [curr]
                            else:
                                current_segment.append(curr)
                        segments.append(current_segment)
                    
                    # 분리된 각 세그먼트별로 밑줄 추가
                    for segment in segments:
                        min_x1 = min(c['x1'] for c in segment)
                        max_x2 = max(c['x2'] for c in segment)
                        # 높이는 해당 세그먼트 내 최대 y2 기준
                        underline_y = max(c['y2'] for c in segment) + underline_offset
                        lines_to_draw_on_page[page_num_of_sentence].append((min_x1, underline_y, max_x2, underline_y))

    if not lines_to_draw_on_page:
        print("  No lines found to underline.")
        return

    os.makedirs(output_folder, exist_ok=True)

    processed_count = 0
    for page_num, line_coords_list in lines_to_draw_on_page.items():
        if not (0 < page_num <= len(page_image_paths)):
            print(f"  Warning: Cannot find image for page number {page_num} (Total pages: {len(page_image_paths)})")
            continue
        
        image_path = page_image_paths[page_num - 1]
        
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"  Error: Failed to load image for page {page_num} ({image_path})")
                continue

            # 중복 라인 제거 후 그리기
            for coords in sorted(list(set(line_coords_list))):
                cv2.line(image, (coords[0], coords[1]), (coords[2], coords[3]), underline_color, underline_thickness)

            base_name, ext = os.path.splitext(os.path.basename(image_path))
            safe_base_name = "".join(c if c.isalnum() or c in ('_','-') else '_' for c in base_name)
            output_filename = f"page_{page_num}_visualized_{safe_base_name}{ext or '.png'}"
            output_path = os.path.join(output_folder, output_filename)
            
            cv2.imwrite(output_path, image)
            print(f"  Page {page_num} underlining complete. Saved to: {output_path}")
            processed_count += 1
        except Exception as e:
            print(f"  Error processing page {page_num}: {e}")

    if processed_count > 0:
        print(f"\nUnderlining task complete for {processed_count} images. Check the '{output_folder}' folder.")
    else:
        print("\nNo images were underlined (or an error occurred during processing).")
    print("--- Underlining task finished ---")

# if __name__ == '__main__':
#   테스트 코드는 새로운 데이터 구조를 반영하여 업데이트 필요
#   예: all_visualization_data와 all_original_ocr_data를 mock 데이터로 제공해야 함
