# 파일명: visualizer.py

import os
import cv2 # OpenCV를 밑줄 그리기에 사용
from collections import defaultdict
# from PIL import Image, ImageDraw # 현재 OpenCV를 사용하므로 PIL은 직접 필요 X

import re

def draw_underlines_for_incorrect_answers_enhanced(
    incorrect_rag_ids: list,          # 오답 판정된 RAG 앵커 ID 튜플 리스트 [(p,b,y,x), ...]
    all_visualization_data: list,     # LLM이 출력한 시각화용 데이터 [[["ID(..)",..], "문장"], ..]
    all_original_ocr_data: list,      # ID가 할당된 전체 원본 OCR 데이터
    page_image_paths: list,           # 페이지 이미지 경로 리스트
    output_folder: str = "underlined_visualization_output",
    underline_color: tuple = (0, 0, 255), # OpenCV BGR: 빨간색
    underline_thickness: int = 2,
    underline_offset: int = 3
):
    """
    오답 RAG ID를 기반으로, LLM의 시각화용 출력과 원본 OCR 데이터를 참조하여
    문장 전체에 해당하는 원본 청크들에 밑줄을 그립니다.
    """
    print(f"\n--- 문장 단위 밑줄 그리기 작업 시작 (visualizer.py) ---")
    if not incorrect_rag_ids:
        print("  오답으로 처리할 ID가 없습니다.")
        return

    if not all_visualization_data:
        print("  오류: 'all_visualization_data'가 비어있습니다. 밑줄을 그릴 수 없습니다.")
        return
        
    if not all_original_ocr_data:
        print("  오류: 'all_original_ocr_data'가 비어있습니다. 밑줄을 그릴 수 없습니다.")
        return

    # 빠른 조회를 위해 원본 OCR 데이터를 딕셔너리로 변환
    ocr_data_lookup = {
        (item['page_num'], item['block_id'], item['y_idx'], item['x_idx']): item
        for item in all_original_ocr_data
    }

    # 오답 ID를 set으로 변환 (리스트 형태의 ID를 튜플로 변환)
    incorrect_anchor_ids_set = {tuple(item) if isinstance(item, list) else item for item in incorrect_rag_ids}
    
    id_pattern = re.compile(r"ID\((\d+),(\d+),(\d+),(\d+)\)")
    lines_to_draw_on_image_page = defaultdict(list)

    # 시각화 데이터를 순회하며 오답 문장 찾기
    for id_list_str, sentence in all_visualization_data:
        if not id_list_str:
            continue
        
        # 문장의 대표 ID(첫번째 ID) 파싱
        match = id_pattern.match(id_list_str[0])
        if not match:
            continue
        
        representative_id = tuple(map(int, match.groups()))

        # 이 문장이 오답 문장인지 확인
        if representative_id in incorrect_anchor_ids_set:
            chunks_for_this_sentence = []
            page_num_of_sentence = -1

            # 이 문장을 구성하는 모든 ID에 대해 원본 OCR 데이터 조회
            for id_str in id_list_str:
                id_match = id_pattern.match(id_str)
                if id_match:
                    id_tuple = tuple(map(int, id_match.groups()))
                    if page_num_of_sentence == -1:
                        page_num_of_sentence = id_tuple[0]
                    
                    if id_tuple in ocr_data_lookup:
                        chunks_for_this_sentence.append(ocr_data_lookup[id_tuple])
            
            # 조회된 청크들의 좌표를 이용해 밑줄 그리기
            if chunks_for_this_sentence:
                min_x1 = min(c['x1'] for c in chunks_for_this_sentence)
                max_x2 = max(c['x2'] for c in chunks_for_this_sentence)
                # 밑줄의 y 위치는 해당 문장을 구성하는 모든 청크들의 가장 아래 y2를 기준으로 계산
                underline_y = max(c['y2'] for c in chunks_for_this_sentence) + underline_offset
                
                lines_to_draw_on_image_page[page_num_of_sentence].append((min_x1, underline_y, max_x2, underline_y))

    if not lines_to_draw_on_image_page:
        print("  밑줄을 그을 대상 라인을 찾지 못했습니다.")
        return

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"  출력 폴더 '{output_folder}' 생성 완료.")

    processed_images_count = 0
    for page_num_to_draw, line_coords_list_on_page in lines_to_draw_on_image_page.items():
        if not (0 < page_num_to_draw <= len(page_image_paths)):
            print(f"  경고: 페이지 번호 {page_num_to_draw}에 해당하는 이미지를 찾을 수 없습니다. (총 페이지 수: {len(page_image_paths)})")
            continue
        
        current_page_image_path = page_image_paths[page_num_to_draw - 1]
        
        try:
            image = cv2.imread(current_page_image_path)
            if image is None:
                print(f"  오류: 페이지 {page_num_to_draw} 이미지 로드 실패 ({current_page_image_path})")
                continue

            # print(f"\n  페이지 {page_num_to_draw} ({os.path.basename(current_page_image_path)})에 밑줄 그리는 중...")
            unique_lines_on_this_page = sorted(list(set(line_coords_list_on_page)))

            for line_coords in unique_lines_on_this_page:
                cv2.line(image, (line_coords[0], line_coords[1]), (line_coords[2], line_coords[3]), underline_color, underline_thickness)

            base_name_orig, ext_orig = os.path.splitext(os.path.basename(current_page_image_path))
            safe_base_name = "".join(c if c.isalnum() or c in ('_','-') else '_' for c in base_name_orig)
            output_filename = f"page_{page_num_to_draw}_visualized_{safe_base_name}{ext_orig if ext_orig else '.png'}"
            output_image_path = os.path.join(output_folder, output_filename)
            
            cv2.imwrite(output_image_path, image)
            print(f"  페이지 {page_num_to_draw} 밑줄 처리 완료. 저장 경로: {output_image_path}")
            processed_images_count += 1
        except Exception as e:
            print(f"  페이지 {page_num_to_draw} 처리 중 오류 발생: {e}")

    if processed_images_count > 0:
        print(f"\n총 {processed_images_count}개의 이미지에 밑줄 작업 완료. '{output_folder}' 폴더를 확인하세요.")
    else:
        print("\n밑줄이 그려진 이미지가 없습니다 (또는 처리 중 오류 발생).")
    print("--- '다음 ID 이전까지' 밑줄 그리기 작업 종료 ---")

# if __name__ == '__main__': 의 테스트 코드는 새로운 데이터 구조를 반영하여 업데이트 필요
# 예: all_visualization_data와 all_original_ocr_data를 mock 데이터로 제공해야 함
