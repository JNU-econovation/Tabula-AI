# 파일명: visualizer.py

import os
import cv2 # OpenCV를 밑줄 그리기에 사용
from collections import defaultdict
# from PIL import Image, ImageDraw # 현재 OpenCV를 사용하므로 PIL은 직접 필요 X

def draw_underlines_for_incorrect_answers_enhanced(
    incorrect_rag_ids: list,          # 오답 판정된 RAG 앵커 ID 튜플 리스트 [(page,block,y,x), ...]
    all_consolidated_data: list,      # 전체 통합 데이터
    all_rag_ready_data: list,         # RAG 준비 데이터: [[[p,b,y,x], [text]], ...]
    page_image_paths: list,           # 페이지 이미지 경로 리스트
    output_folder: str = "underlined_visualization_output", # 출력 폴더명 기본값 변경
    underline_color: tuple = (0, 0, 255), # OpenCV BGR: 빨간색
    underline_thickness: int = 2,
    underline_offset: int = 3        # 텍스트 하단에서 밑줄까지의 y축 오프셋
):
    """
    RAG 오답 ID를 기반으로, all_rag_ready_data의 순서를 활용하여
    '다음 RAG 항목 이전까지'의 원본 청크들에 연속된 밑줄을 그립니다.
    """
    print(f"\n--- '다음 ID 이전까지' 밑줄 그리기 작업 시작 (visualizer.py) ---")
    if not incorrect_rag_ids:
        print("  오답으로 처리할 ID가 없습니다.")
        return

    if not all_consolidated_data:
        print("  오류: 'all_consolidated_data'가 비어있습니다. 밑줄을 그릴 수 없습니다.")
        return
    # all_rag_ready_data는 비어있을 수도 있지만, incorrect_rag_ids가 있다면 사용됨

    incorrect_anchor_ids_set = set(incorrect_rag_ids)
    
    flat_rag_anchor_ids = []
    if all_rag_ready_data: # all_rag_ready_data가 None이거나 비어있지 않을 때만 처리
        for rag_entry in all_rag_ready_data:
            if isinstance(rag_entry, list) and len(rag_entry) > 0 and \
               isinstance(rag_entry[0], list) and len(rag_entry[0]) == 4:
                flat_rag_anchor_ids.append(tuple(rag_entry[0]))
        flat_rag_anchor_ids.sort()

    lines_to_draw_on_image_page = defaultdict(list)

    for anchor_id_tuple in incorrect_anchor_ids_set:
        page_num, block_id, y_idx_anchor, x_idx_anchor = anchor_id_tuple

        original_chunks_on_anchor_line = sorted(
            [item for item in all_consolidated_data 
             if item.get('page_num') == page_num and \
                item.get('block_id') == block_id and \
                item.get('y_idx') == y_idx_anchor],
            key=lambda x: x.get('x_idx', 0) # x_idx가 없는 경우 대비
        )

        if not original_chunks_on_anchor_line:
            continue

        start_underline_x_idx = x_idx_anchor
        end_underline_x_idx = float('inf') 
        
        if flat_rag_anchor_ids: # RAG 데이터가 있을 때만 다음 ID 검색 로직 실행
            try:
                current_anchor_global_idx = flat_rag_anchor_ids.index(anchor_id_tuple)
                for next_global_idx in range(current_anchor_global_idx + 1, len(flat_rag_anchor_ids)):
                    next_rag_id_candidate = flat_rag_anchor_ids[next_global_idx]
                    if next_rag_id_candidate[0] == page_num and \
                       next_rag_id_candidate[1] == block_id and \
                       next_rag_id_candidate[2] == y_idx_anchor:
                        end_underline_x_idx = next_rag_id_candidate[3]
                        break
                    elif next_rag_id_candidate[0] != page_num or \
                         next_rag_id_candidate[1] != block_id or \
                         next_rag_id_candidate[2] != y_idx_anchor:
                        break
            except ValueError:
                pass 

        chunks_for_this_underline = [
            chunk for chunk in original_chunks_on_anchor_line 
            if start_underline_x_idx <= chunk.get('x_idx', float('-inf')) < end_underline_x_idx
        ]
        
        if chunks_for_this_underline:
            min_x1 = min(c['x1'] for c in chunks_for_this_underline)
            max_x2 = max(c['x2'] for c in chunks_for_this_underline)
            actual_underline_y = max(c['y2'] for c in chunks_for_this_underline) + underline_offset
            lines_to_draw_on_image_page[page_num].append((min_x1, actual_underline_y, max_x2, actual_underline_y))

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

"""
if __name__ == '__main__':
    print("="*50)
    print("visualizer.py 직접 실행 테스트")
    print("="*50)

    # 이 테스트를 실행하려면, 아래 mock 데이터들을 실제 데이터 또는 테스트용 데이터로 채워야 합니다.
    # 또한, 이 파일(visualizer.py)과 같은 위치에 테스트용 이미지가 있어야 합니다.

    # --- 테스트 데이터 준비 ---
    # 1. mock_incorrect_rag_ids: 오답으로 판정된 RAG 앵커 ID 리스트
    #    형태: [(page_num, block_id, y_idx, x_idx), ...]
    #    실제 all_rag_ready_data에 있는 ID여야 합니다.
    mock_incorrect_rag_ids = [
        # 예시: (1, 0, 3, 1), (1, 1, 2, 1) # 실제 데이터로 채워주세요
    ]

    # 2. mock_all_consolidated_data: 전체 통합 데이터
    #    형태: [{'page_num': ..., 'block_id': ..., ...}, ...]
    mock_all_consolidated_data = [
        # 예시 (실제 데이터는 훨씬 많고 복잡합니다)
        {'page_num': 1, 'block_id': 0, 'y_idx': 1, 'x_idx': 1, 'text': '첫단어', 'x1': 10, 'y1': 10, 'x2': 50, 'y2': 30, 'llm_status': 'PROCESSED', 'llm_processed_text': '첫단어 수정됨'},
        {'page_num': 1, 'block_id': 0, 'y_idx': 3, 'x_idx': 1, 'text': '오답', 'x1': 10, 'y1': 50, 'x2': 50, 'y2': 70, 'llm_status': 'PROCESSED', 'llm_processed_text': '이것은 오답입니다'},
        {'page_num': 1, 'block_id': 0, 'y_idx': 3, 'x_idx': 2, 'text': '입니다', 'x1': 55, 'y1': 50, 'x2': 100, 'y2': 70, 'llm_status': 'UNPROCESSED_BY_LLM', 'llm_processed_text': '입니다'}, # 이 ID는 오답 RAG ID가 아님
        {'page_num': 1, 'block_id': 0, 'y_idx': 3, 'x_idx': 3, 'text': '정말로', 'x1': 105, 'y1': 50, 'x2': 150, 'y2': 70, 'llm_status': 'MERGED', 'llm_merged_target_id': (1,0,3,1)},
        {'page_num': 1, 'block_id': 0, 'y_idx': 5, 'x_idx': 1, 'text': '다음앵커', 'x1': 10, 'y1': 90, 'x2': 80, 'y2': 110, 'llm_status': 'PROCESSED', 'llm_processed_text': '다음앵커'},
    ]
    
    # 3. mock_all_rag_ready_data: RAG 준비 데이터
    #    형태: [[[page,block,y,x], [text]], ...]
    mock_all_rag_ready_data = [
        [[1, 0, 1, 1], ['첫단어 수정됨']],
        [[1, 0, 3, 1], ['이것은 오답입니다']], # mock_incorrect_rag_ids에 (1,0,3,1) 추가 시 이 항목 대상
        [[1, 0, 5, 1], ['다음앵커']]
    ]
    # mock_incorrect_rag_ids에 (1,0,3,1)을 추가하여 테스트
    if mock_all_rag_ready_data and len(mock_all_rag_ready_data) > 1:
         mock_incorrect_rag_ids.append(tuple(mock_all_rag_ready_data[1][0]))


    # 4. mock_page_image_paths: 페이지 이미지 경로 리스트
    #    실제 이미지가 있는 경로로 설정해야 합니다. (예: 이전 단계에서 생성된 임시 이미지)
    #    테스트를 위해 간단한 더미 이미지를 생성할 수도 있습니다.
    mock_image_folder = "test_images_for_visualizer"
    os.makedirs(mock_image_folder, exist_ok=True)
    mock_page_image_paths = []
    
    # 더미 이미지 생성 (페이지 1만 예시)
    dummy_image_path_page1 = os.path.join(mock_image_folder, "dummy_page_1.png")
    if not os.path.exists(dummy_image_path_page1):
        dummy_img1 = cv2.UMat(200, 300, cv2.CV_8UC3).get() # 300x200 크기의 검은색 이미지
        cv2.putText(dummy_img1, "Page 1", (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.imwrite(dummy_image_path_page1, dummy_img1)
    mock_page_image_paths.append(dummy_image_path_page1)

    if not mock_incorrect_rag_ids:
        print("경고: 테스트용 mock_incorrect_rag_ids가 비어있습니다. 오답 ID를 추가해주세요.")
    elif not mock_page_image_paths or not all(os.path.exists(p) for p in mock_page_image_paths) :
        print(f"경고: 테스트용 이미지 경로({mock_page_image_paths})가 유효하지 않습니다. 실제 이미지 경로를 설정해주세요.")
    else:
        print(f"테스트용 오답 RAG ID: {mock_incorrect_rag_ids}")
        draw_underlines_for_incorrect_answers_enhanced(
            incorrect_rag_ids=mock_incorrect_rag_ids,
            all_consolidated_data=mock_all_consolidated_data,
            all_rag_ready_data=mock_all_rag_ready_data,
            page_image_paths=mock_page_image_paths,
            output_folder=os.path.join(mock_image_folder, "visualized_output") # 테스트 결과 저장 폴더
        )
    
    # 테스트 후 생성된 더미 이미지 및 폴더 삭제 (선택 사항)
    # if os.path.exists(mock_image_folder):
    #     shutil.rmtree(mock_image_folder)
    #     print(f"\n테스트 이미지 폴더 '{mock_image_folder}' 삭제 완료.")

    print("\n--- visualizer.py 직접 실행 테스트 종료 ---")
"""