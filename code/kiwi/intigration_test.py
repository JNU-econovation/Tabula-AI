# intigration_test.py

import os
import shutil # shutil 임포트 확인
import argparse
import config 
from Prompt import gemini_prompt as PROMPT_TEMPLATE
from core import process_document
from visualizer import draw_underlines_for_incorrect_answers_enhanced
import google.generativeai as genai

if __name__ == '__main__':
    if not config.GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY가 config.py를 통해 설정되지 않았습니다.")
    genai.configure(api_key=config.GOOGLE_API_KEY)

    #default_input_file = os.path.join(config.BASE_INPUT_DIR, "예시_한국사.pdf")
    default_input_file = '/Users/ki/Desktop/Google Drive/Dev/Ecode/OCR_Test/예시_한국사.pdf'
    parser = argparse.ArgumentParser(description="코어 처리 및 시각화 통합 테스트 스크립트")
    parser.add_argument(
        "input_file", 
        nargs='?',
        default=default_input_file, 
        help=f"처리할 PDF 또는 이미지 파일의 전체 경로 (기본값: {default_input_file})"
    )
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"오류: 입력 파일 '{args.input_file}'을 찾을 수 없습니다.")
        exit()
    # ... (서비스 계정 파일 확인 등) ...

    print(f"통합 테스트 시작: '{args.input_file}'")
    print("="*70)
    print(f"\n[단계 1] '{args.input_file}' 문서 처리 중...")
    
    # process_document로부터 생성된 임시 폴더 경로도 반환받음
    all_consolidated_data, all_rag_ready_data, image_paths, created_temp_folder_path = process_document(
        input_file_path=args.input_file,
        service_account_file=config.SERVICE_ACCOUNT_FILE,
        temp_base_dir=config.BASE_TEMP_DIR,
        prompt_template=PROMPT_TEMPLATE,
        generation_config=config.GENERATION_CONFIG,
        safety_settings=config.SAFETY_SETTINGS
    )
    print(f"문서 처리 완료. Consolidated: {len(all_consolidated_data)}, RAG_ready: {len(all_rag_ready_data)}, Images: {len(image_paths)}")

    if image_paths and all_consolidated_data: # 시각화를 위한 데이터가 있는지 확인
        # 테스트용 오답 목데이터 (사용자 제공)
        mock_incorrect_rag_ids_to_test = [
            (1, 0, 3, 1), 
            (1, 0, 7, 1),
            (1, 1, 3, 1),
            (1, 1, 4, 1),
            # (1, 1, 17, 1), # 이 ID가 실제 생성된 all_rag_ready_data에 있는지 확인 후 사용
            (2, 0, 6, 1)
        ]
        # 실제 all_rag_ready_data에 있는 ID인지 확인하고 없으면 mock_incorrect_rag_ids_to_test에서 제외하거나 경고
        valid_mock_ids = []
        if all_rag_ready_data: # all_rag_ready_data가 있을 때만 필터링
            rag_ids_set = {tuple(entry[0]) for entry in all_rag_ready_data if isinstance(entry, list) and len(entry) > 0 and isinstance(entry[0], list) and len(entry[0]) == 4}
            for mock_id in mock_incorrect_rag_ids_to_test:
                if mock_id in rag_ids_set:
                    valid_mock_ids.append(mock_id)
                else:
                    print(f"경고: 목업 오답 ID {mock_id}가 생성된 RAG 데이터에 없습니다. 시각화에서 제외됩니다.")
        else: # all_rag_ready_data가 비어있으면 목업 ID도 의미 없음
             print("경고: RAG 데이터가 생성되지 않아 목업 오답 ID를 검증할 수 없습니다.")


        if not valid_mock_ids:
            print("\n[단계 2] 시각화할 유효한 오답 ID가 없습니다. 시각화 단계를 건너<0xEB><0x9C><0x85>니다.")
        elif 'draw_underlines_for_incorrect_answers_enhanced' not in globals(): # visualizer 모듈 임포트 확인
             print("\n[단계 2] 시각화 함수를 찾을 수 없어 시각화를 건너<0xEB><0x9C><0x85>니다.")
        else:
            print(f"\n[단계 2] 오답 시각화 진행 (유효 오답 ID: {valid_mock_ids})...")
            output_visualization_folder = os.path.join(config.BASE_OUTPUT_DIR, "integration_test_visualization_final")
            
            draw_underlines_for_incorrect_answers_enhanced(
                incorrect_rag_ids=valid_mock_ids,
                all_consolidated_data=all_consolidated_data,
                all_rag_ready_data=all_rag_ready_data,
                page_image_paths=image_paths,
                output_folder=output_visualization_folder
            )
            print(f"시각화 결과는 '{output_visualization_folder}' 폴더를 확인하세요.")
    else:
        print("\n문서 처리 결과(이미지 경로 또는 consolidated 데이터)가 없어 시각화를 진행할 수 없습니다.")

    # --- 모든 작업 완료 후 임시 PDF 변환 폴더 최종 정리 ---
    if created_temp_folder_path and os.path.exists(created_temp_folder_path):
        try:
            shutil.rmtree(created_temp_folder_path)
            print(f"\n최종: 임시 PDF 변환 폴더 '{created_temp_folder_path}' 삭제 완료 (테스트 스크립트에서).")
        except Exception as e:
            print(f"\n최종: 임시 PDF 변환 폴더 '{created_temp_folder_path}' 삭제 중 오류: {e}")
    
    print("="*70)
    print("통합 테스트 종료.")