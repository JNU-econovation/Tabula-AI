# 예시: run_pipeline.py (또는 다른 메인 실행 스크립트)

import core # core.py 임포트
import config # 설정 파일 임포트
from Prompt import gemini_prompt as PROMPT_TEMPLATE # 프롬프트 셔플 임포트
# from visualizer import draw_underlines_for_incorrect_answers_enhanced # 나중에 사용

# --- 1. 문서 처리 ---
input_doc_path = "/path/to/your/document.pdf" # 실제 처리할 문서 경로
# config.py에서 필요한 설정값들을 가져와서 사용
# (예: SERVICE_ACCOUNT_FILE, BASE_TEMP_DIR, GENERATION_CONFIG, SAFETY_SETTINGS 등)

print(f"문서 처리 시작: {input_doc_path}")
all_consolidated_data, all_rag_ready_data, image_paths = core.process_document(
    input_file_path=input_doc_path,
    service_account_file=config.SERVICE_ACCOUNT_FILE,
    temp_base_dir=config.BASE_TEMP_DIR, # process_document 함수 내에서 고유 임시 폴더 생성하도록 수정됨
    prompt_template=PROMPT_TEMPLATE,
    generation_config_dict=config.GENERATION_CONFIG, # config 파일의 변수명과 일치하도록 수정
    safety_settings_list=config.SAFETY_SETTINGS   # config 파일의 변수명과 일치하도록 수정
)

if not all_rag_ready_data:
    print("RAG 데이터를 생성하지 못했습니다.")
    # 여기서 실행을 중단하거나 적절한 오류 처리를 할 수 있습니다.
    exit()

print(f"문서 처리 완료. RAG 준비 데이터 {len(all_rag_ready_data)}개 생성됨.")
# 이제 all_consolidated_data, all_rag_ready_data, image_paths 변수를 사용할 수 있습니다.


"""
B. RAG 시스템과의 결합:

인덱싱 단계 (RAG 지식 베이스 구축):

입력: all_rag_ready_data (그리고 각 항목의 ID)
all_rag_ready_data는 [[id_list], [text_string_in_list]] 형태로 구성되어 있습니다.
text_string_in_list[0] (즉, llm_processed_text)를 가져와 **임베딩(embedding)**으로 변환합니다. (예: Google의 text-embedding-004 모델 사용)
생성된 임베딩 벡터를 id_list (페이지, 블록, y, x 좌표 정보) 및 원본 텍스트(text_string_in_list[0])와 함께 **벡터 데이터베이스(Vector DB)**에 저장합니다. 이 ID는 나중에 검색된 내용을 원본 문서의 어느 부분에서 가져왔는지 추적하는 데 사용됩니다.
검색 및 답변 생성 단계:

사용자 질문이 들어오면, 질문 역시 임베딩으로 변환합니다.
이 질문 임베딩을 사용하여 벡터 DB에서 가장 유사한 텍스트 조각(과 그 메타데이터인 ID)들을 검색합니다.
검색된 텍스트 조각들을 컨텍스트로 활용하여 LLM에게 최종 답변 생성을 요청합니다.



C. 오답 시각화 (visualizer.py 활용):

만약 RAG 시스템을 통해 얻은 답변이 "오답"으로 판명되었고, 그 오답의 근거가 된 텍스트 조각들의 ID (incorrect_rag_ids)를 알 수 있다면, 다음과 같이 시각화 기능을 사용할 수 있습니다.

입력:

incorrect_rag_ids: RAG 시스템이 오답의 근거로 사용했다고 알려준 all_rag_ready_data 항목들의 ID 리스트 (예: [(1,0,3,1), (2,0,5,2)]).
all_consolidated_data: 원본 문서를 처리했을 때 core.py가 반환한 전체 통합 데이터. 이 데이터에는 모든 원본 OCR 청크의 좌표와 LLM 처리 상태가 포함되어 있어 밑줄을 그릴 정확한 위치를 찾는 데 사용됩니다.
all_rag_ready_data: 역시 원본 문서를 처리했을 때 core.py가 반환한 데이터. 밑줄 그리기 함수(draw_underlines_for_incorrect_answers_enhanced)의 "다음 ID 전까지" 휴리스틱에 사용될 수 있습니다.
image_paths: 원본 문서를 처리했을 때 core.py가 반환한 페이지 이미지 경로 리스트. 실제 밑줄을 그릴 이미지 파일들입니다.
"""
