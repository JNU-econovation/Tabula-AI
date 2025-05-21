import os
import sys
from typing import Dict, Any
from dataclasses import dataclass

from langchain_core.runnables import RunnableConfig
from core.messages import stream_graph, random_uuid
from layoutparse.parser import create_document_parse_graph

from logs.get_logger import get_logger

# 로그 설정
logger = get_logger()

# 설정 클래스 정의: 파싱에 필요한 기본 설정
@dataclass
class ParseConfig:
    language: str  
    domain_type: str  
    output_dir: str = "result"

# 필수 환경 변수 설정 확인
def check_required_env_vars():
    if "UPSTAGE_API_KEY" not in os.environ:
        logger.error("UPSTAGE_API_KEY 환경 변수가 설정되어 있지 않습니다.")
        return False
    return True
 

def parse_document(file: str, config: ParseConfig) -> Dict[str, Any]:
    """
    PDF 문서 파싱을 위한 메인 함수
    
    Args:
        file: 파일 경로
        config: 파싱 설정 객체
    
    Returns:
        Dict[str, Any]: 파싱 결과 스냅샷
    """
    # 환경 변수 확인
    if not check_required_env_vars():
        sys.exit(1)

    logger.info(f"Parsing file: {file}")
    logger.info(f"Parsing language: {config.language}")
    logger.info(f"Parsing domain: {config.domain_type}")

    # 파서 그래프 생성 시 filepath 전달
    parser_graph = create_document_parse_graph(
        filepath=file,
        output_dir=config.output_dir,
        language=config.language,
        domain_type=config.domain_type
    )

    # 실행 설정
    run_config = RunnableConfig(
        recursion_limit=300,
        configurable={"thread_id": random_uuid()}
    )

    # 입력 데이터 구성
    inputs = {
        "file": file,
        "filepath": file,
        "language": config.language,
        "domain_type": config.domain_type
    }

    # 파서 실행
    logger.info(f"Parsing: '{file}'를 파싱하는 중...")

    # # 실행결과 스트리밍 출력
    # stream_graph(parser_graph, inputs, run_config)

    # # 결과 반환
    # snapshot = parser_graph.get_state(run_config).values
    # snapshot["document_id"] = os.path.splitext(os.path.basename(file))[0]
    # return snapshot

    result = parser_graph.invoke(inputs, config=run_config)
    return result

# def process_document(file_path: str):
#     """문서 처리 파이프라인 (개별 테스트용)"""
#     origin_dir = None
#     try:
#         # 1. 원본 PDF를 result/origin 디렉토리에 복사
#         document_id = os.path.splitext(os.path.basename(file_path))[0]
#         origin_dir = os.path.join("result", "origin")
#         os.makedirs(origin_dir, exist_ok=True)
#         origin_pdf = os.path.join(origin_dir, os.path.basename(file_path))
#         shutil.copy2(file_path, origin_pdf)
        
#         # 2. PDF 파싱
#         config = ParseConfig(
#             language="ko",
#             domain_type="Korean history"
#         )
#         parsed_data = parse_document(file_path, config)
#         if not parsed_data:
#             logger.error("파싱 실패")
#             return
        
#         # 3. 디렉토리 생성
#         create_directories(document_id)
        
#         # 4. 마크다운 생성 및 저장
#         markdown_content = generate_markdown(parsed_data, document_id)
#         save_markdown(markdown_content, document_id)
        
#         logger.info(f"문서 처리 완료: {document_id}")
        
#     except Exception as e:
#         logger.error(f"문서 처리 중 오류 발생: {str(e)}")
#         raise
#     finally:
#         # 5. 임시 디렉토리 정리
#         origin_dirs = [
#             os.path.join("origin", document_id),
#             os.path.join("origin", f"{document_id}_parse")
#         ]
        
#         for origin_dir in origin_dirs:
#             if os.path.exists(origin_dir):
#                 try:
#                     shutil.rmtree(origin_dir)
#                     logger.info(f"임시 디렉토리 정리 완료: {origin_dir}")
#                 except Exception as e:
#                     logger.error(f"임시 디렉토리 정리 중 오류 발생: {str(e)}")


# def create_directories(document_id: str):
#     """필요한 디렉토리 생성 (개별 테스트용)"""
#     directories = [
#         os.path.join("result", "images", document_id),
#         os.path.join("result", "md"),
#         os.path.join("result", "json"),
#         os.path.join("result", "pdf", document_id),
#         os.path.join("result", "origin")
#     ]
    
#     for directory in directories:
#         os.makedirs(directory, exist_ok=True)


# def generate_markdown(parsed_data: Dict[str, Any], document_id: str) -> str:
#     """마크다운 생성 (개별 테스트용)"""
#     markdown = []
    
#     # 텍스트 처리
#     for i, text in enumerate(parsed_data.get("texts", [])):
#         markdown.append(f"## 페이지 {i+1}\n")
#         markdown.append(text + "\n")
        
#         # 해당 페이지의 이미지 처리
#         page_images = [img for img in parsed_data.get("images", []) if img["page"] == i]
#         for img in page_images:
#             img_path = f"../images/{document_id}/image_{img['index']}.jpg"
#             markdown.append(f"![이미지 {img['index']}]({img_path})\n")
    
#     return "\n".join(markdown)


# def save_markdown(content: str, document_id: str):
#     """마크다운 저장 (개별 테스트용)"""
#     md_dir = os.path.join("result", "md")
#     os.makedirs(md_dir, exist_ok=True)
#     markdown_path = os.path.join(md_dir, f"{document_id}.md")
    
#     with open(markdown_path, "w", encoding="utf-8") as f:
#         f.write(content)


# def test_parse():
#     """전체 파싱 프로세스 테스트"""
#     test_file = "/Users/kyeong6/Desktop/korean.pdf"
    
#     if not os.path.exists(test_file):
#         logger.error(f"오류: 파일 '{test_file}'가 존재하지 않습니다.")
#         return
    
#     logger.info("=== 전체 파싱 프로세스 테스트 시작 ===")
#     process_document(test_file)
#     logger.info("=== 전체 파싱 프로세스 테스트 완료 ===")


# def test_parse_document():
#     """parse_document 함수만 테스트"""
#     test_file = "/Users/kyeong6/Desktop/korean.pdf"
    
#     if not os.path.exists(test_file):
#         logger.error(f"오류: 파일 '{test_file}'가 존재하지 않습니다.")
#         return
    
#     config = ParseConfig(
#         language="ko",
#         domain_type="Korean history"
#     )
    
#     logger.info("=== parse_document 함수 테스트 시작 ===")
#     result = parse_document(test_file, config)
#     logger.info(f"파싱 결과: {result}")
#     logger.info("=== parse_document 함수 테스트 완료 ===")


# if __name__ == "__main__":
#     # 테스트할 함수 선택
#     test_parse()  # 전체 프로세스 테스트
#     # test_parse_document()  # parse_document 함수만 테스트
