import os
import sys
from typing import Dict, Any
from dataclasses import dataclass
from langchain_core.runnables import RunnableConfig
from note_sdk.llm import stream_graph, random_uuid
from note_sdk.parsing.parser import create_document_parse_graph
from common_sdk.config import settings as common_settings
from common_sdk.get_logger import get_logger

# 로그 설정
logger = get_logger()

# 설정 클래스 정의: 파싱에 필요한 기본 설정
@dataclass
class ParseConfig:
    language: str # 언어 설정
    domain_type: str # 도메인 타입
    output_dir: str # 결과 디렉토리 경로
    task_id: str # 작업 ID
 

# PDF 문서 파싱을 위한 메인 함수
def parse_document(file: str, config: ParseConfig) -> Dict[str, Any]:
    """
    Args:
        file: 파일 경로
        config: 파싱 설정 객체
    """

    logger.info(f"Parsing file: {file}")
    logger.info(f"Parsing language: {config.language}")
    logger.info(f"Parsing domain: {config.domain_type}")

    # 파서 그래프 생성
    parser_graph = create_document_parse_graph(
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
        "language": config.language,
        "domain_type": config.domain_type,
        "output_dir": config.output_dir,
        "task_id": config.task_id,
        "filetype": "pdf",
        "include_image_in_output": True
    }

    # 파서 실행
    logger.info(f"Parsing: '{file}'")

    result = parser_graph.invoke(inputs, config=run_config)
    return result