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
    language: str  
    domain_type: str  
    output_dir: str = "result"

# 필수 환경 변수 설정 확인
def check_required_env_vars():
    if common_settings.UPSTAGE_API_KEY is None:
        logger.error("Upstage API Key is not set")
        return False
    return True
 

# PDF 문서 파싱을 위한 메인 함수
def parse_document(file: str, config: ParseConfig) -> Dict[str, Any]:
    """
    Args:
        file: 파일 경로
        config: 파싱 설정 객체
    """
    # 환경 변수 확인
    if not check_required_env_vars():
        sys.exit(1)

    logger.info(f"Parsing file: {file}")
    logger.info(f"Parsing language: {config.language}")
    logger.info(f"Parsing domain: {config.domain_type}")

    # 파서 그래프 생성 시 filepath 전달
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
        "file": file,
        "language": config.language,
        "domain_type": config.domain_type
    }

    # 파서 실행
    logger.info(f"Parsing: '{file}'")

    # # 실행결과 스트리밍 출력
    # stream_graph(parser_graph, inputs, run_config)

    # # 결과 반환
    # snapshot = parser_graph.get_state(run_config).values
    # snapshot["document_id"] = os.path.splitext(os.path.basename(file))[0]
    # return snapshot

    result = parser_graph.invoke(inputs, config=run_config)
    return result