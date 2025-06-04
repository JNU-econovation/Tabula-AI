import json
from typing import Dict, Any

from common_sdk import get_logger

logger = get_logger()

class DataProcessor:
    """
    사용자 입력 파싱, 키워드 계층화 및 형식 지정
    """
    
    def __init__(self):
        self.keyword_data: Dict[str, Any] = {}
        self.raw_user_inputs: str = ""

    def set_keyword_data(self, keyword_data: Dict[str, Any]) -> None:
        """
        키워드 데이터 설정
        """
        if not isinstance(keyword_data, dict):
            raise ValueError("Keyword data must be a dictionary")
        
        self.keyword_data = keyword_data  # 단일 키워드 객체 저장
        logger.info(f"Keyword data set successfully: {keyword_data.get('name', 'Unknown')}")
        logger.info(f"Keyword data: {str(keyword_data)[:300]}")
    
    def set_user_inputs(self, user_inputs: str) -> None:
        """
        사용자 입력 데이터 설정 (OCR + LLM 결과)
        """
        if not isinstance(user_inputs, str):
            raise ValueError("User inputs must be a string")
        
        self.raw_user_inputs = user_inputs
        logger.info("User inputs set successfully")
    
    def format_hierarchy_list(self, node: Dict[str, Any], level: int = 0) -> str:
        """
        프롬프트에 전달되는 토큰을 줄이고 LLM이 직관적으로 계층형 구조를 파악할 수 있도록
        JSON 형태의 키워드를 계층형 구조로 변경
        """
        indent = "  " * level
        
        if level == 0:
            # 최상위 레벨은 기호 없이
            result = f"{node['name']}\n"
        else:
            # 레벨에 따라 다른 기호 사용
            prefix = "-" if level == 1 else ("*" if level == 2 else "+")
            result = f"{indent}{prefix} {node['name']}\n"
        
        # 자식 노드가 있으면 재귀적으로 처리
        if "children" in node and node["children"]:
            for child in node["children"]:
                result += self.format_hierarchy_list(child, level + 1)
        
        return result
    
    def extract_user_content(self, user_inputs: str) -> str:
        """
        사용자 입력에서 텍스트 내용만 추출
        """

        if not user_inputs:
            logger.warning("Empty user inputs provided")
            return ""

        try:
            # 문자열을 JSON으로 파싱
            data = json.loads(user_inputs)
            
            # 텍스트 내용만 추출
            texts = []
            for item in data:
                if len(item) >= 2 and len(item[1]) >= 1:
                    texts.append(item[1][0])
            
            # 텍스트를 줄바꿈으로 결합
            return "\n".join(texts)
        except json.JSONDecodeError:
            # JSON 파싱 실패 시 원본 텍스트 반환
            return user_inputs
        
        except Exception as e:
            logger.error(f"Unexpected error in extract_user_content: {e}")
            return user_inputs