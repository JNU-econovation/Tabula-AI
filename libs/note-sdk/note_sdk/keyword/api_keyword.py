import re 
import json

from typing import Dict, Any
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage 

from common_sdk.config import settings as common_settings
from common_sdk.utils import num_tokens_from_string
from common_sdk.prompt_loader import PromptLoader
from common_sdk.get_logger import get_logger

# 로거 설정
logger = get_logger()

class KeywordGuide:
    def __init__(self, domain: str, space_id: str = None, llm_model_name: str = "gpt-4.1-mini", temperature: float = 0):
        """
        Args:
            domain: 도메인 타입
            space_id: 작업 ID
            llm_model_name: 사용할 LLM 모델 이름
            temperature: LLM의 temperature 값
        """
        self.domain_type = domain
        self.space_id = space_id
        self.llm = ChatOpenAI(model=llm_model_name, temperature=temperature, openai_api_key=common_settings.OPENAI_API_KEY_J)

        # 프롬프트 로더 초기화
        self.prompt_loader = PromptLoader()
        self.keyword_prompt = self.prompt_loader.load_prompt("keyword-prompt")

    # 마크다운 내용에서 이미지 및 테이블 링크 제거
    def preprocess_markdown_content(self, content: str) -> str:
        """
        Args:
            content: 원본 마크다운 내용
        """
        # 이미지 링크 제거
        content = re.sub(r"!\[[^\]]*?\]\([^)]*?\)", "", content)
        
        # 테이블 CSV 링크 제거
        content = re.sub(r"\[[^\]]*?\]\([^)]*?\.csv\)", "", content)
        
        return content

    # LLM 호출
    def call_llm(self, system_prompt_content: str, user_prompt_content: str) -> Any:
        """
        Args:
            system_prompt_content: 시스템 프롬프트 내용
            user_prompt_content: 사용자 프롬프트 내용
        """
        messages = [
            SystemMessage(content=system_prompt_content),
            HumanMessage(content=user_prompt_content)
        ]
        
        # 토큰 수 계산
        system_tokens = num_tokens_from_string(system_prompt_content)
        user_tokens = num_tokens_from_string(user_prompt_content)
        total_tokens = system_tokens + user_tokens
        
        logger.info(f"\n=== 토큰 사용량 ===")
        logger.info(f"시스템 프롬프트: {system_tokens} 토큰")
        logger.info(f"사용자 프롬프트: {user_tokens} 토큰")
        logger.info(f"총 토큰 수: {total_tokens} 토큰")
        
        response_obj = None
        try:
            # LLM 호출
            response_obj = self.llm.invoke(messages)
            
            # --- LLM 응답 디버깅 시작 ---
            if response_obj and hasattr(response_obj, 'content'):
                raw_content = response_obj.content
                
                # 응답 토큰 수 계산
                response_tokens = num_tokens_from_string(raw_content)
                logger.info(f"응답 토큰 수: {response_tokens} 토큰")
                logger.info(f"총 처리 토큰 수: {total_tokens + response_tokens} 토큰")
                
                if not raw_content or not raw_content.strip():
                    logger.error("Error: LLM response content is empty")
                    return None
                
                cleaned_json_str = raw_content.strip()
                if cleaned_json_str.startswith("```json") and cleaned_json_str.endswith("```"):
                    cleaned_json_str = cleaned_json_str[len("```json"):].rstrip("`\n ")
                elif cleaned_json_str.startswith("```") and cleaned_json_str.endswith("```"):
                    cleaned_json_str = cleaned_json_str[len("```"):].rstrip("`\n ")

                try:
                    return json.loads(cleaned_json_str)
                except json.JSONDecodeError as e:
                    logger.error(f"JSON 파싱 실패: {e}")
                    return {
                        "name": "문서 분석 실패",
                        "children": [
                            {
                                "name": "분석 오류",
                                "children": [
                                    {
                                        "name": "JSON 파싱 실패",
                                        "children": [
                                            {"name": "LLM 응답 형식 오류"}
                                        ]
                                    }
                                ]
                            }
                        ]
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"LLM 호출 중 오류 발생: {e}")
            return None

    # LLM을 위한 마인드맵 생성용 프롬프트 생성
    def create_mindmap_prompt(self, content: str) -> str:
        # 내용 전처리
        processed_content = self.preprocess_markdown_content(content)
        
        # 프롬프트 템플릿 가져오기
        template = self.keyword_prompt.get("template", "")
        if not template:
            logger.error("프롬프트 템플릿을 찾을 수 없습니다.")
            return ""
            
        # 프롬프트 생성
        try:
            return template.format(content=processed_content)
        except Exception as e:
            logger.error(f"프롬프트 생성 중 오류 발생: {e}")
            return ""

    # 마크다운 파일로부터 마인드맵 구조 생성
    def generate_mindmap_from_markdown(self, md_path: str) -> Dict[str, Any]:
        """
        Args:
            md_path: 마크다운 파일 경로
        """
        try:
            with open(md_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except FileNotFoundError:
            logger.error(f"Error: Markdown file not found at {md_path}")
            return {}
        
        # 사용자 프롬프트 생성
        user_prompt = self.create_mindmap_prompt(content)

        # 마인드맵 생성용 시스템 프롬프트
        system_prompt = "당신은 문서를 분석하여 JSON 형식의 마인드맵을 생성하는 정확하고 지시를 잘 따르는 AI 어시스턴트입니다."
        
        mindmap_data = self.call_llm(system_prompt_content=system_prompt, user_prompt_content=user_prompt)
        
        if isinstance(mindmap_data, dict):
            mindmap_data["document_id"] = Path(md_path).stem
            return mindmap_data
        else:
            logger.error("Error: Failed to generate mindmap data")
            return {}
