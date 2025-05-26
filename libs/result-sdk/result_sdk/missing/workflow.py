import json
import os
from typing import Dict, Any, List, Optional
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from openai import OpenAI

from common_sdk import get_logger
from common_sdk import PromptLoader
from common_sdk import settings
from .processor import DataProcessor

logger = get_logger()

class MissingAnalysisState(TypedDict):
    """
    LangGraph 상태 정의
    """

    # 입력 데이터
    raw_keyword_data: Optional[Dict[str, Any]] # 업로드된 키워드 파일 데이터 → MongoDB 조회 변경 예정
    raw_user_inputs: str # 사용자 입력 원본
    
    # 처리된 데이터
    formatted_hierarchy: Optional[str] # 키워드 계층화 형식
    extracted_user_content: Optional[str] # 추출된 사용자 텍스트
    api_response: Optional[str] # OpenAI API 응답
    
    # 최종 결과
    missing_items: List[str] # 누락된 항목 목록
    
    # 에러 처리
    error: Optional[str] # 에러 메시지

class MissingAnalysisWorkflow:
    """
    누락 분석 LangGraph 워크플로우 관리
    """

    def __init__(self):
        self.openai_client = OpenAI(api_key=settings.OPENAI_API_KEY_B)
        self.prompt_loader = PromptLoader()
        self.data_processor = DataProcessor()
    
    def load_prompt(self) -> str:
        """
        프롬프트 로드
        """

        MISSING_PROMPT_KEY = "missing-prompt"
        
        try:
            prompt_data = self.prompt_loader.load_prompt(MISSING_PROMPT_KEY)["template"]
            return prompt_data
        except Exception as e:
            logger.error(f"failed to load prompt: {e}")
            return None
        
    async def initialize(self, state: MissingAnalysisState) -> MissingAnalysisState:
        """
        초기 설정 및 초기화 노드
        """
        state["missing_items"] = []
        state["error"] = None

        logger.info("Missing Analysis Workflow Start")
        return state
    
    async def formatting_keywords(self, state: MissingAnalysisState) -> MissingAnalysisState:
        """
        키워드 포맷팅 노드
        """
        try:
            raw_data = state["raw_keyword_data"]
            raw_inputs = state["raw_user_inputs"]
        
            if not raw_data:
                raise ValueError("키워드 데이터가 없습니다.")
            
            if not raw_inputs:
                raise ValueError("사용자 입력이 없습니다.")
            
            self.data_processor.set_keyword_data(raw_data)
            self.data_processor.set_user_inputs(raw_inputs)

            formatted_hierarchy = self.data_processor.format_hierarchy_list(raw_data)
            extracted_user_content = self.data_processor.extract_user_content(raw_inputs)
            
            state["formatted_hierarchy"] = formatted_hierarchy
            state["extracted_user_content"] = extracted_user_content
            
            logger.info(f"keyword formatting completed: {len(formatted_hierarchy)} tokens")
            logger.info(f"user input extract completed: {len(extracted_user_content)} tokens")

            return state
        
        except Exception as e:
            logger.error(f"Failed to format keywords: {e}")
            return state
    
    async def process_results(self, state: MissingAnalysisState) -> MissingAnalysisState:
        """
        결과 처리 노드 (OpenAI 요청 → 결과 구성)
        """
        try:
            formatted_hierarchy = state["formatted_hierarchy"]
            extracted_content = state["extracted_user_content"]

            if not formatted_hierarchy or not extracted_content:
                raise ValueError("필요한 데이터가 누락되었습니다.")
            
            # 프롬프트 template 로드
            prompt_template = self.load_prompt()

            # 최종 프롬프트 구성
            final_prompt = prompt_template.format(
                keywords=formatted_hierarchy,
                user_content=extracted_content
            )

            logger.info("OpenAI API call started")

            # OpenAI API 호출
            response = self.openai_client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": "내용 평가 전문가입니다. 누락된 개념을 정확히 찾아내고 결과를 JSON 형식으로 반환해야 합니다."},
                    {"role": "user", "content": final_prompt}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )

            # 응답 내용 저장
            response_content = response.choices[0].message.content
            state["api_response"] = response_content

            missing_items = self._parse_missing_items(response_content)
            state["missing_items"] = missing_items

            logger.info("OpenAI API call completed")
            logger.info(f"found missing items: {len(missing_items)}")

            return state
        
        except Exception as e:
            logger.error(f"failed to process result: {e}")
            return state
        
    async def compile_results(self, state: MissingAnalysisState) -> MissingAnalysisState:
        """
        최종 결과 컴파일 노드
        """
        
        missing_count = len(state["missing_items"])

        if state.get("error"):
            logger.error(f"Missing Analysis Failed: {state['error']}")
        else:
            logger.info(f"Missing Analysis Completed: {missing_count} missing items found")

        return state
    
    def _parse_missing_items(self, response_content: str) -> List[str]:
        """
        누락된 항목 추출
        """
        try:
            result = json.loads(response_content)
            
            # 누락된 항목 추출
            missing_items = []
            if isinstance(result, dict):
                for key in result:
                    if isinstance(result[key], list):
                        missing_items = result[key]
                        break
                else:
                    missing_items = []
            elif isinstance(result, list):
                missing_items = result
            else:
                missing_items = []
                
        except json.JSONDecodeError:
            # JSON 파싱 실패 시, 줄바꿈으로 분리
            missing_items = [item.strip() for item in response_content.split("\n") if item.strip()]
        
        # 중복 제거 및 빈 문자열 제거
        unique_items = []
        for item in missing_items:
            if item and isinstance(item, str) and item not in unique_items:
                # Standardize periods
                if not item.endswith('.'):
                    item += '.'
                unique_items.append(item)
        
        return unique_items
    
    def create_missing_analysis_graph(self) -> StateGraph:
        """
        누락 분석 LangGraph 워크플로우 생성
        """

        workflow = StateGraph(MissingAnalysisState)

        workflow.add_node("initialize", self.initialize)
        workflow.add_node("formatting_keywords", self.formatting_keywords)
        workflow.add_node("process_results", self.process_results)
        workflow.add_node("compile_results", self.compile_results)

        # 시작점 설정
        workflow.set_entry_point("initialize")

        # 엣지 추가
        workflow.add_edge("initialize", "formatting_keywords")
        workflow.add_edge("formatting_keywords", "process_results")
        workflow.add_edge("process_results", "compile_results")
        workflow.add_edge("compile_results", END)

        return workflow.compile()
    
    async def run_analysis(self, keyword_data: Dict[str, Any], user_inputs: str) -> MissingAnalysisState:
        """
        누락 분석 실행
        """

        graph = self.create_missing_analysis_graph()

        initial_state = {
            "raw_keyword_data": keyword_data,
            "raw_user_inputs": user_inputs
        }

        logger.info("Starting Missing Analysis Workflow")

        result = await graph.ainvoke(initial_state)

        logger.info("Missing Analysis Workflow Completed")

        # 결과 반환
        if result.get("error"):
            return {
                "success": False,
                "error": result["error"],
                "missing_items": []
            }
        
        return {
            "success": True,
            "missing_items": result["missing_items"],
            "api_response": result.get("api_response")
        }

        return result