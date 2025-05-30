from typing import Dict, Any, List, Optional
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

from common_sdk import get_logger
from .nodes import MissingAnalysisNodes

logger = get_logger()

class MissingAnalysisState(TypedDict):
    """LangGraph 상태 정의"""
    
    # 입력 데이터
    space_id: Optional[str]  # MongoDB 조회를 위한 _id
    raw_user_inputs: str  # 사용자 입력 원본
    
    # MongoDB 조회 데이터
    keyword_data: Optional[List[Dict[str, Any]]]  # MongoDB에서 조회한 키워드 데이터
    
    # 처리된 데이터
    formatted_hierarchy: Optional[str]  # 키워드 계층화 형식
    extracted_user_content: Optional[str]  # 추출된 사용자 텍스트
    api_response: Optional[str]  # OpenAI API 응답
    
    # 최종 결과
    missing_items: List[str]  # 누락된 항목 목록
    
    # 에러 처리
    error: Optional[str]  # 에러 메시지

class MissingAnalysisWorkflow:
    """누락 분석 LangGraph 워크플로우 관리"""
    
    def __init__(self):
        self.nodes = MissingAnalysisNodes()
    
    def create_missing_analysis_graph(self) -> StateGraph:
        """누락 분석 LangGraph 워크플로우 생성"""
        workflow = StateGraph(MissingAnalysisState)
        
        # 노드 추가
        workflow.add_node("initialize", self.nodes.initialize)
        workflow.add_node("load_keywords_from_db", self.nodes.load_keywords_from_db)
        workflow.add_node("formatting_keywords", self.nodes.formatting_keywords)
        workflow.add_node("process_results", self.nodes.process_results)
        workflow.add_node("compile_results", self.nodes.compile_results)
        
        # 시작점 설정
        workflow.set_entry_point("initialize")
        
        # 엣지 추가
        workflow.add_edge("initialize", "load_keywords_from_db")
        workflow.add_edge("load_keywords_from_db", "formatting_keywords")
        workflow.add_edge("formatting_keywords", "process_results")
        workflow.add_edge("process_results", "compile_results")
        workflow.add_edge("compile_results", END)
        
        return workflow.compile()
    
    async def run_analysis(self, space_id: str, user_inputs: str) -> Dict[str, Any]:
        """누락 분석 실행"""
        graph = self.create_missing_analysis_graph()
        
        initial_state = {
            "space_id": space_id,
            "raw_user_inputs": user_inputs
        }
        
        logger.info(f"Starting Missing Analysis Workflow for space_id: {space_id}")
        
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
    
    # 키워드 분석 결과 MongoDB 저장