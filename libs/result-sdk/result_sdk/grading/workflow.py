# workflow.py

from typing import List
from langgraph.graph import StateGraph, END

from .models import GraphState, PageResult, GradingConfig, EvaluationResponse
from .nodes import GradingNodes
from common_sdk import get_logger

logger = get_logger()

class CorrectionWorkflow:
    """채점 워크플로우"""

    def __init__(self, config: GradingConfig):
        self.config = config
        self.nodes = GradingNodes(config)
    
    def create_correction_graph(self):
        """채점 그래프 생성"""
        graph = StateGraph(GraphState)
        
        graph.add_node("data_input", self.nodes.data_input_node)
        graph.add_node("initialization", self.nodes.initialization_node)
        graph.add_node("process_page", self.nodes.process_page_node)
        graph.add_node("compile_results", self.nodes.compile_results_node)
        
        graph.add_edge("data_input", "initialization")
        graph.add_edge("initialization", "process_page")
        graph.add_conditional_edges(
            "process_page",
            self.nodes.should_end,
            {
                "process_page": "process_page",
                "compile_results": "compile_results"
            }
        )
        graph.add_edge("compile_results", END)
        
        graph.set_entry_point("data_input")
        
        return graph.compile()
    
    async def run_correction(
        self,
        space_id: str,
        index_name: str,
        user_inputs: str,
        lang_type: str = "ko"
    ) -> EvaluationResponse:
        """채점 워크플로우 실행"""
        try:
            correction_graph = self.create_correction_graph()
            
            initial_state: GraphState = {
                "space_id": space_id,
                "index_name": index_name,
                "lang_type": lang_type,
                "user_inputs": user_inputs,
                "prompt_template": "",
                "all_texts": [],
                "pages": {},
                "document_finder": None,
                "wrong_answers_by_page": {},
                "page_results": [],
                "final_results": [],
                "page_processes_pending": False
            }
            
            final_state = await correction_graph.ainvoke(initial_state)
            
            evaluation_response = EvaluationResponse(
                results=final_state["final_results"]
            )
            
            return evaluation_response
            
        except Exception as e:
            logger.error(f"Error in correction process: {str(e)}")
            raise Exception(f"채점 처리 중 오류 발생: {str(e)}")
        
def extract_wrong_answer_ids(evaluation_response: EvaluationResponse) -> List[List[int]]:
    """EvaluationResponse에서 오답의 id 값만 추출하여 리스트로 반환"""
    wrong_ids = []
    
    for page_result in evaluation_response.results:
        for wrong_answer in page_result.wrong_answers:
            wrong_ids.append(wrong_answer.id)
    
    return wrong_ids