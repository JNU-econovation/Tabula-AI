from typing import List, Tuple
from .models import GradingConfig, EvaluationResponse
from .workflow import CorrectionWorkflow, extract_wrong_answer_ids
from common_sdk import settings

class GradingService:
    """자동 채점 Main 클래스"""

    def __init__(self, 
        document_id: str, 
        index_name: str,
        openai_api_keys: List[str] = None,
        model_name: str = "gpt-4.1-mini",
        temperature: float = 0,
        max_tokens: int = 1000
    ):
        """채점 서비스 초기화"""

        if openai_api_keys is None:
                openai_api_keys = [
                    settings.OPENAI_API_KEY_J,
                    settings.OPENAI_API_KEY_K
                ]

        self.config = GradingConfig(
            document_id=document_id,
            index_name=index_name,
            prompt_template="",
            openai_api_keys=openai_api_keys,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )

        self.workflow = CorrectionWorkflow(self.config)

    async def grade(self, user_inputs: str) -> EvaluationResponse:
        """
        채점 실행 (LangGraph 워크플로우)
        """
        return await self.workflow.run_correction(
            document_id=self.config.document_id,
            index_name=self.config.index_name,
            user_inputs=user_inputs
        )
    
    async def grade_with_wrong_ids(self, user_inputs: str) -> Tuple[EvaluationResponse, List[List[int]]]:
        """
        채점 실행 후 오답 ID도 함께 반환
        """
        evaluation_response = await self.grade(user_inputs)
        wrong_ids = extract_wrong_answer_ids(evaluation_response)
        return evaluation_response, wrong_ids
    
    def extract_wrong_ids(self, evaluation_response: EvaluationResponse) -> List[List[int]]:
        """
        채점 결과에서 오답 ID만 추출
        """
        return extract_wrong_answer_ids(evaluation_response)
