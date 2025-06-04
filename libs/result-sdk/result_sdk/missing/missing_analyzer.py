from typing import Dict, Any

from common_sdk import get_logger
from .processor import DataProcessor
from .workflow import MissingAnalysisWorkflow

logger = get_logger()

class MissingAnalyzer:
    """
    누락 분석 실행 Main 클래스
    """

    def __init__(self):
        self.workflow = MissingAnalysisWorkflow()
        self.data_processor = DataProcessor()

    async def analyze(self, space_id: str, user_inputs: str) -> Dict[str, Any]:
        """
        누락 분석 실행
        """
        try:
            logger.info(f"Starting Missing Analysis for space_id: {space_id}")

            # 입력 데이터 검증
            if not space_id:
                raise ValueError("space_id is necessary")
            
            if not user_inputs or not isinstance(user_inputs, str):
                raise ValueError("user_inputs must be a string, not an empty string")

            # 워크플로우 실행
            result = await self.workflow.run_analysis(space_id, user_inputs)

            logger.info("Missing Analysis Completed")

            return result
        
        except ValueError as ve:
            logger.error(f"Input validation error: {ve}")
            return {
                "success": False,
                "error": f"Input validation error: {str(ve)}",
                "missing_items": []
            }
        
        except Exception as e:
            logger.error(f"failed to analyze missing items: {e}")
            return {
                "success": False,
                "error": str(e),
                "missing_items": []
            }