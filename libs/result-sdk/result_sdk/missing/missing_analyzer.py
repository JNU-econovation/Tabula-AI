import json
from typing import Dict, Any, List, Optional

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

    async def analyze(self, keyword_data: Dict[str, Any], user_inputs: str) -> Dict[str, Any]:
        """
        누락 분석 실행
        """
        try:
            logger.info("Starting Missing Analysis")

            result = await self.workflow.run_analysis(keyword_data, user_inputs)

            logger.info("Missing Analysis Completed")

            return result
        
        except Exception as e:
            logger.error(f"failed to analyze missing items: {e}")
            return {
                "success": False,
                "error": str(e),
                "missing_items": []
            }