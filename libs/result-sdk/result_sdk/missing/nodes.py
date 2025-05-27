import json
from typing import Dict, Any, List
from openai import OpenAI

from common_sdk import get_logger, PromptLoader, settings, MongoDB
from .processor import DataProcessor

logger = get_logger()

class MissingAnalysisNodes:
    """
    누락 분석 워크플로우의 개별 노드들
    """
    
    def __init__(self):
        self.openai_client = OpenAI(api_key=settings.OPENAI_API_KEY_B)
        self.prompt_loader = PromptLoader()
        self.data_processor = DataProcessor()
        self.mongodb = MongoDB()
    
    def load_prompt(self) -> str:
        """프롬프트 로드"""
        MISSING_PROMPT_KEY = "missing-prompt"
        
        try:
            prompt_data = self.prompt_loader.load_prompt(MISSING_PROMPT_KEY)["template"]
            return prompt_data
        except Exception as e:
            logger.error(f"failed to load prompt: {e}")
            return None
    
    async def fetch_keyword_data(self, space_id: str) -> List[Dict[str, Any]]:
        """MongoDB에서 키워드 데이터 조회"""
        try:
            from bson import ObjectId
            
            collection = self.mongodb.db.spaces
            document = await collection.find_one({"_id": ObjectId(space_id)})
            
            if not document:
                logger.warning(f"No document found for space_id: {space_id}")
                return []
            
            keyword_data = document.get("keyword", [])
            
            if not isinstance(keyword_data, list):
                logger.warning(f"keyword field is not a list for space_id: {space_id}")
                return []
            
            logger.info(f"Retrieved {len(keyword_data)} keyword items for space_id: {space_id}")
            return keyword_data
            
        except Exception as e:
            logger.error(f"Failed to fetch keyword data from MongoDB: {e}")
            return []
    
    async def initialize(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """초기 설정 및 초기화 노드"""
        state["missing_items"] = []
        state["error"] = None
        state["keyword_data"] = None
        
        logger.info("Missing Analysis Workflow Start")
        return state
    
    async def load_keywords_from_db(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """MongoDB에서 키워드 데이터 로드 노드"""
        try:
            space_id = state["space_id"]
            
            if not space_id:
                raise ValueError("space_id does not exist")
            
            keyword_data = await self.fetch_keyword_data(space_id)
            
            if not keyword_data:
                raise ValueError(f"Keyword data not found for space_id {space_id}")
            
            state["keyword_data"] = keyword_data
            
            logger.info(f"Successfully loaded {len(keyword_data)} keywords from MongoDB")
            return state
        
        except Exception as e:
            logger.error(f"Failed to load keywords from DB: {e}")
            state["error"] = str(e)
            return state
    
    async def formatting_keywords(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """키워드 포맷팅 노드"""
        try:
            keyword_data = state["keyword_data"]
            raw_inputs = state["raw_user_inputs"]
            
            if not keyword_data:
                raise ValueError("Keyword data does not exist")
            
            if not raw_inputs:
                raise ValueError("user input does not exist")
            
            # 첫 번째 키워드만 사용
            if not isinstance(keyword_data, list) or len(keyword_data) == 0:
                raise ValueError("Invalid keyword data format")
            
            main_keyword = keyword_data[0]
            
            self.data_processor.set_keyword_data(main_keyword)
            self.data_processor.set_user_inputs(raw_inputs)
            
            # 단일 키워드 계층구조 포맷팅
            formatted_hierarchy = self.data_processor.format_hierarchy_list(main_keyword)
            extracted_user_content = self.data_processor.extract_user_content(raw_inputs)
            
            state["formatted_hierarchy"] = formatted_hierarchy
            state["extracted_user_content"] = extracted_user_content
            
            logger.info(f"keyword formatting completed: {len(formatted_hierarchy)} tokens")
            logger.info(f"user input extract completed: {len(extracted_user_content)} tokens")
            
            return state
        
        except Exception as e:
            logger.error(f"Failed to format keywords: {e}")
            state["error"] = str(e)
            return state
    
    async def process_results(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """결과 처리 노드 (OpenAI 요청 → 결과 구성)"""
        try:
            formatted_hierarchy = state["formatted_hierarchy"]
            extracted_content = state["extracted_user_content"]
            
            if not formatted_hierarchy or not extracted_content:
                raise ValueError("The required data is missing")
            
            # 프롬프트 template 로드
            prompt_template = self.load_prompt()
            
            if not prompt_template:
                raise ValueError("Failed to load prompt template")
            
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
                    {"role": "system", "content": "당신은 내용 평가 전문가입니다. 누락된 개념을 정확히 찾아내고 결과를 JSON 형식으로 반환해야 합니다."},
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
            state["error"] = str(e)
            return state
    
    async def compile_results(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """최종 결과 컴파일 노드"""
        missing_count = len(state["missing_items"])
        
        if state.get("error"):
            logger.error(f"Missing Analysis Failed: {state['error']}")
        else:
            logger.info(f"Missing Analysis Completed: {missing_count} missing items found")
        
        return state
    
    def _parse_missing_items(self, response_content: str) -> List[str]:
        """누락된 항목 추출"""
        try:
            result = json.loads(response_content)
            
            missing_items = []
            if isinstance(result, dict):
                # 딕셔너리에서 리스트 값을 찾아 추출
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
                # 마침표 표준화
                if not item.endswith('.'):
                    item += '.'
                unique_items.append(item)
        
        return unique_items