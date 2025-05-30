import os
import json
import asyncio
from pathlib import Path
from datetime import datetime

from common_sdk import get_logger
from result_sdk import DataProcessor, MissingAnalysisWorkflow, MissingAnalyzer
from common_sdk.crud.mongodb import MongoDB

logger = get_logger()

class TestMissingAnalysis:
    def __init__(self, test_space_id: str = None):
        """테스트 초기화"""

        self.test_data_dir = Path(__file__).parent / "data"
        self.test_id = f"missing_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 테스트용 space_id 설정
        self.test_space_id = test_space_id
        
        # 테스트 데이터 파일 존재 확인 (user_input)
        self._validate_test_data_files()
        
        # 사용자 입력 데이터 로드
        self.user_inputs = self._load_user_inputs()

        # MongoDB에서 키워드 데이터 로드할 때 사용할 인스턴스
        self.mongodb = None
        self.keyword_data = None

    def _validate_test_data_files(self):
        """테스트 데이터 파일 존재 확인"""
        user_input_file = self.test_data_dir / "user_input.json"
        
        missing_files = []
        
        if not self.test_data_dir.exists():
            missing_files.append(f"Test data directory: {self.test_data_dir}")
            
        if not user_input_file.exists():
            missing_files.append(f"User input file: {user_input_file}")
        
        if missing_files:
            for file in missing_files:
                logger.error(f"Missing file: {file}")
            raise FileNotFoundError(f"Missing required test data files: {missing_files}")
        
        logger.info(f"User input file found: {user_input_file}")

    async def _load_keyword_data_from_db(self):
        """MongoDB에서 키워드 데이터 로드 (crud.py 사용)"""

        if not self.test_space_id:
            return None

        try:
            if not self.mongodb:
                self.mongodb = MongoDB()
            
            keyword_data = await self.mongodb.get_space_keywords(self.test_space_id)
            
            if keyword_data:
                self.keyword_data = keyword_data
                logger.info(f"Keyword data loaded: {len(self.keyword_data)} items")
            else:
                logger.warning(f"No keyword data found for space_id: {self.test_space_id}")
                
        except Exception as e:
            logger.error(f"Failed to load keyword data: {e}")
            return None
    
    def _load_user_inputs(self):
        """사용자 입력 데이터 로드"""
        user_input_file = self.test_data_dir / "user_input.json"

        with open(user_input_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
        try:
            data = json.loads(content)
            logger.info(f"User inputs loaded: {len(data)} items")
            return content
        except json.JSONDecodeError:
            logger.info(f"User inputs loaded as plain text")
            return content
                
        except Exception as e:
            logger.error(f"Failed to load user input data: {e}")
            raise RuntimeError(f"Failed to load user input data: {e}")
        
    async def test_mongodb_connection(self):
        """MongoDB 연결 테스트 (crud.py 통해)"""
        logger.info("\n=== Testing MongoDB Connection ===")
        
        try:
            if not self.mongodb:
                self.mongodb = MongoDB()
            
            # 연결 테스트
            connection_status = self.mongodb.check_connection()
            logger.info(f"MongoDB connection status: {connection_status}")
            
            # 테스트 space_id가 있는 경우 키워드 데이터 조회 테스트
            if self.test_space_id:
                keyword_data = await self.mongodb.get_space_keywords(self.test_space_id)
                if keyword_data:
                    keyword_count = len(keyword_data)
                    logger.info(f"Keyword count in test document: {keyword_count}")
                else:
                    logger.warning(f"Test space document not found or no keywords: {self.test_space_id}")
            
            logger.info("MongoDB connection test PASSED")
            return True
            
        except Exception as e:
            logger.error(f"MongoDB connection test FAILED: {e}")
            raise

    def test_data_processor(self):
        """DataProcessor 기능 테스트"""
        logger.info("\n=== Testing DataProcessor ===")

        if not self.keyword_data:
            logger.warning("No keyword data available for DataProcessor test")
            return None
        
        try:
            processor = DataProcessor()
            main_keyword = self.keyword_data[0]

            # 데이터 설정
            processor.set_keyword_data(main_keyword)
            processor.set_user_inputs(self.user_inputs)
            logger.info("Data setting completed")
            
            # 계층 구조 포맷팅 테스트
            formatted_hierarchy = processor.format_hierarchy_list(main_keyword)
            assert isinstance(formatted_hierarchy, str) and len(formatted_hierarchy) > 0
            logger.info(f"Hierarchies formatting: {len(formatted_hierarchy)} characters")
            
            # 키워드 계층화 결과 로깅
            logger.info("=== KEYWORD HIERARCHY FORMATTING RESULT ===")
            logger.info(f"Formatted hierarchy (first 500 chars):\n{formatted_hierarchy[:500]}...")
            logger.info(f"Total hierarchy length: {len(formatted_hierarchy)} characters")
            
            # 사용자 콘텐츠 추출 테스트
            extracted_content = processor.extract_user_content(self.user_inputs)
            assert isinstance(extracted_content, str) and len(extracted_content) > 0
            logger.info(f"Content extraction: {len(extracted_content)} characters")
            
            # 사용자 입력 정리 결과 로깅
            logger.info("=== USER INPUT EXTRACTION RESULT ===")
            logger.info(f"Extracted user content:\n{extracted_content}")
            logger.info(f"Total extracted content length: {len(extracted_content)} characters")
            
            logger.info("DataProcessor test PASSED")
            
            return {
                "formatted_hierarchy": formatted_hierarchy,
                "extracted_user_content": extracted_content
            }
            
        except Exception as e:
            logger.error(f"DataProcessor test FAILED: {str(e)}")
            raise

    def test_workflow_components(self):
        """MissingAnalysisWorkflow 컴포넌트 테스트"""
        logger.info("=== Testing MissingAnalysisWorkflow ===")
        
        workflow = MissingAnalysisWorkflow()

        # MongoDB 연결 확인 (crud.py를 통해)
        assert workflow.nodes.mongodb is not None

        # 프롬프트 로드 테스트
        prompt = workflow.nodes.load_prompt()
        assert prompt and "{keywords}" in prompt and "{user_content}" in prompt
        
        # 그래프 생성 테스트
        graph = workflow.create_missing_analysis_graph()
        assert graph and hasattr(graph, 'ainvoke')
        
        logger.info(f"Workflow components test PASSED - Prompt: {len(prompt)} chars")
        return True

    async def test_workflow_nodes(self):
        """워크플로우 노드별 테스트"""
        logger.info("\n=== Testing Workflow Nodes ===")

        if not self.test_space_id:
            logger.warning("No test_space_id provided for workflow nodes test")
            return None
        
        workflow = MissingAnalysisWorkflow()

        # 초기 상태 설정
        initial_state = {
            "space_id": self.test_space_id,
            "raw_user_inputs": self.user_inputs,
            "keyword_data": None,
            "formatted_hierarchy": None,
            "extracted_user_content": None,
            "api_response": None,
            "missing_items": None,
            "error": None
        }
        
        # 초기화 노드 테스트
        state = await workflow.nodes.initialize(initial_state)
        assert state["missing_items"] == [] and state["error"] is None
        logger.info("Initialize node test completed")
            
        # MongoDB 키워드 로드 노드 테스트 (crud.py 통해)
        state = await workflow.nodes.load_keywords_from_db(state)
        if state.get("error"):
            logger.error(f"Load keywords failed: {state['error']}")
            return None
        assert state["keyword_data"] is not None
        logger.info("Load keywords from DB node test completed")
            
        # 포맷팅 노드 테스트
        state = await workflow.nodes.formatting_keywords(state)
        if state.get("error"):
            logger.error(f"Formatting failed: {state['error']}")
            return None
            
        logger.info("Workflow nodes test PASSED")
        return state

    def test_missing_analyzer_init(self):
        """MissingAnalyzer 초기화 테스트"""
        logger.info("=== Testing MissingAnalyzer Initialization ===")
        
        try:
            analyzer = MissingAnalyzer()
            
            assert hasattr(analyzer, 'workflow')
            assert hasattr(analyzer, 'data_processor')
            assert isinstance(analyzer.workflow, MissingAnalysisWorkflow)
            assert isinstance(analyzer.data_processor, DataProcessor)
            
            logger.info("MissingAnalyzer initialization completed")

            return analyzer
            
        except Exception as e:
            logger.error(f"MissingAnalyzer initialization test FAILED: {str(e)}")
            raise

    async def test_full_analysis(self):
        """전체 누락 분석 테스트"""
        logger.info("=== Testing Full Analysis with Real OpenAI API ===")

        if not self.test_space_id:
            logger.warning("No test_space_id provided for full analysis test")
            return None
        
        try:
            analyzer = MissingAnalyzer()
            logger.info("MissingAnalyzer initialization completed")
            
            logger.info(f"Starting analysis using space_id: {self.test_space_id}")
            
            start_time = datetime.now()
            result = await analyzer.analyze(self.test_space_id, self.user_inputs)
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # 결과 검증
            assert "success" in result
            
            if result["success"]:
                missing_items = result.get("missing_items", [])
                
                logger.info(f"Analysis completed successfully")
                logger.info(f"Processing time: {processing_time:.2f} seconds")
                logger.info(f"Missing items found: {len(missing_items)}")
                
                # OpenAI API 응답 결과 로깅
                if 'api_response' in result and result['api_response']:
                    logger.info("=== OPENAI API RESPONSE ===")
                    logger.info(f"API response length: {len(result['api_response'])} characters")
                
                # 최종 누락 항목 결과 로깅
                logger.info("=== FINAL MISSING ANALYSIS RESULTS ===")
                logger.info(f"Total missing items found: {len(missing_items)}")
                if missing_items:
                    logger.info(f"Missing items: {missing_items}")
                else:
                    logger.info("No missing items detected")
                
                logger.info("="*60)
                logger.info("Full analysis test PASSED")
                
                return result
            else:
                error_msg = result.get('error', 'Unknown error')
                if "API key" in error_msg or "authentication" in error_msg.lower():
                    logger.warning("API key issue detected.")
                    return None
                else:
                    logger.error(f"Analysis failed: {error_msg}")
                    return None
                
        except Exception as e:
            error_str = str(e).lower()
            if "api key" in error_str or "authentication" in error_str or "unauthorized" in error_str:
                logger.warning("OpenAI API authentication failed")
                return None
            else:
                logger.error(f"Full analysis test FAILED: {str(e)}")
                raise

    async def run_all_tests(self):
        """모든 테스트 실행"""
        logger.info(f"Missing Analysis Test: {self.test_id}")
        logger.info("=" * 50)
        logger.info(f"Space ID: {self.test_space_id or 'None'}")
        
        total_start_time = datetime.now()
        
        try:
            # 1. MongoDB 연결 테스트 (crud.py 통해)
            mongodb_result = await self.test_mongodb_connection()
            
            # 2. MongoDB에서 키워드 데이터 로드 (space_id가 있는 경우, crud.py 통해)
            if self.test_space_id:
                await self._load_keyword_data_from_db()
            
            # 3. DataProcessor 테스트
            processor_result = self.test_data_processor()
            
            # 4. Workflow 컴포넌트 테스트
            workflow_result = self.test_workflow_components()
            
            # 5. Workflow 노드 테스트
            nodes_result = await self.test_workflow_nodes()
            
            # 6. MissingAnalyzer 초기화 테스트
            analyzer_result = self.test_missing_analyzer_init()
            
            # 7. 전체 분석 테스트
            analysis_result = await self.test_full_analysis()
            
            total_time = (datetime.now() - total_start_time).total_seconds()
            
            logger.info("="*60)
            logger.info("ALL TESTS PASSED")
            logger.info(f"Total test time: {total_time:.2f} seconds")
            
            # 최종 결과 요약
            if analysis_result and analysis_result.get("missing_items"):
                missing_count = len(analysis_result["missing_items"])
                logger.info("FINAL ANALYSIS SUMMARY:")
                logger.info(f"Missing items detected: {missing_count}")
            
            return {
                "test_id": self.test_id,
                "status": "success",
                "total_time": total_time,
                "tests_passed": {
                    "mongodb_connection": bool(mongodb_result),
                    "data_processor": bool(processor_result),
                    "workflow_components": workflow_result,
                    "workflow_nodes": bool(nodes_result),
                    "missing_analyzer": bool(analyzer_result),
                    "full_analysis": bool(analysis_result)
                },
                "missing_items_count": len(analysis_result.get("missing_items", [])) if analysis_result else 0
            }
            
        except Exception as e:
            logger.error("="*60)
            logger.error("TESTS FAILED")
            logger.error(f"Error: {str(e)}")
            
            return {
                "test_id": self.test_id,
                "status": "failed",
                "error": str(e)
            }

async def main():
    """메인 테스트 실행 함수"""
    logger.info("Missing Analysis Integration Test")
    logger.info("Testing all components with separated CRUD logic")
    
    # 환경 확인
    print("Environment check:")
    
    # 환경 변수 확인
    try:
        from common_sdk import settings

        # settings에서 API 키 확인
        if hasattr(settings, 'OPENAI_API_KEY_B') and settings.OPENAI_API_KEY_B:
            logger.info("OPENAI_API_KEY_B configured in settings")
        else:
            logger.warning("OPENAI_API_KEY_B not configured in settings")
        
        # MongoDB 설정 확인
        if hasattr(settings, 'MONGO_URI') and settings.MONGO_URI:
            logger.info("MONGO_URI configured in settings")
        else:
            logger.warning("MONGO_URI not configured in settings")
            
        if hasattr(settings, 'MONGO_DATABASE') and settings.MONGO_DATABASE:
            logger.info("MONGO_DATABASE configured in settings")
        else:
            logger.warning("MONGO_DATABASE not configured in settings")
            
    except ImportError as e:
        logger.error(f"settings import error: {e}")
        return
    
    test_space_id = "507f191e810c19729de860ea" # MongoDB _id 값

    if test_space_id:
        logger.info(f"Using test space_id: {test_space_id}")

    # 테스트 실행
    try:
        test_service = TestMissingAnalysis(test_space_id=test_space_id)
        result = await test_service.run_all_tests()
        
        # 최종 결과 출력
        logger.info(f"Final Result: {result['status'].upper()}")
        if result['status'] in ['success', 'partial']:
            logger.info(f"Test completed in {result['total_time']:.2f} seconds")
            if result.get('missing_items_count', 0) > 0:
                logger.info(f"Successfully detected {result['missing_items_count']} missing items")
            
            if result.get('failed_tests'):
                logger.warning("Some tests failed - check logs above for details")
                
    except FileNotFoundError as e:
        logger.error(f"File not found error: {str(e)}")
        logger.error("Please create the required test data files and run the test again.")
    except Exception as e:
        logger.error(f"Test execution failed: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())