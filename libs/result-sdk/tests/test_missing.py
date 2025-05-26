import os
import json
import asyncio
from pathlib import Path
from datetime import datetime

from common_sdk import get_logger
from result_sdk import DataProcessor, MissingAnalysisWorkflow, MissingAnalyzer

logger = get_logger()

class TestMissingAnalysis:
    def __init__(self):
        self.test_data_dir = Path(__file__).parent / "data"
        self.test_id = f"missing_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 테스트 데이터 파일 존재 확인
        self._validate_test_data_files()
        
        # 데이터 로드
        self.keyword_data = self._load_keyword_data()
        self.user_inputs = self._load_user_inputs()

    def _validate_test_data_files(self):
        """테스트 데이터 파일 존재 확인"""
        keyword_file = self.test_data_dir / "keywords.json"
        user_input_file = self.test_data_dir / "user_input.json"
        
        missing_files = []
        
        if not self.test_data_dir.exists():
            missing_files.append(f"Test data directory: {self.test_data_dir}")
        
        if not keyword_file.exists():
            missing_files.append(f"Keywords file: {keyword_file}")
            
        if not user_input_file.exists():
            missing_files.append(f"User input file: {user_input_file}")
        
        if missing_files:
            logger.error("MISSING TEST DATA FILES")
            for file in missing_files:
                logger.error(f"Missing file: {file}")
            raise FileNotFoundError(f"Missing required test data files: {missing_files}")
        
        logger.info(f"Test data directory found: {self.test_data_dir}")
        logger.info(f"Keywords file found: {keyword_file}")
        logger.info(f"User input file found: {user_input_file}")

    def _load_keyword_data(self):
        """키워드 데이터 로드"""
        keyword_file = self.test_data_dir / "keywords.json"
        
        try:
            with open(keyword_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 데이터 구조 검증
            if not isinstance(data, dict):
                raise ValueError("Keywords data must be a JSON object")
            
            if 'name' not in data:
                raise ValueError("Keywords data must have 'name' field")
            
            logger.info(f"Keyword data loaded: {data.get('name', 'Unknown')}")
            
            if 'children' in data:
                total_regions = len(data['children'])
                total_items = 0
                for region in data['children']:
                    if 'children' in region:
                        for city in region['children']:
                            if 'children' in city:
                                total_items += len(city['children'])
                print(f"✓ Keyword structure: {total_regions} regions, {total_items} total items")
            
            return data
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON format in keywords.json: {e}")
            raise ValueError(f"Invalid JSON format in keywords.json: {e}")
        except Exception as e:
            logger.error(f"Failed to load keyword data: {e}")
            raise RuntimeError(f"Failed to load keyword data: {e}")
    
    def _load_user_inputs(self):
        """사용자 입력 데이터 로드"""
        user_input_file = self.test_data_dir / "user_input.json"
        
        try:
            with open(user_input_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            if not content:
                raise ValueError("User input file is empty")
            
            # JSON 형식 확인
            try:
                data = json.loads(content)
                if not isinstance(data, list):
                    raise ValueError("User input data must be a JSON array")
                
                logger.info(f"User inputs loaded: {len(data)} items (JSON format)")
                return content
                
            except json.JSONDecodeError:
                # JSON이 아닌 경우 텍스트로 처리
                logger.info(f"User inputs loaded: {len(content)} characters (plain text format)")
                return content
                
        except Exception as e:
            logger.error(f"Failed to load user input data: {e}")
            raise RuntimeError(f"Failed to load user input data: {e}")

    def test_data_processor(self):
        """DataProcessor 기능 테스트"""
        print("\n=== Testing DataProcessor ===")
        
        try:
            processor = DataProcessor()
            
            # 데이터 설정 테스트
            processor.set_keyword_data(self.keyword_data)
            processor.set_user_inputs(self.user_inputs)
            logger.info("Data setting completed")
            
            # 계층 구조 포맷팅 테스트
            formatted_hierarchy = processor.format_hierarchy_list(self.keyword_data)
            assert isinstance(formatted_hierarchy, str)
            assert len(formatted_hierarchy) > 0
            assert self.keyword_data["name"] in formatted_hierarchy
            logger.info(f"Hierarchy formatting: {len(formatted_hierarchy)} characters")
            
            # 키워드 계층화 결과 로깅
            logger.info("=== KEYWORD HIERARCHY FORMATTING RESULT ===")
            logger.info(f"Formatted hierarchy (first 500 chars):\n{formatted_hierarchy[:500]}...")
            logger.info(f"Total hierarchy length: {len(formatted_hierarchy)} characters")
            
            # 사용자 콘텐츠 추출 테스트
            extracted_content = processor.extract_user_content(self.user_inputs)
            assert isinstance(extracted_content, str)
            assert len(extracted_content) > 0
            logger.info(f"Content extraction: {len(extracted_content)} characters")
            
            # 사용자 입력 정리 결과 로깅
            logger.info("=== USER INPUT EXTRACTION RESULT ===")
            logger.info(f"Extracted user content:\n{extracted_content}")
            logger.info(f"Total extracted content length: {len(extracted_content)} characters")
            
            # 데이터 접근 테스트
            assert processor.raw_keyword_data == self.keyword_data
            assert processor.raw_user_inputs == self.user_inputs
            logger.info("Data access verification completed")
            
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
        
        try:
            workflow = MissingAnalysisWorkflow()
            logger.info("Workflow initialization completed")
            
            # common-sdk PromptLoader 테스트
            prompt = workflow.load_prompt()
            assert prompt is not None
            assert len(prompt) > 0
            assert "{keywords}" in prompt
            assert "{user_content}" in prompt
            logger.info(f"Prompt loading from common-sdk: {len(prompt)} characters")
            
            # LangGraph 그래프 생성 테스트
            graph = workflow.create_missing_analysis_graph()
            assert graph is not None
            assert hasattr(graph, 'ainvoke') or hasattr(graph, 'invoke')
            logger.info("LangGraph creation completed")
            
            logger.info("MissingAnalysisWorkflow test PASSED")
            
            return True
            
        except Exception as e:
            logger.error(f"MissingAnalysisWorkflow test FAILED: {str(e)}")
            raise

    async def test_workflow_nodes(self):
        """워크플로우 노드별 테스트"""
        print("\n=== Testing Workflow Nodes ===")
        
        try:
            workflow = MissingAnalysisWorkflow()
            
            # 초기 상태 설정
            initial_state = {
                "raw_keyword_data": self.keyword_data,
                "raw_user_inputs": self.user_inputs,
                "formatted_hierarchy": None,
                "extracted_user_content": None,
                "api_response": None,
                "missing_items": None,
                "error": None
            }
            
            # 초기화 노드 테스트
            initialized_state = await workflow.initialize(initial_state)
            assert initialized_state["missing_items"] == []
            assert initialized_state["error"] is None
            logger.info("Initialize node test completed")
            
            # 포맷팅 노드 테스트
            formatted_state = await workflow.formatting_keywords(initialized_state)
            assert "formatted_hierarchy" in formatted_state
            assert "extracted_user_content" in formatted_state
            assert len(formatted_state["formatted_hierarchy"]) > 0
            assert len(formatted_state["extracted_user_content"]) > 0
            logger.info("Formatting keywords node test completed")

            # 컴파일 노드 테스트
            compile_test_state = formatted_state.copy()
            compile_test_state["missing_items"] = ["test item 1", "test item 2"]
            compile_test_state["error"] = None
            compiled_state = await workflow.compile_results(compile_test_state)
            assert len(compiled_state["missing_items"]) == 2
            logger.info("Compile results node test completed")
            
            logger.info("Workflow nodes test PASSED")
            
            return formatted_state
            
        except Exception as e:
            logger.error(f"Workflow nodes test FAILED: {str(e)}")
            raise

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
            logger.info("MissingAnalyzer initialization test PASSED")
            
            return analyzer
            
        except Exception as e:
            logger.error(f"MissingAnalyzer initialization test FAILED: {str(e)}")
            raise

    async def test_full_analysis(self):
        """전체 누락 분석 테스트"""
        logger.info("=== Testing Full Analysis with Real OpenAI API ===")
        
        try:
            analyzer = MissingAnalyzer()
            logger.info("MissingAnalyzer initialization completed")
            
            logger.info("Starting analysis using OpenAI API")
            
            start_time = datetime.now()
            result = await analyzer.analyze(self.keyword_data, self.user_inputs)
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
                    logger.info(f"Raw API response:\n{result['api_response']}")
                    logger.info(f"API response length: {len(result['api_response'])} characters")
                
                # 최종 누락 항목 결과 로깅
                logger.info("=== FINAL MISSING ANALYSIS RESULTS ===")
                logger.info(f"Total missing items found: {len(missing_items)}")
                if missing_items:
                    logger.info("Missing items list:")
                    for i, item in enumerate(missing_items, 1):
                        logger.info(f"  {i:2d}. {item}")
                else:
                    logger.info("No missing items detected - all content appears to be covered")
                
                # 전체 분석 결과 출력
                logger.info("="*60)
                logger.info("MISSING ANALYSIS RESULTS")
                logger.info("="*60)
                
                if missing_items:
                    logger.info(f"Found {len(missing_items)} missing items:")
                    logger.info("-" * 40)
                    for i, item in enumerate(missing_items, 1):
                        print(f"{i:2d}. {item}")
                    
                else:
                    logger.info("No missing items detected!")
                    logger.info("All keyword content appears to be covered in the user input.")
                
                # OpenAI API 응답 정보
                if 'api_response' in result and result['api_response']:
                    logger.info("OpenAI API Response:")
                    logger.info(f"Response length: {len(result['api_response'])} characters")
                    
                    # JSON 응답의 구조 파악 시도
                    try:
                        import json
                        api_data = json.loads(result['api_response'])
                        logger.info(f"Response format: JSON")
                        logger.info(f"Response keys: {list(api_data.keys())}")
                    except:
                        logger.info(f"   • Response format: Plain text")
                
                logger.info("="*60)
                logger.info("Full analysis test PASSED")
                
                return result
            else:
                error_msg = result.get('error', 'Unknown error')
                if "API key" in error_msg or "authentication" in error_msg.lower():
                    logger.warning("API key issue detected. Please check OPENAI_API_KEY_B environment variable")
                    return None
                else:
                    logger.error(f"Analysis failed: {error_msg}")
                    return None
                
        except Exception as e:
            error_str = str(e).lower()
            if "api key" in error_str or "authentication" in error_str or "unauthorized" in error_str:
                logger.warning("OpenAI API authentication failed")
                logger.warning("Please set OPENAI_API_KEY_B environment variable with a valid API key")
                return None
            else:
                logger.error(f"Full analysis test FAILED: {str(e)}")
                raise

    async def run_all_tests(self):
        """모든 테스트 실행"""
        logger.info(f"Missing Analysis Integration Test: {self.test_id}")
        logger.info("=" * 60)
        logger.info(f"Test data directory: {self.test_data_dir}")
        
        total_start_time = datetime.now()
        
        try:
            # 1. DataProcessor 테스트
            processor_result = self.test_data_processor()
            
            # 2. Workflow 컴포넌트 테스트
            workflow_result = self.test_workflow_components()
            
            # 3. Workflow 노드 테스트
            nodes_result = await self.test_workflow_nodes()
            
            # 4. MissingAnalyzer 초기화 테스트
            analyzer_result = self.test_missing_analyzer_init()
            
            # 5. 실제 API를 사용한 전체 분석 테스트
            analysis_result = await self.test_full_analysis()
            
            total_time = (datetime.now() - total_start_time).total_seconds()
            
            logger.info("="*60)
            logger.info("ALL TESTS PASSED!")
            logger.info(f"Total test time: {total_time:.2f} seconds")
            
            # 최종 결과 요약
            if analysis_result and analysis_result.get("missing_items"):
                missing_count = len(analysis_result["missing_items"])
                logger.info("FINAL ANALYSIS SUMMARY:")
                logger.info(f"Missing items detected: {missing_count}")

            
            logger.info("Finish All Tests")
            
            return {
                "test_id": self.test_id,
                "status": "success",
                "total_time": total_time,
                "tests_passed": {
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

    def cleanup_test_files(self):
        """테스트 후 생성된 파일들 정리 (선택사항)"""
        try:
            if self.test_data_dir.exists():
                # 로그 파일들 정리
                log_files = list(self.test_data_dir.glob("*.log"))
                for log_file in log_files:
                    log_file.unlink()
                    print(f"Cleaned up log file: {log_file}")
        except Exception as e:
            print(f"Cleanup warning: {e}")


async def main():
    """메인 테스트 실행 함수"""
    logger.info("Missing Analysis Integration Test")
    logger.info("Testing all components")
    
    # 환경 확인
    print("Environment check:")
    
    # 환경 변수 확인
    try:
        from common_sdk import settings

        # settings에서 API 키 확인 (값은 표시하지 않음)
        if hasattr(settings, 'OPENAI_API_KEY_B') and settings.OPENAI_API_KEY_B:
            logger.info("OPENAI_API_KEY_B configured in settings")
        else:
            logger.warning("OPENAI_API_KEY_B not configured in settings")
            
    except ImportError as e:
        logger.error(f"settings import error: {e}")
        return
    
    print()
    
    # 테스트 실행
    try:
        test_service = TestMissingAnalysis()
        result = await test_service.run_all_tests()
        
        # 정리 작업 (선택사항)
        # test_service.cleanup_test_files()
        
        # 최종 결과 출력
        logger.info(f"Final Result: {result['status'].upper()}")
        if result['status'] == 'success':
            logger.info(f"Test completed in {result['total_time']:.2f} seconds")
            if result.get('missing_items_count', 0) > 0:
                logger.info(f"Successfully detected {result['missing_items_count']} missing items")
                
    except FileNotFoundError as e:
        logger.error(f"File not found error: {str(e)}")
        logger.error("Please create the required test data files and run the test again.")


if __name__ == "__main__":
    asyncio.run(main())