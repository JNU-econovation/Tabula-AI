from typing import Dict, Any, List
from pathlib import Path
import fitz
import asyncio
import json
import shutil

from common_sdk import get_logger
from common_sdk.crud.s3 import S3Storage
from common_sdk.crud.mongodb import MongoDB
from common_sdk.sse import update_result_progress

from result_sdk import settings as result_sdk_settings
from result_sdk import process_document
from result_sdk import draw_underlines_for_incorrect_answers_enhanced
from result_sdk.grading import GradingService
from result_sdk.missing import MissingAnalyzer
from common_sdk.prompt_loader import PromptLoader
import google.generativeai as genai

# 로거 설정
logger = get_logger()

# MongoDB 인스턴스
mongodb = MongoDB()

class ResultService:
    def __init__(self, result_id: str, file_paths: List[str], file_name: str, user_id: int, space_id: str):
        self.result_id = result_id
        self.file_paths = file_paths if isinstance(file_paths, list) else [file_paths]
        self.file_name = file_name
        self.user_id = user_id
        self.space_id = space_id
        
        # 의존성 인스턴스들
        self.s3_storage = S3Storage()
        self.mongodb = MongoDB()
        
        # 처리 결과 저장 변수들
        self.png_files: List[str] = []
        self.origin_urls: List[Dict] = []
        self.ocr_results: List[str] = [] # GradingService 입력용 (JSON 문자열화된 RAG 데이터)
        self.all_consolidated_data: List[Dict] = [] # process_document 결과 (시각화용)
        self.all_rag_ready_data: List[List[Any]] = [] # process_document 결과 (시각화 및 채점 입력용)
        self.wrong_answers: List[Dict] = []
        self.wrong_answer_ids: List[List[int]] = [] # draw_underlines_for_incorrect_answers_enhanced 입력용
        self.missing_answers: List[str] = []
        self.highlight_urls: List[str] = []
        self.keyword_data: Dict = None
        self.db_result_id: str = None
    
    async def process_grading(self) -> Dict[str, Any]:
        """메인 처리 로직 (SSE 적용)"""
        try:
            logger.info(f"Result: {self.result_id} - Starting grading process")
            
            # 0. 시작 시점 (0%)
            update_result_progress(self.result_id, 0)
            await asyncio.sleep(0.1)
            
            # 1. PDF → PNG 분리 (0% - 10%)
            await self.convert_pdf_to_png()
            update_result_progress(self.result_id, 10)
            await asyncio.sleep(0.5)
            
            # 2. S3에 원본 저장 (10% - 20%)
            await self.upload_origin_files()
            update_result_progress(self.result_id, 20)
            await asyncio.sleep(0.5)
            
            # 3. OCR + LLM 처리 (20% - 30%)
            await self.process_ocr_and_llm()
            update_result_progress(self.result_id, 30)
            await asyncio.sleep(0.5)
            
            # 4. MongoDB에서 lang_type 조회 (30% - 40%)
            lang_type = await self.get_language_type()
            update_result_progress(self.result_id, 40)
            await asyncio.sleep(0.5)
            
            # 5. MongoDB에서 키워드 데이터 조회 (40% - 50%)
            await self.get_keyword_data()
            update_result_progress(self.result_id, 50)
            await asyncio.sleep(0.5)
            
            # 6. 오답 채점과 누락 판단을 병렬로 처리 (50% - 70%)
            await asyncio.gather(
                self.grade_wrong_answers(lang_type),
                self.detect_missing_answers()
            )
            update_result_progress(self.result_id, 70)
            await asyncio.sleep(0.5)
            
            # 7. 오답 하이라이트 이미지 생성 (70% - 80%)
            await self.generate_highlight_images()
            update_result_progress(self.result_id, 80)
            await asyncio.sleep(0.5)
            
            # 8. 하이라이트 이미지 S3 저장 (80% - 90%)
            await self.upload_highlight_images()
            
            # 9. post_image_url 업데이트
            self.update_post_image_urls()
            update_result_progress(self.result_id, 90)
            await asyncio.sleep(0.5)
            
            # 10. MongoDB에 결과 저장 (90% - 100%)
            await self.save_results_to_db()
            
            # 11. 임시 파일 정리
            self.cleanup_temp_files()
            
            # 12. 최종 완료 (100%) - API 명세에 맞는 Complete 데이터 구성
            results_array = []
            for highlight_data in self.highlight_urls:
                results_array.append({
                    "id": highlight_data.get("id"),
                    "postImageUrl": highlight_data.get("url", "")
                })

            results_array.sort(key=lambda x: x["id"])
            
            complete_data = {
                "resultId": self.db_result_id or self.result_id,
                # "status": "complete",
                "progress": 100,
                "results": results_array
            }
            
            update_result_progress(self.result_id, 100, complete_data)
            
            logger.info(f"Result: {self.result_id} - Grading process completed successfully")
            return self.build_response()
            
        except Exception as e:
            logger.error(f"Result: {self.result_id} - Grading process failed: {str(e)}")
            
            # 에러 발생 시 진행률 -1로 설정
            error_data = {"error": str(e)}
            update_result_progress(self.result_id, -1, error_data)
            
            # 에러 발생 시에도 임시 파일 정리
            self.cleanup_temp_files()
            raise
    
    def cleanup_temp_files(self):
        """임시 파일들 정리"""
        try:
            if self.file_paths:
                # 첫 번째 파일의 부모 디렉토리 (전체 result_id 디렉토리)
                temp_dir = Path(self.file_paths[0]).parent
                
                if temp_dir.exists() and temp_dir.name == self.result_id:
                    shutil.rmtree(temp_dir)
                    logger.info(f"Result: {self.result_id} - Cleaned up temp directory: {temp_dir}")
                
        except Exception as e:
            logger.error(f"Result: {self.result_id} - Error cleaning up temp files: {str(e)}")
    
    # result-sdk 이전 예정
    async def convert_pdf_to_png(self):
        """PDF → PNG 분리 또는 PNG 파일들 정리"""
        try:
            # 여러 파일이 업로드된 경우
            if len(self.file_paths) > 1:
                logger.info(f"Result: {self.result_id} - Multiple PNG files provided: {len(self.file_paths)} files")
                # 모든 파일이 PNG인지 확인하고 그대로 사용
                for file_path in self.file_paths:
                    path = Path(file_path)
                    if path.suffix.lower() not in ['.png', '.jpg', '.jpeg']:
                        raise Exception(f"여러 파일 업로드 시에는 이미지 파일만 허용됩니다: {path.name}")
                
                self.png_files = self.file_paths.copy()
                logger.info(f"Result: {self.result_id} - Using {len(self.png_files)} image files directly")
                return
            
            # 단일 파일인 경우
            file_path = Path(self.file_paths[0])
            
            # PDF가 아닌 경우 (PNG 등) 원본 파일을 그대로 사용
            if file_path.suffix.lower() != '.pdf':
                logger.info(f"Result: {self.result_id} - File is not PDF, using original file: {self.file_name}")
                self.png_files = [str(file_path)]
                return
            
            # PDF → PNG 변환 로직
            logger.info(f"Result: {self.result_id} - Converting PDF to PNG: {self.file_name}")
            
            pdf_document = fitz.open(str(file_path))
            output_dir = file_path.parent / "pages"
            output_dir.mkdir(exist_ok=True)
            
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                mat = fitz.Matrix(150/72, 150/72)
                pix = page.get_pixmap(matrix=mat)
                png_filename = f"page_{page_num + 1:03d}.png"
                png_path = output_dir / png_filename
                pix.save(str(png_path))
                self.png_files.append(str(png_path))
                logger.info(f"Result: {self.result_id} - Page {page_num + 1} converted: {png_filename}")
            
            pdf_document.close()
            logger.info(f"Result: {self.result_id} - PDF conversion completed. Total pages: {len(self.png_files)}")
            
        except Exception as e:
            logger.error(f"Result: {self.result_id} - Error in convert_pdf_to_png: {str(e)}")
            raise Exception(f"파일 처리 중 오류 발생: {str(e)}")

    async def upload_origin_files(self):
        """S3에 원본 파일들 업로드"""
        try:
            logger.info(f"Result: {self.result_id} - Starting origin files upload to S3")
            
            if not self.png_files:
                raise Exception("PNG 파일이 없습니다. convert_pdf_to_png()를 먼저 실행해주세요.")
            
            # 각 PNG 파일을 S3에 업로드
            for i, png_file_path in enumerate(self.png_files):
                page_num = i + 1  # 페이지 번호 (1부터 시작)
                
                # S3에 원본 이미지 업로드
                s3_result = self.s3_storage.upload_origin_image(
                    file_path=png_file_path,
                    user_id=self.user_id,
                    space_id=self.space_id,
                    result_id=self.result_id,
                    page=page_num
                )
                
                # 업로드 결과를 origin_urls 리스트에 저장
                origin_data = {
                    "id": page_num,
                    "s3_key": s3_result["s3_key"],
                    "bucket": s3_result["bucket"],
                    "url": s3_result["url"]
                }
                self.origin_urls.append(origin_data)
                
                logger.info(f"Result: {self.result_id} - Page {page_num} uploaded to S3: {s3_result['s3_key']}")
            
            logger.info(f"Result: {self.result_id} - All origin files uploaded successfully. Total: {len(self.origin_urls)} files")
            
        except Exception as e:
            logger.error(f"Result: {self.result_id} - Error uploading origin files to S3: {str(e)}")
            raise Exception(f"S3 원본 파일 업로드 중 오류 발생: {str(e)}")

    async def process_ocr_and_llm(self):
        """OCR + LLM 처리 (result_sdk.process_document 사용)"""
        try:
            logger.info(f"Result: {self.result_id} - Starting OCR + LLM processing using result_sdk.process_document")
            
            if not self.png_files:
                if not self.file_paths:
                     raise Exception("No input file paths available for OCR+LLM processing.")
                input_for_processing = self.file_paths[0] 
                logger.info(f"Result: {self.result_id} - No pre-converted PNGs (self.png_files empty), using original input: {input_for_processing}")
            else:
                input_for_processing = self.file_paths[0]
                logger.info(f"Result: {self.result_id} - Using original input for process_document: {input_for_processing}, expecting it to use pre-converted PNGs if applicable.")

            if not result_sdk_settings.GOOGLE_API_KEY:
                raise ValueError("GOOGLE_API_KEY is not set in result_sdk.config.settings")
            genai.configure(api_key=result_sdk_settings.GOOGLE_API_KEY)

            try:
                prompt_loader = PromptLoader()
                prompt_data = prompt_loader.load_prompt('ocr-prompt') 
                loaded_prompt_template = prompt_data['template']
                logger.info(f"Result: {self.result_id} - OCR Prompt loaded successfully via PromptLoader.")
            except Exception as e_prompt:
                logger.error(f"Result: {self.result_id} - Failed to load OCR-PROMPT via PromptLoader: {e_prompt}", exc_info=True)
                loaded_prompt_template = "Error loading prompt. OCR Data: {ocr_chunk_list_placeholder}"

            processed_data = process_document(
                input_file_path=input_for_processing,
                service_account_file=result_sdk_settings.SERVICE_ACCOUNT_FILE,
                temp_base_dir=result_sdk_settings.BASE_TEMP_DIR,
                prompt_template=loaded_prompt_template,
                generation_config=result_sdk_settings.GENERATION_CONFIG,
                safety_settings=result_sdk_settings.SAFETY_SETTINGS,
                pre_converted_image_paths=self.png_files # 이미 변환된 PNG 파일 경로 전달
            )
            
            self.all_consolidated_data, self.all_rag_ready_data, image_files_actually_used, _ = processed_data

            if image_files_actually_used and not self.png_files:
                logger.warning(f"Result: {self.result_id} - self.png_files was empty but process_document returned image paths. This is unexpected.")
                self.png_files = image_files_actually_used
            elif image_files_actually_used and set(self.png_files) != set(image_files_actually_used):
                 logger.info(f"Result: {self.result_id} - Image paths from convert_pdf_to_png and process_document differ. Using paths from convert_pdf_to_png.")

            self.ocr_results = json.dumps(self.all_rag_ready_data, ensure_ascii=False) if self.all_rag_ready_data else "[]"
            
            logger.info(f"Result: {self.result_id} - OCR + LLM processing completed. Consolidated items: {len(self.all_consolidated_data)}, RAG items: {len(self.all_rag_ready_data)}")
            logger.info(f"RAG items: {self.all_rag_ready_data}")
            logger.info(f"RAG Data type: {self.all_rag_ready_data}")
            
        except Exception as e:
            logger.error(f"Result: {self.result_id} - Error in OCR + LLM processing: {str(e)}", exc_info=True)
            raise Exception(f"OCR + LLM 처리 중 오류 발생: {str(e)}")

    async def get_language_type(self) -> str:
        """MongoDB에서 언어 타입 조회"""
        try:
            logger.info(f"Result: {self.result_id} - Getting language type for space: {self.space_id}")
            
            # MongoDB에서 해당 space_id의 lang_type 조회
            lang_type = self.mongodb.get_space_lang_type(self.space_id)
            
            if not lang_type:
                logger.warning(f"Result: {self.result_id} - Language type not found for space: {self.space_id}")
                # 기본값으로 한국어 설정
                lang_type = "korean"
            
            logger.info(f"Result: {self.result_id} - Language type retrieved: {lang_type}")
            return lang_type
            
        except Exception as e:
            logger.error(f"Result: {self.result_id} - Error getting language type: {str(e)}")
            # 에러 발생 시 기본값으로 한국어 설정
            logger.info(f"Result: {self.result_id} - Using default language type: korean")
            return "korean"

    async def grade_wrong_answers(self, lang_type: str):
        """오답 채점 (GradingService 사용)"""
        try:
            logger.info(f"Result: {self.result_id} - Starting grading process with language: {lang_type}")
            
            if not self.ocr_results:
                logger.warning(f"Result: {self.result_id} - self.ocr_results is empty. Raising exception.")
                raise Exception("OCR 결과가 없습니다. process_ocr_and_llm()를 먼저 실행해주세요.")
            
            # OCR 결과를 GradingService 입력 형태로 변환
            user_inputs = self.ocr_results
            logger.info(f"Result: {self.result_id} - user_inputs for GradingService (type: {type(user_inputs)}): {user_inputs[:200]}...") # 타입 및 내용 일부 로깅
            
            # GradingService 인스턴스 생성
            grading_service = GradingService(
                space_id=self.space_id,
                lang_type=lang_type
            )
            
            logger.info(f"Result: {self.result_id} - Starting grading with GradingService")
            
            # 채점 실행 및 오답 ID 추출
            evaluation_response, wrong_ids = await grading_service.grade_with_wrong_ids(user_inputs)
            
            # 오답 ID 저장
            self.wrong_answer_ids = wrong_ids if wrong_ids else []
            
            # MongoDB 구조에 맞게 채점 결과 처리
            self.wrong_answers = {"results": []}
            
            if evaluation_response.results:
                logger.info(f"Result: {self.result_id} - Found evaluation results for {len(evaluation_response.results)} pages")
                
                # 페이지별로 오답 정보 처리
                for page_result in evaluation_response.results:
                    page_number = page_result.page
                    
                    # 페이지별 결과 구조 생성
                    page_data = {
                        "page": page_number,
                        "post_image_url": "",
                        "result": []
                    }
                    
                    # 해당 페이지의 오답들 처리
                    for idx, wrong_answer in enumerate(page_result.wrong_answers, 1):
                        wrong_data = {
                            "id": idx,  
                            "wrong": wrong_answer.wrong_answer,  # 틀린 답
                            "feedback": wrong_answer.feedback  # 피드백
                        }
                        page_data["result"].append(wrong_data)
                        
                        logger.info(f"Result: {self.result_id} - Page {page_number}, Wrong answer {idx}: {wrong_answer.wrong_answer}")
                    
                    self.wrong_answers["results"].append(page_data)
            
            else:
                logger.info(f"Result: {self.result_id} - No wrong answers found")
            
            logger.info(f"Result: {self.result_id} - Grading completed. Wrong answer IDs: {len(self.wrong_answer_ids)}, Pages processed: {len(self.wrong_answers['results'])}")
            
        except Exception as e:
            logger.error(f"Result: {self.result_id} - Error in grading process: {str(e)}")
            raise Exception(f"채점 처리 중 오류 발생: {str(e)}")

    async def generate_highlight_images(self):
        """오답 하이라이트 이미지 생성 (result_sdk.draw_underlines_for_incorrect_answers_enhanced 사용)"""
        try:
            logger.info(f"Result: {self.result_id} - Starting highlight image generation using result_sdk")

            if not self.wrong_answer_ids:
                logger.info(f"Result: {self.result_id} - No wrong answer IDs found, skipping highlight generation.")
                return
            if not self.all_consolidated_data:
                logger.warning(f"Result: {self.result_id} - No consolidated data available, skipping highlight generation.")
                return
            if not self.png_files: 
                logger.warning(f"Result: {self.result_id} - No image paths available (self.png_files is empty), skipping highlight generation.")
                return

            output_highlight_folder = Path(self.png_files[0]).parent / "highlights_sdk" 
            if output_highlight_folder.exists():
                shutil.rmtree(output_highlight_folder) 
            output_highlight_folder.mkdir(exist_ok=True)
            
            logger.info(f"Result: {self.result_id} - Highlight images will be saved to: {output_highlight_folder}")
            
            await asyncio.to_thread(
                draw_underlines_for_incorrect_answers_enhanced,
                incorrect_rag_ids=self.wrong_answer_ids,
                all_consolidated_data=self.all_consolidated_data,
                all_rag_ready_data=self.all_rag_ready_data, 
                page_image_paths=self.png_files, 
                output_folder=str(output_highlight_folder),
            )
            
            logger.info(f"Result: {self.result_id} - Highlight image generation completed by result_sdk.")
            
        except Exception as e:
            logger.error(f"Result: {self.result_id} - Error generating highlight images: {str(e)}", exc_info=True)
            raise Exception(f"하이라이트 이미지 생성 중 오류 발생: {str(e)}")

    async def upload_highlight_images(self):
        """하이라이트 이미지 S3 저장 (틀린 답이 없으면 원본 이미지 사용)"""
        try:
            logger.info(f"Result: {self.result_id} - Starting highlight images upload to S3")
            
            # 하이라이트 이미지 디렉토리 확인
            if not self.png_files:
                logger.info(f"Result: {self.result_id} - No PNG files found, skipping highlight upload")
                return
            
            highlight_dir = Path(self.png_files[0]).parent / "highlights_sdk" 
            highlight_files = []
            
            # 하이라이트 이미지가 있는지 확인
            if highlight_dir.exists():
                highlight_files = list(highlight_dir.glob("page_*_visualized_*.png"))
            
            self.highlight_urls = []
            
            # 하이라이트 이미지가 있는 경우
            if highlight_files:
                logger.info(f"Result: {self.result_id} - Found {len(highlight_files)} highlight images, uploading them")

                # 각 하이라이트 이미지를 S3에 업로드
                for highlight_file in highlight_files:
                    try:
                        # 파일명에서 페이지 번호 추출
                        filename = highlight_file.name
                        try:
                            page_num_str = filename.split("page_")[1].split("_")[0]
                            page_num = int(page_num_str)
                        except (IndexError, ValueError) as e_parse:
                            logger.error(f"Result: {self.result_id} - Error parsing page number from filename '{filename}': {e_parse}. Skipping this file.")
                            continue
                        
                        # S3에 하이라이트 이미지 업로드
                        s3_result = self.s3_storage.upload_post_image(
                            file_path=str(highlight_file),
                            user_id=self.user_id,
                            space_id=self.space_id,
                            result_id=self.result_id,
                            page=page_num
                        )
                        
                        # 업로드 결과를 highlight_urls 리스트에 저장
                        highlight_data = {
                            "id": page_num,
                            "s3_key": s3_result["s3_key"],
                            "bucket": s3_result["bucket"],
                            "url": s3_result["url"]
                        }
                        self.highlight_urls.append(highlight_data)
                        
                        logger.info(f"Result: {self.result_id} - Highlight image uploaded to S3: page {page_num} -> {s3_result['s3_key']}")
                        
                    except ValueError as e:
                        logger.error(f"Result: {self.result_id} - Error parsing page number from {filename}: {str(e)}")
                        continue
                    except Exception as e:
                        logger.error(f"Result: {self.result_id} - Error uploading highlight image {filename}: {str(e)}")
                        continue
            
            # 하이라이트 이미지가 없는 경우 (틀린 답이 없는 경우) 원본 이미지를 post로 업로드
            else:
                logger.info(f"Result: {self.result_id} - No highlight images found, uploading original images as post images")
                
                for i, png_file_path in enumerate(self.png_files):
                    try:
                        page_num = i + 1  # 페이지 번호 (1부터 시작)
                        
                        s3_result = self.s3_storage.upload_post_image(
                            file_path=png_file_path,
                            user_id=self.user_id,
                            space_id=self.space_id,
                            result_id=self.result_id,
                            page=page_num
                        )
                        
                        highlight_data = {
                            "id": page_num,
                            "s3_key": s3_result["s3_key"],
                            "bucket": s3_result["bucket"],
                            "url": s3_result["url"]
                        }
                        self.highlight_urls.append(highlight_data)
                        
                        logger.info(f"Result: {self.result_id} - Original image uploaded as post image to S3: page {page_num} -> {s3_result['s3_key']}")
                        
                    except Exception as e:
                        logger.error(f"Result: {self.result_id} - Error uploading original image as post image {png_file_path}: {str(e)}")
                        continue
            
            logger.info(f"Result: {self.result_id} - All post images uploaded successfully. Total: {len(self.highlight_urls)} files")
            
        except Exception as e:
            logger.error(f"Result: {self.result_id} - Error uploading post images to S3: {str(e)}")
            raise Exception(f"S3 하이라이트 이미지 업로드 중 오류 발생: {str(e)}")
    
    def update_post_image_urls(self):
        """하이라이트 이미지 업로드 완료 후 post_image_url 업데이트"""
        try:
            logger.info(f"Result: {self.result_id} - Updating post_image_urls in wrong_answers")
            
            if not self.highlight_urls or not self.wrong_answers.get("results"):
                logger.info(f"Result: {self.result_id} - No highlight URLs or wrong answers to update")
                return
            
            # 페이지별 하이라이트 URL 매핑 생성
            page_url_map = {}
            for highlight_data in self.highlight_urls:
                page_id = highlight_data.get('id')
                url = highlight_data.get('url', '')
                if page_id:
                    page_url_map[page_id] = url
            
            # wrong_answers의 각 페이지에 post_image_url 업데이트
            for page_data in self.wrong_answers["results"]:
                page_number = page_data.get("page")
                if page_number in page_url_map:
                    page_data["post_image_url"] = page_url_map[page_number]
                    logger.info(f"Result: {self.result_id} - Updated post_image_url for page {page_number}")
            
            logger.info(f"Result: {self.result_id} - Post image URLs updated successfully")
            
        except Exception as e:
            logger.error(f"Result: {self.result_id} - Error updating post_image_urls: {str(e)}")

    async def get_keyword_data(self):
        """MongoDB에서 키워드 데이터 조회"""
        try:
            logger.info(f"Result: {self.result_id} - Getting keyword data for space: {self.space_id}")
            
            keyword_data = await self.mongodb.get_space_keywords(self.space_id)
            
            if not keyword_data:
                logger.warning(f"Result: {self.result_id} - No keyword data found for space: {self.space_id}")
                self.keyword_data = None
            else:
                self.keyword_data = keyword_data

                if isinstance(keyword_data, dict):
                    children_count = len(keyword_data.get('children', []))
                    logger.info(f"Result: {self.result_id} - Retrieved keyword data with {children_count} children")
                else:
                    logger.info(f"Result: {self.result_id} - Retrieved keyword data")
            
        except Exception as e:
            logger.error(f"Result: {self.result_id} - Error getting keyword data: {str(e)}")
            self.keyword_data = None

    async def detect_missing_answers(self):
        """누락 답안 판단"""
        try:
            logger.info(f"Result: {self.result_id} - Starting missing answer detection")
            
            if not self.ocr_results:
                logger.warning(f"Result: {self.result_id} - self.ocr_results is empty for missing answers. Raising exception.")
                raise Exception("OCR 결과가 없습니다. process_ocr_and_llm()를 먼저 실행해주세요.")
            
            # 키워드 데이터가 있는지 확인
            if not self.keyword_data:
                logger.warning(f"Result: {self.result_id} - No keyword data available for missing analysis")
                self.missing_answers = []
                return
            
            user_inputs_for_missing = self.ocr_results
            logger.info(f"Result: {self.result_id} - user_inputs for MissingAnalyzer (type: {type(user_inputs_for_missing)}): {user_inputs_for_missing[:200]}...")

            # MissingAnalyzer 인스턴스 생성
            missing_analyzer = MissingAnalyzer()
            
            logger.info(f"Result: {self.result_id} - Starting missing analysis with MissingAnalyzer")
            
            # 누락 분석 실행
            missing_result = await missing_analyzer.analyze(self.space_id, user_inputs_for_missing)
            
            logger.info(f"Result: {self.result_id} - Missing analysis result structure: {type(missing_result)}")
            if isinstance(missing_result, dict):
                logger.info(f"Result: {self.result_id} - Missing analysis result keys: {list(missing_result.keys())}")
            
            # 누락 분석 결과 처리
            if isinstance(missing_result, dict):
                if missing_result.get("success", False):
                    missing_items = missing_result.get("missing_items", [])
                    self.missing_answers = missing_items if isinstance(missing_items, list) else []
                    logger.info(f"Result: {self.result_id} - Found {len(self.missing_answers)} missing items")
                else:
                    error_msg = missing_result.get('error', 'Unknown error')
                    logger.warning(f"Result: {self.result_id} - Missing analysis failed: {error_msg}")
                    self.missing_answers = []
            else:
                logger.warning(f"Result: {self.result_id} - Unexpected missing analysis result type: {type(missing_result)}")
                self.missing_answers = []
            
            logger.info(f"Result: {self.result_id} - Missing answer detection completed")
            
        except Exception as e:
            logger.error(f"Result: {self.result_id} - Error in missing answer detection: {str(e)}")
            logger.error(f"Result: {self.result_id} - Missing analysis exception details", exc_info=True)
            self.missing_answers = []

    async def save_results_to_db(self):
        """MongoDB에 결과 저장"""
        try:
            logger.info(f"Result: {self.result_id} - Starting save results to MongoDB")
            
            # origin_urls에서 URL 정보만 추출하여 저장
            origin_result_url = []
            for origin_url in self.origin_urls:
                origin_result_url.append({
                    "id": origin_url.get("id"),
                    "url": origin_url.get("url"),
                    "s3_key": origin_url.get("s3_key"),
                    "bucket": origin_url.get("bucket")
                })
            
            # MongoDB에 결과 저장
            result_data = self.mongodb.create_result(
                space_id=self.space_id,
                origin_result_url=origin_result_url,
                wrong_answers=self.wrong_answers,
                missing_answers=self.missing_answers
            )
            
            # 저장된 결과 ID를 로그에 기록
            logger.info(f"Result: {self.result_id} - Results saved to MongoDB successfully. DB Result ID: {result_data.get('_id')}")
            
            if result_data.get('_id'):
                self.db_result_id = str(result_data['_id'])
                logger.info(f"Result: {self.result_id} - MongoDB document created with ID: {result_data['_id']}")
            
        except Exception as e:
            logger.error(f"Result: {self.result_id} - Error saving results to MongoDB: {str(e)}")
            raise Exception(f"MongoDB 결과 저장 중 오류 발생: {str(e)}")

    def build_response(self) -> Dict[str, Any]:
        """최종 응답 데이터 구성"""
        return {
            "success": True,
            "response": {
                "resultId": self.result_id
            },
            "error": None
        }