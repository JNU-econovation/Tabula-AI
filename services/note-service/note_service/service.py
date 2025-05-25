import json
import shutil
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from note_sdk.parsing.api_parsing import ParseConfig, parse_document
from note_sdk.keyword.api_keyword import KeywordGuide
from note_sdk.image_processor.api_image_processor import ImageSummary
from note_sdk.vector_store.api_vector_store import VectorLoader
from common_sdk.sse import update_progress
from note_sdk.config import settings
from common_sdk.get_logger import get_logger

logger = get_logger()

# 학습 자료 업로드 클래스
class NoteService:
    def __init__(self, pdf: str, language: str, domain_type: str, task_id: str):
        self.pdf = pdf # PDF 파일 경로
        self.language = language # 언어 설정
        self.domain_type = domain_type # 도메인 타입
        self.task_id = task_id # 작업 아이디
        self.document_id = Path(pdf).stem # 파일명 추출(확장자 제외): 적재할 경우 중복되지 않도록 설정 필요
        self.task_dir = settings.get_task_dir(task_id) # 작업 디렉토리 경로
        
        # 결과 디렉토리 구조
        self.md_dir = self.task_dir / "md"
        self.image_dir = self.task_dir / "images"
        
        # 기능 수행 클래스 초기화
        self.keyword_guide = KeywordGuide(domain=domain_type, task_id=task_id)
        self.image_summary = ImageSummary(task_id=task_id)
        self.vector_loader = VectorLoader(language=language, task_id=task_id)
        
        # 처리 결과 저장
        self._keyword_result: Optional[Dict] = None
        self._current_progress: int = 0

    async def process_document(self) -> Dict[str, Any]:
        try:
            total_start_time = datetime.now()
            logger.info(f"Start Document Processing: {self.document_id}")
            
            # 1. PDF 파싱 (30%)
            logger.info("=== Starting PDF Parsing ===")
            self._current_progress = 30
            update_progress(self.task_id, self._current_progress)
            parsing_result = await self.parse_pdf()
            logger.info("PDF Parsing completed")

            # 파싱 후 딜레이 (2초)
            await asyncio.sleep(2)
            
            # 2. 파싱 결과를 기반으로 비동기 작업 시작
            logger.info("=== Starting Parallel Processing ===")
            tasks = [
                self.process_markdown(),  # 마크다운 처리 및 벡터 DB 적재
                self.generate_keywords(),  # 키워드 생성
                self.process_images_background()  # 이미지 처리(Background)
            ]
            
            # 마크다운 처리 / 키워드 생성은 SSE 포함
            markdown_result, keyword_result, _ = await asyncio.gather(*tasks)
            self._keyword_result = keyword_result

            # 3. 완료 (100%)
            logger.info("=== Processing Completed ===")
            self._current_progress = 100
            update_progress(self.task_id, self._current_progress)

            # 4. 작업 디렉토리 정리
            self.cleanup_task_directory(self.task_dir)
            
            return {
                "document_id": self.document_id,
                "task_id": self.task_id,
                "keyword_result": self._keyword_result,
                "processing_time": (datetime.now() - total_start_time).total_seconds()
            }
            
        except Exception as e:
            logger.error(f"Error occurred during document processing: {str(e)}")
            self._current_progress = -1
            update_progress(self.task_id, self._current_progress)
            raise

    # 진행률 조회 API 응답 형식
    def get_progress(self) -> Dict[str, Any]:
        try:
            current_progress = self.get_current_progress()  # 현재 진행률 조회
            
            if current_progress == 100:
                # 완료된 경우 키워드 결과 포함
                return {
                    "success": True,
                    "response": {
                        "progress": 100,
                        "spaceId": self.task_id,
                        "spaceName": self.document_id,
                        "keywords": self.get_keyword_result()  # 키워드 결과 조회
                    },
                    "error": None
                }
            else:
                # 진행 중인 경우
                return {
                    "success": True,
                    "response": {
                        "progress": current_progress,
                        "status": "processing"
                    },
                    "error": None
                }
                
        except Exception as e:
            return {
                "success": False,
                "response": None,
                "error": str(e)
            }

    def get_current_progress(self) -> int:
        """현재 진행률 반환"""
        return self._current_progress

    def get_keyword_result(self) -> Optional[Dict]:
        """키워드 결과 반환"""
        return self._keyword_result

    # 작업 완료 후 디렉토리 정리: common-sdk로 이전 예정
    def cleanup_task_directory(self, task_dir: Path):
        try:
            if task_dir.exists():
                shutil.rmtree(task_dir)
                logger.info(f"Cleaned up task directory: {task_dir}")
        except Exception as e:
            logger.error(f"Error cleaning up task directory: {str(e)}")

    # PDF 파싱 로직
    async def parse_pdf(self):
        try:
            config = ParseConfig(
                language=self.language,
                domain_type=self.domain_type,
                output_dir=str(self.task_dir),
                task_id=self.task_id
            )
            parsed_data = parse_document(self.pdf, config)
            if not parsed_data:
                raise Exception(f"PDF Parsing Failed: {self.pdf}")
            return parsed_data
        except Exception as e:
            logger.error(f"PDF Parsing Error: {str(e)}")
            raise

    # 마크다운 처리 로직
    async def process_markdown(self):
        try:
            md_path = self.md_dir / f"{self.document_id}.md"
            if not md_path.exists():
                raise FileNotFoundError(f"Markdown file not found: {md_path}")
            
            with open(md_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 벡터 DB 적재(60%)
            update_progress(self.task_id, 60)
            success = await self.vector_loader.process_markdown(content, self.document_id)
            if not success:
                raise Exception("Markdown Vector Database Loading Failed")
            
            # 벡터 DB 적재 후 딜레이 (2초)
            await asyncio.sleep(2)
            
            return success
        except Exception as e:
            logger.error(f"Markdown Processing Error: {str(e)}")
            raise

    # 키워드 생성 로직
    async def generate_keywords(self):
        try:
            md_path = self.md_dir / f"{self.document_id}.md"
            keyword_result = self.keyword_guide.generate_mindmap_from_markdown(str(md_path))
            
            if keyword_result:
                
                keyword_path = settings.get_keyword_path(self.task_id, self.document_id)
                keyword_path.parent.mkdir(exist_ok=True)
                
                with open(keyword_path, 'w', encoding='utf-8') as f:
                    json.dump(keyword_result, f, indent=2, ensure_ascii=False)
            
            # 키워드 생성 완료(90%)
            update_progress(self.task_id, 90)
            
            # 키워드 생성 후 딜레이 (2초)
            await asyncio.sleep(2)

            return keyword_result
        except Exception as e:
            logger.error(f"Keyword Generation Error: {str(e)}")
            raise

    # 이미지 처리 로직
    async def process_images_background(self):
        try:
            md_path = self.md_dir / f"{self.document_id}.md"
            with open(md_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            image_result = await self.image_summary.process_markdown(content, self.document_id)
            
            if image_result and "objects" in image_result:
                # 이미지 처리 전 딜레이 (2초)
                await asyncio.sleep(2)

                await self.vector_loader.process_images(image_result["objects"], self.document_id)

                # 이미지 처리 후 딜레이 (2초)
                await asyncio.sleep(2)
                
            logger.info(f"Background image processing completed for {self.document_id}")
            return True
        except Exception as e:
            logger.error(f"Background Image Processing Error: {str(e)}")
            return False