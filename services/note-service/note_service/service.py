import json
import shutil
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional

from note_sdk.parsing import ParseConfig, parse_document
from note_sdk.keyword import KeywordGuide
from note_sdk.image_processor import ImageSummary
from note_sdk.vector_store import VectorLoader
from common_sdk.sse import update_progress
from common_sdk.crud.mongodb import MongoDB
from note_sdk.config import settings
from common_sdk.exceptions import FileNotFoundError
from common_sdk.get_logger import get_logger
from common_sdk.constants import ProgressPhase, ProgressRange, StatusMessage

# 로거 설정
logger = get_logger()

# MongoDB 인스턴스
mongodb = MongoDB()

# 학습 자료 업로드 클래스
class NoteService:
    def __init__(self, pdf: str, folder_id: str, language: str, domain_type: str, space_id: str, s3_url: str, file_name: str, user_id: int):
        self.pdf = pdf # PDF 파일 경로
        self.folder_id = folder_id # 폴더 아이디
        self.language = language # 언어 설정
        self.domain_type = domain_type # 도메인 타입
        self.space_id = space_id # 작업 아이디
        self.s3_url = s3_url # S3 URL
        self.file_name = file_name # 파일명
        self.user_id = user_id # 사용자 아이디
        self.document_id = Path(pdf).stem # 파일명 추출(확장자 제외)
        self.space_dir = settings.get_space_dir(space_id) # 작업 디렉토리 경로
        
        # 결과 디렉토리 구조
        self.md_dir = self.space_dir / "md"
        self.image_dir = self.space_dir / "images"
        
        # 기능 수행 클래스 초기화
        self.keyword_guide = KeywordGuide(domain=domain_type, space_id=space_id)
        self.image_summary = ImageSummary(space_id=space_id, language=language)
        self.vector_loader = VectorLoader(language=language, space_id=space_id)
        
        # 처리 결과 저장
        self.keyword_result: Optional[Dict] = None
        self.current_progress: int = 0

    async def process_document(self) -> Dict[str, Any]:
        try:
            logger.info(f"User: {self.user_id} Start Document Processing: {self.space_id}")
            
            # 1. PDF 파싱 (0% - 30%)
            logger.info(f"[NoteService] User: {self.user_id} - Starting PDF Parsing")
            self.current_progress = ProgressRange.PDF_PARSING[0]
            update_progress(self.space_id, self.current_progress, {
                "status": StatusMessage.PDF_PARSING,
                "result": {"spaceId": self.space_id}
            })
            
            # 임시 디렉토리 생성 (10%)
            self.current_progress = 10
            update_progress(self.space_id, self.current_progress, {
                "status": "임시 디렉토리 생성",
                "result": {"spaceId": self.space_id}
            })
            
            await self.parse_pdf()
            self.current_progress = ProgressRange.PDF_PARSING[1]
            update_progress(self.space_id, self.current_progress, {
                "status": f"{ProgressPhase.PDF_PARSING} 완료",
                "result": {"spaceId": self.space_id}
            })
            logger.info(f"[NoteService] User: {self.user_id} - PDF Parsing completed")
            
            # 2. 마크다운 처리 및 벡터DB 적재 (30% - 60%)
            logger.info(f"[NoteService] User: {self.user_id} - Starting Markdown Processing")
            self.current_progress = ProgressRange.MARKDOWN_PROCESSING[0]
            update_progress(self.space_id, self.current_progress, {
                "status": StatusMessage.MARKDOWN_PROCESSING,
                "result": {"spaceId": self.space_id}
            })
            
            await self.process_markdown()
            
            self.current_progress = ProgressRange.MARKDOWN_PROCESSING[1]
            update_progress(self.space_id, self.current_progress, {
                "status": f"{ProgressPhase.MARKDOWN_PROCESSING} 완료",
                "result": {"spaceId": self.space_id}
            })
            
            # 3. 키워드 생성 (60% - 90%)
            logger.info(f"[NoteService] User: {self.user_id} - Starting Keyword Generation")
            self.current_progress = ProgressRange.KEYWORD_GENERATION[0]
            update_progress(self.space_id, self.current_progress, {
                "status": StatusMessage.KEYWORD_GENERATION,
                "result": {"spaceId": self.space_id}
            })
            
            keyword_result = await self.generate_keywords()
            self.keyword_result = keyword_result
            
            self.current_progress = ProgressRange.KEYWORD_GENERATION[1]
            update_progress(self.space_id, self.current_progress, {
                "status": f"{ProgressPhase.KEYWORD_GENERATION} 완료",
                "result": {"spaceId": self.space_id}
            })

            # 키워드 결과 추출
            space_name = keyword_result.get("spaceName", self.document_id)
            keywords = keyword_result.get("mindmap", [])

            # 4. MongoDB에 공간 저장 (90% - 100%)
            logger.info(f"[NoteService] User: {self.user_id} - Saving to MongoDB")
            self.current_progress = ProgressRange.DB_STORAGE[0]
            update_progress(self.space_id, self.current_progress, {
                "status": StatusMessage.DB_STORAGE,
                "result": {"spaceId": self.space_id}
            })
            
            mongodb.create_space(
                user_id=self.user_id,
                space_id=self.space_id,
                folder_id=self.folder_id,
                file_url=self.s3_url,
                file_name=self.file_name,
                lang_type=self.language,
                file_domain=self.domain_type,
                space_name=space_name,
                keywords=keywords
            )
            logger.info(f"[NoteService] User: {self.user_id} - MongoDB save Success")

            # 5. SSE 완료 (100%)
            logger.info(f"[NoteService] User: {self.user_id} - Processing Completed")
            self.current_progress = 100
            update_progress(self.space_id, self.current_progress, {
                "status": StatusMessage.COMPLETE,
                "result": {
                    "spaceId": self.space_id,
                    "spaceName": space_name
                }
            })
            
            # 6. 백그라운드 작업 시작 (SSE 종료 후)
            logger.info(f"[NoteService] User: {self.user_id} - Starting Background Tasks")
            
            try:
                # 6.1 이미지 처리
                logger.info(f"[NoteService] User: {self.user_id} - Starting Image Processing")
                await self.process_images_background()
                logger.info(f"[NoteService] User: {self.user_id} - Image Processing Completed")
                
                # 6.2 작업 디렉토리 정리 (모든 작업이 완료된 후)
                self.cleanup_task_directory(self.space_dir)
                logger.info(f"[NoteService] User: {self.user_id} - Finished Background Tasks")
                
            except Exception as e:
                logger.error(f"[NoteService] User: {self.user_id} - Error in background tasks: {str(e)}")
                # 백그라운드 작업 실패 시에도 디렉토리 정리 시도
                self.cleanup_task_directory(self.space_dir)
                raise
            
        except Exception as e:
            logger.error(f"[NoteService] User: {self.user_id} - Error occurred during document processing: {str(e)}")
            self.current_progress = -1
            update_progress(self.space_id, self.current_progress, {
                "status": f"{StatusMessage.ERROR}: {str(e)}",
                "result": {"spaceId": self.space_id}
            })
            # 에러 발생 시에도 디렉토리 정리 시도
            self.cleanup_task_directory(self.space_dir)
            raise

    # 작업 완료 후 디렉토리 정리: common-sdk로 이전 예정
    def cleanup_task_directory(self, space_dir: Path):
        try:
            if space_dir.exists():
                shutil.rmtree(space_dir)
                logger.info(f"[NoteService] User: {self.user_id} - Cleaned up task directory: {space_dir}")
        except Exception as e:
            logger.error(f"[NoteService] User: {self.user_id} - Error cleaning up task directory: {str(e)}")

    # PDF 파싱 로직
    async def parse_pdf(self):

        config = ParseConfig(
            language=self.language,
            domain_type=self.domain_type,
            output_dir=str(self.space_dir),
            space_id=self.space_id,
        )
        parsed_data = parse_document(self.pdf, config)

        return parsed_data

    # 마크다운 처리 로직
    async def process_markdown(self):
        
        md_path = self.md_dir / f"{self.document_id}.md"

        if not md_path.exists():
            logger.error(f"[NoteService] User: {self.user_id} - Markdown file not found: {md_path}")
            raise FileNotFoundError(md_path)
        
        with open(md_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 마크다운 처리 시작 (40%)
        self.current_progress = 40
        update_progress(self.space_id, self.current_progress, {"status": "마크다운 처리 시작"})
        
        # 벡터 DB 적재(50%)
        self.current_progress = 50
        update_progress(self.space_id, self.current_progress, {"status": "벡터 DB 적재 중"})
        
        success = await self.vector_loader.process_markdown(content, self.document_id)
        
        # 벡터 DB 적재 완료 (60%)
        self.current_progress = 60
        update_progress(self.space_id, self.current_progress, {"status": "벡터 DB 적재 완료"})
        
        # 벡터 DB 적재 후 딜레이 (2초)
        await asyncio.sleep(2)
        
        return success

    # 키워드 생성 로직
    async def generate_keywords(self):
        md_path = self.md_dir / f"{self.document_id}.md"
        
        # 키워드 생성 시작 (70%)
        self.current_progress = 70
        update_progress(self.space_id, self.current_progress, {"status": "키워드 생성 시작"})
        
        keyword_result = self.keyword_guide.generate_mindmap_from_markdown(str(md_path))
        
        if keyword_result:
            # 키워드 저장 (80%)
            self.current_progress = 80
            update_progress(self.space_id, self.current_progress, {"status": "키워드 저장 중"})
            
            keyword_path = settings.get_keyword_path(self.space_id, self.document_id)
            keyword_path.parent.mkdir(exist_ok=True)
            
            with open(keyword_path, 'w', encoding='utf-8') as f:
                json.dump(keyword_result, f, indent=2, ensure_ascii=False)
        
        # 키워드 생성 완료(90%)
        self.current_progress = 90
        update_progress(self.space_id, self.current_progress, {"status": "키워드 생성 완료"})
        
        # 키워드 생성 후 딜레이 (2초)
        await asyncio.sleep(2)

        return keyword_result

    # 이미지 처리 로직
    async def process_images_background(self):
        try:
            md_path = self.md_dir / f"{self.document_id}.md"
            with open(md_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            image_result = await self.image_summary.process_markdown(content, self.document_id)
            
            # 이미지 처리 완료
            logger.info(f"[NoteService] User: {self.user_id} - Image Processing Completed")
            return image_result
            
        except Exception as e:
            logger.error(f"[NoteService] User: {self.user_id} - Background Image Processing Error: {str(e)}")
            return False