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
from common_sdk.utils import num_tokens_from_string
from note_sdk.config import settings
from common_sdk.exceptions import FileNotFoundError, TokenExceeded
from common_sdk.get_logger import get_logger
from common_sdk.constants import ProgressPhase, StatusMessage

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
        
        # 진행률 단계 정의
        self.progress_stages = {
            'init': 0,
            'temp_dir': 10,
            'pdf_parsing': 30,
            'md_reading': 35,
            'vector_start': 40,
            'vector_progress': 45,
            'vector_complete': 50,
            'keyword_start': 60,
            'keyword_progress': 70,
            'keyword_save': 75,
            'db_start': 90,
            'db_complete': 95,
            'final': 100
        }

    # 진행률 업데이트 관리
    async def update_progress(self, stage: str, status: str, result: Dict = None):
        if stage in self.progress_stages:
            progress = self.progress_stages[stage]
            # 진행률이 증가하는 경우에만 업데이트
            if progress > self.current_progress:  
                self.current_progress = progress
                update_progress(self.space_id, self.current_progress, {
                    "status": status,
                    "result": result or {"spaceId": self.space_id}
                })
                # 진행률 표시를 위한 딜레이
                await asyncio.sleep(0.5)
                

    async def process_document(self) -> Dict[str, Any]:
        try:
            logger.info(f"User: {self.user_id} Start Document Processing: {self.space_id}")
            
            # 1. PDF 파싱 (0% - 30%)
            await self.update_progress('init', StatusMessage.PDF_PARSING)
            await self.update_progress('temp_dir', "임시 디렉토리 생성")
            
            await self.parse_pdf()
            await self.update_progress('pdf_parsing', f"{ProgressPhase.PDF_PARSING} 완료")
            logger.info(f"[NoteService] User: {self.user_id} - PDF Parsing completed")
            
            # 2. 마크다운 처리 및 벡터DB 적재 (30% - 60%)
            await self.update_progress('md_reading', StatusMessage.MARKDOWN_PROCESSING)
            await self.update_progress('vector_start', "벡터 DB 적재 시작")
            await self.update_progress('vector_progress', "벡터 DB 적재 진행 중")
            
            await self.process_markdown()
            
            await self.update_progress('vector_complete', "벡터 DB 적재 완료")
            
            # 3. 키워드 생성 (60% - 90%)
            await self.update_progress('keyword_start', StatusMessage.KEYWORD_GENERATION)
            await self.update_progress('keyword_progress', "키워드 생성 진행 중")
            
            keyword_result = await self.generate_keywords()
            self.keyword_result = keyword_result
            
            await self.update_progress('keyword_save', "키워드 저장")
            
            # 키워드 결과 추출
            space_name = keyword_result.get("spaceName", self.document_id)
            keywords = keyword_result.get("mindmap", [])

            # 4. MongoDB에 공간 저장 (90% - 100%)
            await self.update_progress('db_start', StatusMessage.DB_STORAGE)
            
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

            await self.update_progress('db_complete', "MongoDB 저장 완료")
            await self.update_progress('final', StatusMessage.COMPLETE, {
                "spaceId": self.space_id,
                "spaceName": space_name
            })
            
            # 5. 백그라운드 작업 시작 (SSE 종료 후)
            logger.info(f"[NoteService] User: {self.user_id} - Starting Background Tasks")
            
            try:
                await self.process_images_background()
                self.cleanup_task_directory(self.space_dir)
                logger.info(f"[NoteService] User: {self.user_id} - Finished Background Tasks")
                
            except Exception as e:
                logger.error(f"[NoteService] User: {self.user_id} - Error in background tasks: {str(e)}")
                self.cleanup_task_directory(self.space_dir)
                raise
            
        except Exception as e:
            logger.error(f"[NoteService] User: {self.user_id} - Error occurred during document processing: {str(e)}")
            self.current_progress = -1
            update_progress(self.space_id, self.current_progress, {
                "status": f"{StatusMessage.ERROR}: {str(e)}",
                "result": {"spaceId": self.space_id}
            })
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
            
        # 토큰 수 체크
        token_count = num_tokens_from_string(content)
        if token_count > settings.MAX_TEXT_TOKEN:
            logger.error(f"[NoteService] User: {self.user_id} - Text token exceeded: {token_count} > {settings.MAX_TEXT_TOKEN}")
            raise TokenExceeded()
        
        success = await self.vector_loader.process_markdown(content, self.document_id)
        return success

    # 키워드 생성 로직
    async def generate_keywords(self):
        md_path = self.md_dir / f"{self.document_id}.md"
        
        with open(md_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 토큰 수 체크
        token_count = num_tokens_from_string(content)
        if token_count > settings.MAX_TEXT_TOKEN:
            logger.error(f"[NoteService] User: {self.user_id} - Token count exceeded: {token_count} > {settings.MAX_TEXT_TOKEN}")
            raise TokenExceeded()
            
        keyword_result = self.keyword_guide.generate_mindmap_from_markdown(str(md_path))
        
        if keyword_result:
            keyword_path = settings.get_keyword_path(self.space_id, self.document_id)
            keyword_path.parent.mkdir(exist_ok=True)
            
            with open(keyword_path, 'w', encoding='utf-8') as f:
                json.dump(keyword_result, f, indent=2, ensure_ascii=False)
        
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