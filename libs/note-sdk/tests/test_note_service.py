import os
import asyncio
import json
from pathlib import Path
from datetime import datetime

from note_sdk.parsing.api_parsing import ParseConfig, parse_document
from note_sdk.keyword.api_keyword import KeywordGuide
from note_sdk.image_processor.api_image_processor import ImageSummary
from note_sdk.vector_store.api_vector_store import VectorLoader
from note_sdk.config import settings
from common_sdk.get_logger import get_logger

logger = get_logger()

class TestNoteService:
    def __init__(self, pdf: str, language: str, domain_type: str, task_id: str):
        self.pdf = pdf  # PDF 파일 경로
        self.language = language  # 언어 설정
        self.domain_type = domain_type  # 도메인 타입
        self.task_id = task_id  # 작업 아이디
        self.document_id = Path(pdf).stem  # 파일명 추출(확장자 제외)
        self.task_dir = settings.get_task_dir(task_id)  # 작업 디렉토리 경로
        
        # 결과 디렉토리 구조
        self.md_dir = self.task_dir / "md"
        self.image_dir = self.task_dir / "images"
        
        # 기능 수행 클래스 초기화
        self.keyword_guide = KeywordGuide(domain=domain_type, task_id=task_id)
        self.image_summary = ImageSummary(task_id=task_id)
        self.vector_loader = VectorLoader(language=language, task_id=task_id)

    async def process_document(self):
        try:
            total_start_time = datetime.now()
            logger.info(f"Start Document Processing: {self.document_id}")
            
            # 1. PDF 파싱
            logger.info("=== Starting PDF Parsing ===")
            parsing_result = await self.parse_pdf()
            logger.info("PDF Parsing completed")
            logger.info(f"Parsing result keys: {parsing_result.keys() if isinstance(parsing_result, dict) else 'Not a dictionary'}")
            
            # 파싱 후 딜레이 (2초)
            await asyncio.sleep(2)
            
            # 2. 파싱 결과를 기반으로 비동기 작업 시작
            logger.info("=== Starting Parallel Processing ===")
            tasks = [
                self.process_markdown(),  # 마크다운 처리 및 벡터 DB 적재
                self.generate_keywords(),  # 키워드 생성
                self.process_images_background()  # 이미지 처리(Background)
            ]
            
            # 마크다운 처리 / 키워드 생성 / 이미지 처리
            markdown_result, keyword_result, image_result = await asyncio.gather(*tasks)
            
            # 3. 완료
            logger.info("=== Processing Completed ===")
            
            # 결과 반환
            return {
                "document_id": self.document_id,
                "status": "success",
                "processing_time": (datetime.now() - total_start_time).total_seconds(),
                "results": {
                    "markdown": markdown_result,
                    "keyword": keyword_result,
                    "image": image_result
                }
            }
            
        except Exception as e:
            logger.error(f"Error occurred during document processing: {str(e)}")
            return {
                "document_id": self.document_id,
                "status": "error",
                "error": str(e)
            }

    async def parse_pdf(self):
        try:
            logger.info(f"Parsing PDF: {self.pdf}")
            
            # 테스트용 PDF 파일을 origin 디렉토리로 복사
            origin_dir = settings.get_origin_dir(self.task_id)
            origin_dir.mkdir(parents=True, exist_ok=True)
            
            # PDF 파일을 origin 디렉토리로 복사
            pdf_filename = os.path.basename(self.pdf)
            origin_pdf_path = origin_dir / pdf_filename
            with open(self.pdf, 'rb') as src, open(origin_pdf_path, 'wb') as dst:
                dst.write(src.read())
            
            # 파싱 설정
            config = ParseConfig(
                language=self.language,
                domain_type=self.domain_type,
                output_dir=str(self.task_dir),
                task_id=self.task_id
            )
            
            # origin 디렉토리의 PDF 파일로 파싱
            parsed_data = parse_document(str(origin_pdf_path), config)
            if not parsed_data:
                raise Exception(f"PDF Parsing Failed: {self.pdf}")
            return parsed_data
        except Exception as e:
            logger.error(f"PDF Parsing Error: {str(e)}")
            raise

    async def process_markdown(self):
        try:
            logger.info("Processing markdown...")
            md_path = self.md_dir / f"{self.document_id}.md"
            if not md_path.exists():
                raise FileNotFoundError(f"Markdown file not found: {md_path}")
            
            with open(md_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            success = await self.vector_loader.process_markdown(content, self.document_id)
            if not success:
                raise Exception("Markdown Vector Database Loading Failed")
            
            return success
        except Exception as e:
            logger.error(f"Markdown Processing Error: {str(e)}")
            raise

    async def generate_keywords(self):
        try:
            logger.info("Generating keywords...")
            md_path = self.md_dir / f"{self.document_id}.md"
            keyword_result = self.keyword_guide.generate_mindmap_from_markdown(str(md_path))
            
            if keyword_result:
                keyword_path = settings.get_keyword_path(self.task_id, self.document_id)
                keyword_path.parent.mkdir(exist_ok=True)
                
                with open(keyword_path, 'w', encoding='utf-8') as f:
                    json.dump(keyword_result, f, indent=2, ensure_ascii=False)
            
            return keyword_result
        except Exception as e:
            logger.error(f"Keyword Generation Error: {str(e)}")
            raise

    async def process_images_background(self):
        try:
            logger.info("Processing images...")
            md_path = self.md_dir / f"{self.document_id}.md"
            with open(md_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            image_result = await self.image_summary.process_markdown(content, self.document_id)
            
            if image_result and "objects" in image_result:
                await self.vector_loader.process_images(image_result["objects"], self.document_id)
                
            logger.info(f"Image processing completed for {self.document_id}")
            return image_result
        except Exception as e:
            logger.error(f"Background Image Processing Error: {str(e)}")
            return False

async def main():
    # 테스트 설정
    TEST_PDF = os.path.join(os.path.dirname(__file__), "data", "test.pdf")
    TEST_LANGUAGE = "ko"
    TEST_DOMAIN_TYPE = "computer science"
    TEST_TASK_ID = "test_task_001"

    # 서비스 인스턴스 생성
    service = TestNoteService(
        pdf=TEST_PDF,
        language=TEST_LANGUAGE,
        domain_type=TEST_DOMAIN_TYPE,
        task_id=TEST_TASK_ID
    )

    # 문서 처리 실행
    result = await service.process_document()
    
    # 결과 출력
    print("\n=== Processing Ended ===")
    # print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    # 테스트 실행
    asyncio.run(main()) 