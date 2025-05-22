import asyncio
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
import json

from layoutparse.parsing import ParseConfig, parse_document
from note_sdk.keyword.api_keyword import KeywordGuide
from note_sdk.image_processor.api_image_processor import ObjectSummary
from note_sdk.vector_store.api_vector_store import VectorLoader

from logs.get_logger import get_logger

# 로그 설정
logger = get_logger()

# 문서 처리 클래스
class DocumentProcessor:
    def __init__(self, pdf: str, language: str, domain_type: str):
        """
        Args:
            pdf: PDF 파일(Multi-part)
            language: 언어 설정 (ko / en)
            domain_type: PDF 도메인 타입
        
        사용자에게 제공받는 값
        """
        self.pdf = pdf
        self.language = language
        self.domain_type = domain_type
        self.document_id = Path(pdf).stem
        self.result_dir = Path("result")
        
        # 결과 디렉토리 구조
        self.md_dir = self.result_dir / "md"
        self.image_dir = self.result_dir / "images" / self.document_id
        
        # 기능 수행 클래스 초기화
        self.keyword_guide = KeywordGuide(domain=domain_type)
        self.object_summary = ObjectSummary()
        self.vector_loader = VectorLoader(language=language)

    # 문서 처리 전체 프로세스
    async def process_document(self) -> Dict[str, Any]:
        try:
            total_start_time = datetime.now()
            logger.info(f"Start Document Processing: {self.document_id}")
            
            # 1. PDF 파싱 및 결과물 저장
            parsing_start_time = datetime.now()
            logger.info("1. PDF Parsing Start")
            config = ParseConfig(
                language=self.language,
                domain_type=self.domain_type,
                output_dir=str(self.result_dir)
            )
            parsed_data = parse_document(self.pdf, config)
            if not parsed_data:
                logger.error("PDF Parsing Failed")
                raise Exception()
            parsing_time = (datetime.now() - parsing_start_time).total_seconds()
            logger.info(f"PDF Parsing Completed({parsing_time:.2f}sec)")
            
            # 2. 마크다운 파일 경로 확인
            md_path = self.md_dir / f"{self.document_id}.md"
            if not md_path.exists():
                logger.error(f"Markdown file not found: {md_path}")
                raise FileNotFoundError()
            logger.info(f"Markdown file path: {md_path}")

            # 3. 순차 처리 시작
            logger.info("3. Sequential Processing Start")
            
            # 3-1. 마크다운 벡터 DB 적재
            markdown_start_time = datetime.now()
            logger.info("3-1. Vector Database Loading Start to Text")
            with open(md_path, 'r', encoding='utf-8') as f:
                content = f.read()
            success = await self.vector_loader.process_markdown(content, self.document_id)
            if not success:
                logger.error("Markdown Vector Database Loading Failed")
                raise Exception()
            markdown_time = (datetime.now() - markdown_start_time).total_seconds()
            logger.info(f"Markdown Vector Database Loading Completed ({markdown_time:.2f}sec)")
            
            # 처리 간 딜레이 (2초)
            await asyncio.sleep(2)
            
            # 3-2. 키워드 생성
            keyword_start_time = datetime.now()
            logger.info("3-2. Keyword Generation Start")
            keyword_result = self.keyword_guide.generate_mindmap_from_markdown(str(md_path))
            keyword_time = (datetime.now() - keyword_start_time).total_seconds()
            logger.info(f"Keyword Generation Completed ({keyword_time:.2f}sec)")

            # 키워드 결과 저장
            if keyword_result:
                keyword_dir = self.result_dir / "keyword"
                keyword_dir.mkdir(exist_ok=True)
                keyword_path = keyword_dir / f"{self.document_id}_mindmap.json"
                
                with open(keyword_path, 'w', encoding='utf-8') as f:
                    json.dump(keyword_result, f, indent=2, ensure_ascii=False)
                logger.info(f"Keyword result saved to {keyword_path}")
            
            # 처리 간 딜레이 (2초)
            await asyncio.sleep(2)
            
            # 3-3. 이미지 처리 및 벡터 DB 적재
            image_start_time = datetime.now()
            logger.info("3-3. Image Processing Start")
            with open(md_path, 'r', encoding='utf-8') as f:
                content = f.read()
            image_result = await self.object_summary.process_markdown(content, self.document_id)
            
            if image_result and "objects" in image_result:
                success = await self.vector_loader.process_images(image_result["objects"], self.document_id)
                if not success:
                    logger.error("Image Vector Database Loading Failed")
                    raise Exception()
            image_time = (datetime.now() - image_start_time).total_seconds()
            logger.info(f"Image Vector Database Loading Completed ({image_time:.2f}sec)")
            
            total_time = (datetime.now() - total_start_time).total_seconds()
            
            return {
                "document_id": self.document_id,
                "processing_time": total_time,
                "parsing_time": parsing_time,
                "markdown_processing_time": markdown_time,
                "keyword_generation_time": keyword_time,
                "image_processing_time": image_time,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error occurred during document processing: {str(e)}")
            return {
                "document_id": self.document_id,
                "status": "error",
                "error": str(e)
            }

async def main():
    # 테스트용 PDF 경로
    pdf = "/Users/kyeong6/Desktop/test/data/hadoop.pdf"
    
    # DocumentProcessor 인스턴스 생성
    processor = DocumentProcessor(
        pdf=pdf,
        language="ko",
        domain_type="Computer Science"
    )
    
    # 문서 처리 실행
    result = await processor.process_document()
    
    # 결과 출력
    if result["status"] == "success":
        print("\n=== 처리 시간 통계 ===")
        print(f"1. PDF 파싱: {result['parsing_time']:.2f}초")
        print(f"2. 마크다운 벡터 DB 적재: {result['markdown_processing_time']:.2f}초")
        print(f"3. 키워드 생성: {result['keyword_generation_time']:.2f}초")
        print(f"4. 이미지 처리: {result['image_processing_time']:.2f}초")
        print(f"총 처리 시간: {result['processing_time']:.2f}초")
    else:
        print(f"문서 처리 실패: {result['error']}")

if __name__ == "__main__":
    asyncio.run(main())
