import os
import pymupdf

from PyPDF2 import PdfReader, PdfWriter

from note_sdk.parsing.state import ParseState
from note_sdk.parsing.base import BaseNode
from note_sdk.config import settings
from common_sdk.get_logger import get_logger

logger = get_logger()

"""
유틸리티 함수 정의
"""
class SplitPDFFilesNode(BaseNode):

    def __init__(self, batch_size=10, test_page=None, **kwargs):
        super().__init__(**kwargs)
        self.name = "SplitPDFNode"
        self.batch_size = batch_size
        self.test_page = test_page

    # 입력 PDF를 여러 개의 작은 PDF 파일로 분할
    def execute(self, state: ParseState) -> ParseState:
        task_id = state["task_id"]
        
        # 임시 디렉토리 생성
        temp_dir = settings.get_temp_dir(task_id)
        os.makedirs(temp_dir, exist_ok=True)

        # PDF 파일 경로 가져오기
        origin_dir = settings.get_origin_dir(task_id)
        pdf_files = list(origin_dir.glob("*.pdf"))
        
        if not pdf_files:
            raise FileNotFoundError(f"No PDF file found in origin directory: {origin_dir}")
        
        # 첫 번째 PDF 파일 사용
        pdf_path = pdf_files[0]
        logger.info(f"Using PDF file: {pdf_path}")

        # PDF 파일 열기
        input_pdf = pymupdf.open(str(pdf_path))
        num_pages = len(input_pdf)
        logger.info(f"File has {num_pages} pages")

        if self.test_page is not None:
            if self.test_page < num_pages:
                num_pages = self.test_page
                logger.info(f"Test mode: processing first {num_pages} pages")

        ret = []
        
        # PDF 분할 작업 시작
        for start_page in range(0, num_pages, self.batch_size):
            # 배치의 마지막 페이지 계산 (전체 페이지 수를 초과하지 않도록)
            end_page = min(start_page + self.batch_size, num_pages) - 1

            # 분할된 PDF 파일명 생성 (임시 디렉토리에 저장)
            output_file = os.path.join(temp_dir, f"{task_id}_{start_page:04d}_{end_page:04d}.pdf")
            logger.info(f"PDF split: {output_file}")

            # 새로운 PDF 파일 생성 및 페이지 삽입
            output_pdf = pymupdf.open()
            output_pdf.insert_pdf(input_pdf, from_page=start_page, to_page=end_page)
            output_pdf.save(output_file)
            output_pdf.close()
            
            # 파일이 실제로 생성되었는지 확인
            if os.path.exists(output_file):
                ret.append(output_file)
                logger.info(f"Successfully created split PDF: {output_file}")
            else:
                logger.error(f"Failed to create split PDF: {output_file}")

        # 원본 PDF 파일 닫기
        input_pdf.close()
        logger.info(f"PDF split completed: {len(ret)} files created")

        if not ret:
            raise Exception("No split PDF files were created")

        return {
            "split_filepaths": ret,
            "filepath": str(pdf_path),  # 원본 PDF 파일 경로 추가
            "filetype": "pdf",
            "temp_dir": temp_dir,
            "task_id": task_id
        }

    def split_pdf(self, pdf_path, output_dir, max_pages):
        """PDF 파일을 페이지별로 분할하는 함수"""
        # settings.get_temp_dir() 사용
        task_id = os.path.splitext(os.path.basename(pdf_path))[0]
        temp_dir = settings.get_temp_dir(task_id)
        os.makedirs(temp_dir, exist_ok=True)
        
        pdf = PdfReader(pdf_path)
        total_pages = len(pdf.pages)
        split_files = []
        
        for i in range(0, total_pages, max_pages):
            end_page = min(i + max_pages, total_pages)
            output_path = os.path.join(temp_dir, f"{task_id}_{i+1:04d}_{end_page:04d}.pdf")
            
            writer = PdfWriter()
            for page in range(i, end_page):
                writer.add_page(pdf.pages[page])
            
            with open(output_path, "wb") as f:
                writer.write(f)
            split_files.append(output_path)
            
        return split_files
