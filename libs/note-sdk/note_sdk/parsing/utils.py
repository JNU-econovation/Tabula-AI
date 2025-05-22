from note_sdk.parsing.state import ParseState
from note_sdk.parsing.base import BaseNode
import pymupdf
import os
from PyPDF2 import PdfReader, PdfWriter

"""
유틸리티 함수 정의
"""
class SplitPDFFilesNode(BaseNode):

    def __init__(self, batch_size=10, test_page=None, **kwargs):
        super().__init__(**kwargs)
        self.name = "SplitPDFNode"
        self.batch_size = batch_size
        self.test_page = test_page

    def execute(self, state: ParseState) -> ParseState:
        """
        입력 PDF를 여러 개의 작은 PDF 파일로 분할합니다.

        :param state: GraphState 객체, PDF 파일 경로와 배치 크기 정보를 포함
        :return: 분할된 PDF 파일 경로 목록을 포함한 GraphState 객체
        """
        filepath = state["filepath"]
        document_id = os.path.splitext(os.path.basename(filepath))[0]
        
        # 임시 디렉토리 생성
        temp_dir = os.path.join("temp", document_id)
        os.makedirs(temp_dir, exist_ok=True)

        # PDF 파일 열기
        input_pdf = pymupdf.open(filepath)
        num_pages = len(input_pdf)
        self.log(f"파일의 전체 페이지 수: {num_pages} Pages.")

        if self.test_page is not None:
            if self.test_page < num_pages:
                num_pages = self.test_page

        ret = []
        # PDF 분할 작업 시작
        for start_page in range(0, num_pages, self.batch_size):
            # 배치의 마지막 페이지 계산 (전체 페이지 수를 초과하지 않도록)
            end_page = min(start_page + self.batch_size, num_pages) - 1

            # 분할된 PDF 파일명 생성 (임시 디렉토리에 저장)
            output_file = os.path.join(temp_dir, f"{document_id}_{start_page:04d}_{end_page:04d}.pdf")
            self.log(f"분할 PDF 생성: {output_file}")

            # 새로운 PDF 파일 생성 및 페이지 삽입
            with pymupdf.open() as output_pdf:
                output_pdf.insert_pdf(input_pdf, from_page=start_page, to_page=end_page)
                output_pdf.save(output_file)
                ret.append(output_file)

        # 원본 PDF 파일 닫기
        input_pdf.close()

        # filepath를 포함하여 반환
        return {
            "split_filepaths": ret,
            "filepath": filepath,  # 원본 filepath 유지
            "filetype": "pdf"      # filetype 추가
        }

    def _split_pdf(self, pdf_path, output_dir, max_pages):
        """PDF 파일을 페이지별로 분할하는 함수"""
        # load/pdf/{document_id} 디렉토리에 저장
        document_id = os.path.splitext(os.path.basename(pdf_path))[0]
        pdf_dir = os.path.join("load", "pdf", document_id)
        os.makedirs(pdf_dir, exist_ok=True)
        
        pdf = PdfReader(pdf_path)
        total_pages = len(pdf.pages)
        split_files = []
        
        for i in range(0, total_pages, max_pages):
            end_page = min(i + max_pages, total_pages)
            output_path = os.path.join(pdf_dir, f"{document_id}_pages_{i+1}-{end_page}.pdf")
            
            writer = PdfWriter()
            for page in range(i, end_page):
                writer.add_page(pdf.pages[page])
            
            with open(output_path, "wb") as f:
                writer.write(f)
            split_files.append(output_path)
            
        return split_files
