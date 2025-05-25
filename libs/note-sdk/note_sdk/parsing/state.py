import operator
from typing import TypedDict, Annotated, List, Dict, Optional
from note_sdk.parsing.element import Element
from langchain_core.documents import Document

"""
파싱 상태 클래스 정의
"""
class ParseState(TypedDict):
    task_id: Annotated[str, "task_id"]  # 작업 ID
    filepath: Annotated[Optional[str], "filepath"]  # 원본 파일 경로 (선택적)
    filetype: Annotated[
        str, "filetype"
    ]  # 파일 타입(PDF, DOCX, PPTX, XLSX) / 현재 PDF만 지원
    split_filepaths: Annotated[List[str], "split_filepaths"]  # 분할한 파일 경로
    working_filepath: Annotated[str, "working_filepath"]  # 현재 작업중인 파일
    output_dir: Annotated[str, "output_dir"]  # 출력 디렉토리
    include_image_in_output: Annotated[
        bool, "include_image_in_output"
    ]  # 출력 디렉토리에 이미지 포함 여부

    metadata: Annotated[
        List[Dict], operator.add
    ]  # parsing metadata (api, model, usage)

    total_cost: Annotated[float, "total_cost"]  # 총 비용

    raw_elements: Annotated[List[Dict], operator.add]  # raw elements from Upstage
    elements_from_parser: Annotated[
        List[Dict], "elements_from_parser"
    ]  # elements after post-processing

    elements: Annotated[List[Element], "elements"]  # Final cleaned elements
    reconstructed_elements: Annotated[
        List[Dict], "reconstructed_elements"
    ]  # reconstructed elements

    export: Annotated[List, operator.add]  # export results

    texts_by_page: Annotated[Dict[int, str], "texts_by_page"]  # texts by page
    images_by_page: Annotated[
        Dict[int, List[Element]], "images_by_page"
    ]  # images by page

    extracted_image_entities: Annotated[
        List[Element], "extracted_image_entities"
    ]  # extracted image entities

    documents: Annotated[List[Document], "documents"]
